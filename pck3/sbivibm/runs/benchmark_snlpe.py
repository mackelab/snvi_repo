import torch
import copy
import time
import numpy as np

from sbi import inference as inference

import pandas as pd
from sbi.utils import BoxUniform

import logging
import math
from typing import Any, Dict, Optional, Tuple

from sbibm.tasks.task import Task

import sbivibm.algorithms.snlpe.spa as spa

def automatic_transform(task):
    prior = task.get_prior_dist()
    support = prior.support        
    transform = torch.distributions.biject_to(support).inv
    return transform

def wrap_prior(prior, transform):
    new_prior = torch.distributions.TransformedDistribution(prior, transform)
    return new_prior 

def wrap_simulator(simulator, transform):
    new_simulator = lambda x: simulator(transform.inv(x))
    return new_simulator



def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    automatic_transforms_enabled: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Runs (S)NLE from `sbi`
Args:
    task: Task instance
    num_observation: Observation number to load, alternative to `observation`
    observation: Observation, alternative to `num_observation`
    num_samples: Number of samples to generate from posterior
    num_simulations: Simulation budget
    num_rounds: Number of rounds
    neural_net: Neural network to use, one of maf / mdn / made / nsf
    hidden_features: Number of hidden features in network
    simulation_batch_size: Batch size for simulator
    training_batch_size: Batch size for training network
    automatic_transforms_enabled: Whether to enable automatic transforms
    mcmc_method: MCMC method
    mcmc_parameters: MCMC parameters
    z_score_x: Whether to z-score x
    z_score_theta: Whether to z-score theta
Returns:
    Samples from posterior, number of simulator calls, log probability of true params if computable
"""
    if task.name == "two_moons":
        import sbivibm.algorithms.snlpe.two_moons.functions as func
        seed = 10
        dim = 2
        x_dim = 2
        automatic_transforms_enabled = False
    elif task.name == "lotka_volterra":
        import algorithms.snlpe.lotka_volterra.functions as func
        raise NotImplementedError("This task does not works ...")
        seed = 10
        dim = 4
        x_dim = 20
        automatic_transforms_enabled = True
    else:
        NotImplementedError("This task was not implemented with SNLPE")

    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)

    if num_rounds == 1:
        log.info(f"Running NLPE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNLPE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)

    x_o = task.get_observation(num_observation)

    prior = task.get_prior_dist()
    #prior = BoxUniform(low=-2 * torch.ones(2), high=2 * torch.ones(2))
    if observation is None:
        observation = task.get_observation(num_observation)
    simulator = task.get_simulator()

    transforms = automatic_transform(task)
    if automatic_transforms_enabled:
        prior = wrap_prior(prior, transforms)
        simulator = wrap_simulator(simulator, transforms)

    flow_lik, flow_post = func.set_up_networks(seed)
    theta = prior.sample((1,))
    x = simulator(theta)

    optimizer_lik = torch.optim.Adam(flow_lik.parameters())
    optimizer_post = torch.optim.Adam(flow_post.parameters())
    decay_rate_post = 0.95

    # test prior pred sampling and sampling for given that

    nbr_rounds = num_rounds
    prob_prior_decay_rate = 0.8
    prob_prior = spa.calc_prob_prior(nbr_rounds, prob_prior_decay_rate)

    print(prob_prior_decay_rate)

    nbr_lik = [num_simulations_per_round for i in range(num_rounds)]
    nbr_epochs_lik = [200]
    if num_rounds > 1:
        nbr_epochs_lik += [50]*(num_rounds-1)
    batch_size = 2000
    batch_size_post = 2000
    nbr_post = [40000]*num_rounds
    nbr_epochs_post = [50]*num_rounds

    x_o_batch_post = torch.zeros(batch_size_post, x_dim)

    for i in range(batch_size_post):
        x_o_batch_post[i, :] = x_o

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start = time.time()

    # TODO check prior and model sim
    models_lik, models_post = spa.inference_spa(flow_lik,
                                                flow_post,
                                                prior,
                                                simulator,
                                                optimizer_lik,
                                                optimizer_post,
                                                decay_rate_post,
                                                x_o.reshape(1, x_dim),
                                                x_o_batch_post,
                                                dim,
                                                prob_prior,
                                                nbr_lik,
                                                nbr_epochs_lik,
                                                nbr_post,
                                                nbr_epochs_post,
                                                batch_size,
                                                batch_size_post)

    end = time.time()
    run_time = end - start
    print(run_time)

    samples = models_post[-1].sample(num_samples, context=x_o.reshape(1, x_dim)).reshape((num_samples, 2)).detach()

    return models_post, samples, num_simulations