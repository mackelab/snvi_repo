import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import likelihood_nn
import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple, List

from sbibm.tasks.task import Task
from sbibm.algorithms.sbi.utils import wrap_simulator_fn


from sbivibm.utils import wrap_posterior, wrap_prior, automatic_transform
from .filter import get_filter, build_classifier, train_classifier, init_classification_data, append_new_classification_data, set_surrogate_likelihood

import pickle
import os





def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    num_simulations_list: List[int]=None,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    neural_net: str = "maf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 1000,
    automatic_transforms_enabled: bool = True,
    z_score_x: bool = True,
    z_score_theta: bool = True,
    vi_parameters={"flow": "spline_autoregressive", "num_flows":5, "loss":"elbo", "learning_rate": 1e-3, "max_num_iters":2000, "check_for_convergence":True, "show_progress_bars":True, "method":"naive"},
    simulation_filter: str = "identity",
    cache_inf: Optional[str] = None,
    **kwargs,
) -> Tuple[list,torch.Tensor, int]:
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
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    log = logging.getLogger(__name__)
    if num_rounds == 1:
        log.info(f"Running NLE")
        num_simulations_per_round = num_simulations
    elif num_simulations_list is not None and num_simulations_list != "None":
        log.info(f"Running SNLE with non uniform simulations sizes")
        num_simulations_list = list(num_simulations_list)
        num_simulations_per_round = min(num_simulations_list)
        assert sum(num_simulations_list) == num_simulations
        assert len(num_simulations_list) == num_rounds
    else:
        log.info(f"Running SNLE")
        num_simulations_per_round = math.floor(num_simulations / num_rounds)


    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warn("Reduced simulation_batch_size to num_simulation_per_round")

    if training_batch_size > num_simulations_per_round:
        training_batch_size = num_simulations_per_round
        if isinstance(samples_to_accept, List) and isinstance(samples_to_accept[0], int) and training_batch_size < samples_to_accept[0]:
            training_batch_size = samples_to_accept[0]
            
        log.warn("Reduced training_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)
    simulator = task.get_simulator(max_calls=num_simulations)

    sim_filter = get_filter(simulation_filter)
    if simulation_filter != "identity":
        theta = prior.sample((1,))
        classifier = build_classifier(theta.shape[-1])


    transforms = automatic_transform(task)
    if automatic_transforms_enabled:
        prior = wrap_prior(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)


    density_estimator_fun = likelihood_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )
    inference_method = inference.SNLE(
        density_estimator=density_estimator_fun, prior=prior,
    )

    proposal = prior
    posteriors = []
    for r in range(num_rounds):

        if num_simulations_list is not None and num_simulations_list != "None":
            num_simulations_per_round = num_simulations_list[r]

        if task.name == "pyloric" and r==0:
            theta, x = task.get_precomputed_dataset()
            theta = theta[:num_simulations_per_round]
            x = x[:num_simulations_per_round]
        else:
            if r==0:
                theta = proposal.sample((num_simulations_per_round,))
            else:
                theta = proposal.sample((num_simulations_per_round,), vi_parameters=vi_parameters)[:num_simulations_per_round]
            log.info(f"Simulating {num_simulations_per_round} samples")
            x = simulator(theta)

        
        idx = sim_filter(theta, x, observation)
        log.info(f"Filtered out {idx.sum()} values")
        if simulation_filter != "identity":
            if r==0:
                classification_data = init_classification_data(theta, idx)
            else:
                classification_data = append_new_classification_data(classification_data, theta, idx)

            classifier = train_classifier(classifier, classification_data,epochs=200)


        density_estimator = inference_method.append_simulations(
            theta[idx], x[idx], from_round=r
        ).train(
            training_batch_size=training_batch_size,
            retrain_from_scratch_each_round=False,
            discard_prior_samples=False,
            show_train_summary=True,
        )
        # Currently direct handling with density estimators leads to error
        posterior = inference_method.build_posterior(
            sample_with="vi",vi_parameters=vi_parameters
        )
        posterior = posterior.set_default_x(observation)
        if r > 0:
            posterior.copy_hyperparameters_from(posteriors[-1])
        if simulation_filter != "identity":
            set_surrogate_likelihood(posterior, classifier)

        posterior.train(**vi_parameters)
        proposal = posterior.set_default_x(observation)
        posteriors.append(posterior)

        if cache_inf is not None:
            # Save inference object...
            inference_method._summary_writer = None
            inference_method._build_neural_net = None
            save = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            with open(save + os.sep + str(cache_inf) + ".pkl", "wb") as handle:
                print(save + os.sep + str(cache_inf) + ".pkl")
                pickle.dump(inference_method, handle)
            torch.save(posterior, save + os.sep + str(cache_inf) + "_posterior.pkl")
            if simulation_filter != "identity":
                torch.save(classifier, save + os.sep + str(cache_inf) + "_classifier.pkl")
            inference_method._summary_writer = inference_method._default_summary_writer()




    if automatic_transforms_enabled:
        for post in posteriors:
            post = wrap_posterior(post, transforms)


    samples = posteriors[-1].sample((num_samples,), vi_parameters=vi_parameters).detach()[:num_samples]

    return posteriors, samples, num_simulations


            
            

		

