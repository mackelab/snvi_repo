from sbibm.algorithms.sbi.utils import wrap_simulator_fn

import torch 
from sbibm.tasks.task import Task
from torch.distributions import Transform, Distribution

def automatic_transform(task:Task):
    """Gives a transform that bijects from the support of the prior to the real (unconstrainted) domain

    Args:
        task (Task): Sbibm task

    Returns:
        Transform: PyTorch transform that bijects prior.support -> real
    """
    prior = task.get_prior_dist()
    support = prior.support        
    transform = torch.distributions.biject_to(support).inv
    return transform

def wrap_prior(prior: Distribution, transform: Transform):
    """Wraps the prior distribution.

    Args:
        prior (Distribution): PyTorch Distributions object
        transform (Transform): PyTorch Transform object

    Returns:
        Distribution: Wrapped prior distribution
    """
    new_prior = torch.distributions.TransformedDistribution(prior, transform)
    return new_prior 


class LikelihoodWrapper:
    def __init__(self, flow, transform):
        self.flow = flow
        self.transform = transform

    def sample(self, *args, **kwargs):
        Y = self.flow.sample(*args, **kwargs)
        return self.transform.inv(Y)

    def log_prob(self, x, theta, *args,**kwargs):
        theta_unconstrained = self.transform(theta)
        log_probs = self.flow.log_prob(x,context=theta_unconstrained,*args, **kwargs)
        dets = self.transform.log_abs_det_jacobian(
            theta, theta_unconstrained
        )
        if dets.ndim > 1 and dets.shape[1] == theta.shape[-1]:
            dets = dets.sum(-1)
        log_probs += dets
        return log_probs 

    def eval(self,*args):
        self.flow.eval(*args)

    def train(self, *args):
        self.flow.train(*args)

    def parameters(self):
        return self.flow.parameters()

    def __call__(self, x):
        theta = x[0]
        theta_unconstrained = self.transform(theta)
        new_x = x
        new_x[0] = theta_unconstrained
        log_probs = self.flow(new_x).squeeze()
        dets = self.transform.log_abs_det_jacobian(
            theta, theta_unconstrained
        )
        if dets.ndim > 1 and dets.shape[1] == theta.shape[-1]:
            dets = dets.sum(-1)
        log_probs += dets
        
        return log_probs


def wrap_posterior(posterior, transform):
    """Transforms the posterior distribution with the given transform.

    Args:
        posterior (Distribution): Pytorch distribution object
        transform (Transform): Pytorch transform object

    Returns:
        posterior: Variational posterior
    """    
    new_q = torch.distributions.TransformedDistribution(posterior._q, transform.inv)
    posterior._q = new_q 
    posterior.net = LikelihoodWrapper(posterior.net, transform)
    return posterior