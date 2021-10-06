import pickle
from sbivibm.tasks import Pyloric
import time
import torch
from sbivibm.runs.filter import set_surrogate_likelihood

task = Pyloric()

with open("../../inf_pyloric_with_classifier_REALLY_FINAL.pkl", "rb") as handle:
    inf = pickle.load(handle)
inf._summary_writer = inf._default_summary_writer()
classifier = torch.load("../../inf_pyloric_with_classifier_REALLY_FINAL_classifier.pkl")

mcmc_posterior = inf.build_posterior(sample_with="vi", vi_parameters={"flow":"affine_autoregressive"})
mcmc_posterior.set_default_x(task.get_observation(2))
set_surrogate_likelihood(mcmc_posterior, classifier)
print(mcmc_posterior)

start_time = time.time()
mcmc_posterior.train(loss="forward_kl", n_particles=1000, max_num_iters=2000, min_num_iters=500)
mcmc_samples = mcmc_posterior.sample((10000,))
end_time = time.time()

print(end_time-start_time)