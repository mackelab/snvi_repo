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
mcmc_posterior = inf.build_posterior(sample_with="mcmc", mcmc_method= "slice_np_vectorized",mcmc_parameters={"num_chains": 100, "thin": 100, "warmup_steps": 100
    ,"init_strategy": "sir"
    ,"sir_batch_size": 1000
    ,"sir_num_batches": 100})
set_surrogate_likelihood(mcmc_posterior, classifier)
mcmc_posterior.set_default_x(task.get_observation(2))

start_time = time.time()
mcmc_samples = mcmc_posterior.sample((10000,))
end_time = time.time()

print(end_time-start_time)