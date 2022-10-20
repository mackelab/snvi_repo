from sbivibm.utils import get_posterior_by_id, get_full_dataset
import torch

import random 
import numpy as np

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print("Script produces 94% valid slurm (RERUNED)")

posterior = get_posterior_by_id("benchmark", "18fd570e-9289-4928-83c9-bf89a7e77f08")
print("Loaded posterior")


print("Start refinement")
posterior.train(loss="renjey_divergence", alpha=0.8, warm_up_rounds=0, n_particles=1000, learning_rate=1e-4, min_num_iters=0, max_num_iters=2000)
posterior.train(loss="renjey_divergence", alpha=0.5, warm_up_rounds=0, n_particles=5000, learning_rate=1e-4, min_num_iters=0, check_for_convergence=False, max_num_iters=2000)

print("Save")
torch.save(posterior, "posterior_reproducible_94_slurm_reruned.pkl")
