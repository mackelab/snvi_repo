from sbivibm.utils import get_posterior_by_id, get_full_dataset
import torch

import random 
import numpy as np

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print("Script fKL till convergence")


posterior = get_posterior_by_id("benchmark", "18fd570e-9289-4928-83c9-bf89a7e77f08")
posterior.train(loss="forward_kl", warm_up_rounds=0, n_particles=5000, learning_rate=1e-3, min_num_iters=0, max_num_iters=3000)

print("Stave")
torch.save(posterior, "posterior_reproducible_fKL.pkl")
