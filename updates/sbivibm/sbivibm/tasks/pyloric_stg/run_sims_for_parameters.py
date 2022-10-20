import random
import time

time.sleep(random.uniform(0,20))

from sbivibm.tasks import Pyloric
import sys,os
import torch

from multiprocessing import Pool
import pandas as pd
import numpy as np
import sys

import time

from pyloric import simulate, summary_stats

NUM_CORES=12

task = Pyloric()
def simulator(parameter):
    seed = int(parameter[-1])
    parameter = parameter[:-1]
    names = task.prior.names
    sample = pd.DataFrame(parameter.reshape(1,-1).numpy(), columns=names)
    x = simulate(sample.loc[0], seed=seed)
    x = summary_stats(x).to_numpy()
    x[np.isnan(x)] = -99 # First try
    return torch.tensor(x)

def main(parameter_path, id, glob_seed):
    glob_seed = 1000000*int(glob_seed) + int(id)
    with open('seed_log.txt', 'a') as file_object:
        file_object.write(f'Used seed: {glob_seed}\n')
    
    torch.manual_seed(glob_seed)
    random.seed(glob_seed)
    np.random.seed(glob_seed)

    thetas = torch.load(parameter_path)
    seed = torch.randint(0,2**32-1, (thetas.shape[0],1)).float()
    paras = torch.hstack((thetas, seed))
    with Pool(NUM_CORES) as pool:
        xs = pool.map(simulator, paras)
    xs = torch.vstack(xs)
    torch.save(thetas, f"thetas_{id}.pkl")
    torch.save(xs, f"xs_{id}.pkl")
    torch.save(seed, f"seed_{id}.pkl")

if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)
    main(*args)
        


