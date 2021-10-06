
from sbibm.tasks import Task
from sbibm.tasks.simulator import Simulator
import torch 

from pyloric import simulate, create_prior, summary_stats
import pandas as pd
import numpy as np 

import os, sys
from pathlib import Path
from multiprocessing import Pool 
import subprocess 
import glob
import time

global nan_replace_glob
global summary_glob
global NAMES 
global CACHE

summary_glob = "summary_statistics"
nan_replace_glob = -99
NAMES_sim = [['AB/PD','AB/PD','AB/PD','AB/PD','AB/PD','AB/PD','AB/PD','AB/PD','LP','LP','LP','LP','LP','LP','LP','LP', 'PY','PY','PY','PY','PY','PY','PY','PY','Synapses',  'Synapses','Synapses','Synapses','Synapses','Synapses','Synapses'],
             ['Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H', 'Leak', 'Na', 'CaT','CaS', 'A', 'KCa', 'Kd', 'H', 'Leak', 'Na', 'CaT', 'CaS', 'A','KCa', 'Kd', 'H', 'Leak', 'AB-LP', 'PD-LP', 'AB-PY', 'PD-PY','LP-PD', 'LP-PY', 'PY-LP']]
NAMES = ["AB/PD_Na","AB/PD_CaT","AB/PD_CaS","AB/PD_A","AB/PDK_Ca","AB/PD_Kd","AB/PD_H","AB/PD_Leak","LP_Na","LP_CaT","LP_CaS","LP_A","LP_KCa","LP_Kd","LP_H","LP_Leak","PY_Na","PY_CaT","PY_CaS","PY_A","PY_KCa","PY_Kd","PY_H","PY_Leak","SynapsesAB-LP","SynapsesPD-LP","SynapsesAB-PY","SynapsesPD-PY","SynapsesLP-PD","SynapsesLP-PY","SynapsesPY-LP"]
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def mp_simulator(parameter):
    seed = int(parameter[-1])
    parameter = parameter[:-1]

    sample = pd.DataFrame(parameter.reshape(1,-1).numpy(), columns=NAMES_sim)
    x = simulate(sample.loc[0], seed=seed)

    if summary_glob == "summary_statistics":
        x = summary_stats(x).to_numpy()
    else:
        x = x["voltage"].flatten()
    x[np.isnan(x)] = nan_replace_glob 

    return torch.tensor(x).float()


def slurm_simulator(thetas, simulation_batches = 500):
    N = thetas.shape[0]
    if N < simulation_batches:
        simulation_batches = N
    jobs = N // simulation_batches

    # Delete intermediate results
    for j in range(jobs):
        subprocess.run(["rm", DIR_PATH + os.sep  + f"thetas_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep  + f"xs_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep  + f"seed_{j}.pkl"])
        
    for fl in glob.glob(DIR_PATH + os.sep  + "slurm-*"):
        os.remove(fl)
    
    # Run the slurm jobs...
    for j in range(jobs):
        torch.save(thetas[j*simulation_batches:(j+1)*simulation_batches,:],DIR_PATH + os.sep  + f"thetas_{j}.pkl")
    
    # Wait to for saving thetas
    time.sleep(10)
    
    for j in range(jobs):
        subprocess.run(["sbatch", DIR_PATH + os.sep  + "run_one.sh", DIR_PATH + os.sep  + f"thetas_{j}.pkl", f"{j}", DIR_PATH, f"--output={DIR_PATH}"])
    
    time.sleep(10)
    
    # Check for complettion
    start_time = time.time()
    jobs_status = np.zeros(jobs)
    i = 0
    while True:
        if (i % 1000) == 0:
            for j in range(jobs):
                jobs_status[j] = os.path.isfile(DIR_PATH + os.sep  + f"xs_{j}.pkl")
            sys.stdout.write(f"\rCompleted {int(jobs_status.sum())}/{jobs} jobs")
            sys.stdout.flush()
            if jobs_status.sum() == jobs:
                break
            current_time = time.time()
            time_till_execution = current_time-start_time 
            if time_till_execution > 300:
                start_time = time.time()
                for j in range(jobs):
                    if not jobs_status[j]:
                        subprocess.run(["sbatch", DIR_PATH + os.sep  + "run_one.sh", DIR_PATH + os.sep  + f"thetas_{j}.pkl", f"{j}", DIR_PATH, f"--output={DIR_PATH}"])

        i += 1
    
    # Wait to receive xs
    time.sleep(10)
    subprocess.run(["scancel", "-n","run_one.sh"])
    
    # if jobs_status.sum() != jobs:
    #     return slurm_simulator(thetas)
    # Append final results
    xs = []
    for j in range(jobs):
        xs.append(torch.load( DIR_PATH + os.sep  + f"xs_{j}.pkl"))
        
    x = torch.vstack(xs)
    
    # Delete intermediate results
    for j in range(jobs):
        subprocess.run(["rm", DIR_PATH + os.sep  + f"thetas_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep  + f"xs_{j}.pkl"])
        subprocess.run(["rm", DIR_PATH + os.sep  + f"seed_{j}.pkl"])
        
    for fl in glob.glob(DIR_PATH + os.sep  + "slurm-*"):
        os.remove(fl)
    return x.float()


class Pyloric(Task):
    def __init__(self, summary="summary_statistics", nan_replace=-99):
        self.summary = summary 
        self.nan_replace = nan_replace
        self.dim_data_unflatten = torch.Size((3,440000))
        self.dim_data_raw = torch.numel(torch.tensor(self.dim_data_unflatten))
        if summary == "summary_statistics":
            dim_data=15
        else:
            dim_data=self.dim_data_raw

        observation_seeds = [4933, "na", 42]


        super().__init__(dim_parameters=31,
                         dim_data=dim_data, 
                         name="pyloric", 
                         name_display="Pyloric STG",
                         num_observations = [1],
                         observation_seeds=observation_seeds,
                         num_posterior_samples=10000,
                         num_reference_posterior_samples=10000,
                         num_simulations=[1000,10000, 100000],
                         path=Path(__file__).parent.absolute())

        self.prior = create_prior()
        self.prior_dist = self.prior.numerical_prior
        self.t = torch.arange(0,11000,0.025)
        self.names = NAMES

    def get_prior(self):
        def prior(num_samples=1):
            return self.prior.sample((num_samples,))
        return prior 

    def unflatten_data(self, data: torch.Tensor) -> torch.Tensor:
        """Unflattens data into multiple observations
        """
        if self.summary is None:
            return data.reshape(-1, *self.dim_data_unflatten)
        else:
            return data.reshape(-1, self.dim_data)

    def get_simulator(self, max_calls=None,nan_replace=0., seed=None, sim_type="slurm", num_cores=8, save_simulations=True):
        if sim_type == "sequential":
            def simulator(parameters):
                num_samples = parameters.shape[0]
                xs = []
                for i in range(num_samples):
                    sample = pd.DataFrame(parameters[i].reshape(1,-1).numpy(), columns=self.prior.names)
                    x = simulate(sample.loc[0], seed=seed)
                    if self.summary == "summary_statistics":
                        x = summary_stats(x).to_numpy()
                    else:
                        x = x["voltage"].flatten()

                    x[np.isnan(x)] = nan_replace
                    xs.append(torch.tensor(x))
                return torch.vstack(xs).float()
            return Simulator(task=self, simulator=simulator, max_calls=max_calls)
        elif sim_type == "parallel":
            def simulator(parameters):
                NUM_SAMPLES = parameters.shape[0]
                seed = torch.randint(0,2**32-1, (NUM_SAMPLES,1)).float()
                paras = torch.hstack((parameters, seed))
                with Pool(num_cores) as pool:
                    xs = pool.map(mp_simulator, paras)
                xs = torch.vstack(xs)
                return xs
            return Simulator(task=self, simulator=simulator, max_calls=max_calls)
        elif sim_type == "slurm":
            return Simulator(task=self, simulator=slurm_simulator, max_calls=max_calls)
        else:
            raise NotImplementedError()

    def get_precomputed_dataset(self):
        
        for i in range(5):
            df_paras = pd.read_pickle(str(Path(__file__).parent.absolute()) + f"/files/all_circuit_parameters_{i}.pkl")
            df_simulation_output = pd.read_pickle(str(Path(__file__).parent.absolute()) +f"/files/all_simulation_outputs_{i}.pkl")
            if i==0:
                thetas = torch.tensor(df_paras.to_numpy()).float()
                xs = torch.tensor(df_simulation_output.to_numpy()[:,:15]).float()
                xs[np.isnan(xs)] = self.nan_replace
            else:
                thetas = torch.vstack([thetas,torch.tensor(df_paras.to_numpy()).float()])
                xs = torch.vstack([xs, torch.tensor(df_simulation_output.to_numpy()[:,:15]).float()])
                xs[np.isnan(xs)] = self.nan_replace
        return thetas, xs
            
                
