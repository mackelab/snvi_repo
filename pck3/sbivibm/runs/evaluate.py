
import torch  
import numpy as np 
import pandas as pd
import uuid
import os 
import typing

from sbibm.tasks import Task 
from sbibm.metrics import c2st, mmd, ppc, ksd, mvn_kl

SEPERATOR = os.path.sep 
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + SEPERATOR + "data" + SEPERATOR
SAMPLE_FOLDER = "samples"
POST_FOLDER = "posteriors"
SAMPEL_FILE = "samples.pt"
PRED_SAMPLE_FILE = "predictive_samples.pt"


def store_results(benchmark_name, task, posteriors, samples, compute_predictives=True):

    # Create folder structure
    folder_name = str(uuid.uuid4())
    path = PATH + benchmark_name

    if not os.path.exists(PATH[:-1]):
        os.mkdir(PATH[:-1])
    if not os.path.exists(path):
        os.mkdir(path)

    while os.path.exists(path + SEPERATOR + folder_name):
        folder_name = str(uuid.uuid4())

    if not os.path.exists(path + SEPERATOR + folder_name):
        os.mkdir(path + SEPERATOR + folder_name)
    if not os.path.exists(path + SEPERATOR + folder_name + SEPERATOR + SAMPLE_FOLDER):
        os.mkdir(path + SEPERATOR + folder_name + SEPERATOR + SAMPLE_FOLDER)
    if not os.path.exists(path + SEPERATOR + folder_name + SEPERATOR + POST_FOLDER):
        os.mkdir(path + SEPERATOR + folder_name + SEPERATOR + POST_FOLDER)

    # Compute predictive samples
    if compute_predictives:
        simulator = task.get_simulator()
        predictive_samples = []
        batch_size = 1000
        for idx in range(int(samples.shape[0] / batch_size)):
            try:
                predictive_samples.append(
                    simulator(samples[(idx * batch_size) : ((idx + 1) * batch_size), :])
                )
            except:
                predictive_samples.append(
                    float("nan") * torch.ones((batch_size, task.dim_data))
                )
        predictive_samples = torch.cat(predictive_samples, dim=0)

    # Saver everythink
    torch.save(samples, path + SEPERATOR + folder_name + SEPERATOR + SAMPLE_FOLDER + SEPERATOR +  SAMPEL_FILE)
    if compute_predictives:
        torch.save(predictive_samples, path + SEPERATOR + folder_name + SEPERATOR + SAMPLE_FOLDER + SEPERATOR + PRED_SAMPLE_FILE)
    torch.save(posteriors[-1], path + SEPERATOR + folder_name + SEPERATOR + POST_FOLDER + SEPERATOR + f"posterior.model")

    return folder_name

def do_not_evaluate_metric(task: Task, cfg: dict, algorithm_params, result_folder, runtime):
    num_observation = cfg.task.num_observation
    num_simulations = cfg.task.num_simulations
    benchmark_name = cfg.name
    observation = task.get_observation(num_observation)
    num_rounds = algorithm_params["num_rounds"]
    algo = cfg.method.method_name.upper()
    if num_rounds == 1:
        # Skip as for single round
        algo = algo[1:]
    if "mcmc" in cfg.method.method_name:
        loss = "na"
        paras = dict(cfg.method.params.mcmc_parameters)
        paras["mcmc_method"] = cfg.method.params.mcmc_method
    else: 
        loss = cfg.task.vi_parameters.loss
        paras = dict(cfg.task.vi_parameters)

    path = PATH + benchmark_name 
    sample_path = path + SEPERATOR + result_folder + SEPERATOR + SAMPLE_FOLDER + SEPERATOR

    df = pd.DataFrame(
        {
            "task": [task.name],
            "algorithm": [algo],
            "loss": [loss],
            "num_rounds": [num_rounds],
            "num_observation": [num_observation],
            "num_simulations": [num_simulations],
            "folder": [result_folder],
            "time": [runtime],
            "parameters": [paras]
        }
    )
    with open(path + SEPERATOR + "benchmark_metrics.csv", "a") as f:
        df.to_csv(f, mode="a", header=f.tell() == 0, index=False) 
    return df

def evaluate_metric(
    task: Task,
    cfg:dict,
    algorithm_paras:dict,
    result_folder: str,
    runtime: float,
):
    r""" Will evaluate the metrics c2st, mmd and mean_dist for a given set of samples """
    num_observation = cfg.task.num_observation
    num_simulations = cfg.task.num_simulations
    benchmark_name = cfg.name

    path = PATH + benchmark_name 
    sample_path = path + SEPERATOR + result_folder + SEPERATOR + SAMPLE_FOLDER + SEPERATOR

    # Load necessary data
    reference_samples = task.get_reference_posterior_samples(num_observation)
    samples = torch.load(sample_path + SAMPEL_FILE)
    predictive_samples = torch.load(sample_path + PRED_SAMPLE_FILE)

    assert reference_samples.shape[0] == samples.shape[0]
    assert reference_samples.shape[0] == predictive_samples.shape[0]

    observation = task.get_observation(num_observation)

    # Compute the metrics
    c2st_accuracy = float(c2st(reference_samples, samples))
    mmd_metric = float(mmd(reference_samples, samples))
    median_dist = float(ppc.median_distance(predictive_samples, observation))
    ksd_metric = float(ksd(task, num_observation, samples, sig2=float(torch.median(torch.pdist(reference_samples))**2), log=False)) 
    mvn_kl_pq = float(mvn_kl.mvn_kl_pq(reference_samples, samples))
    mvn_kl_qp = float(mvn_kl.mvn_kl_qp(reference_samples, samples))

    num_rounds = algorithm_paras["num_rounds"]
    algo = cfg.method.method_name.upper()
    if num_rounds == 1:
        # Skip as for single round
        algo = algo[1:]
    if "mcmc" in cfg.method.method_name:
        loss = "na"
        paras = dict(cfg.method.params.mcmc_parameters)
        paras["mcmc_method"] = cfg.method.params.mcmc_method
    else: 
        loss = cfg.task.vi_parameters.loss
        paras = dict(cfg.task.vi_parameters)


    # Add metrics to benchmark df
    df = pd.DataFrame(
        {
            "task": [task.name],
            "algorithm": [algo],
            "loss": [loss],
            "num_rounds": [num_rounds],
            "num_observation": [num_observation],
            "num_simulations": [num_simulations],
            "c2st": [c2st_accuracy],
            "mmd": [mmd_metric],
            "ksd": [ksd_metric],
            "median_dist": [median_dist],
            "mvn_kl_pq": [mvn_kl_pq],
            "mvn_kl_qp": [mvn_kl_qp],
            "folder": [result_folder],
            "time": [runtime],
            "parameters": [paras]
        }
    )
    with open(path + SEPERATOR + "benchmark_metrics.csv", "a") as f:
        df.to_csv(f, mode="a", header=f.tell() == 0, index=False) 
    return df