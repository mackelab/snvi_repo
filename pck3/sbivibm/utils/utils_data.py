import os, shutil
import pandas as pd

import torch 
import numpy as np


SEPERATOR = os.path.sep 
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + SEPERATOR + "data" + SEPERATOR


def query(name:str, task:str=None, algorithm:str=None, num_simulations:int=None, num_observation:int = None, loss:str=None):
    df = get_full_dataset(name)
    query = ""
    if task is not None:
        if query != "":
            query += "&"
        query += "task==@task"
    if algorithm is not None:
        if query != "":
            query += "&"
        query += "algorithm==@algorithm"
    if num_simulations is not None:
        if query != "":
            query += "&"
        query += "num_simulations==@num_simulations"
    if num_observation is not None:
        if query != "":
            query += "&"
        query += "num_observation==@num_observation"
    if loss is not None:
        if query != "":
            query += "&"
        query += "loss==@loss"
    if query != "":
        df = df.query(query)
        return df
    else:
        return df

def get_posteriors(name:str, task:str=None, algorithm:str=None,num_simulations:int=None, num_observation: int = None, loss:str = None):
    df = query(name, task, algorithm, num_simulations,num_observation, loss)
    folders = df["folder"].tolist() 
    posteriors = dict([(f, []) for f in folders])
    for folder in folders:
        posterior = get_posterior_by_id(name,id)
        posteriors[folder].append(posterior)
    return posteriors

def get_samples(name:str, task:str=None, algorithm:str=None,num_simulations:int=None, loss:str = None, num_observation: int = None):
    df = query(name, task, algorithm, num_simulations,num_observation, loss)
    folders = df["folder"].tolist() 
    samples = dict([(f, []) for f in folders])
    for folder in folders:
        sample = get_samples_by_id(name,folder)
        samples[folder].append(sample)
    return samples 

def get_predictive_samples(name:str, task:str=None, algorithm:str=None,num_simulations:int=None, loss:str = None, num_observation: int = None):
    df = query(name, task, algorithm, num_simulations,num_observation, loss)
    folders = df["folder"].tolist() 
    samples = dict([(f, []) for f in folders])
    for folder in folders:
        sample = get_predictive_samples_by_id(name,folder)
        samples[folder].append(sample)
    return samples 

def get_samples_by_id(name, id):
    path = PATH + name + SEPERATOR
    sample = torch.load(path+id + SEPERATOR + "samples" + SEPERATOR +"samples.pt")
    return sample

def get_predictive_samples_by_id(name, id):
    path = PATH + name + SEPERATOR
    sample = torch.load(path+id + SEPERATOR + "samples" + SEPERATOR +"predictive_samples.pt")
    return sample

def get_posterior_by_id(name, id):
    path = PATH + name + SEPERATOR
    posterior = torch.load(path+id + SEPERATOR + "posteriors" + SEPERATOR +f"posterior.model")
    return posterior

def get_full_dataset(name:str):
    path = PATH + name + SEPERATOR
    df = pd.read_csv(path + "benchmark_metrics.csv") 
    return df

def get_metrics(name:str, task:str, algorithm:str,num_simulations:int):
    df = get_full_dataset(name)
    df = df.query(f"task == '{task}'")
    df = df.query(f"algorithm == '{algorithm}'")
    df = df[df["num_simulations"] == num_simulations]
    return df

def save_figure(fig,name):
    path = PATH + name 
    if not os.path.exists(path + SEPERATOR + "figures"):
        try:
            os.mkdir(path + SEPERATOR +"figures")
        except:
            raise ValueError("Could not create folder") 
    
    fig.savefig(path + SEPERATOR + "figures" + SEPERATOR + name)

def check_for_completness(name, task, algorithm):
    df = get_full_dataset(name)
    df = df.query(f"task=='{task}'")
    df = df[df["algorithm"].str.contains(algorithm, regex=True)]

    num_observations = range(1,11)
    num_simulations = [1000,10000,100000]

    missing = []
    for o in num_observations:
        for s in num_simulations:
            if not (o in df["num_observation"].tolist() and s in df["num_simulations"]).tolist():
                print(f"We miss observation {o} with {s} number of simulations")
    return missing
        


def delete_by_id(name, id):
    path = PATH + name + SEPERATOR
    df = get_full_dataset(name)

    #Delete entry
    idx = df[df["folder"] == id].index[0]
    df_new = df.drop(idx)
    df_new.to_csv(path + "benchmark_metrics.csv", index=False) 
    #Delete folder
    mydir = path + id
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def copy_to_other_dataset(name_from:str, name_to:str, task:str=None, algorithm:str=None,num_simulations:int=None, loss:str = None, num_observation: int = None):
    assert name_from != name_to, "The datasets must have different names"
    # Gather data to copy
    path = PATH + name_from + SEPERATOR
    df = query(name_from, task, algorithm, num_simulations,num_observation, loss)
    folders = df["folder"].tolist() 

    # Copying folders
    path_to_copy = PATH + name_to + SEPERATOR
    for folder in folders:
        src = path+folder
        dest = path_to_copy + folder 
        
        try:
            destinations = shutil.copytree(src, dest)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    # Appending dataframes
    df_dest = get_full_dataset(name_to)
    df_new = df_dest.append(df)
    df_new.to_csv(path_to_copy + "benchmark_metrics.csv", index=False) 
    









