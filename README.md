# Variational methods for simulation based inference

This packages contains the code for reproducing results of the SNVI paper (https://openreview.net/forum?id=kZ0UYdhqkNY).

A user-friendly implementation of the algorithm is implemented in the [sbi toolbox](https://github.com/mackelab/sbi).

### Installation

Install the three packages using 
```bash
pip install -e pck1
pip install -e pck2
pip install -e pck3
```
This should install a script, which is added to your global PATH (at least on Linux). You can call it with the keyword **sbivibm**. With that on could reproduce all our results and more! We use [hydra](https://hydra.cc/docs/intro/), so one also could test different parameterizations.

For example one can use:
```bash
sbivibm experiments=SNL_two_moons_vi_fKL_correction
```
The default for the multirun feature is a slurm-submitit launcher, which settings are likely incompatible with your system. Thus one should switch to the local launcher: submitit_local
```bash
 sbivibm -m hydra/launcher=submitit_local experiments=SNL_two_moons_vi_fKL_correction,SNL_two_moons_vi_rKL_correction
```

### Updates

In the [published version](https://openreview.net/forum?id=kZ0UYdhqkNY) of this paper, we reported a valid rate of 94% for the pyloric task, whereas the [updated version]() reports 86%. In order to achieve a valid rate of 94%, we performed further steps to refine the variational posterior after the last round. This did not involve running further simulations, but we used a different divergence, different number of particles, and more iterations for variational inference after the last round. One can achieve a valid rate of 94% with the procedure run in [this file](https://github.com/mackelab/snvi_repo/tree/main/updates/sbivibm/figures/final_figures/refined_posterior_compute_predictives.py). Code to reproduce the results shown in the main figure of the updated paper is [here](https://github.com/mackelab/snvi_repo/tree/main/updates).