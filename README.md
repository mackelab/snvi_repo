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
