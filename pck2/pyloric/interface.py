import os
from typing import Dict, Optional, Union

import numpy
import numpy as np
import pandas as pd
import torch

import pyximport

# setup_args needed on my MacOS
pyximport.install(
    setup_args={"include_dirs": numpy.get_include()},
    reload_support=True,
    language_level=3,
)

from pyloric.prior import prior_bounds, select_names
from pyloric.simulator import sim_time
from pyloric.summary_statistics import PrinzStats
from pyloric.utils import (
    build_conns,
    build_synapse_q10s,
    create_neurons,
    ensure_array_not_scalar,
    membrane_conductances_replaced_with_defaults,
    q10s_replaced_with_defaults,
    synapses_replaced_with_defaults,
)
from sbi.utils import BoxUniform


def create_prior(
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    customization: Dict = {},
    synapses_log_space: bool = True,
    as_torch_dist: bool = False,
) -> "pd_prior":
    """
    Return prior over circuit parameters of the pyloric network.

    Args:
        lower_bound: Lower bound of the prior. If `None`, use the values used in
            Goncalves et al. 2020. If passed, it must be a 1D array with as many
            elements as `True` values in `customization`.
        upper_bound: Upper bound of the prior. If `None`, use the values used in
            Goncalves et al. 2020. If passed, it must be a 1D array with as many
            elements as `True` values in `customization`.
        customization: If you want to exclude some of the circuit_parameters and use
            constant default values for them, you have to set these entries to `False`
            in the `membrane_gbar` key in the `customization` dictionary. If you want
            to include $Q_{10}$ values, you have to set them in this dictionary.
        synapses_log_space: Whether the synapses will be uniformly distributed in
            logarithmic space or not (=in linear space).
        as_torch_dist: If `False`, the prior will be wrapped to return pandas
            dataframes as samples. If `True`, the samples will be pytorch tensors.

    Returns:
        A uniform prior distribution.
    """

    setups = {
        "membrane_gbar": [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ],
        "Q10_gbar_mem": [False, False, False, False, False, False, False, False],
        "Q10_gbar_syn": [False, False],  # first for glutamate, second for choline
        "Q10_tau_m": [False],
        "Q10_tau_h": [False],
        "Q10_tau_CaBuff": [False],
        "Q10_tau_syn": [False, False],  # first for glutamate, second for choline
    }
    setups.update(customization)
    for key in setups.keys():
        setups[key] = ensure_array_not_scalar(setups[key])

    l_bound, u_bound = prior_bounds(setups, synapses_log_space=synapses_log_space)
    if lower_bound is None:
        lower_bound = l_bound
    if upper_bound is None:
        upper_bound = u_bound
    type_names, channel_names = select_names(setups)

    class pd_prior:
        """Wrapper for the pytorch prior such that it returns pandas samples."""

        def __init__(self, lower, upper, parameter_names):
            self.lower = torch.tensor(lower)
            self.upper = torch.tensor(upper)
            self.names = parameter_names
            self.numerical_prior = BoxUniform(torch.as_tensor(self.lower, dtype=torch.float32), torch.as_tensor(self.upper, dtype=torch.float32))

        def sample(self, sample_shape):
            numerical_sample = self.numerical_prior.sample(sample_shape).numpy()
            return pd.DataFrame(numerical_sample, columns=self.names)

        def log_prob(self, theta):
            numerical_theta = theta.to_numpy()
            return self.numerical_prior.log_prob(numerical_theta)

    prior = pd_prior(lower_bound, upper_bound, [type_names, channel_names])

    if as_torch_dist:
        return prior.numerical_prior
    else:
        return prior


def simulate(
    circuit_parameters: Union[np.array, pd.DataFrame],
    dt: float = 0.025,
    t_max: int = 11000,
    temperature: int = 283,
    noise_std: float = 0.001,
    track_energy: bool = False,
    track_currents: bool = False,
    seed: Optional[int] = None,
    customization: Dict = {},
    defaults: Dict = {},
):
    r"""
    Runs the STG model with a subset of all parameters.

    Args:
        circuit_parameters: Parameters of the circuit model. This should be a pandas
            DataFrame sampled from the prior.
        dt: Step size in milliseconds.
        t_max: Overall runtime of the simulation in milliseconds.
        temperature: Temperature in Kelvin that the simulation is run at.
        noise_std: Standard deviation of the noise added at every time step. Will
            **not** be rescaled with the step-size.
        track_energy: Whether to keep track of and return the energy consumption at any
            step during the simulation. The output dictionary will have the additional
            entry `energy`.
        track_currents: Tracks the conductance values of all channels (also synapses).
            The currents can easily be computed from the conductance values by
            $I = g \cdot (V-E)$. For the calcium channels, the reversal potential of
            the calcium channels is also saved. The output dictionary will have
            additional entries 'membrane_conds', 'synaptic_conds', 'reversal_calcium'.
        seed: Possible seed for the simulation.
        customization:  If you want to exclude some of the `circuit_parameters` and use
            constant default values for them, you have to set these entries to `False`
            in the `use_membrane` key in the `customization` dictionary. If you want
            to include $Q_{10}$ values, you have to set them in the same dictionary and
            append the values of the $Q_{10}$s to the `circuit_parameters`.
        defaults: For all parameters specified as `False` in `customization`, this
            dictionary allows to set the default value, i.e. the value that is used for
            it.
    """

    setup_dict = {
        "membrane_gbar": [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ],
        "Q10_gbar_mem": [False, False, False, False, False, False, False, False],
        "Q10_gbar_syn": [False, False],  # first for glutamate, second for choline
        "Q10_tau_m": [False],
        "Q10_tau_h": [False],
        "Q10_tau_CaBuff": [False],
        "Q10_tau_syn": [False, False],  # first for glutamate, second for choline
    }
    setup_dict.update(customization)
    for key in setup_dict.keys():
        setup_dict[key] = ensure_array_not_scalar(setup_dict[key])

    defaults_dict = {
        "membrane_gbar": [
            ["PM", "PM_4", 0.628e-3],
            ["LP", "LP_3", 0.628e-3],
            ["PY", "PY_4", 0.628e-3],
        ],
        "Q10_gbar_mem": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        "Q10_gbar_syn": [1.5, 1.5],
        "Q10_tau_m": [2.4],
        "Q10_tau_h": [2.8],
        "Q10_tau_CaBuff": [2.0],
        "Q10_tau_syn": [1.7, 1.7],
    }
    defaults_dict.update(defaults)
    for key in defaults_dict.keys():
        defaults_dict[key] = ensure_array_not_scalar(defaults_dict[key])

    membrane_conductances_pd = membrane_conductances_replaced_with_defaults(
        circuit_parameters, defaults_dict
    )
    synaptic_conductances_pd = synapses_replaced_with_defaults(
        circuit_parameters, defaults_dict
    )
    q10_values_pd = q10s_replaced_with_defaults(circuit_parameters, defaults_dict)

    membrane_q10_gbar = q10_values_pd["Q10 gbar"].to_numpy()[0, :8]
    synapse_q10_gbar = build_synapse_q10s(q10_values_pd["Q10 gbar"].to_numpy()[0, 8:10])
    synapse_q10_tau = build_synapse_q10s(q10_values_pd["Q10 tau"].to_numpy()[0, 3:5])
    q10_tau_m = q10_values_pd["Q10 tau"]["m"].to_numpy().tolist() * 7
    q10_tau_h = q10_values_pd["Q10 tau"]["h"].to_numpy().tolist() * 4
    q10_tau_cabuff = q10_values_pd["Q10 tau"]["CaBuff"].to_numpy().tolist()

    # The Q10 of the opening gate of CaS is set. (Caplan 2014)
    q10_tau_m[2] = 2.0

    # The Q10 of the opening gate of KCa is set. (Caplan 2014)
    q10_tau_m[4] = 1.6

    t = np.arange(0, t_max, dt)

    # note: make sure to generate all randomness through self.rng (!)
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()
    I = rng.normal(scale=noise_std, size=(3, len(t)))

    data = sim_time(
        dt,
        t,
        I,
        np.reshape(membrane_conductances_pd.to_numpy(), (3, 8)),
        build_conns(-np.exp(synaptic_conductances_pd.to_numpy()[0])),
        g_q10_conns_gbar=synapse_q10_gbar,
        g_q10_conns_tau=synapse_q10_tau,
        g_q10_memb_gbar=membrane_q10_gbar,
        g_q10_memb_tau_m=q10_tau_m,
        g_q10_memb_tau_h=q10_tau_h,
        g_q10_memb_tau_CaBuff=q10_tau_cabuff,
        temp=temperature,
        num_energy_timesteps=len(t) if track_energy else 0,
        num_energyscape_timesteps=len(t) if track_currents else 0,
        init=None,
        start_val_input=0.0,
        verbose=False,
    )

    results_dict = {"voltage": data["Vs"], "dt": dt, "t_max": t_max}
    if track_energy:
        results_dict.update({"energy": data["energy"]})
    if track_currents:
        results_dict.update({"membrane_conds": data["membrane_conds"]})
        results_dict.update({"synaptic_conds": data["synaptic_conds"]})
        results_dict.update({"reversal_calcium": data["reversal_calcium"]})

    return results_dict


def summary_stats(
    simulation_outputs: Dict, stats_customization: Dict = {}, t_burn_in=1000
) -> pd.DataFrame:
    """
    Return summary statistics of the voltage trace.

    Args:
        simulation_outputs: Dictionary returned by `simulate()`. Contains (at least)
            the voltage traces, the time step, and the duration of the simulation.
        stats_customization: Allows to add summary statistics. Possible keys are:
            `plateau_durations`: Maximum duration of voltage plateaus above -30mV in
                each model neuron.
            `pyloric_like`: bool indicating whether the rhythm was pyloric-like (see
                Prinz 2004 for a definition).
            `pyloric`: bool indicating whether the rhythm was pyloric (see
                Prinz 2004 for a definition).
            `num_bursts`: Integer indicating the number of bursts in each neuron.
            `num_spikes`: Integer indicating the number of spikes in each neuron.
            `voltage_moments`: The first four moments (mean, std, skew, kurtosis) of
                the voltage trace of each neuron.
            `energies`: The energy (in microJoule) consumed by each neuron.
            `energies_per_spike`: The average energy per spike (in microJoule) in each
                neuron.
        t_burn_in: The time (in milliseconds) that should be excluded from the
            computation of summary statistics. The idea is that the rhythm should first
            reach a steady state and only then should one compute the summary
            statistics.

    Returns:
        Summary statistics.
    """

    setups = {
        "cycle_period": True,
        "burst_durations": True,
        "duty_cycles": True,
        "start_phases": True,
        "starts_to_starts": True,
        "ends_to_starts": True,
        "phase_gaps": True,
        "plateau_durations": False,
        "voltage_means": False,
        "voltage_stds": False,
        "voltage_skews": False,
        "voltage_kurtoses": False,
        "num_bursts": False,
        "num_spikes": False,
        "spike_times": False,
        "spike_heights": False,
        "rebound_times": False,
        "energies": False,
        "energies_per_burst": False,
        "energies_per_spike": False,
        "pyloric_like": False,
    }
    setups.update(stats_customization)

    stats_object = PrinzStats(
        setup=setups,
        t_on=t_burn_in,
        t_off=simulation_outputs["t_max"],
        dt=simulation_outputs["dt"],
    )

    ss = stats_object.calc_dict(simulation_outputs)
    return ss
