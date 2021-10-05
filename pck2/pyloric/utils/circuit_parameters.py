import numpy as np
from typing import Dict, Tuple, Optional, List
import pandas as pd
from torch import Tensor
from typing import Union


def create_neurons(neuron_list):
    prinz_neurons = {
        "LP": {
            "LP_0": [100, 0, 8, 40, 5, 75, 0.05, 0.02],  # this3 g_CaS g_A g_Kd
            "LP_1": [100, 0, 6, 30, 5, 50, 0.05, 0.02],  # this2 # KCa, H      # this3
            "LP_2": [100, 0, 10, 50, 5, 100, 0.0, 0.03],
            "LP_3": [100, 0, 4, 20, 0, 25, 0.05, 0.03],
            "LP_4": [100, 0, 6, 30, 0, 50, 0.03, 0.02],  # this2
        },
        "PY": {
            "PY_1": [200, 7.5, 0, 50, 0, 75, 0.05, 0.0],
            # this3  # this3 g_Na, g_CaT, g_CaS
            "PY_0": [100, 2.5, 2, 50, 0, 125, 0.05, 0.01],  # this3
            "PY_3": [400, 2.5, 2, 50, 0, 75, 0.05, 0.0],
            # this3        # this3 g_leak, g_Kd, g_Na
            "PY_5": [500, 2.5, 2, 40, 0, 125, 0.0, 0.02],  # this2 # g_H, g_leak
            "PY_2": [200, 10.0, 0, 50, 0, 100, 0.03, 0.0],  # this3 # CaT Kd H
            "PY_4": [500, 2.5, 2, 40, 0, 125, 0.01, 0.03],  # this2
        },
        "PM": {
            "PM_0": [400, 2.5, 6, 50, 10, 100, 0.01, 0.0],  # this2  g_Na, KCa
            "PM_3": [200, 5.0, 4, 40, 5, 125, 0.01, 0.0],  # this3 CaT, g_A, g_Kd
            "PM_4": [300, 2.5, 2, 10, 5, 125, 0.01, 0.0],
            "PM_1": [100, 2.5, 6, 50, 5, 100, 0.01, 0.0],  # this2
            "PM_2": [200, 2.5, 4, 50, 5, 50, 0.01, 0.0],  # this3
        },
    }
    # Note (PM_0 or PM_1) / (LP_2) / (PY_0) is figure 5a in Prinz 2004.
    # Note (PM_4)         / (LP_3) / (PY_4) is figure 5b in Prinz 2004.

    ret = []
    for n in neuron_list:
        membrane_area = np.asarray(n[2], dtype=np.float64)
        pn = np.asarray(prinz_neurons[n[0]][n[1]], dtype=np.float64)
        neuron = pn * membrane_area
        ret.append(neuron)
    return np.asarray(ret)


def membrane_conductances_replaced_with_defaults(circuit_parameters, defaults_dict):
    default_neurons = create_neurons(defaults_dict["membrane_gbar"])
    default_neurons = np.reshape(default_neurons, (1, 24))
    type_names, cond_names = select_names()
    type_names = type_names[:24]
    cond_names = cond_names[:24]
    default_neurons_pd = pd.DataFrame(default_neurons, columns=[type_names, cond_names])
    for tn, cn in zip(type_names, cond_names):
        if (tn, cn) in circuit_parameters:
            default_neurons_pd.loc[0][tn, cn] = circuit_parameters[tn, cn] * 0.628e-3
    return default_neurons_pd


def synapses_replaced_with_defaults(circuit_parameters, defaults_dict):
    type_names, cond_names = select_names()
    type_names = type_names[24:]
    cond_names = cond_names[24:]
    data_array = []
    for tn, cn in zip(type_names, cond_names):
        if (tn, cn) in circuit_parameters:
            data_array.append(circuit_parameters[tn, cn])
    data_array = np.asarray([data_array])
    default_synapse_values = pd.DataFrame(data_array, columns=[type_names, cond_names])
    return default_synapse_values


def q10s_replaced_with_defaults(circuit_parameters, defaults_dict):
    q10_dict = {
        "membrane_gbar": [
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
        ],
        "synapses": [False, False, False, False, False, False, False],
        "Q10_gbar_mem": [True, True, True, True, True, True, True, True],
        "Q10_gbar_syn": [True, True],  # first for glutamate, second for choline
        "Q10_tau_m": [True],
        "Q10_tau_h": [True],
        "Q10_tau_CaBuff": [True],
        "Q10_tau_syn": [True, True],  # first for glutamate, second for choline
    }
    type_names, cond_names = select_names(q10_dict)

    defaults_pd = dict_to_pd(q10_dict, defaults_dict)

    for tn, cn in zip(type_names, cond_names):
        if (tn, cn) in circuit_parameters:
            defaults_pd.loc[0][tn, cn] = circuit_parameters[tn, cn]

    return defaults_pd


def dict_to_pd(bool_dict, value_dict):
    data = []
    for key in bool_dict.keys():
        for i, entry in enumerate(bool_dict[key]):
            if entry and not isinstance(entry, List):
                data.append(value_dict[key][i])
    data_np = np.asarray([data])
    type_names, cond_names = select_names(bool_dict)
    data_pd = pd.DataFrame(data_np, columns=[type_names, cond_names])
    return data_pd


def build_synapse_q10s(vec):
    """From values of gluatemate and choline, build a 7D vector."""
    return np.asarray([vec[0], vec[1], vec[0], vec[1], vec[0], vec[0], vec[0]])


def build_conns(params):

    # Reversal voltages and dissipation time constants for the synapses, taken from
    # Prinz 2004, p. 1351
    Esglut = -70  # mV
    kminusglut = 40  # ms

    Eschol = -80  # mV
    kminuschol = 100  # ms

    return np.asarray(
        [
            [1, 0, params[0], Esglut, kminusglut],
            [1, 0, params[1], Eschol, kminuschol],
            [2, 0, params[2], Esglut, kminusglut],
            [2, 0, params[3], Eschol, kminuschol],
            [0, 1, params[4], Esglut, kminusglut],
            [2, 1, params[5], Esglut, kminusglut],
            [1, 2, params[6], Esglut, kminusglut],
        ]
    )


def ensure_array_not_scalar(selector):
    if isinstance(selector, bool) or isinstance(selector, float):
        selector = np.asarray([selector])
    return np.asarray(selector)


def select_names(setup: Dict = {}) -> Tuple[List, np.ndarray]:
    """
    Returns the names of all parameters that are selected in the `setup` dictionary.
    """
    default_setup = {
        "membrane_gbar": [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
        ],
        "synapses": [True, True, True, True, True, True, True],
        "Q10_gbar_mem": [False, False, False, False, False, False, False, False],
        "Q10_gbar_syn": [False, False],  # first for glutamate, second for choline
        "Q10_tau_m": [False],
        "Q10_tau_h": [False],
        "Q10_tau_CaBuff": [False],
        "Q10_tau_syn": [False, False],  # first for glutamate, second for choline
    }
    default_setup.update(setup)
    setup = default_setup

    gbar = np.asarray([_channel_names()[c] for c in setup["membrane_gbar"]]).flatten()
    syn = _synapse_names()[setup["synapses"]]
    q10_mem_gbar = _q10_mem_gbar_names()[setup["Q10_gbar_mem"]]
    q10_syn_gbar = _q10_syn_gbar_names()[setup["Q10_gbar_syn"]]
    tau_setups = np.concatenate(
        (
            setup["Q10_tau_m"],
            setup["Q10_tau_h"],
            setup["Q10_tau_CaBuff"],
            setup["Q10_tau_syn"],
        )
    )
    q10_tau = _q10_tau_names()[tau_setups]

    type_names = ["AB/PD"] * sum(setup["membrane_gbar"][0])
    type_names += ["LP"] * sum(setup["membrane_gbar"][1])
    type_names += ["PY"] * sum(setup["membrane_gbar"][2])
    type_names += ["Synapses"] * sum(setup["synapses"])
    type_names += ["Q10 gbar"] * (
        sum(setup["Q10_gbar_mem"]) + sum(setup["Q10_gbar_syn"])
    )
    type_names += ["Q10 tau"] * (
        sum(setup["Q10_tau_m"])
        + sum(setup["Q10_tau_h"])
        + sum(setup["Q10_tau_CaBuff"])
        + sum(setup["Q10_tau_syn"])
    )
    return type_names, np.concatenate((gbar, syn, q10_mem_gbar, q10_syn_gbar, q10_tau))


def _channel_names():
    return np.asarray(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])


def _synapse_names():
    return np.asarray(["AB-LP", "PD-LP", "AB-PY", "PD-PY", "LP-PD", "LP-PY", "PY-LP"])


def _q10_mem_gbar_names():
    return np.asarray(["Na", "CaT", "CaS", "A", "KCa", "Kd", "H", "Leak"])


def _q10_syn_gbar_names():
    return np.asarray(["Glut", "Chol"])


def _q10_tau_names():
    return np.asarray(["m", "h", "CaBuff", "Glut", "Chol"])


def to_pyloric_pd(circuit_parameters: Union[Tensor, np.ndarray]):
    """
    Take an array and return the pandas DataFrame that can be passed to `simulate()`.

    This function will only work if `circuit_parameters` contains the 31 most basic
    parameters (8*3 membrane conductances and 7 synaptic conductances).
    """
    columns = (
        ("AB/PD", "Na"),
        ("AB/PD", "CaT"),
        ("AB/PD", "CaS"),
        ("AB/PD", "A"),
        ("AB/PD", "KCa"),
        ("AB/PD", "Kd"),
        ("AB/PD", "H"),
        ("AB/PD", "Leak"),
        ("LP", "Na"),
        ("LP", "CaT"),
        ("LP", "CaS"),
        ("LP", "A"),
        ("LP", "KCa"),
        ("LP", "Kd"),
        ("LP", "H"),
        ("LP", "Leak"),
        ("PY", "Na"),
        ("PY", "CaT"),
        ("PY", "CaS"),
        ("PY", "A"),
        ("PY", "KCa"),
        ("PY", "Kd"),
        ("PY", "H"),
        ("PY", "Leak"),
        ("Synapses", "AB-LP"),
        ("Synapses", "PD-LP"),
        ("Synapses", "AB-PY"),
        ("Synapses", "PD-PY"),
        ("Synapses", "LP-PD"),
        ("Synapses", "LP-PY"),
        ("Synapses", "PY-LP"),
    )

    if isinstance(circuit_parameters, Tensor):
        pd_params = pd.DataFrame(circuit_parameters.detach().numpy(), columns=columns)
    else:
        circuit_parameters = np.asarray(circuit_parameters)
        if circuit_parameters.ndim == 1:
            circuit_parameters = np.asarray([circuit_parameters])
        pd_params = pd.DataFrame(circuit_parameters, columns=columns)

    return pd_params
