from typing import Dict, List


def energy_of_membrane(simulation_output: Dict) -> List:
    """Returns the energy consumed by any current in any neuron at any time."""

    v_r_na = 50
    v_r_k = -80
    v_r_h = -20
    v_r_leak = -50

    voltage = simulation_output["voltage"]
    membrane_conductances = simulation_output["membrane_conds"]
    reversal_calcium = simulation_output["reversal_calcium"]

    E_Na = membrane_conductances[:, 0, :] * (voltage - v_r_na) ** 2
    E_CaS = membrane_conductances[:, 1, :] * (voltage - reversal_calcium) ** 2
    E_CaT = membrane_conductances[:, 2, :] * (voltage - reversal_calcium) ** 2
    E_A = membrane_conductances[:, 3, :] * (voltage - v_r_k) ** 2
    E_KCa = membrane_conductances[:, 4, :] * (voltage - v_r_k) ** 2
    E_Kd = membrane_conductances[:, 5, :] * (voltage - v_r_k) ** 2
    E_H = membrane_conductances[:, 6, :] * (voltage - v_r_h) ** 2
    E_leak = membrane_conductances[:, 7, :] * (voltage - v_r_leak) ** 2

    return [E_Na, E_CaS, E_CaT, E_A, E_KCa, E_Kd, E_H, E_leak]


def energy_of_synapse(simulation_output: Dict) -> List:
    """Returns the energy consumed by any synaptic current at any time."""

    voltage = simulation_output["voltage"]
    syn_conds = simulation_output["synaptic_conds"]

    rev_g = -70  # glutamate
    rev_c = -80  # choline

    rev_pot = [rev_g, rev_c, rev_g, rev_c, rev_g, rev_g, rev_g]
    postsynaptic_neuron = [1, 1, 2, 2, 0, 2, 1]
    syn_e = []
    for syn_num, post_n in enumerate(postsynaptic_neuron):
        syn_e.append(syn_conds[syn_num] * (voltage[post_n] - rev_pot[syn_num]) ** 2)
    return syn_e
