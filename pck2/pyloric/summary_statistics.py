import numpy as np
import scipy
import scipy.stats
import scipy.signal
from copy import deepcopy
import pandas as pd
import os


########################################################
# This is an overview of Prinz' summary statistics detection routine
# to serve as a reference
#
# Saved Data in Prinz:
# cellmaxnum:   number of voltage maxima
# firstmaxtime
# cellimiends: ends of inter-maximum intervals
# cellimi: durations of IMIs
# meancellimi, mincellimi, maxcellimi

# spikenum: spikes are maxima that overshoot 0mV
# firstspiketime
# cellisiends
# cellisis
# mean, min, maxcellisi

# maxcellisi, mincellisi
# is_modulated_tonic: if largest IMI is less than 5 times the smalles ISI
# isithreshold: 99% of the largest ISI for modulated tonics, ie. only the largest ISI
# becomes an IBI
#               50% of the largest ISI if it is more than 2 times the mean ISI
#               otherwise, the mean ISI
#               but the maximum is always 0.5s

# winspikes: number of spikes in detection window
# winmin, winmax: smallest and largest ISI in the detection window

# for each ISI, if above the ISI threshold, it is classified as an IBI
# cellibiend
# cellibi
# ibinum

# if more than 10 spikes in detection window (5s), cell is tonic if diff between max
# and min isi as a fraction of their average is less than 5%
# a tonic cell is not a modulated tonic

# For non-tonic cells:
# spike_periodic: false if not tonic and no IBIs are saved
# cellperiod: 0 if no IBIs
# cellibispercycle: 0 if no IBIs

# to find out if a cell with IBIs is periodic, go backwards through the IBIs
# if difference between any two consecutive IBIs is less than 5% as a fraction of their
# average, this is consistend with periodic bursting
# so you can read off the burst period 1
# then compare the current ibi to second to last detected, then third, for burst period
# 2 and 3 resp.
# then take the smallest of the three periods, and corresponding ibispercycle

# now to find the number of IBIs per period
# temporary time window : end is middle of last IBI, start is one period before the end
# time (cellperiod * cyclespp)
# find longest IBI that ends within the temp time window. set window end to middle of
# this IBI
# window start is again one period before window end
# count IBIs within this window
# also count IBIs in window one period before this

# Do the same for ICIs, using IMIs instead of ISIs

# Detect for periodicity
# cell-wise:
# for periodicity analysis want more than 5 spikes since simulation start
# if cell tonic, it is periodic, period is mean cell ISI, IBIspercycle=1
# else use above procedure

# whole system
# if all cells are periodic or max periodic:
# find max period
# for each cell, cyclesppis maxperiod / cellperiod, rounded to integer
# count Ibispp or ICIspp

# if cyclesperpp equal values for last cycle, increase concistentcyclesperpp
# same for ibiicipp
# if one of these s true, cell is called consistent
# if consistent:
# averageperiod: average of cyclespp * cellperiod
# maxperioddeviation
# if maxperioddeviation < 0.5% avgperiod:
# period = avgperiod

# Now go through all cells. Tonic cells are to be checked.
# Cells that spike periodically are to be checked if cyclespp * cellibispercycle ==
# ibiicipp and the latter equal ibiicippbefore
# Analogous for non-spiking periodical cells

# If all cells are checked and we have enough time (at least 2 * period), periodic =
# true
# else periodic is false and no consistent cycles per period or IBIs or ICIs are found
# If no periodic spikes, do the same for subthreshold maxima


# for periodic cells, if cyclespp are all 1 andn ibispercycle are all 1, triphasic is
# true
# rhythm is pyloric if bursts are in correct order

# Maxima detected by slope switching from pos to neg
##############################################################


class PrinzStats:
    def __init__(
        self, setup, t_on, t_off, dt,
    ):
        """Compute summary statistics."""
        self.setup = setup
        self.t_on = t_on
        self.t_off = t_off
        self.dt = dt

    def calc_dict(self, simulation_output):

        t_max = self.t_off
        dt = self.dt
        t_on = self.t_on
        t_off = self.t_off

        v = simulation_output["voltage"]
        v = v[:, int(t_on / dt) : int(t_off / dt)]

        if "energy" in simulation_output.keys():
            energy = simulation_output["energy"]
            energy = energy[:, int(t_on / dt) : int(t_off / dt)]
        else:
            energy = np.zeros_like(v)

        # Retrieve summary statistics stored in a dictionary.
        stats = self.calc_summ_stats(v, energy, t_max, dt)

        keys = [
            "cycle_period",
            "burst_durations",
            "duty_cycles",
            "start_phases",
            "starts_to_starts",
            "ends_to_starts",
            "phase_gaps",
            "plateau_durations",
            "voltage_means",
            "voltage_stds",
            "voltage_skews",
            "voltage_kurtoses",
            "num_bursts",
            "num_spikes",
            "spike_times",
            "spike_heights",
            "rebound_times",
            "energies",
            "energies_per_burst",
            "energies_per_spike",
            "pyloric_like",
        ]

        new_dict = {}
        neuron_types = ["PM", "LP", "PY"]
        general_names = []
        specific_names = []
        all_data = []

        for key in keys:
            if self.setup[key]:
                new_dict[key] = stats[key]
                if isinstance(stats[key], float) or isinstance(stats[key], bool):
                    general_names += [key]
                    specific_names += ["Circuit"]
                    all_data += [stats[key]]
                else:
                    if key in [
                        "ends_to_starts",
                        "starts_to_starts",
                        "phase_gaps",
                    ]:
                        all_data += [
                            stats[key][neuron_types[i], neuron_types[j]]
                            for i, j in ((0, 1), (1, 2))
                        ]
                        general_names += [key] * 2
                        specific_names += ["AB/PD-LP", "LP-PY"]
                    elif key == "start_phases":
                        all_data += [stats[key][neuron_types[i]] for i in (1, 2)]
                        general_names += [key] * 2
                        specific_names += ["LP", "PY"]
                    else:
                        for specific_name in stats[key].keys():
                            general_names += [key]
                            specific_names += [specific_name]
                            all_data += [stats[key][specific_name]]

        all_data = np.asarray(all_data)
        general_names = np.asarray(general_names)
        specific_names = np.asarray(specific_names)

        pd_ret = pd.DataFrame(
            np.asarray([all_data]), columns=[general_names, specific_names]
        )

        return pd_ret

    @staticmethod
    def analyse_neuron(t, v_original, energy):
        """Analyse voltage trace of a single neuron."""

        # Minimum number of spikes per second for a neuron not to be considered silent
        silent_thresh = 1
        # Minimum number of spikes per second for a neuron to be considered tonically.
        # firing.
        tonic_thresh = 30
        # Maximum percentage of single-spike bursts for a neuron to be considered.
        # bursting (and not simply spiking)
        burst_thresh = 0.5
        # Minimum number of milliseconds for bursts not to be discounted as single
        # spikes.
        bl_thresh = 40
        # Minimum voltage above which spikes are considered.
        spike_thresh = -10
        # Maximum time between spikes in a given burst (ibi = inter-burst interval).
        ibi_threshold = 150
        # Minimum voltage to be considered for plateaus.
        plateau_threshold = -30

        nan = float("nan")

        t_max = t[-1]
        window_size = int(0.5 / (t[1] - t[0]))
        if window_size % 2 == 0:
            window_size += 1

        voltage_mean = np.mean(v_original)
        voltage_std = np.std(v_original)
        voltage_skew = scipy.stats.skew(v_original)
        voltage_kurtosis = scipy.stats.kurtosis(v_original)

        v = scipy.signal.savgol_filter(v_original, window_size, 3)

        # V = scipy.signal.savgol_filter(V, int(1 / dt), 3)
        # Remaining negative slopes are at spike peaks.
        spike_indices = np.where(
            (v[1:-1] > spike_thresh) & (np.diff(v[:-1]) >= 0) & (np.diff(v[1:]) <= 0)
        )[0]
        spike_times = t[spike_indices]
        num_spikes = len(spike_times)

        spike_heights = v_original[spike_indices]

        # refractory period begins when slopes start getting positive again
        rebound_indices = np.where(
            (v[1:-1] < spike_thresh) & (np.diff(v[:-1]) <= 0) & (np.diff(v[1:]) >= 0)
        )[0]

        # assign rebounds to the corresponding spikes, save their times in rebound_times
        if len(rebound_indices) == 0:
            rebound_times = np.empty_like(spike_times) * nan
        else:
            rebound_times = np.empty_like(spike_times)

            # for each spike, find the corresponding rebound (NaN if it doesn't exist)
            for i in range(num_spikes):
                si = spike_indices[i]
                rebound_ind = rebound_indices[np.argmax(rebound_indices > si)]
                if rebound_ind <= si:
                    rebound_times[i:] = nan
                    break

                rebound_times[i] = t[rebound_ind]

        total_energy = np.sum(energy)

        # neurons with no spikes are boring
        if len(spike_times) == 0:
            neuron_type = "silent"
            burst_start_times = []
            burst_end_times = []
            burst_times = []
            num_bursts = 0
            energy_per_burst = 0

            plateau_durations = nan
            avg_burst_length = nan
            avg_ibi_length = nan
            avg_cycle_length = nan
            avg_spike_length = nan
            mean_energy_per_spike_in_all_bursts = 0.0
        else:

            # calculate average spike lengths, using spikes which have terminated
            last_term_spike_ind = (
                -1 if np.isnan(rebound_times[-1]) else len(spike_times)
            )
            if (
                len(spike_times) == 0 and last_term_spike_ind == -1
            ):  # No terminating spike
                avg_spike_length = nan
            else:
                avg_spike_length = np.mean(
                    rebound_times[:last_term_spike_ind]
                    - spike_times[:last_term_spike_ind]
                )

            # group spikes into bursts, via finding the spikes that are followed by a
            # gap of at least 100 ms
            # The last burst ends at the last spike by convention
            burst_end_spikes = np.append(
                np.where(np.diff(spike_times) >= ibi_threshold)[0], num_spikes - 1
            )

            # The start of a burst is the first spike after the burst ends, or the
            # first ever spike
            burst_start_spikes = np.insert(burst_end_spikes[:-1] + 1, 0, 0)

            # Find the times of the spikes
            burst_start_times = spike_times[burst_start_spikes]
            burst_end_times = spike_times[burst_end_spikes]

            burst_times = np.stack((burst_start_times, burst_end_times), axis=-1)

            burst_lengths = burst_times.T[1] - burst_times.T[0]

            cond = burst_lengths > bl_thresh

            burst_start_times = burst_start_times[cond]
            burst_end_times = burst_end_times[cond]
            burst_times = burst_times[cond]
            burst_lengths = burst_lengths[cond]

            num_bursts = len(burst_times)

            # Energy consumption
            cum_energy_per_spike_in_burst = 0.0
            t = np.asarray(t)
            energies_per_burst = []
            for running_ind in range(len(burst_start_times)):
                burst_start = burst_start_times[running_ind]
                burst_end = burst_end_times[running_ind]
                burst_start_ind = np.where(t == burst_start)[0][0]
                burst_end_ind = np.where(t == burst_end)[0][0]

                # Get energy within bursts.
                # adding 80 cause we want to start 2 ms earlier and end 12 ms later.
                # 2 ms / 0.025 Hz = 80
                energy_within_burst = deepcopy(
                    energy[burst_start_ind - 80 : burst_end_ind + 480]
                )
                energies_per_burst.append(np.sum(energy_within_burst))

                cum_energy_per_spike_in_burst += np.sum(energy_within_burst)
            mean_energy_per_spike_in_all_bursts = (
                cum_energy_per_spike_in_burst / np.maximum(1, num_spikes)
            )
            energy_per_burst = (
                np.mean(energies_per_burst) if energies_per_burst else nan
            )

            # PLATEAUS
            # we cluster the voltage into blocks. Each block starts with the current
            # burst's start time and ends with the next burst's start time. Then,
            # extract the longest sequence of values that are larger than
            # plateau_threshold within each block. Lastly, take the mean of those max
            # values. If no plateaus exist, the longest sequence is defined through the
            # length of the action potentials. Thus, if the length does not exceed some
            # threshold we simply set it to 100.

            longest_list = []
            t = np.asarray(t)
            above_th_all = v > plateau_threshold
            stepping = 10  # subsampling for computational speed
            for running_ind in range(len(burst_start_times)):
                if running_ind == len(burst_start_times) - 1:
                    next_burst_start = spike_times[-1]
                else:
                    next_burst_start = burst_start_times[running_ind + 1]
                burst_start = burst_start_times[running_ind]
                burst_start_ind = np.where(t == burst_start)[0][0]
                next_burst_start_ind = np.where(t == next_burst_start)[0][0]  #

                abouve_th = deepcopy(above_th_all[burst_start_ind:next_burst_start_ind])
                abouve_th = abouve_th[::stepping]
                longest = 0
                current = 0
                for num in abouve_th:
                    if num:
                        current += 1
                    else:
                        longest = max(longest, current)
                        current = 0
                running_ind += 1
                longest_list.append(longest * stepping)
            plateau_durations = np.mean(longest_list) if longest_list else nan
            if plateau_durations < 200:
                # Make sure that the duration of a single spike is not a feature
                plateau_durations = 100
            plateau_durations *= t[1] - t[0]  # convert to ms

            avg_burst_length = np.mean(burst_lengths) if burst_lengths.tolist() else nan

            if len(burst_times) == 1:
                avg_ibi_length = nan

            else:
                ibi_lengths = burst_times.T[0][1:] - burst_times.T[1][:-1]
                avg_ibi_length = np.mean(ibi_lengths) if ibi_lengths.tolist() else nan

            # A neuron is classified as bursting if we can detect multiple bursts and
            # not too many bursts consist of single spikes (to separate bursting
            # neurons from singly spiking neurons).
            if len(burst_times) == 1:
                neuron_type = "non-bursting"
                avg_cycle_length = nan
            else:
                if len(burst_times) / len(spike_times) >= burst_thresh:
                    neuron_type = "bursting"
                else:
                    neuron_type = "non-bursting"

                cycle_lengths = np.diff(burst_times.T[0])
                avg_cycle_length = (
                    np.mean(cycle_lengths) if cycle_lengths.tolist() else nan
                )

        # A neuron is classified as silent if it doesn't spike enough and as tonic if
        # it spikes too much.
        # Recall that tmax is given in ms.
        if len(spike_times) * 1e3 / t_max <= silent_thresh:
            neuron_type = "silent"
        elif len(spike_times) * 1e3 / t_max >= tonic_thresh:
            neuron_type = "tonic"

        return {
            "neuron_type": neuron_type,
            "avg_spike_length": avg_spike_length,
            "num_spikes": num_spikes,
            "spike_times": spike_times,
            "rebound_times": rebound_times,
            "burst_start_times": burst_start_times,
            "burst_end_times": burst_end_times,
            "avg_burst_length": avg_burst_length,
            "avg_cycle_length": avg_cycle_length,
            "avg_ibi_length": avg_ibi_length,
            "plateau_durations": plateau_durations,
            "energies_per_spike": mean_energy_per_spike_in_all_bursts,
            "num_bursts": num_bursts,
            "energies_per_burst": energy_per_burst,
            "energies": total_energy,
            "spike_heights": spike_heights,
            "voltage_means": voltage_mean,
            "voltage_stds": voltage_std,
            "voltage_skews": voltage_skew,
            "voltage_kurtoses": voltage_kurtosis,
        }

    # Analyse voltage traces; check for triphasic (periodic) behaviour
    def analyse_data(self, data, energies, t_max, dt):
        neuron_types = ["PM", "LP", "PY"]
        ref_neuron = neuron_types[0]

        # Percentage of triphasic periods for system to be considered triphasic.
        triphasic_thresh = 0.9
        nan = float("nan")

        t = np.arange(0, t_max, dt)
        v = data

        assert len(v) == len(neuron_types)

        stats = {
            neutype: self.analyse_neuron(t, np.asarray(V), energy)
            for V, neutype, energy in zip(v, neuron_types, energies)
        }

        # if one neuron does not have a periodic rhythm, the whole system is not
        # considered triphasic
        if np.isnan(stats[ref_neuron]["avg_cycle_length"]):
            cycle_period = nan
            period_times = []

            triphasic = False
            period_data = []
        else:
            # The system period is determined by the periods of a fixed neuron (PM)
            ref_stats = stats[ref_neuron]
            period_times = ref_stats["burst_start_times"]
            cycle_period = np.mean(np.diff(period_times))

            # Analyse the periods, store useful data and check if the neuron is
            # triphasic
            n_periods = len(period_times)
            period_data = []
            period_triphasic = np.zeros(n_periods - 1)
            for i in range(n_periods - 1):
                # The start and end times of the given period, and the starts of the
                # neurons' bursts
                # within this period
                pst, pet = period_times[i], period_times[i + 1]
                burst_starts = {}
                burst_ends = {}

                for nt in neuron_types:
                    bs_nt = stats[nt]["burst_start_times"]
                    be_nt = stats[nt]["burst_end_times"]

                    if len(bs_nt) == 0:
                        burst_starts[nt] = []
                        burst_ends[nt] = []
                    else:
                        cond = (pst <= bs_nt) & (bs_nt < pet)
                        burst_starts[nt] = bs_nt[cond]
                        burst_ends[nt] = be_nt[cond]

                # A period is classified as triphasic if all neurons start to burst
                # once within the period
                if np.all([len(burst_starts[nt]) == 1 for nt in neuron_types]):
                    period_triphasic[i] = 1
                    period_data.append(
                        {nt: (burst_starts[nt], burst_ends[nt]) for nt in neuron_types}
                    )

            # if we have at least two periods and most of them are triphasic, classify
            # the system as triphasic
            if n_periods >= 2:
                triphasic = np.mean(period_triphasic) >= triphasic_thresh
            else:
                triphasic = False

        stats.update(
            {
                "cycle_period": cycle_period,
                "period_times": period_times,
                "triphasic": triphasic,
                "period_data": period_data,
            }
        )

        return stats

    def calc_summ_stats(self, data, energies, tmax, dt):
        """
        Compute features of single neurons and use them to return circuit-level stats.
        """
        neuron_types = ["PM", "LP", "PY"]
        # Percentage of pyloric periods for triphasic system to be pyloric_like.
        pyloric_thresh = 0.7

        nan = float("nan")

        single_neuron_stats = self.analyse_data(data, energies, tmax, dt)

        burst_durations = {}
        duty_cycles = {}
        start_phases = {}
        starts_to_starts = {}
        ends_to_starts = {}
        phase_gaps = {}
        plateau_durations = {}
        energy = {}
        num_bursts = {}
        energy_per_burst = {}
        total_energy = {}
        num_spikes = {}
        spike_times = {}
        spike_heights = {}
        rebound_times = {}
        voltage_mean = {}
        voltage_std = {}
        voltage_skew = {}
        voltage_kurtosis = {}

        for nt in neuron_types:
            burst_durations[nt] = single_neuron_stats[nt]["avg_burst_length"]
            duty_cycles[nt] = burst_durations[nt] / single_neuron_stats["cycle_period"]
            plateau_durations[nt] = single_neuron_stats[nt]["plateau_durations"]
            energy[nt] = single_neuron_stats[nt]["energies_per_spike"] / 1000 * dt
            num_bursts[nt] = single_neuron_stats[nt]["num_bursts"]
            energy_per_burst[nt] = (
                single_neuron_stats[nt]["energies_per_burst"] / 1000 * dt
            )
            total_energy[nt] = single_neuron_stats[nt]["energies"] / 1000 * dt
            num_spikes[nt] = single_neuron_stats[nt]["num_spikes"]
            spike_times[nt] = single_neuron_stats[nt]["spike_times"]
            spike_heights[nt] = single_neuron_stats[nt]["spike_heights"]
            rebound_times[nt] = single_neuron_stats[nt]["rebound_times"]
            voltage_mean[nt] = single_neuron_stats[nt]["voltage_means"]
            voltage_std[nt] = single_neuron_stats[nt]["voltage_stds"]
            voltage_skew[nt] = single_neuron_stats[nt]["voltage_skews"]
            voltage_kurtosis[nt] = single_neuron_stats[nt]["voltage_kurtoses"]

            if not single_neuron_stats["triphasic"]:
                for nt2 in neuron_types:
                    ends_to_starts[nt, nt2] = nan
                    phase_gaps[nt, nt2] = nan
                    starts_to_starts[nt, nt2] = nan

                start_phases[nt] = nan
            else:
                # triphasic systems are candidate pyloric-like systems, so we collect
                # some information
                for nt2 in neuron_types:
                    list_es = [
                        e[nt2][0] - e[nt][1] for e in single_neuron_stats["period_data"]
                    ]
                    ends_to_starts[nt, nt2] = np.mean(list_es) if list_es else nan
                    phase_gaps[nt, nt2] = (
                        ends_to_starts[nt, nt2] / single_neuron_stats["cycle_period"]
                    )
                    list_ss = [
                        e[nt2][0] - e[nt][0] for e in single_neuron_stats["period_data"]
                    ]
                    starts_to_starts[nt, nt2] = np.mean(list_ss) if list_ss else nan

                start_phases[nt] = (
                    starts_to_starts[neuron_types[0], nt]
                    / single_neuron_stats["cycle_period"]
                )

        # The three conditions from Prinz' paper must hold (most of the time) for the
        # system to be considered pyloric-like
        pyloric_analysis = np.asarray(
            [
                (
                    e[neuron_types[1]][0] - e[neuron_types[2]][0],
                    e[neuron_types[1]][1] - e[neuron_types[2]][1],
                    e[neuron_types[0]][1] - e[neuron_types[1]][0],
                )
                for e in single_neuron_stats["period_data"]
            ]
        )
        pyloric_like = False
        if single_neuron_stats["triphasic"]:
            all_pyloric_analyses = np.all(pyloric_analysis <= 0, axis=1)
            mean_of_pylorics = np.mean(all_pyloric_analyses)
            if mean_of_pylorics >= pyloric_thresh:
                pyloric_like = True

        single_neuron_stats.update(
            {
                "cycle_period": single_neuron_stats["cycle_period"],
                "burst_durations": burst_durations,
                "duty_cycles": duty_cycles,
                "start_phases": start_phases,
                "starts_to_starts": starts_to_starts,
                "ends_to_starts": ends_to_starts,
                "phase_gaps": phase_gaps,
                "plateau_durations": plateau_durations,
                "pyloric_like": pyloric_like,
                "energies_per_spike": energy,
                "num_bursts": num_bursts,
                "energies_per_burst": energy_per_burst,
                "energies": total_energy,
                "num_spikes": num_spikes,
                "spike_times": spike_times,
                "spike_heights": spike_heights,
                "rebound_times": rebound_times,
                "voltage_means": voltage_mean,
                "voltage_stds": voltage_std,
                "voltage_skews": voltage_skew,
                "voltage_kurtoses": voltage_kurtosis,
            }
        )

        # This is just for plotting purposes, a convenience hack
        for nt in neuron_types:
            single_neuron_stats[nt]["period_times"] = single_neuron_stats[
                "period_times"
            ]

        return single_neuron_stats
