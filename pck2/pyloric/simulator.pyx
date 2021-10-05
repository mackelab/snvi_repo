#!/usr/bin/env python
import tqdm

from libc.math cimport exp, log

import numpy as np
cimport numpy as np
import time

import cython
cimport cython


ctypedef double dtype

cdef dtype mhtt(dtype V, dtype num, dtype den):
    return 1.0 / (1 + exp((V + num) / den))

cdef dtype mhtt2(dtype V, dtype num1, dtype den1, dtype num2, dtype den2):
    return 1.0 / (exp((V + num1) / den1) + exp((V + num2) / den2))

cdef dtype getINatauh(dtype V):
    return (1.34 / (1 + exp((V + 62.9) / -10))) * (1.5 + 1 / (1 + exp((V + 34.9) / 3.6)))

cdef dtype getIHtaum(dtype V):
    return 2.0 / (exp(-14.59 - 0.086 * V) + exp(-1.87 + 0.0701 * V))


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sim_time(
    dtype dt,
    np.ndarray[dtype] t,
    Ix_,
    modelx_,
    conns_,
    g_q10_conns_gbar,
    g_q10_conns_tau,
    g_q10_memb_gbar,
    g_q10_memb_tau_m,
    g_q10_memb_tau_h,
    g_q10_memb_tau_CaBuff,
    dtype temp,
    num_energy_timesteps: int,
    num_energyscape_timesteps: int,
    init = None,
    start_val_input=0.0,
    bint verbose = True
):
    """
    Simulates the model for a specified time duration.

    Args:
        dt: Timestep (mS)
        t: Array of time values - should be np.arange(0,tmax,dt)
        Ix_: Input currents for each neuron
        modelx_: Model to use for each neuron
        conns_: List of connections in the form [ #out, #in, strength, Es, kminus ]
            Units: strength: mS, Es: mV, kminus: ms
        temp: Temperature at which the simulation is run.
        num_energy_timesteps: Until which time step the energy should be stored. The
            vector that stores the energy has as many entries as time steps to be
            stored. Hence, if `num_energy_timesteps=0` the memory costs are smaller.
        init: Whether to use a custom initialization for the gating variables. If not
            `None`, set the initial values for voltage, Ca concentration and state
            variables.
        start_val_input: Initial value for the some of the gating variables when
            `init=None`.
        verbose: Whether to show a progressbar during the simulation.

    Returns:
        Dictionary.
    """

    cdef dtype Nval = 0.0

    # Globals cannot yet be defined as C type variables in Cython, so we define them here
    cdef np.ndarray[dtype] INadata     = np.asarray([ 3   , 25.5  , -5.29 , 48.9  , 5.18  , 2.64  , -2.52 , 120   , -25   , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  ])
    cdef np.ndarray[dtype] ICaTdata    = np.asarray([ 3   , 27.1  , -7.2  , 32.1  , 5.5   , 43.4  , -42.6 , 68.1  , -20.5 , Nval  , Nval  , 210   , -179.6, 55    , -16.9 , Nval  , Nval  ])
    cdef np.ndarray[dtype] ICaSdata    = np.asarray([ 3   , 33    , -8.1  , 60    , 6.2   , 2.8   , 14    , 27    , 10    , 70    , -13   , 120   , 300   , 55    , 9     , 65    , -16   ])
    cdef np.ndarray[dtype] IAdata      = np.asarray([ 3   , 27.2  , -8.7  , 56.9  , 4.9   , 23.2  , -20.8 , 32.9  , -15.2 , Nval  , Nval  , 77.2  , -58.4 , 38.9  , -26.5 , Nval  , Nval  ])
    cdef np.ndarray[dtype] IKCadata    = np.asarray([ 4   , 28.3  , -12.6 , Nval  , Nval  , 180.6 , -150.2, 46    , -22.7 , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  ])
    cdef np.ndarray[dtype] IKddata     = np.asarray([ 4   , 12.3  , -11.8 , Nval  , Nval  , 14.4  , -12.8 , 28.3  , -19.2 , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  ])
    cdef np.ndarray[dtype] IHdata      = np.asarray([ 1   , 75    , 5.5   , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  , Nval  ])

    cdef dtype Esglut = -70
    cdef dtype kminusglut = 40

    cdef dtype Eschol = -80
    cdef dtype kminuschol = 100

    # Q10s for synapses
    cdef np.ndarray[dtype, ndim=1] g_temp_factor_conns   = np.asarray(g_q10_conns_gbar) ** ((temp - 283) / 10)
    cdef np.ndarray[dtype, ndim=1] tau_temp_factor_conns = np.asarray(g_q10_conns_tau) ** ((temp - 283) / 10)

    # Q10s for membrane gbar
    cdef np.ndarray[dtype, ndim=1] g_temp_factor_memb    = np.asarray(g_q10_memb_gbar) ** ((temp - 283) / 10)

    # Q10s for membrane tau m
    cdef np.ndarray[dtype, ndim=1] tau_temp_factor_m  = np.asarray(g_q10_memb_tau_m) ** ((temp - 283) / 10)

    # Q10s for membrane tau h
    cdef np.ndarray[dtype, ndim=1] tau_temp_factor_h  = np.asarray(g_q10_memb_tau_h) ** ((temp - 283) / 10)

    # Q10s for membrane tau CaBuff
    cdef np.ndarray[dtype, ndim=1] tau_temp_factor_CaBuff  = np.asarray(g_q10_memb_tau_CaBuff) ** ((temp - 283) / 10)

    # Convert the arguments from lists to numpy arrays
    cdef np.ndarray[dtype, ndim=2] Ix = np.asarray(Ix_)
    cdef np.ndarray[dtype, ndim=2] modelx = np.asarray(modelx_)
    cdef np.ndarray[dtype, ndim=2] conns = np.asarray(conns_)
    conns[:,2] = conns[:,2] * g_temp_factor_conns

    cdef int n = len(modelx) # number of neurons
    cdef int m = len(conns) # number of connections

    # Counting variables
    cdef size_t i, j, k

    modelx = np.asarray(modelx)

    # adding the Q10-value
    #cdef dtype g_q10 = 1.5
    #cdef dtype g_temp_factor = g_q10 ** ((temp - 283) / 10)

    # Parameters
    cdef np.ndarray[dtype] gNax    = modelx.T[0] * g_temp_factor_memb[0]
    cdef np.ndarray[dtype] gCaTx   = modelx.T[1] * g_temp_factor_memb[1]
    cdef np.ndarray[dtype] gCaSx   = modelx.T[2] * g_temp_factor_memb[2]
    cdef np.ndarray[dtype] gAx     = modelx.T[3] * g_temp_factor_memb[3]
    cdef np.ndarray[dtype] gKCax   = modelx.T[4] * g_temp_factor_memb[4]
    cdef np.ndarray[dtype] gKdx    = modelx.T[5] * g_temp_factor_memb[5]
    cdef np.ndarray[dtype] gHx     = modelx.T[6] * g_temp_factor_memb[6]
    cdef np.ndarray[dtype] gleakx  = modelx.T[7] * g_temp_factor_memb[7]

    # Constants
    cdef dtype C = 0.6283e-3

    cdef dtype ENa = 50
    cdef dtype EK = -80
    cdef dtype EH = -20
    cdef dtype Eleak = -50

    cdef dtype Catau = 200
    cdef dtype f = 14961
    cdef dtype CaExt = 3000
    cdef dtype Ca0 = 0.05

    cdef dtype R = 8.31451e3                   # mJ / (mol * K)
    cdef dtype F = 96485.3415;                 # C / mol
    cdef dtype z = 2                           # Ca is divalent
    cdef dtype RToverzF = R * temp / (z * F)      # mJ / (mol * K) * K / (C / mol) = mV

    cdef dtype Vth = -35
    cdef dtype Delta = 5

    cdef int nsteps = len(t)

    cdef dtype tau_q10_m = 1.7
    cdef dtype tau_q10_h = 2.8
    cdef dtype tau_q10_CaBuff = 1.7

    ####################################

    # Neuron state variables
    cdef np.ndarray[dtype, ndim=2] Vx   = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] Cax  = np.empty_like(Ix)

    cdef np.ndarray[dtype, ndim=2] mNax    = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] mCaTx   = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] mCaSx   = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] mAx     = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] mKCax   = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] mKdx    = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] mHx     = np.empty_like(Ix)

    cdef np.ndarray[dtype, ndim=2] hNax    = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] hCaTx   = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] hCaSx   = np.empty_like(Ix)
    cdef np.ndarray[dtype, ndim=2] hAx     = np.empty_like(Ix)

    # Synapse state variables
    cdef np.ndarray[dtype, ndim=2] sx      = np.zeros((m, nsteps))
    cdef np.ndarray[dtype, ndim=2] ICax      = np.zeros((m, nsteps))

    # Currents (only for current timestep)
    cdef np.ndarray[dtype] cNax = np.zeros(n)           # mS
    cdef np.ndarray[dtype] cCaTx = np.zeros(n)          # mS
    cdef np.ndarray[dtype] cCaSx = np.zeros(n)          # mS
    cdef np.ndarray[dtype] cAx = np.zeros(n)            # mS
    cdef np.ndarray[dtype] cKCax = np.zeros(n)          # mS
    cdef np.ndarray[dtype] cKdx = np.zeros(n)           # mS
    cdef np.ndarray[dtype] cHx = np.zeros(n)            # mS
    cdef np.ndarray[dtype] cleakx = np.zeros(n)         # mS

    # Synapse state variables
    cdef np.ndarray[dtype] csx      = np.empty(m)
    cdef np.ndarray[dtype] Icsx     = np.empty(n)
    cdef np.ndarray[dtype] ECax     = np.empty(n)

    cdef dtype Vcoeff, Vinf, Vinf_

    cdef dtype mNatau, mNainf
    cdef dtype mCaTtau, mCaTinf
    cdef dtype mCaStau, mCaSinf
    cdef dtype mAtau, mAinf
    cdef dtype mKCatau, mKCainf
    cdef dtype mKdtau, mKdinf
    cdef dtype mHtau, mHinf

    cdef dtype hNatau, hNainf
    cdef dtype hCaTtau, hCaTinf
    cdef dtype hCaStau, hCaSinf
    cdef dtype hAtau, hAinf

    cdef dtype Cainf

    cdef dtype e, stau, sinf

    cdef int npost, npre

    cdef dtype start_val = start_val_input

    cdef np.ndarray[dtype] current_energy = np.asarray([0.0, 0.0, 0.0])
    cdef np.ndarray[dtype] current_synaptic_energy = np.asarray([0.0, 0.0, 0.0])
    cdef np.ndarray[dtype, ndim=2] energy_over_time = np.empty((3, num_energy_timesteps))

    cdef np.ndarray[dtype, ndim = 2] cond_na = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_cat = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_cas = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_a = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_kca = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_kd = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_h = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_leak = np.empty((3, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] cond_syn = np.empty((7, num_energyscape_timesteps))
    cdef np.ndarray[dtype, ndim = 2] reversal_calcium = np.empty((3, num_energyscape_timesteps))

    ### Always the case if SNPE calls this, init data is passed via constructor (not supported yet)
    if init == None:        # default: simulation from initial point
        for j in range(n):

            Vx[j, 0] = Eleak

            init_inf = False
            if init_inf == True:
                mNainf = mhtt(Vx[j, 0], INadata[1], INadata[2])
                mCaTinf = mhtt(Vx[j, 0], ICaTdata[1], ICaTdata[2])
                mCaSinf = mhtt(Vx[j, 0], ICaSdata[1], ICaSdata[2])
                mAinf = mhtt(Vx[j, 0], IAdata[1], IAdata[2])
                mKCainf = (Cax[j, 0] / (Cax[j, 0] + 3)) * mhtt(Vx[j, 0], IKCadata[1], IKCadata[2])
                mKdinf = mhtt(Vx[j, 0], IKddata[1], IKddata[2])
                mHinf = mhtt(Vx[j, 0], IHdata[1], IHdata[2])
                hNainf = mhtt(Vx[j, 0], INadata[3], INadata[4])
                hCaTinf = mhtt(Vx[j, 0], ICaTdata[3], ICaTdata[4])
                hCaSinf = mhtt(Vx[j, 0], ICaSdata[3], ICaSdata[4])
                hAinf = mhtt(Vx[j, 0], IAdata[3], IAdata[4])

                Cax[j, 0] = Ca0

                mNax[j, 0] = mNainf
                mCaTx[j, 0] = mCaTinf
                mCaSx[j, 0] = mCaSinf
                mAx[j, 0] = mAinf
                mKCax[j, 0] = mKCainf
                mKdx[j, 0] = mKdinf
                mHx[j, 0] = mHinf

                hNax[j, 0] = hNainf
                hCaTx[j, 0] = hCaTinf
                hCaSx[j, 0] = hCaSinf
                hAx[j, 0] = hAinf
            else:
                Cax[j, 0] = Ca0

                hNax[j, 0] = 1
                hCaTx[j, 0] = 1
                hCaSx[j, 0] = 1
                hAx[j, 0] = 1

                mNax[j, 0] = start_val
                mCaTx[j, 0] = start_val
                mCaSx[j, 0] = start_val
                mAx[j, 0] = start_val
                mKCax[j, 0] = start_val
                mKdx[j, 0] = start_val
                mHx[j, 0] = start_val
                hNax[j, 0] = start_val
                hCaTx[j, 0] = start_val
                hCaSx[j, 0] = start_val
                hAx[j, 0] = start_val

            # ICa is calculated later

    else:                  # simulation from given points
        for i in range(n):
            data = init[i]
            Vx[i, 0] = data[0]
            Cax[i, 0] = data[1]

            mNax[i, 0] = data[2]
            mCaTx[i, 0] = data[3]
            mCaSx[i, 0] = data[4]
            mAx[i, 0] = data[5]
            mKCax[i, 0] = data[6]
            mKdx[i, 0] = data[7]
            mHx[i, 0] = data[8]

            hNax[i, 0] = data[10]
            hCaTx[i, 0] = data[11]
            hCaSx[i, 0] = data[12]
            hAx[i, 0] = data[13]

        for i in range(m):
            sx[i, 0] = init[n][i]

    #################################################################################
    # The rest of the function body should be identical in HH.py and cHH.pyx
    # Debug variables
    logs = { 'I' : Ix, 'Is' : np.empty_like(mNax), 'INa' : np.empty_like(mNax), 'ICaT' : np.empty_like(mCaTx), 'ICaS' : np.empty_like(mCaSx), 'IA' : np.empty_like(mAx), \
             'IKCa' : np.empty_like(mKCax), 'IKd' : np.empty_like(mKdx), 'IH' : np.empty_like(mHx), 'Ileak' : np.empty_like(mHx),
              'mNa' : mNax, 'mCaT' : mCaTx, 'mCaS' : mCaSx, 'mA' : mAx, 'mKCa' : mKCax, 'mKd' : mKdx, 'mH' : mHx,
              'hNa' : hNax, 'hCaT' : hCaTx, 'hCaS' : hCaSx, 'hA' : hAx, 's' : sx }

    # Calculate Ca current at time 0
    for j in range(n): # n is the number of neurons
        ECax[j] = -RToverzF * log(Cax[j, 0]/CaExt)             # mV * 1 = mV
        ICax[j, 0] = (gCaTx[j] * (mCaTx[j, 0] ** ICaTdata[0]) * hCaTx[j, 0] + \
                      gCaSx[j] * (mCaSx[j, 0] ** ICaSdata[0]) * hCaSx[j, 0]) * (Vx[j, 0] - ECax[j])     # mS??? * mV = muA

    if verbose:
        iterlist = tqdm.tqdm(range(1, nsteps))
    else:
        iterlist = range(1,nsteps) # nsteps = len(t)
    for i in iterlist:
        # Calculate synaptic currents
        for k in range(n): # n = len(modelx)
            csx[k] = Icsx[k] = 0

        current_synaptic_energy[0] = 0.0
        current_synaptic_energy[1] = 0.0
        current_synaptic_energy[2] = 0.0

        # # store energy in vector
        # if i < num_energy_timesteps:
        for k in range(m): # m = len(conns)
            npost = int(conns[k,0])
            csx[npost] += -conns[k,2] * sx[k, i-1]                  # positive currents inhibit spiking in our model
            Icsx[npost] += -conns[k,2] * sx[k, i-1] * conns[k,3]   # mS * 1 * mV = muA

        if i < num_energy_timesteps:
            for k in range(m): # m = len(conns)
                npost = int(conns[k,0])
                current_synaptic_energy[npost] += -conns[k,2] * sx[k, i-1] * (Vx[npost, i-1] - conns[k,3]) ** 2
                if current_synaptic_energy[npost] < 0.0:
                    print('problem, synaptic cost < 0.0!')
        if i < num_energyscape_timesteps:
            for k in range(m): # m = len(conns)
                cond_syn[k, i] = -conns[k,2] * sx[k, i-1]

        # Update V and [Ca] for all neurons
        for j in range(n):
            # Exponential Euler
            cNax[j] = gNax[j] * (mNax[j, i - 1] ** INadata[0]) * hNax[j, i - 1]         # mS
            cCaTx[j] = gCaTx[j] * (mCaTx[j, i - 1] ** ICaTdata[0]) * hCaTx[j, i - 1]    # mS
            cCaSx[j] = gCaSx[j] * (mCaSx[j, i - 1] ** ICaSdata[0]) * hCaSx[j, i - 1]    # mS
            cAx[j] = gAx[j] * (mAx[j, i - 1] ** IAdata[0]) * hAx[j, i - 1]              # mS
            cKCax[j] = gKCax[j] * (mKCax[j, i - 1] ** IKCadata[0])                      # mS
            cKdx[j] = gKdx[j] * (mKdx[j, i - 1] ** IKddata[0])                          # mS
            cHx[j] = gHx[j] * (mHx[j, i - 1] ** IHdata[0])                              # mS
            cleakx[j] = gleakx[j]                                                       # mS

            # store energy in vector
            if i < num_energy_timesteps:
                # instantaneous energy stemming from membrane currents
                current_energy[j] = cNax[j] * (Vx[j, i-1] - ENa) ** 2 +\
                    cCaTx[j] * (Vx[j, i-1] - ECax[j]) ** 2 +\
                    cCaSx[j] * (Vx[j, i-1] - ECax[j]) ** 2 +\
                    cAx[j] * (Vx[j, i-1] - EK) ** 2 +\
                    cKCax[j] * (Vx[j, i-1] - EK) ** 2 +\
                    cKdx[j] * (Vx[j, i-1] - EK) ** 2 +\
                    cHx[j] * (Vx[j, i-1] - EH) ** 2 +\
                    cleakx[j] * (Vx[j, i-1] - Eleak) ** 2

                # add energy from the synapses
                current_energy[j] = current_energy[j] + current_synaptic_energy[j]
                energy_over_time[j, i] = current_energy[j]

            if i < num_energyscape_timesteps:
                reversal_calcium[j, i] = ECax[j]
                cond_na[j, i] = cNax[j]
                cond_cat[j, i] = cCaTx[j]
                cond_cas[j, i] = cCaSx[j]
                cond_a[j, i] = cAx[j]
                cond_kca[j, i] = cKCax[j]
                cond_kd[j, i] = cKdx[j]
                cond_h[j, i] = cHx[j]
                cond_leak[j, i] = cleakx[j]

            # Calculate Ca reversal potential using Nernst equation
            ECax[j] = RToverzF * log(CaExt / Cax[j, i-1])  # mV * 1 = mV

            ICax[j, i] = (cCaTx[j] + cCaSx[j]) * (Vx[j, i-1] - ECax[j])  # mS??? * mV = muA

            # Catau is a constant defined above
            Cainf = Ca0 - f * ICax[j, i]  # (muM / muA) * muA = muM
            Cax[j, i] = Cainf + (Cax[j, i-1] - Cainf) * exp(-dt * tau_temp_factor_CaBuff[0] / Catau)  # muM; Exponent: ms / ms = 1

            Vcoeff = csx[j] + cNax[j] + cCaTx[j] + cCaSx[j] + cAx[j] + cKCax[j] + cKdx[j] + cHx[j] + cleakx[j] # mS
            Vinf_ = Icsx[j] + cNax[j] * ENa + cCaTx[j] * ECax[j] + cCaSx[j] * ECax[j] + cAx[j] * EK + cKCax[j] * EK + cKdx[j] *\
                    EK + cHx[j] * EH + cleakx[j] * Eleak + Ix[j, i]
            if Vcoeff == 0:
                Vx[j, i] = Vx[j, i-1] + dt * Vinf_ / C
            else:
                Vinf = Vinf_ / Vcoeff  # muA / mS = mV
                Vx[j, i] = Vinf + (Vx[j, i-1] - Vinf) * exp(-dt * Vcoeff / C)  # ms * mS / muF = 1

        # Update gating variables
        for j in range(n):
            # t_m * dm/dt = m_inf - m
            # Prinz used a truncating Forward Euler scheme for the gating variables
            # Use old values for V instead of new ones? i -> i-1
            mNainf = mhtt(Vx[j, i-1], INadata[1], INadata[2])
            mNatau = INadata[5] + INadata[6] * mhtt(Vx[j, i-1], INadata[7], INadata[8])       # ms
            mNax[j, i] = mNainf + (mNax[j, i-1] - mNainf) * exp(-dt * tau_temp_factor_m[0] / mNatau)

            mCaTinf = mhtt(Vx[j, i-1], ICaTdata[1], ICaTdata[2])
            mCaTtau = ICaTdata[5] + ICaTdata[6] * mhtt(Vx[j, i-1], ICaTdata[7], ICaTdata[8])  # ms
            mCaTx[j, i] = mCaTinf + (mCaTx[j, i-1] - mCaTinf) * exp(-dt * tau_temp_factor_m[1] / mCaTtau)

            mCaSinf = mhtt(Vx[j, i-1], ICaSdata[1], ICaSdata[2])
            mCaStau = ICaSdata[5] + ICaSdata[6] * mhtt2(Vx[j, i-1], ICaSdata[7], ICaSdata[8], ICaSdata[9], ICaSdata[10])    # ms
            mCaSx[j, i] = mCaSinf + (mCaSx[j, i-1] - mCaSinf) * exp(-dt * tau_temp_factor_m[2] / mCaStau)

            mAinf = mhtt(Vx[j, i-1], IAdata[1], IAdata[2])
            mAtau = IAdata[5] + IAdata[6] * mhtt(Vx[j, i-1], IAdata[7], IAdata[8])            # ms
            mAx[j, i] = mAinf + (mAx[j, i-1] - mAinf) * exp(-dt * tau_temp_factor_m[3] / mAtau)

            mKCainf = (Cax[j, i-1] / (Cax[j, i-1] + 3)) * mhtt(Vx[j, i-1], IKCadata[1], IKCadata[2])
            mKCatau = IKCadata[5] + IKCadata[6] * mhtt(Vx[j, i-1], IKCadata[7], IKCadata[8])  # ms
            mKCax[j, i] = mKCainf + (mKCax[j, i-1] - mKCainf) * exp(-dt * tau_temp_factor_m[4] / mKCatau)

            mKdinf = mhtt(Vx[j, i-1], IKddata[1], IKddata[2])
            mKdtau = IKddata[5] + IKddata[6] * mhtt(Vx[j, i-1], IKddata[7], IKddata[8])       # ms
            mKdx[j, i] = mKdinf + (mKdx[j, i-1] - mKdinf) * exp(-dt * tau_temp_factor_m[5] / mKdtau)

            mHinf = mhtt(Vx[j, i-1], IHdata[1], IHdata[2])
            mHtau = getIHtaum(Vx[j, i-1])
            mHx[j, i] = mHinf + (mHx[j, i-1] - mHinf) * exp(-dt * tau_temp_factor_m[6] / mHtau)

            hNainf = mhtt(Vx[j, i-1], INadata[3], INadata[4])
            hNatau = getINatauh(Vx[j, i-1])                   # ms
            hNax[j, i] = hNainf + (hNax[j, i-1] - hNainf) * exp(-dt * tau_temp_factor_h[0] / hNatau)

            hCaTinf = mhtt(Vx[j, i-1], ICaTdata[3], ICaTdata[4])
            hCaTtau = ICaTdata[11] + ICaTdata[12] * mhtt(Vx[j, i-1], ICaTdata[13], ICaTdata[14])      # ms
            hCaTx[j, i] = hCaTinf + (hCaTx[j, i-1] - hCaTinf) * exp(-dt * tau_temp_factor_h[1] / hCaTtau)

            hCaSinf = mhtt(Vx[j, i-1], ICaSdata[3], ICaSdata[4])
            hCaStau = ICaSdata[11] + ICaSdata[12] * mhtt2(Vx[j, i-1], ICaSdata[13], ICaSdata[14], ICaSdata[15], ICaSdata[16])     # ms
            hCaSx[j, i] = hCaSinf + (hCaSx[j, i-1] - hCaSinf) * exp(-dt * tau_temp_factor_h[2] / hCaStau)

            hAinf = mhtt(Vx[j, i-1], IAdata[3], IAdata[4])
            hAtau = IAdata[11] + IAdata[12] * mhtt(Vx[j, i-1], IAdata[13], IAdata[14])                # ms
            hAx[j, i] = hAinf + (hAx[j, i-1] - hAinf) * exp(-dt * tau_temp_factor_h[3] / hAtau)

        for k in range(m):
            # Rewritten to avoid overflow under standard conditions
            npre = int(conns[k,1])
            e = exp((Vth - Vx[npre, i-1]) / Delta)
            sinf = 1 / (1 + e)
            stau =  conns[k,4] * (1 - sinf) / tau_temp_factor_conns[k]     # 1 / ms^-1 = ms

            sx[k, i] = sx[k, i-1] + (sinf - sx[k, i-1]) * dt / stau  # ms / ms = 1
            if dt > stau:
                sx[k, i] = sinf

    membrane_conds = np.asarray([cond_na, cond_cat, cond_cas, cond_a,
                                       cond_kca, cond_kd, cond_h, cond_leak])
    membrane_conds = np.transpose(membrane_conds, (1,0,2))
    ret = {
        'Vs': Vx,
        'Cas': Cax,
        'ICas': ICax,
        'logs': logs,
        'energy': energy_over_time / (0.628*1e-3),
        'membrane_conds': membrane_conds / (0.628*1e-3),
        'synaptic_conds': cond_syn / (0.628*1e-3),
        'reversal_calcium': reversal_calcium,
    }
    return ret
