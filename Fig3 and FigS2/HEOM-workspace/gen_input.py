import json
import numpy as np
import armadillo as arma
from bath_gen_Drude_PSD import generate

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================
#                                       Auxiliary functions     
# ==============================================================================================

def delta(m, n):
    return 1 if m == n else 0

def get_Hs(n_el, n_ph, dG, V, w_c, g_DA, t_DA, E_R):

    nfock = n_el * n_ph
    hams = np.zeros((nfock, nfock), dtype = complex)

    E_D = dG / 2
    E_A = - dG / 2
    g_D = g_DA * w_c / 2
    g_A = - g_DA * w_c / 2

    for i in range(n_el):
        for j in range(n_el):
            for m in range(n_ph):
                for n in range(n_ph):
                    
                    hams[n_el * m + i, n_el * n + j] = ( (E_D + g_D**2 / w_c + m * w_c) * delta(i, 0) * delta(j, 0) * delta(m, n)

                        + (E_A + g_A**2 / w_c + E_R + m * w_c) * delta(i, 1) * delta(j, 1) * delta(m, n)

                        + (V + (g_D + g_A) * t_DA / w_c) * delta(i, 0) * delta(j, 1) * delta(m, n)

                        + (V + (g_D + g_A) * t_DA / w_c) * delta(i, 1) * delta(j, 0) * delta(m, n)

                        + g_D * delta(i, 0) * delta(j, 0) * (np.sqrt(n + 1) * delta(m, n + 1) + np.sqrt(n) * delta(m, n - 1)) 

                        + g_A * delta(i, 1) * delta(j, 1) * (np.sqrt(n + 1) * delta(m, n + 1) + np.sqrt(n) * delta(m, n - 1)) 
                        
                        + t_DA * delta(i, 0) * delta(j, 1) * (np.sqrt(n + 1) * delta(m, n + 1) + np.sqrt(n) * delta(m, n - 1)) 

                        + t_DA * delta(i, 1) * delta(j, 0) * (np.sqrt(n + 1) * delta(m, n + 1) + np.sqrt(n) * delta(m, n - 1)) 

                        )

    return hams

def get_Qs(n_el, n_ph):

    nfock = n_el * n_ph
    Qs = np.zeros((nfock, nfock), dtype = complex)

    for i in range(n_el):
        for j in range(n_el):
            for m in range(n_ph):
                for n in range(n_ph):

                    Qs[n_el * m + i, n_el * n + j] = delta(i, 1) * delta(j, 1) * delta(m, n)

    return Qs

def get_rho0(n_el, n_ph):

    nfock = n_el * n_ph
    rho0 = np.zeros((nfock, nfock), dtype = complex)

    for i in range(nfock):
        for j in range(nfock):

            rho0[i, j] = delta(i, 0) * delta(j, 0)
    
    return rho0

# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================

class parameters:

    # ===== DEOM propagation scheme =====
    dt = 0.025 * fs_to_au
    t = 5000 * fs_to_au    # plateau time as 20ps for HEOM
    nt = int(t / dt)
    nskip = 100

    lmax = 100
    nmax = 1000000
    ferr = 1.0e-07

    # system parameters
    n_el = 2
    dG   = 0.4 / conv
    V    = 30 * cm_to_au
    w_c  = 0.2 / conv
    
    # === select a situation ===
    # outside cavity use
    # n_ph = 1
    # g_DA = 0.0
    # t_DA = 0.0

    # inside cavity use
    n_ph = 6
    g_DA = 0.5
    t_DA = 0.5 * cm_to_au

    # ===========================

    # total number of states
    nfock = n_el * n_ph

    # ===== Drude-Lorentz model =====
    temp    = 300 / au_to_K                             # temperature
    nmod    = 1                                         # number of dissipation modes

    # Bath I parameters, Drude-Lorentz model
    lam     = 0.2 / conv
    gamma   = 20 * cm_to_au                      # bath characteristic frequency
    
    # PSD scheme
    pade    = 1                            # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsd    = 2                            # number of Pade terms

    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    
    Qs = get_Qs(n_el, n_ph)
    hams = get_Hs(n_el, n_ph, dG, V, w_c, g_DA, t_DA, lam)
    rho0 = get_rho0(n_el, n_ph)
    
# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # bath
    temp = parameters.temp
    nmod = parameters.nmod
    lam = parameters.lam
    gamma = parameters.gamma
    pade = parameters.pade
    npsd = parameters.npsd
    # system
    nfock = parameters.nfock
    hams = parameters.hams
    rho0 = parameters.rho0
    # system-bath
    Qs = parameters.Qs
    # DEOM
    dt = parameters.dt
    nt = parameters.nt
    nskip = parameters.nskip
    lmax = parameters.lmax
    nmax = parameters.nmax
    ferr = parameters.ferr

# ==============================================================================================================================
    # hidx
    ini['hidx']['trun'] = 0
    ini['hidx']['lmax'] = lmax
    ini['hidx']['nmax'] = nmax
    ini['hidx']['ferr'] = ferr

	# bath
    ini['bath']['temp'] = temp
    ini['bath']['nmod'] = nmod
    ini['bath']['jomg'] = [{"jdru":[(lam, gamma)]}]
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd

    jomg = ini['bath']['jomg']
    nind = 0
    for m in range(nmod):
        try:
            ndru = len(jomg[m]['jdru'])
        except:
            ndru = 0
        try:
            nsdr = len(jomg[m]['jsdr'])
        except:
            nsdr = 0
        nper = ndru + 2 * nsdr + npsd
        nind += nper

    etal, etar, etaa, expn, delr = generate (temp, npsd, pade, jomg)
    
    mode = np.zeros((nind), dtype = int)

    arma.arma_write(mode, 'inp_mode.mat')
    arma.arma_write(delr, 'inp_delr.mat')
    arma.arma_write(etal, 'inp_etal.mat')
    arma.arma_write(etar, 'inp_etar.mat')
    arma.arma_write(etaa, 'inp_etaa.mat')
    arma.arma_write(expn, 'inp_expn.mat')

    np.savetxt("etal", etal)
    np.savetxt("expn", expn)

    # dissipation mode
    qmds = np.zeros((nmod, nfock, nfock), dtype = complex)
    qmds[0,:,:] = Qs       # the electron-phonon interaction, H_sb = R ⊗ \sum_j c_j x_j, described by J(w)

    arma.arma_write (hams,ini['syst']['hamsFile'])
    arma.arma_write (qmds,ini['syst']['qmdsFile'])
    arma.arma_write (rho0,'inp_rho0.mat')

    jsonInit = {"deom":ini,
                "rhot":{
                    "dt": dt,
                    "nt": nt,
                    "nk": nskip,
					"xpflag": 1,
					"staticErr": 0,
                    "rho0File": "inp_rho0.mat",
                    "sdipFile": "inp_sdip.mat",
                    "pdipFile": "inp_pdip.mat",
					"bdipFile": "inp_bdip.mat"
                },
            }
    
    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4)
