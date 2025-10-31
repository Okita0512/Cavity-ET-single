"""
Reproducing [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)] with Fermi's Golden Rule Rates
Composed by Wenxiang Ying, 08/25/2025
"""

from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':18}
fig = plt.figure(figsize=(7, 7), dpi = 128)

# ==============================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
meV_to_au = 1 / (conv * 1000)               # 1 meV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
K_to_au = 1.0 / 3.1577464e+05               # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.
# ==============================================================================

# Model Parameters
Delta = 245 * cm_to_au      # diabatic coupling
T = 300 * K_to_au           # temperature
beta = 1.0 / T

# phonon bath parameter
lam = 1.0 / conv            # reorganization energy
gamma = 20 * cm_to_au
alpha = 2 * lam / gamma

# effective bath parameter
w_c = 2.0 / conv            # cavity frequency
t_DA = 69 * cm_to_au
g_DA = 0

# ==============================================================================

def coth(x):
    return 1. / np.tanh(x)

# Brownian spectral density
def J_Brownian(x, Lam, Gamma, ws):
    return 2 * Lam * Gamma * ws**2 * x / ((ws**2 - x**2)**2 + (Gamma * x)**2)

# discretization of Ohmic spectral density (with exponential cutoff)
def bathParam_2(ωc, alpha, ndof):     # for bath descritization

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):

        ω[d] =  - ωc * np.log(1 - (d + 1)/(ndof + 1))
        c[d] =  np.sqrt(alpha * ωc * ω[d]/ (2 * (ndof + 1)))

    return c, ω

# for efficient discretization of Jeff(w)
def bathParam_cav(Lam, Gamma, ws, ndof):

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))

    ω =  np.linspace(1e-8 * meV_to_au, w_c + 20 * Gamma, ndof)
    dω = ω[1] - ω[0]
    for j in range(ndof):
        c[j] =  np.sqrt((2 / np.pi) * ω[j] * J_Brownian(ω[j], Lam, Gamma, ws) * dω)
    
    return c, ω

# terms that going to be sum
def g_j(c_j, w_j, t):
    return - (c_j**2 / (w_j**2)) * (coth(beta*w_j/2) * (1 - np.cos(w_j * t)) + 1.0j * np.sin(w_j * t))

# FT 
def DFT(w_c, Gamma_c):

    ndof = 100
    Q = w_c / Gamma_c
    print("Cavity Q factor:", Q)
    ndof_c = 300    # make sure error of reorg < 1%

    UP = 2**5 * fs_to_au
    dt = 2**(-19) * fs_to_au
    nfft = int(UP / dt)
    print(nfft)

    t = - np.linspace(0, UP, nfft)
    ft = np.zeros((nfft), dtype = complex)
    A_t = np.zeros((nfft), dtype = complex)
    gt = np.zeros((nfft), dtype = complex)
    ft_cav = np.zeros((nfft), dtype = complex)

    c, ω = bathParam_2(gamma, alpha, ndof) 
    # debug: check if the reorganization energy is correct
    print("E_R / 1.0 eV = 1? ==>", np.sum(c**2 / ω) / (1.0 / conv))

    for j in range(len(c)):
        ft += g_j(c[j], ω[j], t)

    C, W = bathParam_cav(1.0 / (2 * w_c**2), Gamma_c, w_c, ndof_c) 
    # debug: check if the reorganization energy is correct
    print("E_R * (2 w_c**2) = 1? ==>", np.sum(C**2 / (2 * W**2)) * (2 * w_c**2))

    for j in range(len(C)):
        
        A_t += (C[j]**2 / W[j]) * (- np.cos(W[j] * t) + 1.0j * np.sin(W[j] * t) * coth(beta * W[j] / 2))
        gt += t_DA**2 * C[j]**2 * (np.cos(W[j] * t) * coth(beta * W[j] / 2) - 1.0j * np.sin(W[j] * t))
    
        ft_cav += g_DA**2 * w_c**2 * g_j(C[j], W[j], t) # * 2 * W[j]

    ht = (Delta + t_DA * g_DA * w_c * A_t) * (Delta + t_DA * g_DA * w_c * A_t)
    ft_cav += ft
    
    C_t_out = 2 * Delta**2 * np.exp(ft)
    C_t_cav = 2 * (ht + gt) * np.exp(ft_cav)

#    plt.plot(t, C_t_out, '-', color = 'black', label = 'Ct Outside Cavity')
#    plt.plot(t, C_t_cav, '--', color = 'red', label = 'Ct Inside Cavity')
#    plt.legend(frameon = False)
#    plt.show()

    return (2 * np.pi) * t / (UP * dt), np.real(fft(C_t_cav)) * dt

# ==============================================================================
# Evaluate Marcus rate, Eq. 38 of [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]
w = - np.linspace(0, 3 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
Exp_1 = np.exp(- beta * (- w - lam)**2 / (4 * lam))
Exp_2 = np.exp(- beta * (- w - lam - w_c)**2 / (4 * lam))
k_Marcus_out = Prefactor * Delta**2 * Exp_1
k_Marcus_in = Prefactor * (Delta**2 * Exp_1 + t_DA**2 * Exp_2)

# ==============================================================================
plt.semilogy(- w * conv, k_Marcus_out * conv, '-', linewidth = 3, color = '#444444', label = 'Marcus Outside Cavity')
plt.semilogy(- w * conv, k_Marcus_in * conv, '--', linewidth = 3, color = "#FD0000", label = 'Marcus Inside Cavity')

Gamma_c = 1000 * meV_to_au
t, y_c = DFT(w_c, Gamma_c)
plt.semilogy(- t * conv, y_c * conv, '-', linewidth = 2.5, color = 'orange', label = r'$\mathcal{Q} = 2$') # , label = r'$200$ meV'

# save data
np.savetxt("dG", t[:100] * conv)
np.savetxt("k_in", y_c[:100] * conv)
# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
x1, x2 = 0, 3      # x-axis range: (x1, x2)
y1, y2 = 1e-8, 1e-1     # y-axis range: (y1, y2)

plt.xlim(x1, x2)
plt.ylim(y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(1)
x_minor_locator = MultipleLocator(0.2)
# y_major_locator = MultipleLocator(0.5)
# y_minor_locator = MultipleLocator(0.1)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8, labelsize = 10)
ax.tick_params(which = 'minor', length = 4)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 20, which = 'both', direction = 'in')

# RHS y-axis
ax2 = ax.twinx()
# ax2.yaxis.set_major_locator(y_major_locator)
# ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 8)
ax2.tick_params(which = 'minor', length = 4)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'- $\Delta$ G (eV)', font = 'Times New Roman', size = 20)
ax.set_ylabel(r'Rate (eV / $\hbar$)', font = 'Times New Roman', size = 20)
# ax.set_title('Gamma=0.001 eV', font = 'Times New Roman', size = 20)

# legend location, font & markersize
ax.legend(loc = 'upper right', prop = font, markerscale = 1, frameon = False)
plt.legend(frameon = False)

# plt.show()

plt.savefig("Q = 2.pdf", bbox_inches='tight')