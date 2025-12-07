"""
Reproducing [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)] with Fermi's Golden Rule Rates
Composed by Wenxiang Ying, 08/25/2025
"""

from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
import math
from numpy.fft import rfft, irfft
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':14}
fig = plt.figure(figsize=(7, 7), dpi = 128)

# ==============================================================================
# conversion factors
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
g_DA = 0.5

# ==============================================================================
# Evaluate Marcus rate, Eq. 38 of [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]
from math import factorial
from scipy.special import assoc_laguerre as laguerre
def FC(m, n):   # Franck–Condon coefficients, see https://doi.org/10.1002/qua.560240843

    gamma = np.sqrt(2) * g_DA

    if m >= n:
        return np.sqrt(factorial(n)/factorial(m)) * (gamma/np.sqrt(2))**(m-n) * laguerre(gamma**2/2, n, m-n) * np.exp(-gamma**2/4)
    else:
        return np.sqrt(factorial(m)/factorial(n)) * (-gamma/np.sqrt(2))**(n-m) * laguerre(gamma**2/2, m, n-m) * np.exp(-gamma**2/4)
    
w = - np.linspace(0, 4 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
L_max = 10           # number of Franck–Condon terms

# Marcus without loss
Exp = np.zeros((L_max, 1000), dtype= float)
for m in range(L_max):
    Exp[m, :] = np.exp(- beta * (- w - lam - m * w_c)**2 / (4 * lam))

k_Marcus_out = Prefactor * Delta**2 * Exp[0, :]
k_Marcus_in = 0
for m in range(L_max):
    k_Marcus_in += Delta**2 * FC(0, m)**2 * Exp[m, :]
    if m >= 1:
        k_Marcus_in += t_DA**2 * FC(1, m)**2 * Exp[m, :]
k_Marcus_in = k_Marcus_in * Prefactor

# ==============================================================================
# GMJ with loss (Eq. 45 of the paper – slow cavity, convolution trick)

# Brownian spectral density
def J_Brownian(x, Lam, Gamma, ws):
    return 2 * Lam * Gamma * ws**2 * x / ((ws**2 - x**2)**2 + (Gamma * x)**2)

# for efficient discretization of Jeff(w)
def bathParam_cav(Lam, Gamma, ws, ndof):

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))

    ω =  np.linspace(1e-8 * meV_to_au, w_c + 7 * Gamma, ndof)
    dω = ω[1] - ω[0]
    for j in range(ndof):
        c[j] =  np.sqrt((2 / np.pi) * ω[j] * J_Brownian(ω[j], Lam, Gamma, ws) * dω)

    return c, ω

# instanteneous Marcus (GMJ Eq. 45)
Gamma_c = 2000 * meV_to_au
ndof_c = 1000
C, W = bathParam_cav(1.0 / (2 * w_c**2), Gamma_c, w_c, ndof_c) 
print("E_R * (2 w_c**2) = 1? ==>", np.sum(C**2 / (2 * W**2)) * (2 * w_c**2))
Expp = np.exp(- np.sum((g_DA * (w_c / W) * C)**2))
Prefactor_inn = Prefactor * Delta**2 * Expp

# m = 0
I_0 = np.exp(- beta * (- w - lam)**2 / (4 * lam))

# m = 1
I_1 = 0
for i in range(len(C)):
    g_i = g_DA * (w_c / W[i]) * C[i]
    I_1 += g_i**2 * np.exp(- beta * (- w - lam - W[i])**2 / (4 * lam))

# general treatment of p-fold sum using FFT convolution
def p_fold_sum(C, W, p):
    
    dW = W[1] - W[0]
    eps = 1e-15
    W_safe = np.clip(W, eps, None)
    a = (g_DA * (w_c / W_safe) * C)**2          
    n = a.size
    Lp = p*n - (p - 1)                          
    nfft = 1 << (Lp - 1).bit_length()

    A = rfft(a, nfft)
    conv_p = irfft(A**p, nfft)[:Lp]             

    S = p*W[0] + dW*np.arange(Lp)
    denom = 4.0 * lam
    u = - w - lam
    D2 = np.subtract.outer(u, S)**2
    K = np.exp(- beta * D2 / denom)

    return (1.0 / factorial(p)) * (K @ conv_p)

I = I_0 + I_1
for p in range(2, L_max):
    I += p_fold_sum(C, W, p)

k_Marcus_inn = Prefactor_inn * I   # GMJ Eq. 45

# ==============================================================================
# Full generalized Marcus–Jortner rate (Eq. 43) and its fast-cavity limit (Eq. 44)

# ----- common definitions -----
u = - w - lam
dW = W[1] - W[0]

# cavity-mode couplings and coefficients
gk = g_DA * (w_c / W) * C   # effective electronic–photonic coupling g'_k
ck = C                      # spectral coefficients c_k

H_DA = Delta
H_AD = Delta
t_DA_p = t_DA               # t'_{DA}
t_AD_p = t_DA               # t'_{AD}

# base GMJ prefactor: sqrt(pi beta / E_R) * exp(-sum |g_k|^2)
Prefactor_GMJ = Prefactor * Expp

I_base = I                  # this is \mathcal{I}(u) appearing with H_DA H_AD in Eq. 45

# ----- first line of Eq. 43: same as Eq. 45 -----
term0 = H_DA * H_AD * I_base

# ----- second line of Eq. 43: single-mode correction -----
# A_k = c_k^2 t'^2 - g_k c_k (H_DA t' + H_AD t')
A = ck**2 * t_DA_p * t_AD_p - gk * ck * (H_DA * t_AD_p + H_AD * t_DA_p)

u_grid = u
I_grid = I_base
term1 = np.zeros_like(u_grid)
for j in range(len(W)):
    shifted = u_grid - W[j]
    I_shift = np.interp(shifted, u_grid, I_grid, left=0.0, right=0.0)
    term1 += A[j] * I_shift

# ----- third line of Eq. 43: two-mode correction -----
d = ck * gk
conv_d = np.convolve(d, d)              # sum over k_beta, k_gamma of d_beta d_gamma
E_min = 2.0 * W[0]
E_sum = E_min + dW * np.arange(conv_d.size)

term2 = np.zeros_like(u_grid)
for idx2 in range(conv_d.size):
    shifted = u_grid - E_sum[idx2]
    I_shift = np.interp(shifted, u_grid, I_grid, left=0.0, right=0.0)
    term2 += conv_d[idx2] * I_shift
term2 *= t_DA_p * t_AD_p

k_GMJ_43 = Prefactor_GMJ * (term0 + term1 + term2)

# ----- fast-cavity limit: Eq. 44 (instantaneous cavity) -----
Gamma_c_fast = 2000 * meV_to_au
ndof_c_fast  = 1000
C_fast, W_fast = bathParam_cav(1.0 / (2 * w_c**2), Gamma_c_fast, w_c, ndof_c_fast)
print("E_R * (2 w_c**2) (fast cavity) = ", np.sum(C_fast**2 / (2 * W_fast**2)) * (2 * w_c**2))

Expp_1 = np.exp(- beta * (- w - lam)**2 / (4 * lam))
Expp_2 = np.zeros_like(w)
for j in range(len(C_fast)):
    Expp_2 += C_fast[j]**2 * np.exp(- beta * (- w - lam - W_fast[j])**2 / (4 * lam))

k_GMJ_44 = Prefactor * (Delta**2 * Expp_1 + t_DA**2 * Expp_2)

# ==============================================================================
# plotting

plt.semilogy(- w * conv, k_Marcus_out * conv, '-', linewidth = 3, color = '#444444',
             label = 'Outside Cavity (Marcus)')
# plt.semilogy(- w * conv, k_Marcus_in * conv, '--', linewidth = 3, color = "#FD0000",
#              label = 'Inside Cavity (MJ, no loss)')
plt.semilogy(- w * conv, k_Marcus_inn * conv, '--', linewidth = 3, color = "orange",
             label = r'Inside Cavity (GMJ, Eq. 45)')
plt.semilogy(- w * conv, k_GMJ_44 * conv, '--', linewidth = 3, color = 'green',
             label = r'Inside Cavity (GMJ, Eq. 44)')
plt.semilogy(- w * conv, k_GMJ_43 * conv, '-', linewidth = 3, color = '#FD0000',
             label = r'Inside Cavity (GMJ, Eq. 43)')

data1 = np.loadtxt("dG", dtype = float)
data2 = np.loadtxt("k_in", dtype = float)
plt.semilogy(- data1, data2, 'o', markersize = 4, markeredgewidth = 2, markerfacecolor = "#0385FF",
             markeredgecolor = '#0385FF', label = r'Inside Cavity (FGR)')

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
x1, x2 = 0, 4      # x-axis range: (x1, x2)
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
ax.set_xlabel(r'- $\Delta$ G$_0$ (eV)', font = 'Times New Roman', size = 20)
ax.set_ylabel(r'Rate (eV / $\hbar$)', font = 'Times New Roman', size = 20)
# ax.set_title('Gamma=0.001 eV', font = 'Times New Roman', size = 20)

# legend location, font & markersize
ax.legend(loc = 'upper right', prop = font, markerscale = 1, frameon = False) # , bbox_to_anchor = (0.1, 0)
plt.legend(frameon = False)

# plt.show() 

plt.savefig("Model_C_Q = 1_GMJ_43_44_45.pdf", bbox_inches='tight')
