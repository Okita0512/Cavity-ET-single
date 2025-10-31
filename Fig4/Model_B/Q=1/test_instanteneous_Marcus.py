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
Delta = 30 * cm_to_au      # diabatic coupling
T = 300 * K_to_au           # temperature
beta = 1.0 / T

# phonon bath parameter
lam = 0.2 / conv            # reorganization energy
gamma = 20 * cm_to_au
alpha = 2 * lam / gamma

# effective bath parameter
w_c = 0.2 / conv            # cavity frequency
t_DA = 0.5 * cm_to_au
g_DA = 0.5 

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

# ==============================================================================
# Evaluate Marcus rate, Eq. 38 of [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]
from math import factorial
from scipy.special import assoc_laguerre as laguerre
def FC(m, n):   # Frank-Condon coefficients, see https://doi.org/10.1002/qua.560240843

    gamma = np.sqrt(2) * g_DA

    if m >= n:
        return np.sqrt(factorial(n)/factorial(m)) * (gamma/np.sqrt(2))**(m-n) * laguerre(gamma**2/2, n, m-n) * np.exp(-gamma**2/4)
    else:
        return np.sqrt(factorial(m)/factorial(n)) * (-gamma/np.sqrt(2))**(n-m) * laguerre(gamma**2/2, m, n-m) * np.exp(-gamma**2/4)
    
w = - np.linspace(0, 0.8 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
L_max = 10           # number of Frank-Condon terms

# Marcus without loss
Exp = np.zeros((L_max, 1000), dtype= float)
for m in range(L_max):
    Exp[m, :] = np.exp(- beta * (- w - lam - m * w_c)**2 / (4 * lam))

print("the value of FC_00:", FC(0, 0)**2, ', expected to be 0.78 as reported in [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]')

k_Marcus_out = Prefactor * Delta**2 * Exp[0, :]
k_Marcus_in = 0
for m in range(L_max):
    k_Marcus_in += Delta**2 * FC(0, m)**2 * Exp[m, :] 
    if m >= 1:
        k_Marcus_in += t_DA**2 * FC(1, m)**2 * Exp[m, :] 
k_Marcus_in = k_Marcus_in * Prefactor

# ==============================================================================
# Marcus with loss (instanteneous Marcus) -- need to fix...
Gamma_c = 200 * meV_to_au
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

k_Marcus_inn = Prefactor_inn * I

# ==============================================================================
plt.semilogy(- w * conv, k_Marcus_out * conv, '-', linewidth = 3, color = '#444444', label = 'Marcus Outside Cavity')
plt.semilogy(- w * conv, k_Marcus_in * conv, '--', linewidth = 3, color = "#FD0000", label = 'Marcus Inside Cavity')
plt.semilogy(- w * conv, k_Marcus_inn * conv, '--', linewidth = 3, color = "orange", label = 'Marcus Inside Cavity_Q = 1')

data1 = np.loadtxt("dG", dtype = float)
data2 = np.loadtxt("k_in", dtype = float)
plt.semilogy(data1, data2, 'o', markersize = 6, markeredgewidth = 2, markeredgecolor = "#0385FF", markerfacecolor = '#0385FF', label = 'FGR Inside Cavity')

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
x1, x2 = 0, 0.8      # x-axis range: (x1, x2)
y1, y2 = 1e-5, 5e-4     # y-axis range: (y1, y2)

plt.xlim(x1, x2)
plt.ylim(y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(0.2)
x_minor_locator = MultipleLocator(0.1)
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

plt.savefig("Q = 1_test.pdf", bbox_inches='tight')