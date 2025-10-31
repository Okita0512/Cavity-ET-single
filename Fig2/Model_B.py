from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':18}
fig = plt.figure(figsize=(14, 7), dpi = 128)

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
w_c0 = 0.2 / conv            # cavity frequency
t_DA = 0.5 * cm_to_au
g_DA = 0.5 

# ==============================================================================
plt.subplot(1,2,1)

w = - np.linspace(0, 0.8 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
Exp = np.exp(- beta * (- w - lam)**2 / (4 * lam))
k_Marcus_out = Prefactor * Delta**2 * Exp

plt.semilogy(- w * conv, k_Marcus_out * conv, '-', linewidth = 3, color = '#444444', label = 'Outside Cavity (Marcus)', zorder = 1)
# plt.semilogy(- w * conv, k_Marcus_in * conv, '--', linewidth = 3, color = "#FD0000", label = 'Inside Cavity (MJ)')

k1 = Prefactor * Delta**2 * np.exp(- beta * (- (- 0.4/conv) - lam)**2 / (4 * lam))
plt.scatter(0.4, k1 * conv, s = 150, c = "#FD0000", marker = 'o', zorder = 2)

k2 = Prefactor * Delta**2 * np.exp(- beta * (- (- 0.6/conv) - lam)**2 / (4 * lam))
plt.scatter(0.6, k2 * conv, s = 150, c = "#0037FD", marker = 'o', zorder = 3)

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
x1, x2 = 0, 0.8      # x-axis range: (x1, x2)
y1, y2 = 1e-8, 5e-4     # y-axis range: (y1, y2)

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
# ax2 = ax.twinx()
# ax2.yaxis.set_major_locator(y_major_locator)
# ax2.yaxis.set_minor_locator(y_minor_locator)
# ax2.tick_params(which = 'major', length = 8)
# ax2.tick_params(which = 'minor', length = 4)
# ax2.axes.yaxis.set_ticklabels([])

# plt.tick_params(which = 'both', direction = 'in')
# plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'- $\Delta$ G$_0$ (eV)', font = 'Times New Roman', size = 20)
ax.set_ylabel(r'Rate (eV / $\hbar$)', font = 'Times New Roman', size = 20)
# ax.set_title('Gamma=0.001 eV', font = 'Times New Roman', size = 20)

# legend location, font & markersize
ax.legend(loc = 'lower left', prop = font, markerscale = 1, frameon = False)
# plt.legend(frameon = False)

# ==============================================================================

plt.subplot(1,2,2)

# ==============================================================================

# Evaluate Marcus rate, Eq. 39 of [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]
from math import factorial
from scipy.special import assoc_laguerre as laguerre

w = - 0.4 / conv
w_c = np.linspace(0, 0.8 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
L_max = 10           # number of Frank-Condon terms
Exp = np.zeros((L_max, 1000), dtype= float)
for m in range(L_max):
    Exp[m, :] = np.exp(- beta * (- w - lam - m * w_c)**2 / (4 * lam))

k_Marcus_out = Prefactor * Delta**2 * np.exp(- beta * (- w - lam)**2 / (4 * lam))
k_Marcus_in = np.zeros((1000), dtype= float)

for i in range(1, len(w_c)):

    def FC(m, n):   # Frank-Condon coefficients, see https://doi.org/10.1002/qua.560240843

        gamma = np.sqrt(2) * g_DA / np.sqrt(w_c[i] / w_c0)

        if m >= n:
            return np.sqrt(factorial(n)/factorial(m)) * (gamma/np.sqrt(2))**(m-n) * laguerre(gamma**2/2, n, m-n) * np.exp(-gamma**2/4)
        else:
            return np.sqrt(factorial(m)/factorial(n)) * (-gamma/np.sqrt(2))**(n-m) * laguerre(gamma**2/2, m, n-m) * np.exp(-gamma**2/4)

    for m in range(L_max):
        k_Marcus_in[i] += Delta**2 * FC(0, m)**2 * Exp[m, i] 
        if m >= 1:
            k_Marcus_in[i] += t_DA**2 * FC(1, m)**2 * Exp[m, i] * w_c[i] / w_c0

k_Marcus_in = k_Marcus_in * Prefactor

# x-axis and LHS y-axis
ax = plt.gca()

# scale for major and minor locator
x_major_locator = MultipleLocator(0.2)
x_minor_locator = MultipleLocator(0.1)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

ax.plot(w_c * conv, (k_Marcus_in / k_Marcus_out), '-', linewidth = 3, color = "#FD0000", label = 'Inside Cavity (MJ)')
ax.vlines(x = 0.2, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#FD0000")
ax.vlines(x = 0.2 / 2, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#FD0000", alpha = .39)
ax.vlines(x = 0.2 / 3, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#FD0000", alpha = .17)
ax.vlines(x = 0.2 / 4, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#FD0000", alpha = .08)
ax.vlines(x = 0.2 / 5, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#FD0000", alpha = .04)
ax.vlines(x = 0.2 / 6, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#FD0000", alpha = .02)

ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(axis = 'x', which = 'major', length = 8, labelsize = 10, pad = 10)
ax.tick_params(axis = 'x', which = 'minor', length = 4)
ax.tick_params(axis = 'y', which = 'major', length = 8, labelsize = 10)
ax.tick_params(axis = 'y', which = 'minor', length = 4)

ax.tick_params('x', labelsize = 20, which = 'both', direction = 'in')
ax.tick_params('y', labelsize = 20, which = 'both', direction = 'in', color = "#FD0000")
ax.spines['left'].set_color("#FD0000")
ax.spines['left'].set_linewidth(2)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
for y1_label_temp in y1_label:
    y1_label_temp.set_fontname('Times New Roman')
    y1_label_temp.set_color("#FD0000")

# x and y range of plotting 
x1, x2 = 0, 0.8      # x-axis range: (x1, x2)
y1, y2 = 0.0, 3.0     # y-axis range: (y1, y2)

plt.xlim(x1, x2)
plt.ylim(y1, y2)

# ==============================================================================
# RHS
# Evaluate Marcus rate, Eq. 38 of [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]
w = - 0.6 / conv
w_c = np.linspace(0, 0.8 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
L_max = 10           # number of Frank-Condon terms
Exp = np.zeros((L_max, 1000), dtype= float)
for m in range(L_max):
    Exp[m, :] = np.exp(- beta * (- w - lam - m * w_c)**2 / (4 * lam))

k_Marcus_out = Prefactor * Delta**2 * np.exp(- beta * (- w - lam)**2 / (4 * lam))
k_Marcus_in = np.zeros((1000), dtype= float)

for i in range(1, len(w_c)):

    def FC(m, n):   # Frank-Condon coefficients, see https://doi.org/10.1002/qua.560240843

        gamma = np.sqrt(2) * g_DA / np.sqrt(w_c[i] / w_c0)

        if m >= n:
            return np.sqrt(factorial(n)/factorial(m)) * (gamma/np.sqrt(2))**(m-n) * laguerre(gamma**2/2, n, m-n) * np.exp(-gamma**2/4)
        else:
            return np.sqrt(factorial(m)/factorial(n)) * (-gamma/np.sqrt(2))**(n-m) * laguerre(gamma**2/2, m, n-m) * np.exp(-gamma**2/4)

    for m in range(L_max):
        k_Marcus_in[i] += Delta**2 * FC(0, m)**2 * Exp[m, i] 
        if m >= 1:
            k_Marcus_in[i] += t_DA**2 * FC(1, m)**2 * Exp[m, i] * w_c[i] / w_c0

k_Marcus_in = k_Marcus_in * Prefactor

# RHS y-axis
ax2 = ax.twinx()
ax2.plot(w_c * conv, (k_Marcus_in / k_Marcus_out) / 1e2, '-', linewidth = 3, color = "#0037FD", label = 'Inside Cavity (MJ)')
ax.vlines(x = 0.4, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#0037FD")
ax.vlines(x = 0.4 / 2, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#0037FD", alpha = .39)
ax.vlines(x = 0.4 / 3, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#0037FD", alpha = .17)
ax.vlines(x = 0.4 / 4, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#0037FD", alpha = .08)
ax.vlines(x = 0.4 / 5, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#0037FD", alpha = .04)
ax.vlines(x = 0.4 / 6, ymin = 0.0, ymax = 3.0, linestyles='--', color = "#0037FD", alpha = .02)

y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 8, labelsize = 10)
ax2.tick_params(which = 'minor', length = 4)
# ax2.axes.yaxis.set_ticklabels([])

ax2.tick_params('y', labelsize = 20, which = 'both', direction = 'in', color = "#0037FD")
ax.spines['right'].set_color("#0037FD")
ax.spines['right'].set_linewidth(2)

y2_label = ax2.get_yticklabels()
for y2_label_temp in y2_label:
    y2_label_temp.set_fontname('Times New Roman')
    y2_label_temp.set_color("#0037FD")

y1, y2 = 0.0, 3.0     # y-axis range: (y1, y2)
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'$\hbar \omega$ (eV)', fontname = 'Times New Roman', size = 20)
ax.set_ylabel(r'$k_\mathrm{in}~/~k_\mathrm{out}$', fontname = 'Times New Roman', fontsize = 20, color = "#FD0000", labelpad = 8)
ax2.set_ylabel(r'$k_\mathrm{in}~/~k_\mathrm{out}$ ($\times 10^{2}$)', fontname = 'Times New Roman', fontsize = 20, color = "#0037FD", labelpad = 8)

# legend location, font & markersize
# ax.legend(loc = 'upper right', prop = font, markerscale = 1, frameon = False)
# plt.legend(frameon = False)

# plt.show()

plt.savefig("Model B.pdf", bbox_inches='tight')