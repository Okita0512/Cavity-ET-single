from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
from matplotlib.ticker import FuncFormatter
import numpy as np
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':16}
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
T = 30000 * K_to_au           # temperature
beta = 1.0 / T

# phonon bath parameter
lam = 0.2 / conv            # reorganization energy
gamma = 20 * cm_to_au
alpha = 2 * lam / gamma

# effective bath parameter
w_c = 0.2 / conv            # cavity frequency
t_DA = 0.5 * cm_to_au
g_DA = 0.5 

# ==============================================================================

def coth(x):
    return np.cosh(x) / np.sinh(x)

# discretization
def bathParam(ωc, alpha, ndof):     # for bath descritization

    c = np.zeros(( ndof ))
    ω = np.zeros(( ndof ))
    for d in range(ndof):

        ω[d] =  - ωc * np.log(1 - (d + 1)/(ndof + 1))
        c[d] =  np.sqrt(alpha * ωc * ω[d]/ (2 * (ndof + 1)))

    return c, ω

# terms that going to be sum
def g_j(c_j, w_j, t):
    return - (c_j**2 / (w_j**2)) * (coth(beta*w_j/2) * (1 - np.cos(w_j * t)) + 1.0j * np.sin(w_j * t))

# FT 
def DFT(w_c):

    ndof = 100

    UP = 2**6 * fs_to_au
    dt = 2**(-12) * fs_to_au
    nfft = int(UP / dt)
    print(nfft)

    t = - np.linspace(0, UP, nfft)
    ft = np.zeros((nfft), dtype = complex)

    c, ω = bathParam(gamma, alpha, ndof) 
    for j in range(len(c)):
        ft += g_j(c[j], ω[j], t)

    print("reorganization energy / 0.2 eV", np.sum(c**2 / ω) * conv / 0.2)

    ht = Delta + t_DA * g_DA * (- np.cos(w_c * t) + 1.0j * np.sin(w_c * t) * coth(beta * w_c / 2))
    ht = ht * (Delta + t_DA * g_DA * (- np.cos(w_c * t) + 1.0j * np.sin(w_c * t) * coth(beta * w_c / 2)))
    gt = t_DA**2 * (np.cos(w_c * t) * coth(beta * w_c / 2) - 1.0j * np.sin(w_c * t))
    ft_cav = ft - g_DA**2 * ((1 - np.cos(w_c * t)) * coth(beta * w_c / 2) + 1.0j * np.sin(w_c * t))
    
    C_t_out = 2 * Delta**2 * np.exp(ft) # * np.exp(1.0j * lam * t)
    C_t_cav = 2 * (ht + gt) * np.exp(ft_cav)

#    plt.plot(t / fs_to_au, C_t_out, '-', color = 'black', label = 'Ct Outside Cavity')
#    plt.plot(t / fs_to_au, np.real(C_t_cav), '--', color = 'red', label = 'Ct Inside Cavity - real')
#    plt.plot(t / fs_to_au, np.imag(C_t_cav), '--', color = 'blue', label = 'Ct Inside Cavity - imag')

# debug
#    ht = Delta + t_DA * g_DA * (- 1 + (w_c * t)**2 / 2 + 1.0j * 2 * t / beta)
#    ht = ht * (Delta + t_DA * g_DA * (- 1 + (w_c * t)**2 / 2 + 1.0j * 2 * t / beta))
#     ht = Delta + t_DA * g_DA * (- 1 + 1.0j * 2 * t / beta)
#     ht = ht * (Delta + t_DA * g_DA * (- 1 + 1.0j * 2 * t / beta))
#     gt = t_DA**2 * ((2 - (w_c * t)**2) / (beta * w_c) - 1.0j * w_c * t)
#     ft_cav = ft - g_DA**2 * (w_c * t**2 / beta + 1.0j * w_c * t)
# 
#     C_t_cav = 2 * (ht + gt) * np.exp(ft_cav)
#     plt.plot(t / fs_to_au, np.real(C_t_cav), '--', color = 'green', label = 'Ct Inside Cavity -real (Approx)')
#     plt.plot(t / fs_to_au, np.imag(C_t_cav), '--', color = 'cyan', label = 'Ct Inside Cavity -imag (Approx)')
# 
#     plt.legend(frameon = False)
#     plt.show()

    return (2 * np.pi) * t / (UP * dt), np.real(fft(C_t_out)) * dt, np.real(fft(C_t_cav)) * dt

# Evaluate Marcus rate, Eq. 39 of [A. Semenov and A. Nitzan, J. Chem. Phys. 150, 174122 (2019)]
w = np.linspace(0, 0.8 / conv, 1000)
Prefactor = np.sqrt(np.pi * beta / lam)
E_R_prime = lam + g_DA**2 * w_c
Prefactor_in = np.sqrt(np.pi * beta / E_R_prime)
Exp = np.exp(- beta * (w - lam)**2 / (4 * lam))
Exp_in = np.exp(- beta * (w - E_R_prime)**2 / (4 * E_R_prime))

k_Marcus_out = Prefactor * Delta**2 * Exp
k_Marcus_in = t_DA**2 * 2 / (beta * w_c) - t_DA**2 * 2 * g_DA**2 / (beta * E_R_prime)
k_Marcus_in += Delta**2 + (-w / E_R_prime) * 2 * Delta * t_DA * g_DA - t_DA**2 * (- g_DA**2 * ((-w) / E_R_prime)**2 + w_c / (2 * E_R_prime))
# k_Marcus_in += (beta * w_c / (4 * E_R_prime)) * (2 * Delta * t_DA * g_DA * w_c - t_DA**2 * (2 * (-w + E_R_prime - g_DA**2 * w_c) + (-w + E_R_prime)**2 / E_R_prime))

k_Marcus_in *= Prefactor_in * Exp_in

k_Marcus_in2 = 0 # t_DA**2 * 2 / (beta * w_c) - t_DA**2 * 2 * g_DA**2 / (beta * E_R_prime)
k_Marcus_in2 += Delta**2 # + (-w / E_R_prime) * 2 * Delta * t_DA * g_DA - t_DA**2 * (- g_DA**2 * ((-w) / E_R_prime)**2 + w_c / (2 * E_R_prime))
k_Marcus_in2 *= Prefactor_in * Exp_in
# ==============================================================================
t, y, y_c = DFT(w_c)

scale = 1e5

plt.semilogy(w * conv, k_Marcus_out * conv * scale, '-', linewidth = 3, color = '#444444') #, label = 'Outside Cavity (MJ)')
plt.semilogy(w * conv, k_Marcus_in2 * conv * scale, '--', linewidth = 3, color = "#08F12F", label = 'Inside Cavity (Marcus, Eq. 22)')
plt.semilogy(w * conv, k_Marcus_in * conv * scale, '--', linewidth = 3, color = "#FD0000", label = 'Inside Cavity (Marcus, Eq. 28)')

plt.semilogy( - t * conv, y * conv * scale, 'o', markersize = 10, markeredgewidth = 2, markeredgecolor = "k", markerfacecolor = 'none') #, label = 'Outside Cavity (FGR)')
plt.semilogy( - t * conv, y_c * conv * scale, 'o', markersize = 6, markeredgewidth = 2, markeredgecolor = "#0385FF", markerfacecolor = '#0385FF') #, label = 'Inside Cavity (FGR)')

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
x1, x2 = 0, 0.8      # x-axis range: (x1, x2)
y1, y2 = 2, 4     # y-axis range: (y1, y2)

plt.xlim(x1, x2)
plt.ylim(y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(0.2)
x_minor_locator = MultipleLocator(0.1)
# y_major_locator = MultipleLocator(0.5)
# y_minor_locator = MultipleLocator(0.1)

from matplotlib.ticker import FixedLocator, FuncFormatter
# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)

ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
ax.yaxis.set_major_locator(FixedLocator([2, 3, 4]))

# ax.yaxis.set_major_locator(y_major_locator)
# ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8, labelsize = 10)
ax.tick_params(which = 'minor', length = 4)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 20, which = 'both', direction = 'in', pad = 6)

# RHS y-axis
# ax2 = ax.twinx()
# # ax2.yaxis.set_major_locator(y_major_locator)
# # ax2.yaxis.set_minor_locator(y_minor_locator)
# ax2.tick_params(which = 'major', length = 8)
# ax2.tick_params(which = 'minor', length = 4)
# ax2.axes.yaxis.set_ticklabels([])

# plt.tick_params(which = 'both', direction = 'in')
# plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'- $\Delta$ G$_0$ (eV)', font = 'Times New Roman', size = 20)
ax.set_ylabel(r'Rate ($\times 10^{-5}$ eV / $\hbar$)', font = 'Times New Roman', size = 20)
# ax.set_title('Gamma=0.001 eV', font = 'Times New Roman', size = 20)

# legend location, font & markersize
ax.legend(loc = 'lower left', prop = font, markerscale = 1, frameon = False)
# plt.legend(frameon = False)

# plt.show()

plt.savefig("Model B - high T.pdf", bbox_inches='tight')