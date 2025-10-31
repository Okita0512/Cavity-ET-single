import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':24}
fig = plt.figure(figsize=(5, 5), dpi = 128)

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
    return np.cosh(x) / np.sinh(x)

# Brownian spectral density
def J_Brownian(x, Lam, Gamma, ws):
    return 2 * Lam * Gamma * ws**2 * x / ((ws**2 - x**2)**2 + (Gamma * x)**2)

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
    return - (c_j**2 / (w_j**2)) * (coth(beta*w_j/2) * (1 - np.cos(w_j * t)) - 1.0j * np.sin(w_j * t))

ndof = 100

UP = 2**5 * fs_to_au
dt = 2**(-10) * fs_to_au
nfft = int(UP / dt)
print(nfft)

t = np.linspace(0, UP, nfft)
ft = np.zeros((nfft), dtype = complex)

c, ω = bathParam(gamma, alpha, ndof) 
for j in range(len(c)):
    ft += g_j(c[j], ω[j], t)

print("reorganization energy", np.sum(c**2 / ω) * conv)

ht = Delta + t_DA * g_DA * (- np.cos(w_c * t) + 1.0j * np.sin(w_c * t) * coth(beta * w_c / 2))
ht = ht * (Delta + t_DA * g_DA * (- np.cos(w_c * t) + 1.0j * np.sin(w_c * t) * coth(beta * w_c / 2)))
gt = t_DA**2 * (np.cos(w_c * t) * coth(beta * w_c / 2) + 1.0j * np.sin(w_c * t))
ft_cav = ft - g_DA**2 * ((1 - np.cos(w_c * t)) * coth(beta * w_c / 2) - 1.0j * np.sin(w_c * t))

C_t_out = 2 * Delta**2 * np.exp(ft) # * np.exp(1.0j * lam * t)
C_t_cav = 2 * (ht + gt) * np.exp(ft_cav)

plt.plot(t / fs_to_au, np.real(C_t_out) * 1e6, '-',  linewidth = 3, color = '#444444', label = 'Outside Cavity')
plt.plot(t / fs_to_au, np.real(C_t_cav) * 1e6, '--', linewidth = 3, color = "#FD0000", label = 'Inside Cavity')

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
x1, x2 = 0, 5      # x-axis range: (x1, x2)
y1, y2 = -2.6, 3     # y-axis range: (y1, y2)

plt.xlim(x1, x2)
plt.ylim(y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(2)
x_minor_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
y_minor_locator = MultipleLocator(0.2)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8, labelsize = 10)
ax.tick_params(which = 'minor', length = 4)

plt.tick_params(labelsize = 20, which = 'both', direction = 'in')

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 8)
ax2.tick_params(which = 'minor', length = 4)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
# ax.set_xlabel(r'time (fs)', font = 'Times New Roman', size = 20)
# ax.set_ylabel(r'amplitude (a.u.$^{2}$ $\times 10^6$)', font = 'Times New Roman', size = 20)
# ax.set_title('Real Part', font = 'Times New Roman', size = 30)

# legend location, font & markersize
# ax.legend(loc = 'upper right', prop = font, markerscale = 1, frameon = False)
# plt.legend(frameon = False)



plt.savefig("Fig5-2.pdf", bbox_inches='tight')