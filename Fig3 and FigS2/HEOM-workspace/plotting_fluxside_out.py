import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from matplotlib.pyplot import MultipleLocator, tick_params
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

def delta(m, n):
    return 1 if m == n else 0

# ==============================================================================================

lw = 3.0
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 24}

unitlen = 8
fig = plt.figure(figsize=(1.0 * unitlen, 1.0 * unitlen), dpi = 128)

# ==============================================================================================
#                                      Fig 2a: time dependent rate  
# ==============================================================================================
n_el = 2
n_ph = 1
nfock = n_el * n_ph

data1 = np.loadtxt("./rho-out.dat", dtype = float)

PRt_1 = 0

for i in range(n_el):
    for j in range(n_el):
        for m in range(n_ph):
            for n in range(n_ph):
                
                PRt_1 += data1[:, (nfock * (n_el * m + i) + (n_el * n + j)) + 1] * delta(i, 0) * delta(j, 0) * delta(m, n)

dt = (data1[1,0] - data1[0,0])
rate_1 = np.zeros((len(PRt_1) - 1), dtype=float)

chi_eq = 0.0

for n in range(1, len(rate_1)):
    rate_1[n] = - np.real( ((PRt_1[n] - PRt_1[n - 1]) / dt) / ( PRt_1[n] + chi_eq * (PRt_1[n] - 1) ))

# convert to the unit of eV
rate_1 = rate_1 * conv

time_1 = data1[:,0]

# plot the time-dependent rate constant, unit = eV / hbar
ratio = 1e-5
plt.plot(time_1[0:-1] / (1000 * fs_to_au), rate_1 / ratio, "-", linewidth = lw, color = 'tab:blue', label = r"Outside Cavity")

# x and y range of plotting 
time = 5
y1, y2 = 3, 5.8     # y-axis range: (y1, y2)

plt.xlim(-0.1, time)
plt.ylim(y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(1)
x_minor_locator = MultipleLocator(0.2)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30, pad = 10)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'time (ps)', size = 24)
ax.set_ylabel(r'$k\ (\times 10^{-5}\ \mathrm{eV} / \hbar)$', size = 24)
ax.legend(frameon = False, loc = 'upper right', prop = font_legend, markerscale = 1)



# plt.show()
plt.savefig("fig_flux_side_out.pdf", bbox_inches='tight')

