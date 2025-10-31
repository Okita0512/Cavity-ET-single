import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':14}
fig = plt.figure(figsize=(7, 7), dpi = 128)
lw = 1.5

# ==============================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
meV_to_au = 1 / (conv * 1000)               # 1 meV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
K_to_au = 1.0 / 3.1577464e+05               # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.
# ==============================================================================
data11 = np.loadtxt("./Model_A/Q=inf/dG", dtype = float)
data12 = np.loadtxt("./Model_A/Q=inf/k_out", dtype = float)
data13 = np.loadtxt("./Model_A/Q=inf/k_in", dtype = float)
plt.semilogy(data11, data12, '-', linewidth = 2.5, color = 'black', label = 'Outside Cavity')
plt.semilogy(data11, data13, 'o-', linewidth = lw, color = 'darkred', label = r'$\mathcal{Q} = \infty$')

data21 = np.loadtxt("./Model_A/Q=10/dG", dtype = float)
data22 = np.loadtxt("./Model_A/Q=10/k_in", dtype = float)
plt.semilogy(- data21, data22, 'o-', linewidth = lw, color = 'orange', label = r'$\mathcal{Q} = 10$') # , label = r'$200$ meV'

data31 = np.loadtxt("./Model_A/Q=5/dG", dtype = float)
data32 = np.loadtxt("./Model_A/Q=5/k_in", dtype = float)
plt.semilogy(- data31, data32, 'o-', linewidth = lw, color = 'green', label = r'$\mathcal{Q} = 5$') # , label = r'$500$ meV'

data41 = np.loadtxt("./Model_A/Q=2/dG", dtype = float)
data42 = np.loadtxt("./Model_A/Q=2/k_in", dtype = float)
plt.semilogy(- data41, data42, 'o-', linewidth = lw, color = 'cyan', label = r'$\mathcal{Q} = 2$') # , label = r'$1000$ meV'

data51 = np.loadtxt("./Model_A/Q=1/dG", dtype = float)
data52 = np.loadtxt("./Model_A/Q=1/k_in", dtype = float)
plt.semilogy(- data51, data52, 'o-', linewidth = lw, color = 'blue', label = r'$\mathcal{Q} = 1$') # , label = r'$2000$ meV'

data61 = np.loadtxt("./Model_A/Q=0.5/dG", dtype = float)
data62 = np.loadtxt("./Model_A/Q=0.5/k_in", dtype = float)
plt.semilogy(- data61, data62, 'o-', linewidth = lw, color = 'violet', label = r'$\mathcal{Q} = 0.5$', alpha = .7) # , label = r'$4000$ meV'

data71 = np.loadtxt("./Model_A/Q=0.2/dG", dtype = float)
data72 = np.loadtxt("./Model_A/Q=0.2/k_in", dtype = float)
plt.semilogy(- data71, data72, 'o-', linewidth = lw, color = 'purple', label = r'$\mathcal{Q} = 0.2$', alpha = .7) # , label = r'$4000$ meV'

data81 = np.loadtxt("./Model_A/Q=0/dG", dtype = float)
data82 = np.loadtxt("./Model_A/Q=0/k_in", dtype = float)
plt.semilogy(- data81, data82, 'o', linewidth = lw, color = 'red', label = r'$\mathcal{Q}~\to~0$') # , label = r'$4000$ meV'

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
ax.legend(loc = 'lower left', prop = font, markerscale = 1, bbox_to_anchor = (0.06, 0), frameon = False)
# plt.legend(frameon = False)

# plt.show()

plt.savefig("Model A.pdf", bbox_inches='tight')