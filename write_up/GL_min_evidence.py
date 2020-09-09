import pickle

from chisq_fit3 import *
from tqdm import tqdm
import matplotlib as mpl

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
# font={#’family’:’normal’,
#       'weight':'bold',
#       'size':30}
# mpl.rc('font',**font)

color_blind_palette = [(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115),
                       (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]

color_blind_palette = [(a / 255, b / 255, c / 255) for a, b, c in color_blind_palette]

# directory = f"posterior_data/{today.year}_{today.month}_{today.day}/"
directory = f"posterior_data/{today.year}_{9}_{5}/"

points = 3000
no_samples = 30
model1 = model27
model2 = model28

model3 = model60
model4 = model61

GL_min_graph = 7
GL_mins = pickle.load(open(f"{directory}evidence_GL_mins_points{points}_samples{no_samples}.pcl", "rb"))

# results1 = pickle.load(open(f"{directory}evidence_results1_points{points}_samples{no_samples}.pcl", "rb"))
# results2 = pickle.load(open(f"{directory}evidence_results2_points{points}_samples{no_samples}.pcl", "rb"))

results1 = pickle.load(open(f"{directory}evidence_N{N}_model27_points{points}_samples{no_samples}.pcl", "rb"))
results2 = pickle.load(open(f"{directory}evidence_N{N}_model28_points{points}_samples{no_samples}.pcl", "rb"))
results = results1 - results2

results = results[GL_mins >= GL_min_graph]
GL_mins = GL_mins[GL_mins >= GL_min_graph]

plt.errorbar(GL_mins, numpy.mean(results, axis=1), yerr=numpy.std(results, axis=1), capsize=5, color=color_blind_palette[0], label=r'$log(E)$', ls='', marker='_')

ax = plt.gca()
ax.set_ylabel(r"$log(E)$")

# Show the Jeffrey's scale 
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [-1, -1], [1, 1], color=color_blind_palette[7], alpha=0.2, label=r'$0 < log(E) < 1$')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [1, 1], [2.5, 2.5], color=color_blind_palette[4], alpha=0.2, label=r'$1 < log(E) < 2.5$')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [2.5, 2.5], [max(numpy.mean(results, axis=1)) * 1.1, max(numpy.mean(results, axis=1)) * 1.1], color=color_blind_palette[2], alpha=0.2, label=r'$log(E) > 2.5$')

# Add in the frequentist side of things
pvalues_1 = []
pvalues_2 = []

# An assortment of starting values - try them all
x0_s = [[0.0014, -0.05, 0.58, -0.038, 1.064, 0.684, 0.453],
        [0.0014, -0.134, 0.608, -0.06, 1.064, 0.6844, 0.454],
        [0, -0.04, 0.657, -0.038, 1, 2 / 3, 0.8],
        [1.02040644e-03, -7.03445237e-01, 1.85354748e+00, -1.83766936e-01,
        1.06990729e+00, 6.84724598e-01, 1.00000000e-03],
        [0, -1.1, 0.657, -0.038, 1, 2 / 3, 0.8],
        [0.00231032942, -1.68948597, 0.582233883, -0.0863432296, 1.08681148, 0.669969122, 1.20784972], 
        [0.0014094110011229537, -0.9956587394857183, 100, -15.520408778140792, 1.0837290017931231, 0.6779685700464381, 0.0004295598691863679],
        [-0.016, -0.97, 20, -2.2, 0.98, 0.713, 0.0018],
        [-1.89020457e-02, -9.80798332e-01, 2.00000000e+01, -3.49234687e+00,
        1.06376347e+00, 7.11750219e-01, 3.86805210e-03],
        [-0.01892748112187912, -0.9961520257453251, 100, -17.51002928127534, 1.0638925005064692, 0.7122102338205046, 0.0007525097395268309],
        [-0.01612187, -0.87133898, 1.       , -0.44064317/4, 0.97991928,
        0.7127984, 0.01029297]]


try:
  pvalues_1 = pickle.load(open(f"{today}_N{N}_pvalues_1.pcl", "rb"))
  # raise(Exception)

except:
  pvalues_1 = []
  pvalues_2 = []
  for GL_min in tqdm(GL_mins):
    g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
    
    cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut, N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    # alpha_range = [-0.1, 0.1]
    # c_range = [-2, 2]
    # f0_range = [-20, 20]
    # f1_range = [-20, 20]
    # lambduh_range = [-1, 3]
    # nu_range = [0.3, 1.1]
    # omega_range = [0, 2]

    alpha_range = [-numpy.inf, numpy.inf]
    c_range = [-numpy.inf, numpy.inf]
    f0_range = [-numpy.inf, numpy.inf]
    f1_range = [-numpy.inf, numpy.inf]
    lambduh_range = [-numpy.inf, numpy.inf]
    nu_range = [0, numpy.inf]
    omega_range = [0, numpy.inf]

    if model1 is model50 and model2 is model51:
      bounds = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
      [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])
    
    if model1 is model27 and model2 is model28:
      bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
      [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])
      # x0 = [0, 0, 0.657, -0.038, 1, 2 / 3, 0.8]

    kwargs = {"m_s": m_s_cut, "N_s": N_s_cut, "g_s": g_s_cut, "L_s": L_s_cut, "Bbar_s": Bbar_s_cut}

    max_pvalue1 = 0
    for x0 in x0_s:
      res1 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model1),
      kwargs=kwargs, method='dogbox')
      chisq1 = chisq_calc(res1.x, cov_inv, model1, **kwargs)
      dof = g_s_cut.shape[0] - len(res1.x)
      p1 = chisq_pvalue(dof, chisq1)
      if p1 > max_pvalue1:
        max_pvalue1 = p1

    pvalues_1.append(max_pvalue1)

    max_pvalue2 = 0
    for x0 in x0_s:
      res2 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model2),
      kwargs=kwargs, method='dogbox')
      chisq2 = chisq_calc(res2.x, cov_inv, model2, **kwargs)
      dof = g_s_cut.shape[0] - len(res2.x)
      p2 = chisq_pvalue(dof, chisq2)
      if p2 > max_pvalue2:
        max_pvalue2 = p2

    pvalues_2.append(max_pvalue2)

  pickle.dump(pvalues_1, open(f"{today}_N{N}_pvalues_1.pcl", "wb"))
  pickle.dump(pvalues_2, open(f"{today}_N{N}_pvalues_2.pcl", "wb"))


pvalues_1 = pickle.load(open(f"{today}_N{N}_pvalues_1.pcl", "rb"))
pvalues_2 = pickle.load(open(f"{today}_N{N}_pvalues_2.pcl", "rb"))

ax = plt.gca()
fig = plt.gcf()
ax.set_xlabel(r"$gL_{min}$")
ax.set_xlim(GL_min_graph, 40)

ax2 = ax.twinx()
ax2.set_ylabel(r'$p-value$')
ax2.plot(GL_mins, pvalues_1, marker='s', markerfacecolor='none', label=r"$\log(g)$", ls='', color=color_blind_palette[1])
ax2.plot(GL_mins, pvalues_2, marker='s', markerfacecolor='none', label=r"$\log(L)$", ls='', color=color_blind_palette[3])

# Show our acceptance threshold (alpha)
ax2.plot(numpy.array(GL_mins) * 2 - 10, [0.05, ] * len(GL_mins), color='grey')


fig.tight_layout()

ax.tick_params(direction='in')
ax2.tick_params(direction='in')

if N == 2:
  ax.legend(loc=(0.02, 0.4))
ax2.legend(loc='upper right')

# Align 0 for both axes
left1, right1 = ax.get_ylim()
left2, right2 = ax2.get_ylim()

ax.set_ylim(1.2, max(numpy.mean(results, axis=1)) * 1.1)
ax2.set_ylim(0, right2)

# Shift the axis a bit below 0
change = 0.02
a, b = ax.get_ylim()
ax.set_ylim(a - (b - a) * change, b)

a, b = ax2.get_ylim()
ax2.set_ylim(a - (b - a) * change, b)

ax.set_yscale('log')
ax2.annotate(r"$p = 0.05$", xy=(36, 0.06), color='grey')

plt.show()
