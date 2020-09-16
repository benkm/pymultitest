import pickle

from chisq_functions import *
from tqdm import tqdm
import matplotlib as mpl

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
# font={#’family’:’normal’,
#       'weight':'bold',
#       'size':30}
# mpl.rc('font',**font)

alpha_range = [-numpy.inf, numpy.inf]
c_range = [-numpy.inf, numpy.inf]
f0_range = [-numpy.inf, numpy.inf]
f1_range = [-numpy.inf, numpy.inf]
lambduh_range = [-numpy.inf, numpy.inf]
nu_range = [0, numpy.inf]
omega_range = [0, numpy.inf]

bounds = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
  [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])
    

color_blind_palette = [(0, 0, 0), (230, 159, 0), (86, 180, 233), (0, 158, 115),
                       (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]

color_blind_palette = [(a / 255, b / 255, c / 255) for a, b, c in color_blind_palette]

in_directory = "output_data/GL_min_bayes/"
out_directory = "output_data/GL_min_evidence/"

points = 800
model1 = NC_logg
model2 = NC_logL
GL_max = 76.8
tag = ""
# prior_name = "normal"
prior_name = "nu_0.2_1.5"
N = 4

GL_min_graph = 7
GL_mins = pickle.load(open(f"{in_directory}GL_mins_{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}.pcl", "rb"))

results1 = pickle.load(open(f"{in_directory}results_1_{model1.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}.pcl", "rb"))
results2 = pickle.load(open(f"{in_directory}results_2_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}.pcl", "rb"))
results = results1 - results2
pdb.set_trace()

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

# Bbar_s = ["0.400", "0.420", "0.440", "0.460", "0.480", "0.500"]
Bbar_s = ["0.420", "0.480"]

Bbar_list = []
for i in range(len(Bbar_s)):
  for j in range(i + 1, len(Bbar_s)):
    Bbar_list.append([Bbar_s[i], Bbar_s[j]])

# An assortment of starting values - try them all
x0 = [0, 0.657, -0.038, 1, 2 / 3]

try:
  pvalues_1 = pickle.load(open(f"{out_directory}{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}_pvalues_1.pcl", "rb"))
  raise(Exception)

except:
  pvalues_1 = numpy.zeros((len(GL_mins), len(Bbar_list)))
  pvalues_2 = numpy.zeros((len(GL_mins), len(Bbar_list)))

  for i in range(len(Bbar_list)):
    Bbar_1, Bbar_2 = Bbar_list[i]

    samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

    for j, GL_min in tqdm(enumerate(GL_mins)):
      g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

      cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
      cov_1_2 = numpy.linalg.cholesky(cov_matrix)
      cov_inv = numpy.linalg.inv(cov_1_2)

      res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)

      res1 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model1))
      chisq = chisq_calc(res1.x, cov_inv, model1, res_function)
      dof = g_s_cut.shape[0] - len(res1.x)
      p1 = chisq_pvalue(dof, chisq)

      # if (Bbar_1 == "0.420" and Bbar_2 == "0.480") and abs(GL_min - 25.6) < 0.1:
        # pdb.set_trace()

      # if p1 > 0.95:
      #   pdb.set_trace()

      pvalues_1[j, i] = p1

      res2 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model2))
      chisq = chisq_calc(res2.x, cov_inv, model2, res_function)
      dof = g_s_cut.shape[0] - len(res2.x)
      p2 = chisq_pvalue(dof, chisq)

      pvalues_2[j, i] = p2

  pickle.dump(pvalues_1, open(f"{out_directory}{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}_pvalues_1.pcl", "wb"))
  pickle.dump(pvalues_2, open(f"{out_directory}{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}_pvalues_2.pcl", "wb"))


pvalues_1 = pickle.load(open(f"{out_directory}{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}_pvalues_1.pcl", "rb"))
pvalues_2 = pickle.load(open(f"{out_directory}{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}_pvalues_2.pcl", "rb"))

ax = plt.gca()
fig = plt.gcf()
ax.set_xlabel(r"$gL_{min}$")
if N == 2:
  ax.set_xlim(GL_min_graph, 40)
if N == 4:
  ax.set_xlim(GL_min_graph, 35)

ax2 = ax.twinx()
ax2.set_ylabel(r'$p-value$')

pvalues_1_ext = pvalues_1.reshape(len(GL_mins) * len(Bbar_list))
pvalues_2_ext = pvalues_2.reshape(len(GL_mins) * len(Bbar_list))

GL_mins_ext = GL_mins.reshape((1, len(GL_mins))).repeat(len(Bbar_list), axis=1).reshape(len(GL_mins) * len(Bbar_list))

ax2.plot(GL_mins_ext, pvalues_1_ext, marker='s', markerfacecolor='none', label=r"$\log(g)$", ls='', color=color_blind_palette[1])
ax2.plot(GL_mins_ext, pvalues_2_ext, marker='^', markerfacecolor='none', label=r"$\log(L)$", ls='', color=color_blind_palette[3])

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

if N == 2:
  ax.set_ylim(1.2, max(numpy.mean(results, axis=1)) * 1.1)
if N == 4:
  ax.set_ylim(2, max(numpy.mean(results, axis=1)) * 1.1)

ax2.set_ylim(0, right2)

# Shift the axis a bit below 0
change = 0.02
a, b = ax.get_ylim()
ax.set_ylim(a - (b - a) * change, b)

a, b = ax2.get_ylim()
ax2.set_ylim(a - (b - a) * change, b)

# ax.set_yscale('log')
ax2.annotate(r"$p = 0.05$", xy=(36, 0.06), color='grey')

plt.show()
