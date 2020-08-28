import pickle

from chisq_fit3 import *
from tqdm import tqdm

directory = f"posterior_data/{today.year}_{today.month}_{today.day}/"
points = 200
no_samples = 10

GL_min_graph = 10
GL_mins = numpy.sort(list(set(numpy.around(g_s * L_s, 1))))

# Because the model has 5 parameters we need at least 6 data points
GL_maxi = numpy.around(numpy.sort(g_s * L_s)[-6], 1)
GL_max = 76.8

results = pickle.load(open(f"{directory}evidence_results_points{points}_samples{no_samples}.pcl", "rb"))
results = results[GL_mins >= GL_min_graph]
GL_mins = GL_mins[GL_mins >= GL_min_graph]

results = results[GL_mins <= GL_maxi]
GL_mins = GL_mins[GL_mins <= GL_maxi]

plt.errorbar(GL_mins, numpy.mean(results, axis=1), yerr=numpy.std(results, axis=1), capsize=5, color='k', label='log(Evidence)')
ax = plt.gca()
ax.set_ylabel("log(Evidence)")

# Add in the frequentist side of things
pvalues_1 = []
pvalues_2 = []

for GL_min in tqdm(GL_mins):
  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
  
  cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut, N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut)
  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  alpha_range = [-0.1, 0.1]
  f0_range = [-10, 10]
  f1_range = [-10, 10]
  lambduh_range = [0, 2]
  nu_range = [0.5, 0.9]

  bounds = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])

  x0 = [0, 0.657, -0.038, 1, 2 / 3]

  kwargs = {"m_s": m_s_cut, "N_s": N_s_cut, "g_s": g_s_cut, "L_s": L_s_cut, "Bbar_s": Bbar_s_cut}

  res50 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model50),
  kwargs=kwargs, method='dogbox')
  chisq50 = chisq_calc(res50.x, cov_inv, model50, **kwargs)
  dof = g_s_cut.shape[0] - len(res50.x)
  pvalues_1.append(chisq_pvalue(dof, chisq50))

  res51 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model51),
  kwargs=kwargs, method='dogbox')
  chisq51 = chisq_calc(res51.x, cov_inv, model51, **kwargs)
  dof = g_s_cut.shape[0] - len(res51.x)
  pvalues_2.append(chisq_pvalue(dof, chisq51))


ax = plt.gca()
fig = plt.gcf()

ax2 = ax.twinx()
ax2.set_ylabel('p-value')
ax2.plot(GL_mins, pvalues_1, marker='s', markerfacecolor='none', label="model1")
ax2.plot(GL_mins, pvalues_2, marker='s', markerfacecolor='none', label="model2")

fig.tight_layout()

ax.tick_params(direction='in')
ax2.tick_params(direction='in')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')


plt.show()
