import pickle

from chisq_fit3 import *
from tqdm import tqdm

# directory = f"posterior_data/{today.year}_{today.month}_{today.day}/"
directory = f"posterior_data/{today.year}_{8}_{31}/"

points = 1000
no_samples = 20
model1 = model27
model2 = model28

GL_min_graph = 7
GL_mins = pickle.load(open(f"{directory}evidence_GL_mins_points{points}_samples{no_samples}.pcl", "rb"))

# results1 = pickle.load(open(f"{directory}evidence_results1_points{points}_samples{no_samples}.pcl", "rb"))
# results2 = pickle.load(open(f"{directory}evidence_results2_points{points}_samples{no_samples}.pcl", "rb"))

results1 = pickle.load(open(f"{directory}evidence_model27_points{points}_samples{no_samples}.pcl", "rb"))
results2 = pickle.load(open(f"{directory}evidence_model28_points{points}_samples{no_samples}.pcl", "rb"))
results = results1 - results2

results = results[GL_mins >= GL_min_graph]
GL_mins = GL_mins[GL_mins >= GL_min_graph]

plt.errorbar(GL_mins, numpy.mean(results, axis=1), yerr=numpy.std(results, axis=1), capsize=5, color='k', label='log(Evidence)', ls='')
ax = plt.gca()
ax.set_ylabel("log(Evidence)")

# Show the Jeffrey's scale 
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [0, 0], [1, 1], color='g', alpha=0.2, label='Insignificant 0 < x < 1')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [1, 1], [2.5, 2.5], color='r', alpha=0.2, label='Significant 1 < x < 2.5')
plt.fill_between([min(GL_mins) / 2, max(GL_mins) * 2], [2.5, 2.5], [max(numpy.mean(results, axis=1)) * 1.1, max(numpy.mean(results, axis=1)) * 1.1], color='b', alpha=0.2, label='Very Significant, x > 2.5')

# Add in the frequentist side of things
pvalues_1 = []
pvalues_2 = []

for GL_min in tqdm(GL_mins):
  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
  
  cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut, N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut)
  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  alpha_range = [-0.1, 0.1]
  c_range = [-2, 2]
  f0_range = [-4, 4]
  f1_range = [-2, 2]
  lambduh_range = [-1, 3]
  nu_range = [0.3, 1.1]
  omega_range = [0, 2]

  if model1 is model50 and model2 is model51:
    bounds = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])
    x0 = [0, 0.657, -0.038, 1, 2 / 3]
  
  if model1 is model27 and model2 is model28:
    bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
    [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])
    # x0 = [0, 0, 0.657, -0.038, 1, 2 / 3, 0.8]
    x0 = [-0.01612187, -0.87133898,  4.        , -0.44064317,  0.97991928, 0.7127984 ,  0.01029297]


  kwargs = {"m_s": m_s_cut, "N_s": N_s_cut, "g_s": g_s_cut, "L_s": L_s_cut, "Bbar_s": Bbar_s_cut}

  res1 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model1),
  kwargs=kwargs, method='dogbox')
  chisq1 = chisq_calc(res1.x, cov_inv, model1, **kwargs)
  dof = g_s_cut.shape[0] - len(res1.x)
  pvalues_1.append(chisq_pvalue(dof, chisq1))

  res2 = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model2),
  kwargs=kwargs, method='dogbox')
  chisq2 = chisq_calc(res2.x, cov_inv, model2, **kwargs)
  dof = g_s_cut.shape[0] - len(res2.x)
  pvalues_2.append(chisq_pvalue(dof, chisq2))


ax = plt.gca()
fig = plt.gcf()
ax.set_xlabel("GL_min")
ax.set_xlim(GL_min_graph, 40)

ax2 = ax.twinx()
ax2.set_ylabel('p-value')
ax2.plot(GL_mins, pvalues_1, marker='s', markerfacecolor='none', label="model1", ls='')
ax2.plot(GL_mins, pvalues_2, marker='s', markerfacecolor='none', label="model2", ls='')

# Show our acceptance threshold (alpha)
ax2.plot(GL_mins, [0.05, ] * len(GL_mins), ls='--', label='alpha = 0.05', color='r')


fig.tight_layout()

ax.tick_params(direction='in')
ax2.tick_params(direction='in')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Align 0 for both axes
left1, right1 = ax.get_ylim()
left2, right2 = ax2.get_ylim()

ax.set_ylim(0, right1)
ax2.set_ylim(0, right2)

plt.show()
