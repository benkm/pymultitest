## TO INVESTIGATE THE 5 PARAMETER MODEL
from chisq_functions import *
from bayes_functions import *
from tqdm import tqdm
import argparse
import sys

# PARAMETERS
N = 4
GL_max = 76.8
points = 10
tag = ""
prior_name = "8param"
no_samples = 8

Bbar_1 = sys.argv[1]
Bbar_2 = sys.argv[2]
GL_min = float(sys.argv[3])


alpha_range = [-0.1, 0.1]
f0_range = [0, 1]
f1_range = [-2, 2]
lambduh_range = [0, 2]
# nu_range = [0.5, 0.9]
nu_range = [0.2, 1.5]

# 8 param fit
alpha_range = [-0.1, 0.1]
c1_range = [-100, 100]
c2_range = [-100, 100]
f0_range = [0, 1]
f1_range = [-2, 2]
lambduh_range = [0, 2]
nu_range = [0, 2]
omega_range = [0, 2]

model1 = param_8g
model2 = param_8L

# Where the results are saved
directory = f'output_data/GL_min_bayes/'

prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
prior_range = [alpha_range, c1_range, c2_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]

n_params = len(prior_range)

# Because the model has n parameters we need at least (n+1) data points

# All Bbar_s have the same g_s and L_s, so choose the first one wlog.
samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

GL_maxi = numpy.around(numpy.sort(g_s * L_s)[-(n_params + 1)], 1)
GL_max = 76.8

GL_mins = numpy.sort(list(set(numpy.around(g_s * L_s, 1))))
GL_mins = GL_mins[GL_mins <= GL_maxi]

# Apply a lower-bound GL_min, as we aren't particularly interested in results
# below a certain threshold
GL_mini = 10
GL_mins = GL_mins[GL_mins >= GL_mini]


def f0(i):
  tag = f"{Bbar_1}_{Bbar_2}_{i}"

  samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

  g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

  cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)

  analysis1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, return_analysis_small=True)

  analysis2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                              N, g_s, Bbar_s, L_s, samples, m_s,
                              n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                              tag=tag, return_analysis_small=True)

  return numpy.array([analysis1[0], analysis2[0]])


p = Pool()
results = numpy.array(p.map(f0, range(no_samples), chunksize=1))
p.close()

results1 = results[:, 0]
results2 = results[:, 1]

tag = f"Bbar_1{Bbar_1}_Bbar_2{Bbar_2}_GL_min{GL_min}"

pickle.dump(results1, open(f"{directory}results_1_{model1.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}.pcl", "wb"))
pickle.dump(results2, open(f"{directory}results_2_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}.pcl", "wb"))
pickle.dump(GL_mins, open(f"{directory}GL_mins_{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmax{GL_max:.1f}_p{points}.pcl", "wb"))
