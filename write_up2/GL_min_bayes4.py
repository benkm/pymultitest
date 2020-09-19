## TO INVESTIGATE THE 5 PARAMETER MODEL
from chisq_functions import *
from bayes_functions import *
from tqdm import tqdm
import argparse
import sys

# PARAMETERS
N = 4
GL_max = 76.8
points = 500
tag = ""
prior_name = "update_2"
no_samples = 40

run_id = int(sys.argv[1])

GL_mins = numpy.array([3.2, 4, 4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])

Bbar_s = ["0.420", "0.440", "0.460", "0.480", "0.500"]
Bbar_list = []
for i in range(len(Bbar_s)):
  for j in range(i + 1, len(Bbar_s)):
    Bbar_list.append([Bbar_s[i], Bbar_s[j]])

Bbar_id = run_id // (len(GL_mins) * no_samples)
GL_min_id = (run_id // no_samples) % len(GL_mins)
i = run_id % no_samples

Bbar_1, Bbar_2 = Bbar_list[Bbar_id]
GL_min = GL_mins[GL_min_id]

print("Hi this is Python here")
print(f"run_id = {run_id}")
print(f"GL_min = {GL_min}")
print(f"Bbar_1 = {Bbar_1}")
print(f"Bbar_2 = {Bbar_2}")
print(f"i = {i}")


alpha_range = [-0.1, 0.1]
f0_range = [0, 1]
f1_range = [-2, 2]
lambduh_range = [0, 2]
# nu_range = [0.5, 0.9]
nu_range = [0.2, 1.5]

# 8 param fit
if prior_name == "smaller_nu":
  alpha_range = [-0.1, 0.1]
  c1_range = [-100, 100]
  c2_range = [-100, 100]
  f0_range = [0, 1]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0.2, 1.5]
  omega_range = [0, 2]


Update1
alpha_range = [-0.01, 0.01]
c1_range = [-100, 100]
c2_range = [-200, 200]
f0_range = [0, 1]
f1_range = [-4, 4]
lambduh_range = [0.5, 1.5]
nu_range = [0.4, 1.5]
omega_range = [0, 4]

# Update2
if prior_name == "update_2":
  alpha_range = [-0.1, 0.1]
  c1_range = [-200, 200]
  c2_range = [-400, 400]
  f0_range = [0, 1]
  f1_range = [-4, 4]
  lambduh_range = [0.5, 1.5]
  nu_range = [0.4, 2]
  omega_range = [0, 4]

model1 = param_8g
model2 = param_8L

# Where the results are saved
directory = f'output_data/GL_min_bayes4/'

prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
prior_range = [alpha_range, c1_range, c2_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]

n_params = len(prior_range)

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

tag = f"{Bbar_1}_{Bbar_2}_{i}"

samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)

tag = f"{Bbar_1}_{Bbar_2}_GL_min{GL_min:.1f}"

analysis1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                            N, g_s, Bbar_s, L_s, samples, m_s,
                            n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                            tag=tag, return_analysis_small=True, keep_GLmax=False)

analysis2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                            N, g_s, Bbar_s, L_s, samples, m_s,
                            n_live_points=points, sampling_efficiency=0.3, clean_files=True,
                            tag=tag, return_analysis_small=True, keep_GLmax=False)

pickle.dump(analysis1[0], open(f"{directory}results_1_{model1.__name__}_{tag}_prior{prior_name}_N{N}_p{points}.pcl", "wb"))
pickle.dump(analysis2[0], open(f"{directory}results_2_{model2.__name__}_{tag}_prior{prior_name}_N{N}_p{points}.pcl", "wb"))
pickle.dump(GL_mins, open(f"{directory}GL_mins_{model1.__name__}_{model2.__name__}_{tag}_prior{prior_name}_N{N}_p{points}.pcl", "wb"))
