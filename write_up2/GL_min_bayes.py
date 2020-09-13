## TO INVESTIGATE THE 5 PARAMETER MODEL
from chisq_functions import *
from bayes_functions import *
from tqdm import tqdm

# PARAMETERS
N = 4
Bbar_s = ["0.400", "0.420", "0.440", "0.460", "0.480", "0.500"]
GL_min = 25.6
GL_max = 76.8
points = 5000
tag = ""
prior_name = "normal"

alpha_range = [-0.1, 0.1]
f0_range = [-4, 4]
f1_range = [-2, 2]
lambduh_range = [0, 2]
nu_range = [0.5, 0.9]

model1 = no_corrections_logg
model2 = no_corrections_logL

# Where the results are saved
directory = f'output_data/GL_min_bayes/'

prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
n_params = len(prior_range)

Bbar_list = []
for i in range(len(Bbar_s)):
  for j in range(i + 1, len(Bbar_s)):
    Bbar_list.append([Bbar_s[i], Bbar_s[j]])

# Because the model has n parameters we need at least (n+1) data points

# All Bbar_s have the same g_s and L_s, so choose the first one wlog.
Bbar_1, Bbar_2 = Bbar_list[0]
samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

GL_maxi = numpy.around(numpy.sort(g_s * L_s)[-(n_params + 1)], 1)
GL_max = 76.8

GL_mins = numpy.sort(list(set(numpy.around(g_s * L_s, 1))))
GL_mins = GL_mins[GL_mins <= GL_maxi]

# Apply a lower-bound GL_min, as we aren't particularly interested in results
# below a certain threshold
GL_mini = 5
GL_mins = GL_mins[GL_mins >= GL_mini]

results1 = numpy.zeros((len(GL_mins), len(Bbar_list)))
results2 = numpy.zeros((len(GL_mins), len(Bbar_list)))
results = numpy.zeros((len(GL_mins), 2, len(Bbar_list)))


def f0(GL_min):
  results_piece1 = numpy.zeros(len(Bbar_list))
  results_piece2 = numpy.zeros(len(Bbar_list))

  for i in tqdm(range(len(Bbar_list))):
    Bbar_1, Bbar_2 = Bbar_list[i]

    samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

    g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

    cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)

    analysis1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                                N, g_s, Bbar_s, L_s, samples, m_s,
                                n_live_points=points, sampling_efficiency=0.8, clean_files=False,
                                tag=tag, return_analysis_small=True)

    analysis2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                                N, g_s, Bbar_s, L_s, samples, m_s,
                                n_live_points=points, sampling_efficiency=0.8, clean_files=False,
                                tag=tag, return_analysis_small=True)

    results_piece1[i] = analysis1[0]
    results_piece2[i] = analysis2[0]

  return numpy.array([results_piece1, results_piece2])


# for j, GL_min in enumerate(GL_mins):
#   results[j] = f0(GL_min)

p = Pool(4)
results = numpy.array(p.map(f0, GL_mins, chunksize=1))
p.close()

results1 = results[:, 0, :]
results2 = results[:, 1, :]

pickle.dump(results1, open(f"{directory}results_1_{model1.__name__}_{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{points}.pcl", "wb"))
pickle.dump(results2, open(f"{directory}results_2_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{points}.pcl", "wb"))
pickle.dump(GL_mins, open(f"{directory}GL_mins_{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{points}.pcl", "wb"))
