# Investigate the effect of point count on the Bayesian Evidence
from bayes_functions import *
from chisq_functions import *
from tqdm import tqdm

# PARAMETERS
N = 4
Bbar_1 = "0.420"
Bbar_2 = "0.480"
GL_min = 25.6
GL_max = 76.8
tag = ""
prior_name = "normal"
no_samples = 5

alpha_range = [-0.1, 0.1]
f0_range = [-4, 4]
f1_range = [-2, 2]
lambduh_range = [0, 2]
nu_range = [0.5, 0.9]

model1 = NC_logg
model2 = NC_logL

# Where the results are saved
directory = f'output_data/point_evidence/'

prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
n_params = len(prior_range)

point_range = (10 ** numpy.linspace(2, 3.5, 20)).astype(int)

samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)


def f0(points):
  results_piece1 = numpy.zeros(no_samples)
  results_piece2 = numpy.zeros(no_samples)

  for i in tqdm(range(no_samples)):

    analysis1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory,
                                N, g_s, Bbar_s, L_s, samples, m_s,
                                n_live_points=points, sampling_efficiency=0.8, clean_files=False,
                                tag=f"{i}", return_analysis_small=True)

    analysis2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory,
                                N, g_s, Bbar_s, L_s, samples, m_s,
                                n_live_points=points, sampling_efficiency=0.8, clean_files=False,
                                tag=f"{i}", return_analysis_small=True)

    results_piece1[i] = analysis1[0]
    results_piece2[i] = analysis2[0]

  return numpy.array([results_piece1, results_piece2])


p = Pool(4)
results = numpy.array(p.map(f0, point_range, chunksize=1))
p.close()

results1 = results[:, 0, :]
results2 = results[:, 1, :]

pdb.set_trace()
pickle.dump(results1, open(f"{directory}results_1_{model1.__name__}_{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}.pcl", "wb"))
pickle.dump(results2, open(f"{directory}results_2_{model2.__name__}_{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}.pcl", "wb"))
pickle.dump(point_range, open(f"{directory}point_range_{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}.pcl", "wb"))
