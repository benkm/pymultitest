from bayes_functions import *

N = 4
Bbar_1 = "0.420"
Bbar_2 = "0.480"
model = no_corrections_logg
GL_min = 25.6
GL_max = 76.8
points = 100
tag=""


alpha_range = [-0.1, 0.1]
f0_range = [-4, 4]
f1_range = [-2, 2]
lambduh_range = [0, 2]
nu_range = [0.5, 0.9]

prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
n_params = len(prior_range)

samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)

directory = f'output_data/evidence/'


analysis1 = run_pymultinest(prior_range, no_corrections_logg, GL_min, GL_max, n_params, directory,
                            N, g_s, Bbar_s, L_s, samples, m_s,
                            n_live_points=points, sampling_efficiency=0.8, clean_files=False,
                            tag=tag)

analysis2 = run_pymultinest(prior_range, no_corrections_logL, GL_min, GL_max, n_params, directory,
                            N, g_s, Bbar_s, L_s, samples, m_s,
                            n_live_points=points, sampling_efficiency=0.8, clean_files=False,
                            tag=tag)
