from chisq_functions import *


# Fit specifics
N = 4
Bbar_1 = "0.420"
Bbar_2 = "0.480"
model = no_corrections_logg
GL_min = 25.6
GL_max = 76.8

alpha_range = [-numpy.inf, numpy.inf]
c_range = [-numpy.inf, numpy.inf]
f0_range = [-numpy.inf, numpy.inf]
f1_range = [-numpy.inf, numpy.inf]
lambduh_range = [-numpy.inf, numpy.inf]
nu_range = [0, numpy.inf]
omega_range = [0, numpy.inf]

bounds5 = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])


# Load in data
samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

# Check that all data has the same N
assert len(set(N_s)) == 1

g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)


# Minimizer starting point
x0 = [0, 0.657, -0.038, 1, 2 / 3]
method = 'dogbox'

directory = f'output_data/chisq_fit/'

res = least_squares(res_function, x0, bounds=bounds5, args=(cov_inv, model), method=method)
plot_fit(res, cov_matrix, model, directory, GL_min, GL_max, N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, incl_K1=True)

plt.savefig(f"{directory}fit_graph_{model.__name__}_x0{x0}_{method}.png")
plt.show()

chisq = chisq_calc(res.x, cov_inv, model, res_function)
dof = g_s_cut.shape[0] - len(res.x)
p = chisq_pvalue(dof, chisq)
print(f"chisq = {chisq}")
print(f"chisq/dof = {chisq / dof}")
print(f"pvalue = {p}")

numpy.save(f"{directory}best_fit_params_{model.__name__}_x0{x0}_{method}.npy", numpy.array(res.x))
