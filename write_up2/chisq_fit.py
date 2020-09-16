from chisq_functions import *


# Fit specifics
N = 4
Bbar_1 = "0.400"
Bbar_2 = "0.440"
model = quadratic_negative
GL_min = 16
GL_max = 76.8

alpha_range = [-0.1, 0.1]
c_range = [-numpy.inf, numpy.inf]
f0_range = [0.4, 0.8]
f1_range = [-0.2, 0.2]
f2_range = [-1, 1]
f_range = [-10, 10]
lambduh_range = [0, 2]
nu_range = [0, 2]
omega_range = [0, numpy.inf]

alpha_range = [-numpy.inf, numpy.inf]
c1_range = [-numpy.inf, numpy.inf]
c2_range = [-numpy.inf, numpy.inf]
f0_range = [0, 1]
f1_range = [-numpy.inf, numpy.inf]
lambduh_range = [0, 2]
nu_range = [0, numpy.inf]
omega_range = [0, numpy.inf]


bounds4 = ([alpha_range[0], f_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f_range[1], lambduh_range[1], nu_range[1]])

bounds5 = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])

bounds6 = ([alpha_range[0], f0_range[0], f1_range[0], f2_range[0], lambduh_range[0], nu_range[0]],
           [alpha_range[1], f0_range[1], f1_range[1], f2_range[1], lambduh_range[1], nu_range[1]])


bounds8 = ([alpha_range[0], c1_range[0], c2_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
           [alpha_range[1], c1_range[1], c2_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])



# Load in data
# samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl', keep_1_Bbar=True, Bbar_special="0.420")
samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')


# Check that all data has the same N
assert len(set(N_s)) == 1

g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)


# Minimizer starting point
# x0 = [0, 1, 1, 2 / 3]
# x0 = [0, 0.657, -0.038, 1, 2 / 3]
# x0 = [0, 0.657, -0.038, 0.001, 1, 2 / 3]
# x0 = [-9.64e-06, 0.4, -0.08, 0.36, 1, 0.9]
# # x0 = [-0.00313578,  0.84747275, -1.        ,  1.13686033,  1.24574919,
# #         0.86559021]
x0 = [-6.72070595e-05, 4.18032790e-01, 1.17430348e-01, -1.92958327e-01,
        1.14282757e+00, 7.60070699e-01]
x8 = [0, 0, 0, 0.5, -0.1, 1, 2 / 3, 0.8]

method = 'dogbox'

directory = f'output_data/chisq_fit/'

res = least_squares(res_function, x8, bounds=bounds8, args=(cov_inv, model), method=method)
plot_fit(res.x, cov_matrix, model, directory, GL_min, GL_max, N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, incl_K1=True)

plt.savefig(f"{directory}fit_graph_{model.__name__}_GL_min{GL_min}_x0{x0}_{method}.png")
plt.show()

chisq = chisq_calc(res.x, cov_inv, model, res_function)
dof = g_s_cut.shape[0] - len(res.x)
p = chisq_pvalue(dof, chisq)
print(f"chisq = {chisq}")
print(f"chisq/dof = {chisq / dof}")
print(f"pvalue = {p}")

numpy.save(f"{directory}best_fit_params_{model.__name__}_x0{x0}_{method}.npy", numpy.array(res.x))


# f0 = 0.4
# f1 = -0.080
# f2 = 0.48
# res_x = [-9.64e-06, f0, f1, f2, 1, 0.9]

# res_x = [-9.64e-06, 0.40, -0.08, +0.48, 1, 0.95]
# res_x = [-9.64e-06, 0.46, -0.245, 0.7, 1, 0.95]
# res_x = [-9.64e-06, 0.46, 0.2, 0.4, 1, 0.95]


# plot_fit(res_x, cov_matrix, model, directory, GL_min, GL_max, N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, incl_K1=True)
# print(chisq_calc(res_x, cov_inv, model, res_function))
