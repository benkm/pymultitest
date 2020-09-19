from chisq_functions import *

# Fit specifics
N = 4
Bbar_1 = "0.440"
Bbar_2 = "0.500"
model = param_8g
GL_min = 24
GL_max = 76.8
prior_name = "update_1"
no_samples = 5

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


if prior_name == "update_2":
  alpha_range = [-0.1, 0.1]
  c1_range = [-200, 200]
  c2_range = [-400, 400]
  f0_range = [0, 1]
  f1_range = [-4, 4]
  lambduh_range = [0.5, 1.5]
  nu_range = [0.4, 2]
  omega_range = [0, 4]


if prior_name == "update_1":
  alpha_range = [-0.01, 0.01]
  c1_range = [-100, 100]
  c2_range = [-200, 200]
  f0_range = [0, 1]
  f1_range = [-4, 4]
  lambduh_range = [0.5, 1.5]
  nu_range = [0.4, 1.5]
  omega_range = [0, 4]


bounds4 = ([alpha_range[0], f_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f_range[1], lambduh_range[1], nu_range[1]])


bounds5 = ([alpha_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0]],
    [alpha_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1]])


bounds6 = ([alpha_range[0], f0_range[0], f1_range[0], f2_range[0], lambduh_range[0], nu_range[0]],
           [alpha_range[1], f0_range[1], f1_range[1], f2_range[1], lambduh_range[1], nu_range[1]])


bounds8 = ([alpha_range[0], c1_range[0], c2_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
           [alpha_range[1], c1_range[1], c2_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])


samples, N_s, g_s, L_s, Bbar_s, m_s = pickle.load(open(f'input_data/all_data_N{N}.pcl', 'rb'))


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
# x8 = [-2.11357238e-04,  1.81807344e+01, -4.13420167e+01,  5.97804995e-01,
#        -3.00156844e-01,  1.07987095e+00,  9.11785984e-01,  7.61712915e-01]

method = 'dogbox'

directory = 'output_data/chisq_fit_sigma/'


# Load in data
# samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl', keep_1_Bbar=True, Bbar_special="0.420")
samples, N_s, g_s, L_s, Bbar_s, m_s = load_in_data(f'input_data/Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl')

# Check that all data has the same N
assert len(set(N_s)) == 1

g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

size = samples_cut.shape[1]

numpy.random.seed(436347)
boot_samples = numpy.random.randint(0, size, size=(no_samples, size))

pvalues = numpy.zeros(no_samples)
param_estimates = numpy.zeros((no_samples, len(model.__code__.co_varnames[4:])))

for i in range(no_samples):
  samples_cut = samples_cut.T[boot_samples[i]].T
  m_s_cut = numpy.mean(samples_cut, axis=1)

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
  # x8 = [-2.11357238e-04, 1.81807344e+01, -4.13420167e+01, 5.97804995e-01,
  #        -3.00156844e-01, 1.07987095e+00, 9.11785984e-01, 7.61712915e-01]

  method = 'dogbox'

  directory = f'output_data/chisq_fit/'

  res = least_squares(res_function, x8, bounds=bounds8, args=(cov_inv, model), method=method)

  chisq = chisq_calc(res.x, cov_inv, model, res_function)
  dof = g_s_cut.shape[0] - len(res.x)
  p = chisq_pvalue(dof, chisq)
  print(f"chisq = {chisq}")
  print(f"chisq/dof = {chisq / dof}")
  print(f"pvalue = {p}")
  print(f"dof = {dof}")
  print(f"res_x = {res.x}")

  tag = f"{Bbar_1}_{Bbar_2}_GL_min{GL_min:.1f}"

  pvalues[i] = p
  param_estimates[i] = numpy.array(res.x)

pickle.dump(pvalues, open(f"{directory}pvalues_{model.__name__}_tag{tag}_N{N}_GLmax{GL_max:.1f}_prior{prior_name}.pcl", "wb"))
pickle.dump(param_estimates, open(f"{directory}param_estimates_{model.__name__}_tag{tag}_N{N}_GLmax{GL_max:.1f}_prior{prior_name}.pcl", "wb"))
