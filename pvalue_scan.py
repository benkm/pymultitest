from chisq_fit import *
import os

today = datetime.date.fromtimestamp(time.time())

directory = f"posterior_data/{today.year}_{today.month}_{today.day}/"

if not os.path.isdir(directory):
  os.makedirs(directory)

model = model23
x = x23

param_names = model.__code__.co_varnames[4:]

GL_min_ratio = 1.9

pvalues = {}
fit_values = {}

alpha_range = [-0.1, 0.1]
c_range = [-10, 10]
f0_range = [-100, 100]
f1_range = [-10, 10]
lambduh_range = [0, 2]
nu_range = [0.5, 0.9]
omega_range = [0, 2]

bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
    [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])

for GL_min in numpy.sort(list(set(g_s * L_s))):
  for GL_max in numpy.sort(list(set(g_s * L_s))):
    if GL_max > GL_min * GL_min_ratio:
        g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)

        cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut,
                                      N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut)

        kwargs = {"m_s": m_s_cut, "N_s": N_s_cut, "g_s": g_s_cut, "L_s": L_s_cut, "Bbar_s": Bbar_s_cut}
        cov_1_2 = numpy.linalg.cholesky(cov_matrix)
        cov_inv = numpy.linalg.inv(cov_1_2)

        res = least_squares(res_function, x, bounds=bounds, args=(cov_inv, model), method='dogbox', kwargs=kwargs)
        chisq = chisq_calc(res.x, cov_inv, model, **kwargs)
        dof = g_s_cut.shape[0] - len(res.x)

        if dof > 0:
          p = chisq_pvalue(dof, chisq)
          pvalues[(f"{GL_min:.1f}", f"{GL_max:.1f}")] = p
          for i, param in enumerate(param_names):
            fit_values[(f"{GL_min:.1f}", f"{GL_max:.1f}", param)] = res.x[i]

data = [pvalues, fit_values, bounds]

pickle.dump(data, open(f"{directory}{model.__name__}_best_fit_data.pcl", "wb"))

# Plot the results
