from chisq_fit import *
import pymultinest
import os
import pickle

today = datetime.date.fromtimestamp(time.time())


# Function that returns likelihood functions
def likelihood_maker(n_params, cov_inv, model):
  def loglike(cube, ndim, nparams):
    params = []
    for i in range(n_params):
      params.append(cube[i])
    
    chisq = chisq_calc(params, cov_inv, model)

    return -0.5 * chisq

  return loglike


def prior_maker(prior_range):
  def prior(cube, ndim, nparams):
    for i in range(len(prior_range)):
      cube[i] = cube[i] * (prior_range[i][1] - prior_range[i][0]) + prior_range[i][0]

  return prior


def run_pymultinest(prior_range, model, GL_min, GL_max, n_params, directory, n_live_points=400):
  if not os.path.isdir(directory):
    os.makedirs(directory)

  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)

  cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut)
  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  likelihood_function = likelihood_maker(n_params, cov_inv, model)
  prior = prior_maker(prior_range)

  basename = f"{directory}{model.__name__}_GLmin{GL_min}_GLmax{GL_max}"

  # Get the parameter names of the model from the function
  # Ignore the first 4 parameters because they aren't fitting parameters
  param_names = model.__code__.co_varnames[4:]

  pymultinest.run(likelihood_function, prior, n_params, outputfiles_basename=basename, resume=False, n_live_points=n_live_points)
  json.dump(param_names, open(basename + 'params.json', 'w')) # save parameter names

  analysis = pymultinest.Analyzer(outputfiles_basename=basename, n_params=n_params)

  stats = analysis.get_stats()

  E, delta_E = stats['global evidence'], stats['global evidence error']

  sigma_1_range = [analysis.get_stats()['marginals'][i]['1sigma'] for i in range(n_params)]
  sigma_2_range = [analysis.get_stats()['marginals'][i]['2sigma'] for i in range(n_params)]

  posterior_data = analysis.get_equal_weighted_posterior()
  
  analysis_data = [E, delta_E, sigma_1_range, sigma_2_range, posterior_data]

  pickle.dump(analysis_data, open(f"{basename}_analysis.pcl", "wb"))

  return analysis


if __name__ == "__main__":
  # Default prior range
  alpha_range = [-1, 1]
  c_range = [-10, 10]
  f0_range = [-100, 100]
  f1_range = [-10, 10]
  lambduh_range = [-10, 10]
  nu_range = [0.5, 0.9]

  GL_min = 12.8
  GL_max = 76.8

  prior_range = [alpha_range, c_range, f0_range, f1_range, lambduh_range, nu_range]
  n_params = len(prior_range)

  # Where the results are saved
  directory = f'model_output_{today.year}_{today.month}_{today.day}/'

  # analysis = run_pymultinest(prior_range, model6, GL_min, GL_max, n_params, directory)
  # analysis = run_pymultinest(prior_range, model7, GL_min, GL_max, n_params, directory)
 
  alpha_range = [-0.1, 0.1]
  c_range = [-2, 0]
  f0_range = [-20, 20]
  f1_range = [-1, 1]
  lambduh_range = [-10, 10]
  nu_range = [0.5, 0.9]
  omega_range = [-0.1, 0.1]

  prior_range = [alpha_range, c_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]
  n_params = len(prior_range)

  analysis = run_pymultinest(prior_range, model8, GL_min, GL_max, n_params, directory, n_live_points=2000)
  