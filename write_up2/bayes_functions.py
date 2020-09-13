from chisq_functions import *
import pymultinest
import os
import pickle
from multiprocessing import Pool, current_process


# Function that returns likelihood functions
def likelihood_maker(n_params, cov_inv, model, res_function):
  def loglike(cube, ndim, nparams):
    params = []
    for i in range(n_params):
      params.append(cube[i])
    
    chisq = chisq_calc(params, cov_inv, model, res_function)

    return -0.5 * chisq

  return loglike


def prior_maker(prior_range):
  def prior(cube, ndim, nparams):
    for i in range(len(prior_range)):
      cube[i] = cube[i] * (prior_range[i][1] - prior_range[i][0]) + prior_range[i][0]

  return prior


def run_pymultinest(prior_range, model, GL_min, GL_max, n_params, directory,
                    N, g_s, Bbar_s, L_s, samples, m_s, prior_name="",
                    n_live_points=400, likelihood_kwargs={}, INS=False, clean_files=False,
                    sampling_efficiency=0.3, return_analysis_small=False, tag=""): # Reccomended for Bayesian evidence

  if not os.path.isdir(directory):
    os.makedirs(directory)

  g_s_cut, Bbar_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s)

  cov_matrix, different_ensemble = cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut)

  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  res_function = make_res_function(N, m_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)

  likelihood_function = likelihood_maker(n_params, cov_inv, model, res_function)

  prior = prior_maker(prior_range)

  basename = f"{directory}{model.__name__}{tag}_prior{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{n_live_points}"

  # Save the priors into a file
  pickle.dump(prior_range, open(f"{directory}priors_{prior_name}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{n_live_points}.pcl", "wb"))

  # Get the parameter names of the model from the function
  # Ignore the first 4 parameters because they aren't fitting parameters
  param_names = model.__code__.co_varnames[4:]

  pymultinest.run(likelihood_function, prior, n_params, outputfiles_basename=basename, resume=False,
                  n_live_points=n_live_points, sampling_efficiency=sampling_efficiency,
                  evidence_tolerance=0.1, importance_nested_sampling=INS
                  )
  # save parameter names
  f = open(basename + '.paramnames', 'w')
  for i in range(len(param_names)):
    f.write(f"{param_names[i]}\n")

  f.close()

  # Save the prior ranges
  f = open(basename + '.ranges', 'w')
  for i in range(len(param_names)):
    f.write(f"{param_names[i]} {prior_range[i][0]} {prior_range[i][1]}\n")
  
  f.close()
  # Also save as a json
  json.dump(param_names, open(f'{basename}params.json', 'w'))

  analysis = pymultinest.Analyzer(outputfiles_basename=basename, n_params=n_params)

  stats = analysis.get_stats()

  E, delta_E = stats['global evidence'], stats['global evidence error']

  sigma_1_range = [analysis.get_stats()['marginals'][i]['1sigma'] for i in range(n_params)]
  sigma_2_range = [analysis.get_stats()['marginals'][i]['2sigma'] for i in range(n_params)]
  median = [analysis.get_stats()['marginals'][i]['median'] for i in range(n_params)]

  posterior_data = analysis.get_equal_weighted_posterior()
  
  analysis_data = [E, delta_E, sigma_1_range, sigma_2_range, posterior_data, median]

  if not clean_files:
    pickle.dump(analysis_data, open(f"{basename}_analysis.pcl", "wb"))

  # Make a cut down version for the purpose of quicker transfer
  analysis_small = [E, delta_E, sigma_1_range, sigma_2_range, median]

  print(f"{current_process()}: saving {basename}_analysis_small.pcl")
  pickle.dump(analysis_small, open(f"{basename}_analysis_small.pcl", "wb"))

  if clean_files:
    # Remove the remaining saved files to conserve disk space
    print(f"Removing files : {basename}*")
    os.popen(f'rm {basename}ev.dat')
    os.popen(f'rm {basename}live.points')
    os.popen(f'rm {basename}.paramnames')
    os.popen(f'rm {basename}params.json')
    os.popen(f'rm {basename}phys_live.points')
    os.popen(f'rm {basename}post_equal_weights.dat')
    os.popen(f'rm {basename}post_separate.dat')
    os.popen(f'rm {basename}.ranges')
    os.popen(f'rm {basename}resume.dat')
    os.popen(f'rm {basename}stats.dat')
    os.popen(f'rm {basename}summary.txt')
    os.popen(f'rm {basename}.txt')

  if return_analysis_small:
    return analysis_small

  else:
    return analysis
