from chisq_fit import *
import pymultinest
import os

today = datetime.date.fromtimestamp(time.time())

# Where the results are saved
datafile = f'./model_output_{today.year}_{today.month}_{today.day}/'

if not os.path.isdir(f'model_output_{today.year}_{today.month}_{today.day}'):
  os.makedirs(f'model_output_{today.year}_{today.month}_{today.day}')

# Input data
GL_min = 12.8
GL_max = 76.8
no_samples = 500


# Use the same cut as Andreas for now
g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)


cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

# To start with I'm going to integrate the likelihood function in the cubiod formed
# by the two best parameter fits as the corners
res = least_squares(res_function, x0, args=(cov_inv, model1), method='lm')
res2 = least_squares(res_function, x0, args=(cov_inv, model2), method='lm')

params1 = numpy.array(res.x)
params2 = numpy.array(res2.x)


# Describes how broad a prior is used
prior_size = 10


# Use uniform priors
def prior(cube, ndim, nparams):
  for i in range(len(params1)):
    maxi = max(params1[i], params2[i])
    mini = min(params1[i], params2[i])
    Delta = (maxi - mini)
    cube[i] = cube[i] * prior_size * Delta + mini - Delta


def prior_7_params(cube, ndim, nparams):
  nu_min, nu_max = 0.5, 0.9
  f0_min, f0_max = -20, 20
  lambduh_min, lambduh_max = -10, 10
  omega_min = - 1 / nu_max
  #               alpha, c, f0, f1, lambduh, nu, omega
  parameter_mins = [-1, -0.9, f0_min, 3, -10, nu_min, -0.01]
  parameter_maxs = [1, -0.5, f0_max, 8, 10, nu_max, 0.01]

  for i in range(len(parameter_mins)):
    cube[i] = cube[i] * (parameter_maxs[i] - parameter_mins[i]) + parameter_mins[i]


def prior_6_params(cube, ndim, nparams):
  nu_min, nu_max = 0.5, 0.9
  f0_min, f0_max = -20, 20
  lambduh_min, lambduh_max = -10, 10
  #                alpha, c,   f0,  f1, lambduh, nu
  parameter_mins = [-10, -1, f0_min, -10, -10, nu_min]
  parameter_maxs = [10, 1, f0_max, 10, 10, nu_max]

  for i in range(len(parameter_mins)):
    cube[i] = cube[i] * (parameter_maxs[i] - parameter_mins[i]) + parameter_mins[i]


def loglike1(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu, omega = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6]
  x = alpha, c, f0, f1, lambduh, nu, omega

  chisq = chisq_calc(x, cov_inv, model1)

  loglikelihood = numpy.sum(-0.5 * chisq)

  return loglikelihood


def loglike2(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu, omega = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6]

  x = alpha, c, f0, f1, lambduh, nu, omega

  chisq = chisq_calc(x, cov_inv, model2)

  loglikelihood = numpy.sum(-0.5 * chisq)

  return loglikelihood


def loglike3(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
  x = alpha, c, f0, f1, lambduh, nu

  chisq = chisq_calc(x, cov_inv, model3)

  loglikelihood = numpy.sum(-0.5 * chisq)

  return loglikelihood


def loglike1_small(cube, ndim, nparams):
  alpha, f0, f1, lambduh, nu = cube[0], cube[2], cube[3], cube[4], cube[5]

  x = alpha, f0, f1, lambduh, nu

  chisq = chisq_calc(x, cov_inv, model1_small)

  loglikelihood = numpy.sum(-0.5 * chisq)

  return loglikelihood


def loglike2_small(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu, omega = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6]

  x = alpha, f0, f1, lambduh, nu

  chisq = chisq_calc(x, cov_inv, model2_small)

  loglikelihood = numpy.sum(-0.5 * chisq)

  return loglikelihood


def loglike5(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
  x = alpha, c, f0, f1, lambduh, nu
  chisq = chisq_calc(x, cov_inv, model5)
  loglikelihood = numpy.sum(-0.5 * chisq)

  return loglikelihood


# number of dimensions our problem has
parameters = ["alpha", "c", "a", "b", "lambduh", "nu", "omega"]
n_params = len(parameters)


def run_pymultinest(likelihood_function, model, label, prior, n_params):
  # run MultiNest
  pymultinest.run(likelihood_function, prior, n_params, outputfiles_basename=datafile + label, resume=False)
  json.dump(parameters, open(datafile + label + 'params.json', 'w')) # save parameter names

  run = pymultinest.Analyzer(outputfiles_basename=datafile + label, n_params=n_params)

  a_lnZ = run.get_stats()['global evidence']
  print()
  print('************************')
  print('MAIN RESULT: Evidence Z ')
  print('************************')
  print(f'log Z for model {label} = {a_lnZ:.1f}\n')

  return a_lnZ


def calculate_evidence(prior_size_factor):
  print("Running MULTINEST")
  print("===============================================================")

  def prior_6_param_evidence(cube, ndim, nparams):
    # Calculated on 17th June 2020 from .get_stats()['marginals'][i]['1simga']
    model1_range = [[0.0005853572784567271, 0.001866498744194777],
                    [-0.018228204591096673, -0.011775230591446972],
                    [-11.422102875021858, -9.621105891652844],
                    [1.7677927812576606, 1.7988108519915715],
                    [1.0270122283734873, 1.1110952600277195],
                    [0.6661373392944578, 0.692576691079186]]
    model2_range = [[-0.022319786471839402, -0.019287358551606647],
                    [-0.06706385049369484, -0.061266141742762334],
                    [-9.191897024381701, -7.8282552205573825],
                    [1.9523234804972314, 1.9853359088313542],
                    [1.0453962785708737, 1.1307129943447916],
                    [0.7111315082611166, 0.739203322019145]]

    # param_names = ['alpha', 'c', 'f0', 'f1', 'lambduh', 'nu']

    for i in range(6):
      mini = min(model1_range[i][0], model2_range[i][0])
      maxi = max(model1_range[i][1], model2_range[i][1])
      diff = maxi - mini

      bottom = mini - diff * ((prior_size_factor - 1) / 2)
      top = maxi + diff * ((prior_size_factor - 1) / 2)

      # print(f"Range on {param_names[i]} : ({bottom}, {top})")

      cube[i] = cube[i] * (top - bottom) + bottom
      
  # Run model 3 (log(g))
  label3 = f'3_prior_{prior_size_factor:.1f}'
  pymultinest.run(loglike3, prior_6_param_evidence, 6, outputfiles_basename=datafile + label3, resume=False)
  json.dump(parameters, open(datafile + label3 + 'params.json', 'w')) # save parameter names

  analysis3 = pymultinest.Analyzer(outputfiles_basename=datafile + label3, n_params=6)
  numpy.save(f"{datafile}posterior_data{label3}.npy", analysis3.get_equal_weighted_posterior())

  param_range3 = [analysis3.get_stats()['marginals'][i]['1sigma'] for i in range(6)]
  E3 = analysis3.get_stats()['global evidence']
  E3_delta = analysis3.get_stats()['global evidence error']

  # Run model 5 (log(1 / L))
  label5 = f'5_prior_{prior_size_factor:.1f}'
  pymultinest.run(loglike5, prior_6_param_evidence, 6, outputfiles_basename=datafile + label5, resume=False)
  json.dump(parameters, open(datafile + label5 + 'params.json', 'w')) # save parameter names

  analysis5 = pymultinest.Analyzer(outputfiles_basename=datafile + label5, n_params=6)
  numpy.save(f"{datafile}posterior_data{label5}.npy", analysis5.get_equal_weighted_posterior())

  param_range5 = [analysis5.get_stats()['marginals'][i]['1sigma'] for i in range(6)]
  E5 = analysis5.get_stats()['global evidence']
  E5_delta = analysis5.get_stats()['global evidence error']

  # The errors are independent so the error of the subtraction calculates them in quadrature
  error = numpy.sqrt(E5_delta ** 2 + E3_delta ** 2)

  evidence = E3 - E5

  print(f"logE = {evidence} +- {error}")

  return evidence, error


results = []
errors = []
for prior_size_factor in 2 ** numpy.arange(4):
  result, error = calculate_evidence(prior_size_factor)

  results.append(result)
  errors.append(error)

results = numpy.array(results)
errors = numpy.array(errors)

numpy.save(f"evidence_array_{today.year}_{today.month}_{today.day}.npy", results)
numpy.save(f"evidence_error_array_{today.year}_{today.month}_{today.day}.npy", errors)


# model1_lnZ = run_pymultinest(loglike1, model1, '1', prior_7_params)
# model2_lnZ = run_pymultinest(loglike2, model2, '2', prior_7_params)
# model3_lnZ = run_pymultinest(loglike3, model3, '3', prior_6_params, 6)

# model5_lnZ = run_pymultinest(loglike5, model5, '5', prior_6_params, 6)

# E = numpy.exp(model1_lnZ - model2_lnZ)

# Get posterior distributions
# analysis1 = pymultinest.analyse.Analyzer(n_params, outputfiles_basename=datafile + '1')
# numpy.save(f"{datafile}posterior_data1.npy", analysis1.get_equal_weighted_posterior())

# analysis2 = pymultinest.analyse.Analyzer(n_params, outputfiles_basename=datafile + '2')
# numpy.save(f"{datafile}posterior_data2.npy", analysis2.get_equal_weighted_posterior())


# analysis3 = pymultinest.analyse.Analyzer(n_params, outputfiles_basename=datafile + '3')
# numpy.save(f"{datafile}posterior_data3.npy", analysis3.get_equal_weighted_posterior())

# analysis5 = pymultinest.analyse.Analyzer(n_params, outputfiles_basename=datafile + '5')
# numpy.save(f"{datafile}posterior_data5.npy", analysis5.get_equal_weighted_posterior())

# param_range3 = [analysis3.get_stats()['marginals'][i]['1sigma'] for i in range(6)]
# E3 = analysis3.get_stats()['global evidence']
# E3_delta = analysis3.get_stats()['global evidence error']

# param_range5 = [analysis5.get_stats()['marginals'][i]['1sigma'] for i in range(6)]
# E5 = analysis5.get_stats()['global evidence']
# E5_delta = analysis5.get_stats()['global evidence error']


# Run also for the smaller models without corrections to scaling
# model1_small_lnZ = run_pymultinest(loglike1_small, model1_small, '1_small')
# model2_small_lnZ = run_pymultinest(loglike2_small, model2_small, '2_small')

# analysis1_small = pymultinest.analyse.Analyzer(n_params - 2, outputfiles_basename=datafile + '1_small')
# numpy.save(f"{datafile}posterior_data1_small.npy", analysis1_small.get_equal_weighted_posterior())

# analysis2_small = pymultinest.analyse.Analyzer(n_params - 2, outputfiles_basename=datafile + '2_small')
# numpy.save(f"{datafile}posterior_data2_small.npy", analysis2_small.get_equal_weighted_posterior())
