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


cov_matrix = cov_matrix_calc(samples_cut, m_s_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)

# To start with I'm going to integrate the likelihood function in the cubiod formed
# by the two best parameter fits as the corners
res = least_squares(res_function, x0, args=(cov_inv, model1), method='lm')
res2 = least_squares(res_function, x0, args=(cov_inv, model2), method='lm')

params1 = numpy.array(res.x)
params2 = numpy.array(res2.x)


# Describes how broad a prior is used
prior_size = 100


# Use uniform priors
def prior(cube, ndim, nparams):
  for i in range(len(params1)):
    maxi = max(params1[i], params2[i])
    mini = min(params1[i], params2[i])
    Delta = (maxi - mini)
    cube[i] = cube[i] * prior_size * Delta + mini - Delta


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


# number of dimensions our problem has
parameters = ["alpha", "c", "a", "b", "lambduh", "nu", "omega"]
n_params = len(parameters)


def run_pymultinest(likelihood_function, model, label):
  # run MultiNest
  pymultinest.run(likelihood_function, prior, n_params, outputfiles_basename=datafile + label, resume=False)
  json.dump(parameters, open(datafile + label + 'params.json', 'w')) # save parameter names

  # plot the distribution of a posteriori possible models
  plt.figure()
  plt.plot(g_s_cut * L_s_cut, m_s_cut / g_s_cut, '+ ', color='red', label='data')
  run = pymultinest.Analyzer(outputfiles_basename=datafile + label, n_params=n_params)
  for (alpha, c, f0, f1, lambduh, nu, omega) in run.get_equal_weighted_posterior()[::100,:-1]:
    plt.plot(g_s_cut * L_s_cut, model(N_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, alpha, c, f0, f1, lambduh, nu, omega) / g_s_cut, '-', color='blue', alpha=0.3, label='data')

  plt.savefig(datafile + 'posterior' + label + '.pdf')
  plt.close()

  a_lnZ = run.get_stats()['global evidence']
  print()
  print('************************')
  print('MAIN RESULT: Evidence Z ')
  print('************************')
  print(f'log Z for model {label} = {a_lnZ:.1f}\n')

  return a_lnZ


model1_lnZ = run_pymultinest(loglike1, model1, '1')
model2_lnZ = run_pymultinest(loglike2, model2, '2')

E = numpy.exp(model1_lnZ - model2_lnZ)

# Get posterior distributions
analysis1 = pymultinest.analyse.Analyzer(n_params, outputfiles_basename=datafile + '1')
numpy.save(f"{datafile}posterior_data1_size{prior_size}.npy", analysis1.get_equal_weighted_posterior())

analysis2 = pymultinest.analyse.Analyzer(n_params, outputfiles_basename=datafile + '2')
numpy.save(f"{datafile}posterior_data2_size{prior_size}.npy", analysis2.get_equal_weighted_posterior())
