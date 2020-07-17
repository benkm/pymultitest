import pickle
import numpy
import matplotlib.pyplot as plt
import os
import pdb

from scanf import scanf
from scipy.optimize import least_squares, minimize
from scipy.special import gammaincc


## -------------------- STEP 1 : Load in the data -----------------------------
with open('Ben.pcl', 'rb') as pickle_file:
  data = pickle.load(pickle_file, encoding='latin1')

  Bbar_s = []
  N_s = []
  g_s = []
  L_s = []
  m_s = []

  # Array to hold all of the data
  samples = []

  for key in data:
    if key[:16] == "DBinder_crossing":
      print(key)

      # Extract the parameter values
      Bbar, N, g, L = scanf("DBinder_crossing_B%f_%d_%f_%d", key)

      # Append them to the lists of parameter values
      Bbar_s.append(Bbar)
      N_s.append(N)
      g_s.append(g)
      L_s.append(L)
      
      # Extract the observed mass value
      m_s.append(data[key][4][0])

      # Now extract the 500 bootstrap samples
      samples.append(data[key][4][2])

  samples = numpy.array(samples)

# Turn data into numpy arrays
N_s = numpy.array(N_s)
g_s = numpy.array(g_s)
L_s = numpy.array(L_s)
Bbar_s = numpy.array(Bbar_s)
m_s = numpy.array(m_s)

# Remove nan values
keep = numpy.logical_not(numpy.isnan(samples))[:, 0]
samples = samples[keep]
N_s = N_s[keep]
g_s = g_s[keep]
L_s = L_s[keep]
Bbar_s = Bbar_s[keep]
m_s = m_s[keep]


## ---- STEP 2 : Define the models and the best fit (minimizer) locations -----
Z0 = 0.252731


# The one-loop expression as in the IRReg paper
def mPT_1loop(g, N):
  return - g * Z0 * (2 - 3 / N ** 2)


def K1(g, N):
  return numpy.log((g / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def K2(L, N):
  return numpy.log(((1 / L) / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def model1(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K1(g, N))


def model2(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K2(L, N))


# No corrections to scaling
def model3(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K1(g, N))


def model4(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K2(L, N))


# Starting points for minimizer - based on data from Hansenbusch and the
# EFT work of Antonin
x_EFT = [0, -0.04, 0.657, -0.038, 1, 2 / 3, 0.8]
x_EFT_small = [0, 0.657, -0.038, 1, 2 / 3]

# Based on the best fit for model2, using nu, omega can be less than 0
x2 = [-0.0221, 0.018, 0.548, -0.0614, 1.072, 0.72, -0.729]
x2_small = [-0.0221, 0.548, -0.0614, 1.072, 0.72]


x1_bounded = [0.0014, -0.134, 0.608, -0.06, 1.064, 0.6844, 0.454]
x1_bounded_small = [0.0014, 0.608, -0.06, 1.064, 0.6844]
x2_bounded = [-0.016, -0.97, 20, -2.2, 0.98, 0.713, 0.0018]
x2_bounded_small = [-0.016, 20, -2.2, 0.98, 0.713]


## STEP 3 : --------------- Define a function that finds the lowest gL_min of a
## fit that gives an acceptable p-value ---------------------------------------
def cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s):
  GL_s = g_s * L_s

  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10), 
                           GL_s <= GL_max * (1 + 10 ** -10))

  return g_s[keep], Bbar_s[keep], N_s[keep], L_s[keep], samples[keep], m_s[keep]


def cov_matrix_calc(data):
  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = data

  # In reality the covariance between different ensembles is 0. We can set it as
  # such to get a more accurate calculation of the covariance matrix
  different_N = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_g = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_L = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))

  # Check if two ensembles have different physical parameters, e.g. they are different
  # ensembles
  for i in range(samples_cut.shape[0]):
    for j in range(samples_cut.shape[0]):
      different_N[i, j] = N_s_cut[i] != N_s_cut[j]
      different_g[i, j] = g_s_cut[i] != g_s_cut[j]
      different_L[i, j] = L_s_cut[i] != L_s_cut[j]

  # This is true if two data points come different simulations
  different_ensemble = numpy.logical_or(different_N,
                       numpy.logical_or(different_L,
                                        different_g))

  # When calculating the covariance matrix, we know the covariance between
  # different ensembles is zero.
  size = samples_cut.shape[0]
  cov_matrix = numpy.zeros((size, size))
  cov_matrix2 = numpy.cov(samples_cut)

  for i in range(size):
    for j in range(size):
      if different_ensemble[i, j] == 0:
        cov_matrix[i, j] = cov_matrix2[i, j]
        # cov_matrix[i, j] = numpy.mean((samples_cut[i] - m_s_cut[i]) * (samples_cut[j] - m_s_cut[j]))
      # else the value remains zero as there is no covariance between samples from different ensembles

  return cov_matrix


def res_function(x, data, model, cov_inv):
  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = data

  # Caculate the residuals between the model and the data
  predictions = model(N_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, *x)

  residuals = m_s_cut - predictions

  normalized_residuals = numpy.dot(cov_inv, residuals)

  return normalized_residuals


def chisq_calc(x, data, model, cov_inv):
  normalized_residuals = res_function(x, data, model, cov_inv)

  chisq = numpy.sum(normalized_residuals ** 2)

  return chisq


def chisq_pvalue(k, x):
  """k is the rank, x is the chi-sq value. Basically k is the number of degrees of
  freedom equal to the number of data points minus the number of fitting parameters"""
  return gammaincc(k / 2, x / 2)


def get_minimum_acceptable_gL_min(model, x0, bounds=None):
  gL_s = g_s * L_s
  gL_max = max(gL_s)
  gL_min_list = numpy.sort(list(set(gL_s)))[::-1]

  gL_best = None

  for i in range(3, len(gL_min_list)):
    gL_min = gL_min_list[i]

    g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(gL_min, gL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
    data = g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut

    cov_matrix = cov_matrix_calc(data)
    cov_1_2 = numpy.linalg.cholesky(cov_matrix)
    cov_inv = numpy.linalg.inv(cov_1_2)

    if bounds is None:
      res = least_squares(res_function, x0, args=(data, model, cov_inv), method='lm')

    else:
      res = least_squares(res_function, x0, bounds=bounds, args=(data, model, cov_inv), method='dogbox')

    chisq = chisq_calc(res.x, data, model, cov_inv)
    dof = g_s_cut.shape[0] - len(res.x)
    p_value = chisq_pvalue(dof, chisq)
    print(f"model = {model.__name__}")
    print(f"gL_min = {gL_min}")
    print(f"chisq = {chisq}")
    print(f"pvalue = {p_value}")

    if p_value > 0.05:
      gL_best = gL_min

    i = i + 1

  return gL_best


bounds = ([-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, 0, -numpy.inf, 0],
            [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf])

bounds_small = ([-numpy.inf, -numpy.inf, -numpy.inf, 0, -numpy.inf],
          [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf])

x_s = [(x_EFT, x_EFT_small), (x2, x2_small), (x1_bounded, x1_bounded_small), (x2_bounded, x2_bounded_small)]
bounds_s = [(None, None), (None, None), (bounds, bounds_small), (bounds, bounds_small)]
names = ["EFT", "x2", "x1_bounded", "x2_bounded"]

try:
  gL_mins = pickle.load(open('gL_mins.pcl', "rb"))

except:
  gL_mins = {}
  for i in range(len(x_s)):
    x, x_small = x_s[i]
    bound, bound_small = bounds_s[i]
    gL_mins[names[i]] = {}
    for model in [model1, model2]:
      gL_mins[names[i]][model.__name__] = get_minimum_acceptable_gL_min(model, x, bounds=bound)

    for model in [model3, model4]:
      gL_mins[names[i]][model.__name__] = get_minimum_acceptable_gL_min(model, x_small, bounds=bound_small)
 
  pickle.dump(gL_mins, open("gL_mins.pcl", "wb"))

## STEP 4: --------- With these fits, perform a bootstrap on the fit to get the
## range of the resulting parameters obtained (also on the chisq and p-value) -


def colors(g, mini, maxi):
  two_thirds = (maxi * 2 + mini) / 3

  if mini <= g <= two_thirds:
    fraction = ((g - mini) / (two_thirds - mini))
    return ((1 - fraction), fraction, 0)

  elif two_thirds <= g <= maxi:
    fraction = ((g - two_thirds) / (maxi - two_thirds))
    return (0, (1 - fraction), fraction)

  else:
    raise(ValueError)


def plot_fit(res, cov_matrix, model_function, GL_min, GL_max, data, directory=False,
             ext=1, alpha=0, lambduh=0, incl_K1=False, incl_alpha=False, K=K1):
  """
    ext : extension factor towards origin - model is plotted to 1 / (GL_max * ext)
  """
  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = data

  # if not os.path.isdir(directory):
  #   os.makedirs(directory)

  N = 2

  std_diag = numpy.diag(cov_matrix) ** 0.5

  for Bbar in set(Bbar_s):
    for g in set(g_s_cut):
      entries = numpy.argwhere(numpy.logical_and(g_s_cut == g, Bbar_s_cut == Bbar))[:, 0]

      # Now sort by L values
      sort = numpy.argsort(L_s_cut[entries])

      plt.errorbar(1 / (g * L_s_cut[entries][sort]), m_s_cut[entries][sort] / g, yerr=std_diag[entries][sort] / g, ls='', label=f'g = {g}, Bbar = {Bbar}', color=colors(g, 0.1, 0.6))
      plt.scatter(1 / (g * L_s_cut[entries][sort]), m_s_cut[entries][sort] / g, facecolors='none', edgecolors=colors(g, 0.1, 0.6))

      L_range = numpy.linspace(GL_min / g, GL_max * ext / g, 1000)

      predictions = model_function(N, g, L_range, Bbar, *res.x)

      # Plot a smooth line for the model
      plt.plot(1 / (g * L_range), predictions / g, color=colors(g, 0.1, 0.6))

      if incl_alpha:
        # Plot the mPT_1loop + alpha * g ** 2 contribution
        plt.plot([1 / (GL_max * ext), 1 / GL_min], [(mPT_1loop(g, N) + g ** 2 * alpha)/ g, (mPT_1loop(g, N) + g ** 2 * alpha) / g], color=colors(g, 0.1, 0.6), ls='--', label='mPT_1loop + alpha term')

      if incl_K1:
        K1_term = - lambduh * g ** 2 * K1(g, N)
        # Plot the mPT_1loop + alpha * g ** 2 contribution and the K1 contribution
        plt.plot([1 / (GL_max * ext), 1 / GL_min], [(mPT_1loop(g, N) + g ** 2 * alpha + K1_term) / g, (mPT_1loop(g, N) + g ** 2 * alpha + K1_term) / g], color=colors(g, 0.1, 0.6), ls='--', label='mPT_1loop + alpha + K1 term')

  # Also add to the plot the 1-loop mass value
  plt.plot([0, 1 / GL_min], [mPT_1loop(g, N) / g, mPT_1loop(g, N) / g], color='k', ls='--', label='mPT_1loop')

  plt.xlabel("1 / gL")
  plt.ylabel("value / g")
  # plt.legend()
  # plt.savefig(f"{directory}best_fit_{model_function.__name__}_GLmin{GL_min}_GLmax{GL_max}_{today.year}_{today.month}_{today.day}.png")
  plt.show()
  plt.close()


## PART A: ----------------- Implement double-bootstrap -----------------------
# N1 = 50
# N2 = samples.shape[1]
# numpy.random.seed(98329782)
# resamples = numpy.random.randint(0, 500, size=(N1, N2))


def model_string_to_function(string):
  if string == 'model1':
    return model1
  if string == 'model2':
    return model2
  if string == 'model3':
    return model3
  if string == 'model4':
    return model4


# # Use the best fits from each of the models
# results_A = {}
# for j, fit_type in enumerate(gL_mins):
#   x, x_small = x_s[j]
#   bound, bound_small = bounds_s[j]
#   results_A[fit_type] = {}

#   for model_string in gL_mins[fit_type]:
#     gL_min = gL_mins[fit_type][model_string]
#     gL_s = g_s * L_s
#     gL_max = max(gL_s)

#     if gL_min is not None:
#       results_A[fit_type][model_string] = {}
#       fit = results_A[fit_type][model_string]

#       model = model_string_to_function(model_string)

#       if model_string in ['model3', 'model4']:
#         x = x_small
#         bound = bound_small

#       # Record the parameter values of the best fit
#       fit['params'] = {}
#       for param in model.__code__.co_varnames[4:]:
#         fit['params'][param] = numpy.zeros(N1)

#       # Record the chisq and p value of the fit
#       fit['chisq'] = numpy.zeros(N1)
#       fit['p_value'] = numpy.zeros(N1)

#       # Record the covariance matrix
#       g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(gL_min, gL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
      
#       n = g_s_cut.shape[0]
#       fit['cov'] = numpy.zeros((N1, n, n))

#       for i in range(N1):
#         resample = numpy.transpose(numpy.transpose(samples_cut)[resamples[i]])

#         # Need to set m_s_cut to a new value
#         m_s_resample = numpy.mean(resample, axis=1)

#         data = g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, resample, m_s_resample

#         cov_matrix = cov_matrix_calc(data)
#         cov_1_2 = numpy.linalg.cholesky(cov_matrix)
#         cov_inv = numpy.linalg.inv(cov_1_2)

#         if bound is None:
#           res = least_squares(res_function, x, args=(data, model, cov_inv), method='lm')

#         else:
#           res = least_squares(res_function, x, bounds=bound, args=(data, model, cov_inv), method='dogbox')

#         chisq = chisq_calc(res.x, data, model, cov_inv)
#         dof = g_s_cut.shape[0] - len(res.x)
#         p_value = chisq_pvalue(dof, chisq)

#         print(f"model = {model.__name__}")
#         print(f"gL_min = {gL_min}")
#         print(f"chisq = {chisq}")
#         print(f"pvalue = {p_value}")

#         # Add the results to the data we've collected
#         for k, param in enumerate(model.__code__.co_varnames[4:]):
#           fit['params'][param][i] = res.x[k]

#         fit['chisq'][i] = chisq
#         fit['p_value'][i] = p_value
#         fit['cov'][i] = cov_matrix

#       # Calculate the mean and 1 and 2 sigma ranges for each parameter
#       fit['params']['mean'] = {}
#       fit['params']['std'] = {}
#       fit['params']['sigma1'] = {}
#       fit['params']['sigma2'] = {}

#       sigma_1 = []
#       sigma_2 = []
#       sig_1_size = 0.68
#       sig_2_size = 0.95

#       for k, param in enumerate(model.__code__.co_varnames[4:]):
#         fit['params']['mean'][param] = numpy.mean(fit['params'][param])
#         fit['params']['std'][param] = numpy.mean(fit['params'][param])

#         lower = numpy.percentile(fit['params'][param], (0.5 - (sig_1_size / 2)) * 100)
#         upper = numpy.percentile(fit['params'][param], (0.5 + (sig_1_size / 2)) * 100)

#         lower2 = numpy.percentile(fit['params'][param], (0.5 - (sig_2_size / 2)) * 100)
#         upper2 = numpy.percentile(fit['params'][param], (0.5 + (sig_2_size / 2)) * 100)

#         fit['params']['sigma1'][param] = (lower, upper)
#         fit['params']['sigma2'][param] = (lower2, upper2)


## PART B: Take covariance matrix as a constant and use samples
results_B = {}
N = samples.shape[1]


def cov_matrix_calc_const(data):
  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = data

  # In reality the covariance between different ensembles is 0. We can set it as
  # such to get a more accurate calculation of the covariance matrix
  different_N = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_g = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_L = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))

  # Check if two ensembles have different physical parameters, e.g. they are different
  # ensembles
  for i in range(samples_cut.shape[0]):
    for j in range(samples_cut.shape[0]):
      different_N[i, j] = N_s_cut[i] != N_s_cut[j]
      different_g[i, j] = g_s_cut[i] != g_s_cut[j]
      different_L[i, j] = L_s_cut[i] != L_s_cut[j]

  # This is true if two data points come different simulations
  different_ensemble = numpy.logical_or(different_N,
                       numpy.logical_or(different_L,
                                        different_g))

  # When calculating the covariance matrix, we know the covariance between
  # different ensembles is zero.
  size = samples_cut.shape[0]
  cov_matrix = numpy.zeros((size, size))

  for i in range(size):
    for j in range(size):
      if different_ensemble[i, j] == 0:
        cov_matrix[i, j] = numpy.mean((samples_cut[i] - m_s_cut[i]) * (samples_cut[j] - m_s_cut[j]))
      # else the value remains zero as there is no covariance between samples from different ensembles

  return cov_matrix


for j, fit_type in enumerate(gL_mins):
  x, x_small = x_s[j]
  bound, bound_small = bounds_s[j]
  results_B[fit_type] = {}

  for model_string in gL_mins[fit_type]:
    gL_min = gL_mins[fit_type][model_string]
    gL_s = g_s * L_s
    gL_max = max(gL_s)

    if gL_min is not None:
      results_B[fit_type][model_string] = {}
      fit = results_B[fit_type][model_string]

      model = model_string_to_function(model_string)

      if model_string in ['model3', 'model4']:
        x = x_small
        bound = bound_small

      # Record the parameter values of the best fit
      fit['params'] = {}
      for param in model.__code__.co_varnames[4:]:
        fit['params'][param] = numpy.zeros(N)

      # Record the chisq and p value of the fit
      fit['chisq'] = numpy.zeros(N)
      fit['p_value'] = numpy.zeros(N)

      # Record the covariance matrix
      g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(gL_min, gL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
      n = g_s_cut.shape[0]
      data = g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut

      cov_matrix = cov_matrix_calc_const(data)
      cov_1_2 = numpy.linalg.cholesky(cov_matrix)
      cov_inv = numpy.linalg.inv(cov_1_2)

      fit['cov'] = cov_inv

      for i in range(N):
        # Need to set m_s_cut to a new value
        m_s_resample = samples_cut[:, i]

        # NOTE: res_function doesn't use the samples part of data
        data = g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples, m_s_resample

        if bound is None:
          res = least_squares(res_function, x, args=(data, model, cov_inv), method='lm')

        else:
          res = least_squares(res_function, x, bounds=bound, args=(data, model, cov_inv), method='dogbox')

        # plot_fit(res, cov_matrix, model, gL_min, gL_max, data, ext=10, alpha=res.x[0], lambduh=res.x[4], incl_K1=True)
        chisq = chisq_calc(res.x, data, model, cov_inv) * 0.5
        dof = g_s_cut.shape[0] - len(res.x)
        p_value = chisq_pvalue(dof, chisq)

        print(f"model = {model.__name__}")
        print(f"gL_min = {gL_min}")
        print(f"chisq = {chisq}")
        print(f"pvalue = {p_value}")

        # pdb.set_trace()
        # Add the results to the data we've collected
        for k, param in enumerate(model.__code__.co_varnames[4:]):
          fit['params'][param][i] = res.x[k]

        fit['chisq'][i] = chisq
        fit['p_value'][i] = p_value

      # Calculate the mean and 1 and 2 sigma ranges for each parameter
      fit['params']['mean'] = {}
      fit['params']['std'] = {}
      fit['params']['sigma1'] = {}
      fit['params']['sigma2'] = {}

      sigma_1 = []
      sigma_2 = []
      sig_1_size = 0.68
      sig_2_size = 0.95
      for k, param in enumerate(model.__code__.co_varnames[4:]):
        fit['params']['mean'][param] = numpy.mean(fit['params'][param])

        lower = numpy.percentile(fit['params'][param], (0.5 - (sig_1_size / 2)) * 100)
        upper = numpy.percentile(fit['params'][param], (0.5 + (sig_1_size / 2)) * 100)

        lower2 = numpy.percentile(fit['params'][param], (0.5 - (sig_2_size / 2)) * 100)
        upper2 = numpy.percentile(fit['params'][param], (0.5 + (sig_2_size / 2)) * 100)

        fit['params']['sigma1'][param] = (lower, upper)
        fit['params']['sigma2'][param] = (lower2, upper2)
