import json
import sys
import numpy
import pickle
from numpy import log, exp, pi
import scipy.stats, scipy
# import pymultinest
import matplotlib.pyplot as plt
import pdb
import datetime
import time
from scipy.special import gammaincc
from scipy.optimize import minimize, least_squares
from scanf import scanf

today = datetime.date.fromtimestamp(time.time())


# Input data
GL_min = 12.8
GL_max = 76.8
no_samples = 500
Z0 = 0.252731


# The one-loop expression as in the IRReg paper
def mPT_1loop(g, N):
  return - g * Z0 * (2 - 3 / N ** 2)


# Load in Andreas' pickled data
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


# Calculate the lambda terms
def K1(g, N):
  return numpy.log((g / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def K2(L, N):
  return numpy.log(((1 / L) / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def Omega(g, L, c, omega):
  return 1 / (1 + c * (g * L) ** -omega)


# Used to alter the range of data being considered
def cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s):
  GL_s = g_s * L_s

  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10), 
                           GL_s <= GL_max * (1 + 10 ** -10))

  return g_s[keep], Bbar_s[keep], N_s[keep], L_s[keep], samples[keep], m_s[keep]


# Model 1 using g as a regulator
def model1(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega(g, L, c, omega)) - 1) * f0 - lambduh * K1(g, N))


# Model 2 using (1 / L) as a regulator
def model2(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega(g, L, c, omega)) - 1) * f0 - lambduh * K2(L, N))


def model1_small(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (f1 * Bbar - 1) * f0 - lambduh * K1(g, N))


def model2_small(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (f1 * Bbar - 1) * f0 - lambduh * K2(L, N))


# Use the same cut as Andreas for now
g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)

# Best fit params
alpha_fit = 0.0018
c_fit = 0.024
f0_fit = -64.3
f1_fit = 1.1131
lambduh_fit = 1.057
nu_fit = 0.674
omega_fit = 0.800


x0 = [alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit, omega_fit]
x1 = [alpha_fit, f0_fit, f1_fit, lambduh_fit, nu_fit]


def cov_matrix_calc(samples_cut, m_s_cut):
  # In reality the covariance between different ensembles is 0. We can set it as
  # such to get a more accurate calculation of the covariance matrix
  different_N = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_g = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_L = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))

  # Check if two ensembles have different physical parameters, e.g. they are different
  # ensembles
  for i in range(samples_cut.shape[0]):
    for j in range(samples_cut.shape[0]):
      different_N[i, j] = N_s[i] != N_s[j]
      different_g[i, j] = g_s[i] != g_s[j]
      different_L[i, j] = L_s[i] != L_s[j]

  # This is true if two data points come different simulations
  different_ensemble = numpy.logical_or(different_N,
                       numpy.logical_or(different_L,
                                        different_g))

  # Find the covariance matrix - method 1 - don't enforce independence between ensembles
  # Calculate by hand so that the means from 
  size = samples_cut.shape[0]
  cov_matrix = numpy.zeros((size, size))
  for i in range(size):
    for j in range(size):
      if different_ensemble[i, j] == 0:
        cov_matrix[i, j] = numpy.mean((samples_cut[i] - m_s_cut[i]) * (samples_cut[j] - m_s_cut[j]))
      # else the value remains zero as there is no covariance between samples from different ensembles

  return cov_matrix


cov_matrix = cov_matrix_calc(samples_cut, m_s_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)


def chisq_calc(x, cov_inv, model_function):
  # Caculate the residuals between the model and the data
  predictions = model_function(N_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, *x)

  residuals = m_s_cut - predictions

  normalized_residuals = numpy.dot(cov_inv, residuals)

  chisq = numpy.sum(normalized_residuals ** 2)

  return chisq


def chisq_pvalue(k, x):
  """k is the rank, x is the chi-sq value. Basically k is the number of degrees of
  freedom equal to the number of data points minus the number of fitting parameters"""
  return gammaincc(k / 2, x / 2)


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


def plot_fit(res, cov_matrix, model_function, ext=1, alpha=0, lambduh=0, incl_K1=False, incl_alpha=False, K=K1):
  """
    ext : extension factor towards origin - model is plotted to 1 / (GL_max * ext)
  """
  N = 2

  std_diag = numpy.diag(cov_matrix) ** 0.5

  for Bbar in set(Bbar_s_cut):
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
  plt.savefig(f"graphs/model_fit{model_function.__name__}_{today.year}_{today.month}_{today.day}.png")
  plt.show()
  plt.close()


# Try using the scipy least-squares method with Nelder-Mead
def res_function(x, cov_inv, model_function):

  # Caculate the residuals between the model and the data
  predictions = model_function(N_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, *x)

  residuals = m_s_cut - predictions

  normalized_residuals = numpy.dot(cov_inv, residuals)

  return normalized_residuals


if __name__ == '__main__':
  x_dict = {}
  x_dict[model1], x_dict[model2] = x0, x0
  x_dict[model1_small], x_dict[model2_small] = x1, x1

  # Plot model 1
  res = least_squares(res_function, x_dict[model1], args=(cov_inv, model1), method='lm')
  plot_fit(res, cov_matrix, model1, ext=10, alpha=res.x[0], lambduh=res.x[4], incl_K1=True)

  for model in [model2, model1_small, model2_small]:
    res = least_squares(res_function, x_dict[model], args=(cov_inv, model), method='lm')
    plot_fit(res, cov_matrix, model, ext=10, alpha=res.x[0])
