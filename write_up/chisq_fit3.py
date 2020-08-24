import datetime
import json
import pdb
import pickle
import sys
import time
import os

# import pymultinest
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.stats
from numpy import exp, log, pi
from scanf import scanf
from scipy.optimize import least_squares, minimize
from scipy.special import gammaincc

today = datetime.date.fromtimestamp(time.time())


# Input data
GL_min = 8
GL_max = 76.8
no_samples = 500
Z0 = 0.252731


# The one-loop expression as in the IRReg paper
def mPT_1loop(g, N):
  return - g * Z0 * (2 - 3 / N ** 2)


# Load in Andreas' pickled data
with open('Ben2.pcl', 'rb') as pickle_file:
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


def Omega_2(g, L, c, omega):
  return 1 + c * (g * L) ** -omega


# Used to alter the range of data being considered
def cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s):
  GL_s = g_s * L_s

  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10), 
                           GL_s <= GL_max * (1 + 10 ** -10))

  return g_s[keep], Bbar_s[keep], N_s[keep], L_s[keep], samples[keep], m_s[keep]


# Motivated by the functional form of the theory
def model27(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K1(g, N))


def model28(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K2(L, N))


# Use the same cut as Andreas for now
g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)


# x27 = [0.0014, -0.05, 0.58, -0.038, 1.064, 0.684, 0.453]
# x27 = [0.0014, -0.134, 0.608, -0.06, 1.064, 0.6844, 0.454]
x27 = [0, -0.04, 0.657, -0.038, 1, 2 / 3, 0.8]
x28 = [-0.016, -0.97, 20, -2.2, 0.98, 0.713, 0.0018]
x28 = [-0.01612187, -0.87133898,  4.        , -0.44064317,  0.97991928,
        0.7127984 ,  0.01029297]
x28 = [-0.01612187, -0.87133898,  1.        , -0.44064317/4,  0.97991928,
        0.7127984 ,  0.01029297]


def cov_matrix_calc(samples_cut, m_s_cut, N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut):
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

  # Find the covariance matrix - method 1 - don't enforce independence between ensembles
  # Calculate by hand so that the means from 
  size = samples_cut.shape[0]
  cov_matrix = numpy.zeros((size, size))
  for i in range(size):
    for j in range(size):
      if different_ensemble[i, j] == 0:
        cov_matrix[i, j] = numpy.mean((samples_cut[i] - m_s_cut[i]) * (samples_cut[j] - m_s_cut[j]))
      # else the value remains zero as there is no covariance between samples from different ensembles

  return cov_matrix, different_ensemble


cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)


def chisq_calc(x, cov_inv, model_function, **kwargs):
  # Caculate the residuals between the model and the data
  normalized_residuals = res_function(x, cov_inv, model_function, **kwargs)

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


def plot_fit(res, cov_matrix, model_function, directory, GL_min, GL_max, ext=1, alpha=0, lambduh=0, incl_K1=False, incl_alpha=False, K=K1,
             m_s_cut=m_s_cut, N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut, Bbar_s_cut=Bbar_s_cut):
  """
    ext : extension factor towards origin - model is plotted to 1 / (GL_max * ext)
  """
  if not os.path.isdir(directory):
    os.makedirs(directory)

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
  plt.savefig(f"{directory}best_fit_{model_function.__name__}_GLmin{GL_min}_GLmax{GL_max}_{today.year}_{today.month}_{today.day}.png")
  plt.show()
  plt.close()


# Try using the scipy least-squares method with Nelder-Mead
def res_function(x, cov_inv, model_function, prior=False, prior_values=None, prior_sigmas=None,
                m_s=m_s_cut, N_s=N_s_cut, g_s=g_s_cut, L_s=L_s_cut, Bbar_s=Bbar_s_cut):

  # Caculate the residuals between the model and the data
  predictions = model_function(N_s, g_s, L_s, Bbar_s, *x)

  residuals = m_s - predictions

  normalized_residuals = numpy.dot(cov_inv, residuals)

  # Include a prior in the form of extra residuals
  if prior:
    extra_piece = []

    # Get the parameter names
    param_names = numpy.array(model_function.__code__.co_varnames[4:])

    for entry in prior_values:
      index = numpy.argwhere(param_names == entry)[0][0]
      
      extra_piece.append((x[index] - prior_values[entry]) / prior_sigmas[entry])

    extra_piece = numpy.array(extra_piece)

    normalized_residuals = numpy.concatenate((normalized_residuals, extra_piece))

  return normalized_residuals


if __name__ == '__main__':
  directory = f'best_fit_graphs/{today.year}_{today.month}_{today.day}/'

  alpha_range = [-0.1, 0.1]
  c_range = [-10, 10]
  f0_range = [-100, 100]
  f1_range = [-20, 20]
  lambduh_range = [0, 2]
  nu_range = [0.5, 0.9]
  omega_range = [0, 2]

  alpha_range = [-0.1, 0.1]
  c_range = [-2, 2]
  f0_range = [-1, 1]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0.5, 0.9]
  omega_range = [0, 2]

  # alpha_range = [-0.2, 0.2]
  # c_range = [-2, 2]
  # f0_range = [-4, 4]
  # f1_range = [-2, 2]
  # lambduh_range = [-1, 3]
  # nu_range = [0.3, 1.1]
  # omega_range = [0, 4]

  bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
    [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])

  # res27 = least_squares(res_function, x27, bounds=bounds, args=(cov_inv, model27), method='dogbox')
  # plot_fit(res27, cov_matrix, model27, directory, GL_min, GL_max, ext=10, alpha=res27.x[0], lambduh=res27.x[4], incl_K1=True)
  # chisq27 = chisq_calc(res27.x, cov_inv, model27)
  # dof = g_s_cut.shape[0] - len(res27.x)
  # p27 = chisq_pvalue(dof, chisq27)
  # print(f"chisq = {chisq27}")
  # print(f"chisq/dof = {chisq27 / dof}")
  # print(f"pvalue = {p27}")
  # numpy.save(f"{directory}model27_best_fit_params.npy", numpy.array(res27.x))

  # res28 = least_squares(res_function, x28, args=(cov_inv, model28), method='lm')
  res28 = least_squares(res_function, x28, bounds=bounds, args=(cov_inv, model28), method='dogbox')
  plot_fit(res28, cov_matrix, model28, directory, GL_min, GL_max, ext=10, alpha=res28.x[0], lambduh=res28.x[4], incl_K1=True)
  chisq28 = chisq_calc(res28.x, cov_inv, model28)
  dof = g_s_cut.shape[0] - len(res28.x)
  p28 = chisq_pvalue(dof, chisq28)
  print(f"chisq = {chisq28}")
  print(f"chisq/dof = {chisq28 / dof}")
  print(f"pvalue = {p28}")
  numpy.save(f"{directory}model28_best_fit_params.npy", numpy.array(res28.x))
