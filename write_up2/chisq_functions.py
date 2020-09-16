import json
import pdb
import pickle
import sys
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
from tqdm import tqdm


# The one-loop expression as in the IRReg paper
def mPT_1loop(g, N):
  Z0 = 0.252731
  return - g * Z0 * (2 - 3 / N ** 2)


# with open('Ben2.pcl', 'rb') as pickle_file:
# with open(f'Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl', 'rb') as pickle_file:


def load_in_data(filename, keep_1_Bbar=False, Bbar_special=None):
  # Load in Andreas' pickled data
  with open(filename, 'rb') as pickle_file:
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
        if keep_1_Bbar:
          if abs(Bbar - float(Bbar_special)) < 10 ** -6:

            Bbar_s.append(Bbar)
            N_s.append(N)
            g_s.append(g)
            L_s.append(L)
            
            # Extract the observed mass value
            m_s.append(data[key][4][0])

            # Now extract the 500 bootstrap samples
            samples.append(data[key][4][2])

        else:
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

  return samples, N_s, g_s, L_s, Bbar_s, m_s


# Calculate the lambda terms
def K1(g, N):
  return numpy.log((g / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def K2(L, N):
  return numpy.log(((1 / L) / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


# Used to alter the range of data being considered
def cut(GL_min, GL_max, g_s, Bbar_s, L_s, samples, m_s):
  GL_s = g_s * L_s

  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10), 
                           GL_s <= GL_max * (1 + 10 ** -10))

  return g_s[keep], Bbar_s[keep], L_s[keep], samples[keep], m_s[keep]


# Motivated by the functional form of the theory
def model27(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / (1 + c * (g * L) ** -omega)) - f0) / f1) - lambduh * K1(g, N))


def model28(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / (1 + c * (g * L) ** -omega)) - f0) / f1) - lambduh * K2(L, N))


# Asymptotic model
def model40(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (r1 * Bbar + r2 * numpy.log(g * L) - r3) - lambduh * K1(g, N))


def model41(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (r1 * Bbar + r2 * numpy.log(g * L) - r3) - lambduh * K2(L, N))


# No corrections to scaling model
def NC_logg(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K1(g, N))


def NC_logL(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K2(L, N))


def NC_logg_single(N, g, L, Bbar, alpha, f, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * f - lambduh * K1(g, N))


def NC_logL_single(N, g, L, Bbar, alpha, f, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * f - lambduh * K2(L, N))


def NC_logg_f_nu(N, g, L, Bbar, alpha, f, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * f ** (numpy.log(nu) / (g * L)) - lambduh * K1(g, N))


def NC_logL_f_nu(N, g, L, Bbar, alpha, f, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * f ** (numpy.log(nu) / (g * L)) - lambduh * K2(L, N))


def quadratic(N, g, L, Bbar, alpha, f0, f1, f2, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (- f1 + numpy.sqrt(f1 ** 2 - 2 * (f0 - Bbar) * f2)) / f2 - lambduh * K1(g, N))


def QN(N, g, L, Bbar, alpha, f0, f1, f2, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (- f1 - numpy.sqrt(f1 ** 2 - 2 * (f0 - Bbar) * f2)) / f2 - lambduh * K1(g, N))


def QN_L(N, g, L, Bbar, alpha, f0, f1, f2, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (- f1 - numpy.sqrt(f1 ** 2 - 2 * (f0 - Bbar) * f2)) / f2 - lambduh * K2(L, N))


def param_8g(N, g, L, Bbar, alpha, c1, c2, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / (1 + (c1 + c2 * Bbar) * (g * L) ** -omega)) - f0) / f1) - lambduh * K1(g, N))


def param_8L(N, g, L, Bbar, alpha, c1, c2, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / (1 + (c1 + c2 * Bbar) * (g * L) ** -omega)) - f0) / f1) - lambduh * K2(L, N))


# More accurate asymptotic model
def model60(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (r1 + r2 * numpy.log(g * L))) - r3) - lambduh * K1(g, N))


def model61(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (r1 + r2 * numpy.log(g * L))) - r3) - lambduh * K2(L, N))


# To get asymptotic solutions from the full model
def get_rs(c, f0, f1, omega):
  c_prime = c + 1

  r1 = f1 * c_prime
  r2 = f1 * omega
  r3 = f0 / f1

  return r1, r2, r3


def get_back(r1, r2, r3, f0):
  f1 = f0 / r3
  omega = r2 / f1
  c_prime = r1 / f1
  c = c_prime - 1

  return c, f0, f1, omega


def cov_matrix_calc(g_s_cut, L_s_cut, m_s_cut, samples_cut):
  # In reality the covariance between different ensembles is 0. We can set it as
  # such to get a more accurate calculation of the covariance matrix
  different_g = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
  different_L = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))

  # Check if two ensembles have different physical parameters, e.g. they are different
  # ensembles
  for i in range(samples_cut.shape[0]):
    for j in range(samples_cut.shape[0]):
      different_g[i, j] = g_s_cut[i] != g_s_cut[j]
      different_L[i, j] = L_s_cut[i] != L_s_cut[j]

  # This is true if two data points come different simulations
  different_ensemble = numpy.logical_or(different_L, different_g)

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


def chisq_calc(x, cov_inv, model_function, res_function):
  # Caculate the residuals between the model and the data
  normalized_residuals = res_function(x, cov_inv, model_function)

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


def plot_fit(res_x, cov_matrix, model_function, directory, GL_min, GL_max,
             N, m_s, g_s, L_s, Bbar_s, ext=100, incl_alpha=False, incl_K1=False):
  """
    ext : extension factor towards origin - model is plotted to 1 / (GL_max * ext)
  """
  if not os.path.isdir(directory):
    os.makedirs(directory)

  std_diag = numpy.diag(cov_matrix) ** 0.5

  param_names = model_function.__code__.co_varnames[4:]

  alpha_index = numpy.argwhere(numpy.array(param_names) == "alpha")
  lambduh_index = numpy.argwhere(numpy.array(param_names) == "lambduh")

  alpha = res_x[alpha_index[0, 0]]
  lambduh = res_x[lambduh_index[0, 0]]

  for Bbar in set(Bbar_s):
    for g in set(g_s):
      entries = numpy.argwhere(numpy.logical_and(g_s == g, Bbar_s == Bbar))[:, 0]

      # Now sort by L values
      sort = numpy.argsort(L_s[entries])

      plt.errorbar(1 / (g * L_s[entries][sort]), m_s[entries][sort] / g, yerr=std_diag[entries][sort] / g, ls='', label=f'g = {g}, Bbar = {Bbar}', color=colors(g, 0.1, 0.6))
      plt.scatter(1 / (g * L_s[entries][sort]), m_s[entries][sort] / g, facecolors='none', edgecolors=colors(g, 0.1, 0.6))

      L_range = numpy.linspace(GL_min / g, GL_max * ext / g, 1000)

      predictions = model_function(N, g, L_range, Bbar, *res_x)

      # Plot a smooth line for the model
      plt.plot(1 / (g * L_range), predictions / g, color=colors(g, 0.1, 0.6))

      if incl_alpha:
        # Plot the mPT_1loop + alpha * g ** 2 contribution
        plt.plot([1 / (GL_max * ext), 1 / GL_min], [(mPT_1loop(g, N) + g ** 2 * alpha) / g, (mPT_1loop(g, N) + g ** 2 * alpha) / g], color=colors(g, 0.1, 0.6), ls='--', label='mPT_1loop + alpha term')

      if incl_K1:
        K1_term = - lambduh * g ** 2 * K1(g, N)
        # Plot the mPT_1loop + alpha * g ** 2 contribution and the K1 contribution

        plt.plot([1 / (GL_max * ext), 1 / GL_min], [(mPT_1loop(g, N) + g ** 2 * alpha + K1_term) / g, (mPT_1loop(g, N) + g ** 2 * alpha + K1_term) / g], color=colors(g, 0.1, 0.6), ls='--', label='mPT_1loop + alpha + K1 term')

  # Also add to the plot the 1-loop mass value
  plt.plot([0, 1 / GL_min], [mPT_1loop(g, N) / g, mPT_1loop(g, N) / g], color='k', ls='--', label='mPT_1loop')

  plt.xlabel("1 / gL")
  plt.ylabel("value / g")


# Try using the scipy least-squares method with Nelder-Mead
def make_res_function(N, m_s, g_s, L_s, Bbar_s):
  def res_function(x, cov_inv, model_function):

    # Caculate the residuals between the model and the data
    predictions = model_function(N, g_s, L_s, Bbar_s, *x)

    residuals = m_s - predictions

    normalized_residuals = numpy.dot(cov_inv, residuals)

    return normalized_residuals

  return res_function
