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

Bbar_1 = "0.420"
Bbar_2 = "0.480"
N = 4

# The one-loop expression as in the IRReg paper
def mPT_1loop(g, N):
  return - g * Z0 * (2 - 3 / N ** 2)


# Load in Andreas' pickled data
# with open('Ben2.pcl', 'rb') as pickle_file:
with open(f'Ben_N={N}_B={Bbar_1}_B={Bbar_2}.pcl', 'rb') as pickle_file:
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
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / (1 + c * (g * L) ** -omega)) - f0) / f1) - lambduh * K1(g, N))


def model28(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / (1 + c * (g * L) ** -omega)) - f0) / f1) - lambduh * K2(L, N))


# Asymptotic model
def model40(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (r1 * Bbar + r2 * numpy.log(g * L) - r3) - lambduh * K1(g, N))


def model41(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (r1 * Bbar + r2 * numpy.log(g * L) - r3) - lambduh * K2(L, N))


# No corrections to scaling model
def model50(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K1(g, N))


def model51(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar - f0) / f1) - lambduh * K2(L, N))


# More accurate asymptotic model
def model60(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (r1 + r2 * numpy.log(g * L))) - r3) - lambduh * K1(g, N))


def model61(N, g, L, Bbar, alpha, r1, r2, r3, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (r1 + r2 * numpy.log(g * L))) - r3) - lambduh * K2(L, N))


g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)


# Bbar_s / (c_prime * f1+ omega * numpy.log(g_s * L_s) * f1) - f0/f1
# ((Bbar_s / (1 + (c_prime - 1) * (g_s * L_s) ** -omega)) - f0)/f1
x27 = [0.0014, -0.05, 0.58, -0.038, 1.064, 0.684, 0.453]
x27 = [0.0014, -0.134, 0.608, -0.06, 1.064, 0.6844, 0.454]
x27 = [0, -0.04, 0.657, -0.038, 1, 2 / 3, 0.8]
x27 = [ 1.02040644e-03, -7.03445237e-01,  1.85354748e+00, -1.83766936e-01,
        1.06990729e+00,  6.84724598e-01,  1.00000000e-03]
x27 = [0, -1.1, 0.657, -0.038, 1, 2 / 3, 0.8]
# x27 = [0.00231032942, -1.68948597, 0.582233883, -0.0863432296, 1.08681148, 0.669969122, 1.20784972]
# x27 = [0.0014094110011229537, -0.9956587394857183, 100, -15.520408778140792, 1.0837290017931231, 0.6779685700464381, 0.0004295598691863679]

x28 = [-0.016, -0.97, 20, -2.2, 0.98, 0.713, 0.0018] 
x28 = [-1.89020457e-02, -9.80798332e-01,  2.00000000e+01, -3.49234687e+00,
        1.06376347e+00,  7.11750219e-01,  3.86805210e-03]
x28 = [-0.01892748112187912, -0.9961520257453251, 100, -17.51002928127534, 1.0638925005064692, 0.7122102338205046, 0.0007525097395268309]
x28 = [-0.01612187, -0.87133898,  1.        , -0.44064317/4,  0.97991928,
        0.7127984 ,  0.01029297]
x41 = [-0.016, -17.6, -1.25, -9.06, 0.98, 0.713]
x61 = [-2.07208844e-02, -0.05671968881968792, -0.02002526730378901, 
       -5.403922057787865, 1.07421302e+00, 7.22702801e-01]


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

# The following prior comes as a result of the asymptotic study
# x27 = [1.00933027e-03, -1 + 5.5e-04, 1000, -99.32, 1.07010446e+00, 6.74677964e-01, -7.0056e-06]


def make_cov_matrix_calc(N_s_cut, g_s_cut, L_s_cut):
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

  return cov_matrix_calc


cov_matrix_calc = make_cov_matrix_calc(N_s_cut, g_s_cut, L_s_cut)

cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut)
cov_1_2 = numpy.linalg.cholesky(cov_matrix)
cov_inv = numpy.linalg.inv(cov_1_2)


def chisq_calc(x, cov_inv, model_function, res_function, **kwargs):
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
def make_res_function(m_s, N_s, g_s, L_s, Bbar_s):
  def res_function(x, cov_inv, model_function):

    # Caculate the residuals between the model and the data
    predictions = model_function(N_s, g_s, L_s, Bbar_s, *x)

    residuals = m_s - predictions

    normalized_residuals = numpy.dot(cov_inv, residuals)
    
    return normalized_residuals

  return res_function


res_function = make_res_function(m_s_cut, N_s_cut, g_s_cut, L_s_cut, Bbar_s_cut)


def use_asymptotic_model(model, model_asy, bounds, bounds_asy, x0_asy, cov_inv, res_function, f0=100):
  res_asy = least_squares(res_function, x0_asy, bounds=bounds_asy, args=(cov_inv, model_asy), method='dogbox')

  alpha = res_asy.x[0]
  r1 = res_asy.x[1]
  r2 = res_asy.x[2]
  r3 = res_asy.x[3]
  lambduh = res_asy.x[4]
  nu = res_asy.x[5]
  
  c, f0, f1, omega = get_back(r1, r2, r3, 100)

  x0 = [alpha, c, f0, f1, lambduh, nu, omega]

  res = least_squares(res_function, x0, bounds=bounds, args=(cov_inv, model), method='dogbox')

  return res


if __name__ == '__main__':
  directory = f'best_fit_graphs/{today.year}_{today.month}_{today.day}/'

  # Test the interoperability between model28 and model61
  c_prime = 10 ** -3
  omega = 10 ** -4
  f1 = -100
  f0 = 1000
  c = -1 + c_prime
  alpha = 0
  nu = 2 / 3
  lambduh = 1

  # r1 = f1 * c_prime
  # r2 = f1 * omega
  # r3 = f0 / f1
  r1, r2, r3 = get_rs(c, f0, f1, omega)

  predictions1 = model28(N, g_s, L_s, Bbar_s, alpha, c, f0, f1, lambduh, nu, omega)
  predictions2 = model61(N, g_s, L_s, Bbar_s, alpha, r1, r2, r3, lambduh, nu)

  # Take it back now y'all
  c, f0, f1, omega = get_back(r1, r2, r3, f0)
  predictions3 = model28(N, g_s, L_s, Bbar_s, alpha, c, f0, f1, lambduh, nu, omega)
  predictions4 = model61(N, g_s, L_s, Bbar_s, alpha, r1, r2, r3, lambduh, nu)

  # alpha_range = [-0.1, 0.1]
  # c_range = [-10, 10]
  # f0_range = [-20, 20]
  # f1_range = [-20, 20]
  # lambduh_range = [0, 2]
  # nu_range = [0.5, 0.9]
  # omega_range = [0, 2]

  alpha_range = [-numpy.inf, numpy.inf]
  c_range = [-numpy.inf, numpy.inf]
  f0_range = [-numpy.inf, numpy.inf]
  f1_range = [-numpy.inf, numpy.inf]
  lambduh_range = [-numpy.inf, numpy.inf]
  nu_range = [0, numpy.inf]
  omega_range = [0, numpy.inf]

  # alpha_range = [-0.1, 0.1]
  # c_range = [-2, 2]
  # f0_range = [-1, 1]
  # f1_range = [-2, 2]
  # lambduh_range = [0, 2]
  # nu_range = [0.5, 0.9]
  # omega_range = [0, 2]

  # alpha_range = [-0.2, 0.2]
  # c_range = [-2, 2]
  # f0_range = [-4, 4]
  # f1_range = [-2, 2]
  # lambduh_range = [-1, 3]
  # nu_range = [0.3, 1.1]
  # omega_range = [0, 4]

  bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
    [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])

  res27 = least_squares(res_function, x27, bounds=bounds, args=(cov_inv, model27), method='dogbox')
  plot_fit(res27, cov_matrix, model27, directory, GL_min, GL_max, ext=10, alpha=res27.x[0], lambduh=res27.x[4], incl_K1=True)
  chisq27 = chisq_calc(res27.x, cov_inv, model27, res_function)
  dof = g_s_cut.shape[0] - len(res27.x)
  p27 = chisq_pvalue(dof, chisq27)
  print(f"chisq = {chisq27}")
  print(f"chisq/dof = {chisq27 / dof}")
  print(f"pvalue = {p27}")
  numpy.save(f"{directory}model27_best_fit_params.npy", numpy.array(res27.x))

  # x27 = [1.00933027e-03, -1 + 5.5e-04, 1000, -99.32, 1.07010446e+00, 6.74677964e-01, -7.0056e-06]

  # res28 = least_squares(res_function, x28, args=(cov_inv, model28), method='lm')
  res28 = least_squares(res_function, x28, bounds=bounds, args=(cov_inv, model28), method='dogbox')
  plot_fit(res28, cov_matrix, model28, directory, GL_min, GL_max, ext=10, alpha=res28.x[0], lambduh=res28.x[4], incl_K1=True)
  chisq28 = chisq_calc(res28.x, cov_inv, model28, res_function)
  dof = g_s_cut.shape[0] - len(res28.x)
  p28 = chisq_pvalue(dof, chisq28)
  print(f"chisq = {chisq28}")
  print(f"chisq/dof = {chisq28 / dof}")
  print(f"pvalue = {p28}")
  numpy.save(f"{directory}model28_best_fit_params.npy", numpy.array(res28.x))

  bounds_small = ([alpha_range[0], -50, -50, -50, lambduh_range[0], nu_range[0]],
    [alpha_range[1], 50, 50, 50, lambduh_range[1], nu_range[1]])

  res60 = least_squares(res_function, x61, bounds=bounds_small, args=(cov_inv, model60), method='dogbox')
  plot_fit(res60, cov_matrix, model60, directory, GL_min, GL_max, ext=10, alpha=res60.x[0], lambduh=res60.x[4], incl_K1=True)
  chisq60 = chisq_calc(res60.x, cov_inv, model60, res_function)
  dof = g_s_cut.shape[0] - len(res60.x)
  p60 = chisq_pvalue(dof, chisq60)
  print(f"chisq = {chisq60}")
  print(f"chisq/dof = {chisq60 / dof}")
  print(f"pvalue = {p60}")
  numpy.save(f"{directory}model60_best_fit_params.npy", numpy.array(res60.x))

  res61 = least_squares(res_function, x61, bounds=bounds_small, args=(cov_inv, model61), method='dogbox')
  plot_fit(res61, cov_matrix, model61, directory, GL_min, GL_max, ext=10, alpha=res61.x[0], lambduh=res61.x[4], incl_K1=True)
  chisq61 = chisq_calc(res61.x, cov_inv, model61, res_function)
  dof = g_s_cut.shape[0] - len(res61.x)
  p61 = chisq_pvalue(dof, chisq61)
  print(f"chisq = {chisq61}")
  print(f"chisq/dof = {chisq61 / dof}")
  print(f"pvalue = {p61}")
  numpy.save(f"{directory}model61_best_fit_params.npy", numpy.array(res61.x))


  # Use the asymptotic model to get better fits:
  res27_new = use_asymptotic_model(model27, model60, bounds, bounds_small, x61, cov_inv, res_function)
  plot_fit(res27, cov_matrix, model27, directory, GL_min, GL_max, ext=10, alpha=res27.x[0], lambduh=res27.x[4], incl_K1=True)
  chisq27_new = chisq_calc(res27_new.x, cov_inv, model27, res_function)
  dof = g_s_cut.shape[0] - len(res27_new.x)
  p27_new = chisq_pvalue(dof, chisq27_new)
  print(f"chisq = {chisq27_new}")
  print(f"chisq/dof = {chisq27_new / dof}")
  print(f"pvalue = {p27_new}")

  res28_new = use_asymptotic_model(model28, model61, bounds, bounds_small, x61, cov_inv, res_function)
  plot_fit(res28, cov_matrix, model28, directory, GL_min, GL_max, ext=10, alpha=res28.x[0], lambduh=res28.x[4], incl_K1=True)
  chisq28_new = chisq_calc(res28_new.x, cov_inv, model28, res_function)
  dof = g_s_cut.shape[0] - len(res28_new.x)
  p28_new = chisq_pvalue(dof, chisq28_new)
  print(f"chisq = {chisq28_new}")
  print(f"chisq/dof = {chisq28_new / dof}")
  print(f"pvalue = {p28_new}")



  x41 = [-1.61922402e-02, -1 + 5.157e-04, 100, -11.163, 9.81685228e-01,  6.84870629e-01, 1.805e-04]
# x41 = [-1.61922402e-02, -1 + 5.157e-04, 1000, -111.63, 9.81685228e-01,  6.84870629e-01, -1.805e-05] 