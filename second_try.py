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

# Input data
GL_min = 12.8
GL_max = 76.8
no_samples = 500

# Extract the dependency between the coupling, g, and the 1-loop mass under perterbation
# theory
g_s = [0.1, 0.2, 0.3, 0.5, 0.6]
mPT_1loop_s = [-0.031591, -0.063182, -0.094774, -0.157956, -0.189548]

# Fit a straight line to the mPT_1loop
curve = numpy.polyfit(g_s, mPT_1loop_s, 1)

def mPT_1loop(g):
  return curve[1] + g * curve[0]


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


K1_s = K1(g_s, N_s)
K2_s = K2(L_s, N_s)


# Used to alter the range of data being considered
def cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, K1_s, K2_s, samples, m_s):
  GL_s = g_s * L_s

  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10), 
                                          GL_s <= GL_max * (1 + 10 ** -10))

  return g_s[keep], Bbar_s[keep], N_s[keep], L_s[keep], K1_s[keep], K2_s[keep], samples[keep], m_s[keep]


# Model 1 using g as a regulator
def model1(g_s, L_s, Bbar_s, K1_s, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g_s) + g_s ** 2 * (alpha + (g_s * L_s) ** (-1 / nu) * ((f1 * Bbar_s) / (1 + c * (g_s * L_s) ** -omega) - 1) * f0 - lambduh * K1_s)


# Model 2 using (1 / L) as a regulator
def model2(g_s, L_s, Bbar_s, K2_s, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g_s) + g_s ** 2 * (alpha + (g_s * L_s) ** (-1 / nu) * ((f1 * Bbar_s) / (1 + c * (g_s * L_s) ** -omega) - 1) * f0 - lambduh * K2_s)


# Test out Andreas' best fit parameters
g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, K1_s_cut, K2_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max,
                                                          g_s, Bbar_s, N_s, L_s, K1_s, K2_s, samples, m_s)

# Best fit params
alpha_fit = 0.0018  # Got 0.00183
c_fit = 0.024 # Got -0.0602
f0_fit = -64.3 # Got -64.2
f1_fit = 1.1131 # Got 1.102
lambduh_fit = 1.057 # Got 1.056
nu_fit = 0.677 # Got 0.602
omega_fit = 0.800  # 0.798


# Find the covariance matrix - method 1 - don't enforce independence between ensembles
cov_matrix = numpy.cov(samples_cut)

# Decompose the matrix using Cholskey decomposition
cov_1_2 = numpy.linalg.cholesky(cov_matrix)

# Invert the covarience matrix sqrt
cov_inv = numpy.linalg.inv(cov_1_2)

# In reality the covariance between different ensembles is 0. We can set it as
# such to get a more accurate calculation of the covariance matrix
different_N = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
different_g = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))
different_L = numpy.zeros((samples_cut.shape[0], samples_cut.shape[0]))

for i in range(samples_cut.shape[0]):
  for j in range(samples_cut.shape[0]):
    different_N[i, j] = N_s[i] != N_s[j]
    different_g[i, j] = g_s[i] != g_s[j]
    different_L[i, j] = L_s[i] != L_s[j]

# This is true if two data points come from the same original function
different_ensemble = numpy.logical_or(different_N,
                     numpy.logical_or(different_L,
                                      different_g))

# Set the covariance between different ensembles to 0
cov_matrix2 = numpy.where(different_ensemble, 0, cov_matrix)
cov2_1_2 = numpy.linalg.cholesky(cov_matrix2)
cov2_inv = numpy.linalg.inv(cov2_1_2)


def chisq_calc(x, cov_inv):
  alpha, c, f0, f1, lambduh, nu, omega = x

  # Caculate the residuals between the model and the data
  predictions = model1(g_s_cut, L_s_cut, Bbar_s_cut, K1_s_cut, alpha, c, f0, f1, lambduh, nu, omega)

  residuals = m_s_cut - predictions

  normalized_residuals = numpy.dot(cov_inv, residuals)

  chisq = numpy.sum(normalized_residuals ** 2)

  return chisq


def chisq_pvalue(k, x):
  "k is the rank, x is the chi-sq value"
  return gammaincc(k / 2, x / 2)


res = minimize(chisq_calc, [alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit, omega_fit], method='CG', args=(cov_inv, ))
res2 = minimize(chisq_calc, [alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit, omega_fit], method='CG', args=(cov2_inv, ))

chisq = chisq_calc(res.x, cov_inv)
pvalue = chisq_pvalue(7, chisq)

chisq2 = chisq_calc(res2.x, cov2_inv)
pvalue2 = chisq_pvalue(7, chisq2)


def plot_fit(res):
  alpha_fit2, c_fit2, f0_fit2, f1_fit2, lambduh_fit2, nu_fit2, omega_fit2 = res.x
/
  predictions2 = model1(g_s_cut, L_s_cut, Bbar_s_cut, K1_s_cut,
                       alpha_fit2, c_fit2, f0_fit2, f1_fit2, lambduh_fit2, nu_fit2, omega_fit2)

  plt.errorbar(g_s_cut * L_s_cut, m_s_cut / g_s_cut, yerr=numpy.diag(cov_matrix) ** 0.5 / g_s_cut, ls='', label='data')
  plt.scatter(g_s_cut * L_s_cut, predictions / g_s_cut, color='r', label='prediction')

  plt.xlabel("gL")
  plt.ylabel("value / g")
  plt.legend()
  plt.close()

# Compare to Andreas results
x0 = [alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit, omega_fit]
chisq_andreas = chisq_calc(x0, cov_inv)


# Try using the scipy least-squares method with Nelder-Mead
def res_function(x, cov_inv):
  alpha, c, f0, f1, lambduh, nu, omega = x

  # Caculate the residuals between the model and the data
  predictions = model1(g_s_cut, L_s_cut, Bbar_s_cut, K1_s_cut, alpha, c, f0, f1, lambduh, nu, omega)

  residuals = m_s_cut - predictions

  normalized_residuals = numpy.dot(cov_inv, residuals)

  return normalized_residuals


res3 = least_squares(res_function, x0, args=(cov_inv, ), method='lm')
res4 = least_squares(res_function, x0, args=(cov2_inv, ), method='lm')

chisq3 = chisq_calc(res3.x, cov_inv)
chisq4 = chisq_calc(res4.x, cov2_inv)
