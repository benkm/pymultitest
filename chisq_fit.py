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


def Omega_2(g, L, c, omega):
  return 1 + c * (g * L) ** -omega


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


def model3(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) * (1 + c * numpy.log(g * L))) - 1) * f0 - lambduh * K1(g, N))


def model4(N, g, L, Bbar, alpha, c, f0, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (Bbar * (1 + c * numpy.log(g * L))) * f0 - lambduh * K1(g, N))


# 6 parameter alternative model
def model5(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) * (1 + c * numpy.log(g * L))) - 1) * f0 - lambduh * K2(L, N))


# Model 3 but with the correct Omega expression
def model6(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / (1 + c * numpy.log(g * L))) - 1) * f0 - lambduh * K1(g, N))


# Model 5 but with the correct Omega expression
def model7(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / (1 + c * numpy.log(g * L))) - 1) * f0 - lambduh * K2(L, N))


# Model 1 but with the correct Omega expression
def model8(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K1(g, N))


# Model 2 but with the correct Omega expression
def model10(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K2(L, N))


# Look at c varying between different g
def c(g, c1, c2, c3, c5, c6):
  return  numpy.where(g == 0.1, c1,
            numpy.where(g == 0.2, c2,
              numpy.where(g == 0.3, c3,
                numpy.where(g == 0.5, c5,
                  numpy.where(g == 0.6, c6, numpy.nan)
                )
              )
            )
          )


def model9(N, g, L, Bbar, alpha, c1, c2, c3, c5, c6, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / (1 + c(g, c1, c2, c3, c5, c6) * numpy.log(g * L))) - 1) * f0 - lambduh * K1(g, N))


# A copy of model 8 for the purpose of differentiating model 8 with prior, to model
# 8 without prior.
def model81(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K1(g, N))


def model82(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K1(g, N))


# Like model 81 but with K2
def model101(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K2(L, N))


def model102(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((f1 * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K2(L, N))


# Like model 8 but f1 -> 1 / f1
def model12(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 * Omega_2(g, L, c, omega))) - 1) * f0 - lambduh * K1(g, N))


# Like model 10 but f1 -> 1 / f1
def model13(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 * Omega_2(g, L, c, omega))) - 1) * f0 - lambduh * K2(L, N))


# f1 / (1 + c * (gL)^omega) --> 1 / (f1 + c * (gL)^omega)
def model14(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 + c * (g * L) ** -omega)) - 1) * f0 - lambduh * K1(g, N))


def model15(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 + c * (g * L) ** -omega)) - 1) * f0 - lambduh * K2(L, N))


# f1 / (1 + c * (gL)^omega) --> 1 / (f1 + (gL)^omega / c)
def model16(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 + (g * L) ** -omega / c)) - 1) * f0 - lambduh * K1(g, N))


def model17(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 + (g * L) ** -omega / c)) - 1) * f0 - lambduh * K2(L, N))


# No corrections to scaling - for the purpose of finding the best f1
def model18(N, g, L, Bbar, alpha, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (Bbar * f1 - 1) * f0 - lambduh * K1(g, N))


f1_special = 1.70277772


def model22(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((((f1 + f1_special) * Bbar) / Omega_2(g, L, c, omega)) - 1) * f0 - lambduh * K1(g, N))


def model23(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 + (1 / f1_special) + c * (g * L) ** -omega)) - 1) * f0 - lambduh * K1(g, N))


# Same as model23 except with K2
def model24(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 + (1 / f1_special) + c * (g * L) ** -omega)) - 1) * f0 - lambduh * K2(L, N))


def model25(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 * (1 + c * (g * L) ** -omega))) - 1) * f0 - lambduh * K1(g, N))


def model26(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * ((Bbar / (f1 * (1 + c * (g * L) ** -omega))) - 1) * f0 - lambduh * K2(L, N))


# Motivated by the functional form of the theory
def model27(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K1(g, N))


def model28(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K2(L, N))


def model29(N, g, L, Bbar, alpha, c, f0, f1, lambduh1, lambduh2, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh1 * K1(g, N) - lambduh2 * K2(L, N))


# Model 27 with one parameter removed in each case
def model31(N, g, L, Bbar, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * ((g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K1(g, N))


def model32(N, g, L, Bbar, alpha, f0, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 - 0.0461 * (g * L) ** -omega))) - 0.657) / f1) - lambduh * K1(g, N))


def model33(N, g, L, Bbar, alpha, c, f1, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - 0.657) / f1) - lambduh * K1(g, N))


def model34(N, g, L, Bbar, alpha, c, f0, lambduh, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / -0.0380) - lambduh * K1(g, N))


def model35(N, g, L, Bbar, alpha, c, f0, f1, nu, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - 1 * K1(g, N))


def model36(N, g, L, Bbar, alpha, c, f0, f1, lambduh, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-3 / 2) * (((Bbar / ((1 + c * (g * L) ** -omega))) - f0) / f1) - lambduh * K1(g, N))


def model37(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -0.8))) - f0) / f1) - lambduh * K1(g, N))


# Let's try removing even more parameters!
def model40(N, g, L, Bbar, alpha, c, f1, lambduh, nu):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-1 / nu) * (((Bbar / ((1 + c * (g * L) ** -0.8))) - 0.657) / f1) - lambduh * K1(g, N))


def model41(N, g, L, Bbar, alpha, c, f1, lambduh, omega):
  return mPT_1loop(g, N) + g ** 2 * (alpha + (g * L) ** (-3 / 2) * (((Bbar / ((1 + c * (g * L) ** -omega))) - 0.657) / f1) - lambduh * K1(g, N))


# This is the same as model 8 but with alpha and lambduh set to their mean values
# This is done purely in order to better explore the likelihood landscape of the
# remaining 5 parameters
# def model20(N, g, L, Bbar, alpha, c, f0, f1, lambduh, nu, omega):
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
x3 = [alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit]
x4 = [alpha_fit, c_fit, f0_fit, lambduh_fit, nu_fit]
x8 = [0.00015, -0.7, -10.5, 1.70, 1.08, 0.676, 0.8]
x9 = [0.00015, -0.7, -0.7, -0.7, -0.7, -0.7, -10.5, 1.70, 1.08, 0.676, 0.8]
# x10 = [-0.017, 1.75, -9, 5.25, 1, 0.71, -0.087]  # Outdated as uses omega < 0
x10 = [-1.56964558e-02, -5.08092294e-01, -9.06697176e+00, 9.70731725e-01,
        9.70743246e-01, 7.12407520e-01, 8.12431112e-02]
x18 = [alpha_fit, f0_fit, f1_fit, lambduh_fit, nu_fit]
x22 = [alpha_fit, c_fit, f0_fit, f1_fit - f1_special, lambduh_fit, nu_fit, omega_fit]
x_odd = numpy.array([ 9.84309254e-04,  2.65788448e+07, -1.00850915e+01,  4.82541893e+07,
        1.07051641e+00,  6.84771439e-01, -2.19677215e-02])
x23 = [ 1.39774886e-03, -8.16216752e-02, -1.00925303e+01,  2.05436109e-02,
        1.06381225e+00,  6.84422890e-01,  4.53802770e-01]
x24 = [-0.015, -0.4, -9, 0.35, 0.96, 0.71, 0.1]
x25 = [0.0014, -0.05, -10, 0.6, 1.064, 0.684, 0.453]
x26 = [-0.015, -0.4, -9, 1, 0.96, 0.71, 0.1]
# x27 = [0.0014, -0.05, 0.58, -0.038, 1.064, 0.684, 0.453]
x27 = [0.0014, -0.134, 0.608, -0.06, 1.064, 0.6844, 0.454]
x28 = [-0.016, -0.97, 20, -2.2, 0.98, 0.713, 0.0018]
# x29 = [0.0014, -0.134, 0.608, -0.06, 1.064, 0, 0.6844, 0.454]
x29 = [-0.016, -0.995, 100, -11, 0, 0.981, 0.713, 0.000353]

x31 = [-0.134, 0.608, -0.06, 1.064, 0.6844, 0.454]
x32 = [0.0014, 0.608, -0.06, 1.064, 0.6844, 0.454]
x33 = [0.0014, -0.134, -0.06, 1.064, 0.6844, 0.454]
x34 = [0.0014, -0.134, 0.608, 1.064, 0.6844, 0.454]
x35 = [0.0014, -0.134, 0.608, -0.06, 0.6844, 0.454]
x36 = [0.0014, -0.134, 0.608, -0.06, 1.064, 0.454]
x37 = [0.0014, -0.134, 0.608, -0.06, 1.064, 0.6844]

x40 = [0.0014, -0.134, -0.06, 1.064, 0.6844]
x41 = [0.0014, -0.134, -0.06, 1.064, 0.454]




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


# For the calculation of f1
GL_min2 = 32
g_s_cut2, Bbar_s_cut2, N_s_cut2, L_s_cut2, samples_cut2, m_s_cut2 = cut(GL_min2, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)
cov_matrix2, different_ensemble2 = cov_matrix_calc(samples_cut2, m_s_cut2)
cov2_1_2 = numpy.linalg.cholesky(cov_matrix2)
cov2_inv = numpy.linalg.inv(cov2_1_2)


# Extract the difference between two Bbar values for the same ensemble
g_s_new = []
L_s_new = []
N_s_new = []
Bbar_s_new1 = []
Bbar_s_new2 = []

samples_new = []

m_s_new1 = []
m_s_new2 = []

counter = 0

for i in range(len(g_s_cut)):
  for k in range(i, len(g_s_cut)):
    if i != k and different_ensemble[i, k] == 0:
      # These configs are the same but with different ensembles
      if Bbar_s_cut[i] > Bbar_s_cut[k]:
        g_s_new.append(g_s_cut[i])
        L_s_new.append(L_s_cut[i])
        N_s_new.append(N_s_cut[i])
        Bbar_s_new1.append(Bbar_s_cut[k])
        Bbar_s_new2.append(Bbar_s_cut[i])
        m_s_new2.append(m_s_cut[i])
        m_s_new1.append(m_s_cut[k])


        samples_new.append(samples_cut[i] - samples_cut[k])

      else:
        g_s_new.append(g_s_cut[k])
        L_s_new.append(L_s_cut[k])
        N_s_new.append(N_s_cut[k])
        Bbar_s_new2.append(Bbar_s_cut[k])
        Bbar_s_new1.append(Bbar_s_cut[i])
        m_s_new2.append(m_s_cut[k])
        m_s_new1.append(m_s_cut[i])

        samples_new.append(samples_cut[k] - samples_cut[i])

g_s_new = numpy.array(g_s_new)
L_s_new = numpy.array(L_s_new)
N_s_new = numpy.array(N_s_new)
m_s_new1 = numpy.array(m_s_new1)
m_s_new2 = numpy.array(m_s_new2)


samples_new = numpy.array(samples_new)
m_s_new = m_s_new2 - m_s_new1

# Bbar_s_new should be filled with identical values
assert len(set(Bbar_s_new1)) == 1
assert len(set(Bbar_s_new2)) == 1

diff_Bbar = Bbar_s_new2[0] - Bbar_s_new1[0]

cov_matrix_new = numpy.diag(numpy.diag(numpy.cov(samples_new)))
cov_1_2_new = numpy.linalg.cholesky(cov_matrix_new)
cov_inv_new = numpy.linalg.inv(cov_1_2_new)

x3_diff_Bbar = [c_fit, f0_fit * f1_fit, nu_fit]


# In the following model F takes on the role of f0 * f1
# Looking at the difference between mass values at two BBars
def model3_diff_Bbar(N, g, L, diff_Bbar, c, F, nu):
  return g ** 2 * (g * L) ** (-1 / nu) * F * diff_Bbar * (1 + c * numpy.log(g * L))


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


def plot_fit_diff_Bbar(res, cov_matrix, model_function, ext=1):
  """
    ext : extension factor towards origin - model is plotted to 1 / (GL_max * ext)
  """
  N = 2

  std_diag = numpy.diag(cov_matrix) ** 0.5

  for g in set(g_s_new):
    entries = numpy.argwhere(g_s_new == g)[:, 0]

    # Now sort by L values
    sort = numpy.argsort(L_s_new[entries])

    plt.errorbar(1 / (g * L_s_new[entries][sort]), m_s_new[entries][sort] / g, yerr=std_diag[entries][sort] / g, ls='', label=f'g = {g}', color=colors(g, 0.1, 0.6))
    plt.scatter(1 / (g * L_s_new[entries][sort]), m_s_new[entries][sort] / g, facecolors='none', edgecolors=colors(g, 0.1, 0.6))

    L_range = numpy.linspace(GL_min / g, GL_max * ext / g, 1000)

    predictions = model_function(N, g, L_range, diff_Bbar, *res.x)

    # Plot a smooth line for the model
    plt.plot(1 / (g * L_range), predictions / g, color=colors(g, 0.1, 0.6))

  plt.xlabel("1 / gL")
  plt.ylabel("value / g")
  # plt.legend()
  plt.savefig(f"graphs/model_fit{model_function.__name__}_{today.year}_{today.month}_{today.day}.png")
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

  bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
    [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])

  # x_dict = {}
  # x_dict[model1], x_dict[model2] = x0, x0
  # x_dict[model1_small], x_dict[model2_small] = x1, x1

  # Plot model 1
  # res = least_squares(res_function, x_dict[model1], args=(cov_inv, model1), method='lm')
  # plot_fit(res, cov_matrix, model1, ext=10, alpha=res.x[0], lambduh=res.x[4], incl_K1=True)
  # chisq1 = chisq_calc(res.x, cov_inv, model1)
  # p1 = chisq_pvalue(g_s_cut.shape[0] - len(res.x), chisq1)

  # Plot model 3
  # res3 = least_squares(res_function, x3, args=(cov_inv, model3), method='lm')
  # plot_fit(res3, cov_matrix, model3, directory, GL_min, GL_max, ext=10, alpha=res3.x[0], lambduh=res3.x[4], incl_K1=True)
  # chisq3 = chisq_calc(res3.x, cov_inv, model3)
  # dof = g_s_cut.shape[0] - len(res3.x)
  # p3 = chisq_pvalue(dof, chisq3)
  # print(f"chisq = {chisq3}")
  # print(f"chisq/dof = {chisq3 / dof}")
  # print(f"pvalue = {p3}")
  # numpy.save(f"{directory}model3_best_fit_params.npy", numpy.array(res3.x))

  # Plot model 5
  # res5 = least_squares(res_function, x3, args=(cov_inv, model5), method='lm')
  # plot_fit(res5, cov_matrix, model5, ext=10, alpha=res5.x[0], lambduh=res5.x[4], incl_K1=True)
  # chisq5 = chisq_calc(res5.x, cov_inv, model5)
  # p5 = chisq_pvalue(g_s_cut.shape[0] - len(res5.x), chisq5)

  # Plot model 6
  # res6 = least_squares(res_function, x3, args=(cov_inv, model6), method='lm')
  # plot_fit(res6, cov_matrix, model6, directory, GL_min, GL_max, ext=10, alpha=res6.x[0], lambduh=res6.x[4], incl_K1=True)
  # chisq6 = chisq_calc(res6.x, cov_inv, model6)
  # dof = g_s_cut.shape[0] - len(res6.x)
  # p6 = chisq_pvalue(dof, chisq6)
  # print(f"chisq = {chisq6}")
  # print(f"chisq/dof = {chisq6 / dof}")
  # print(f"pvalue = {p6}")
  # numpy.save(f"{directory}model6_best_fit_params.npy", numpy.array(res6.x))

  # Plot model 7
  # res7 = least_squares(res_function, x3, args=(cov_inv, model7), method='lm')
  # plot_fit(res7, cov_matrix, model7, directory, GL_min, GL_max, ext=10, alpha=res7.x[0], lambduh=res7.x[4], incl_K1=True)
  # chisq7 = chisq_calc(res7.x, cov_inv, model7)
  # dof = g_s_cut.shape[0] - len(res7.x)
  # p7 = chisq_pvalue(dof, chisq7)
  # print(f"chisq = {chisq7}")
  # print(f"chisq/dof = {chisq7 / dof}")
  # print(f"pvalue = {p7}")
  # numpy.save(f"{directory}model7_best_fit_params.npy", numpy.array(res7.x))


  # # Plot model 8
  # bounds = ([-numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, -numpy.inf, 0],
  #           [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf])
  # res8 = least_squares(res_function, x8, bounds=bounds, args=(cov_inv, model8), method='dogbox')
  # plot_fit(res8, cov_matrix, model8, directory, GL_min, GL_max, ext=10, alpha=res8.x[0], lambduh=res8.x[4], incl_K1=True)
  # chisq8 = chisq_calc(res8.x, cov_inv, model8)
  # dof = g_s_cut.shape[0] - len(res8.x)
  # p8 = chisq_pvalue(dof, chisq8)
  # print(f"chisq = {chisq8}")
  # print(f"chisq/dof = {chisq8 / dof}")
  # print(f"pvalue = {p8}")
  # numpy.save(f"{directory}model8_best_fit_params.npy", numpy.array(res8.x))
  # pdb.set_trace()

  # res10 = least_squares(res_function, x10, bounds=bounds, args=(cov_inv, model10), method='dogbox')
  # plot_fit(res10, cov_matrix, model10, directory, GL_min, GL_max, ext=10, alpha=res10.x[0], lambduh=res10.x[4], incl_K1=True)
  # chisq10 = chisq_calc(res10.x, cov_inv, model10)
  # dof = g_s_cut.shape[0] - len(res10.x)
  # p10 = chisq_pvalue(dof, chisq10)
  # print(f"chisq = {chisq10}")
  # print(f"chisq/dof = {chisq10 / dof}")
  # print(f"pvalue = {p10}")
  # numpy.save(f"{directory}model10_best_fit_params.npy", numpy.array(res10.x))
  # pdb.set_trace()


  # x22 = res8.x - numpy.array([0, 0, 0, f1_special, 0, 0, 0])
  # res22 = least_squares(res_function, x22, args=(cov_inv, model22), method='dogbox')
  # plot_fit(res22, cov_matrix, model22, directory, GL_min, GL_max, ext=10, alpha=res22.x[0], lambduh=res22.x[4], incl_K1=True)
  # chisq22 = chisq_calc(res22.x, cov_inv, model22)
  # dof = g_s_cut.shape[0] - len(res22.x)
  # p22 = chisq_pvalue(dof, chisq22)
  # print(f"chisq = {chisq22}")
  # print(f"chisq/dof = {chisq22 / dof}")
  # print(f"pvalue = {p22}")
  # numpy.save(f"{directory}model22_best_fit_params.npy", numpy.array(res22.x))

  # pdb.set_trace()

  # x23 = res8.x
  # f1 = x23[3]
  # c = x23[1]
  # x23[3] = 1 / f1 - 1 / f1_special
  # x23[1] = c / f1
  # pdb.set_trace()
  # res23 = least_squares(res_function, x23, bounds=bounds, args=(cov_inv, model23), method='dogbox')
  # plot_fit(res23, cov_matrix, model23, directory, GL_min, GL_max, ext=10, alpha=res23.x[0], lambduh=res23.x[4], incl_K1=True)
  # chisq23 = chisq_calc(res23.x, cov_inv, model23)
  # dof = g_s_cut.shape[0] - len(res23.x)
  # p23 = chisq_pvalue(dof, chisq23)
  # print(f"chisq = {chisq23}")
  # print(f"chisq/dof = {chisq23 / dof}")
  # print(f"pvalue = {p23}")
  # numpy.save(f"{directory}model23_best_fit_params.npy", numpy.array(res23.x))


  # x24 = res10.x
  # f1 = x24[3]
  # c = x24[1]
  # x24[3] = 1 / f1 - 1 / f1_special
  # x24[1] = c / f1

  # res24 = least_squares(res_function, x24, bounds=bounds, args=(cov_inv, model24), method='dogbox')
  # plot_fit(res24, cov_matrix, model24, directory, GL_min, GL_max, ext=10, alpha=res24.x[0], lambduh=res24.x[4], incl_K1=True)
  # chisq24 = chisq_calc(res24.x, cov_inv, model24)
  # dof = g_s_cut.shape[0] - len(res24.x)
  # p24 = chisq_pvalue(dof, chisq24)
  # print(f"chisq = {chisq24}")
  # print(f"chisq/dof = {chisq24 / dof}")
  # print(f"pvalue = {p24}")
  # numpy.save(f"{directory}model24_best_fit_params.npy", numpy.array(res24.x))
  # pdb.set_trace()

  # res25 = least_squares(res_function, x25, bounds=bounds, args=(cov_inv, model25), method='dogbox')
  # plot_fit(res25, cov_matrix, model25, directory, GL_min, GL_max, ext=10, alpha=res25.x[0], lambduh=res25.x[4], incl_K1=True)
  # chisq25 = chisq_calc(res25.x, cov_inv, model25)
  # dof = g_s_cut.shape[0] - len(res25.x)
  # p25 = chisq_pvalue(dof, chisq25)
  # print(f"chisq = {chisq25}")
  # print(f"chisq/dof = {chisq25 / dof}")
  # print(f"pvalue = {p25}")
  # numpy.save(f"{directory}model25_best_fit_params.npy", numpy.array(res25.x))


  # res26 = least_squares(res_function, x26, bounds=bounds, args=(cov_inv, model26), method='dogbox')
  # plot_fit(res26, cov_matrix, model26, directory, GL_min, GL_max, ext=10, alpha=res26.x[0], lambduh=res26.x[4], incl_K1=True)
  # chisq26 = chisq_calc(res26.x, cov_inv, model26)
  # dof = g_s_cut.shape[0] - len(res26.x)
  # p26 = chisq_pvalue(dof, chisq26)
  # print(f"chisq = {chisq26}")
  # print(f"chisq/dof = {chisq26 / dof}")
  # print(f"pvalue = {p26}")
  # numpy.save(f"{directory}model26_best_fit_params.npy", numpy.array(res26.x))

  res27 = least_squares(res_function, x27, bounds=bounds, args=(cov_inv, model27), method='dogbox')
  plot_fit(res27, cov_matrix, model27, directory, GL_min, GL_max, ext=10, alpha=res27.x[0], lambduh=res27.x[4], incl_K1=True)
  chisq27 = chisq_calc(res27.x, cov_inv, model27)
  dof = g_s_cut.shape[0] - len(res27.x)
  p27 = chisq_pvalue(dof, chisq27)
  print(f"chisq = {chisq27}")
  print(f"chisq/dof = {chisq27 / dof}")
  print(f"pvalue = {p27}")
  numpy.save(f"{directory}model27_best_fit_params.npy", numpy.array(res27.x))


  # res28 = least_squares(res_function, x28, bounds=bounds, args=(cov_inv, model28), method='dogbox')
  # plot_fit(res28, cov_matrix, model28, directory, GL_min, GL_max, ext=10, alpha=res28.x[0], lambduh=res28.x[4], incl_K1=True)
  # chisq28 = chisq_calc(res28.x, cov_inv, model28)
  # dof = g_s_cut.shape[0] - len(res28.x)
  # p28 = chisq_pvalue(dof, chisq28)
  # print(f"chisq = {chisq28}")
  # print(f"chisq/dof = {chisq28 / dof}")
  # print(f"pvalue = {p28}")
  # numpy.save(f"{directory}model28_best_fit_params.npy", numpy.array(res28.x))

  # print("model31")
  # res31 = least_squares(res_function, x31, args=(cov_inv, model31))
  # plot_fit(res31, cov_matrix, model31, directory, GL_min, GL_max, ext=10, alpha=res31.x[0], lambduh=res31.x[4], incl_K1=True)
  # chisq31 = chisq_calc(res31.x, cov_inv, model31)
  # dof = g_s_cut.shape[0] - len(res31.x)
  # p31 = chisq_pvalue(dof, chisq31)
  # print(f"chisq = {chisq31}")
  # print(f"chisq/dof = {chisq31 / dof}")
  # print(f"pvalue = {p31}")
  # print("\n")
  # numpy.save(f"{directory}model31_best_fit_params.npy", numpy.array(res31.x))

  # print("model32")
  # res32 = least_squares(res_function, x32, args=(cov_inv, model32))
  # plot_fit(res32, cov_matrix, model32, directory, GL_min, GL_max, ext=10, alpha=res32.x[0], lambduh=res32.x[4], incl_K1=True)
  # chisq32 = chisq_calc(res32.x, cov_inv, model32)
  # dof = g_s_cut.shape[0] - len(res32.x)
  # p32 = chisq_pvalue(dof, chisq32)
  # print(f"chisq = {chisq32}")
  # print(f"chisq/dof = {chisq32 / dof}")
  # print(f"pvalue = {p32}")
  # print("\n")
  # numpy.save(f"{directory}model32_best_fit_params.npy", numpy.array(res32.x))

  # print("model33")
  # res33 = least_squares(res_function, x33, args=(cov_inv, model33))
  # plot_fit(res33, cov_matrix, model33, directory, GL_min, GL_max, ext=10, alpha=res33.x[0], lambduh=res33.x[4], incl_K1=True)
  # chisq33 = chisq_calc(res33.x, cov_inv, model33)
  # dof = g_s_cut.shape[0] - len(res33.x)
  # p33 = chisq_pvalue(dof, chisq33)
  # print(f"chisq = {chisq33}")
  # print(f"chisq/dof = {chisq33 / dof}")
  # print(f"pvalue = {p33}")
  # print("\n")
  # numpy.save(f"{directory}model33_best_fit_params.npy", numpy.array(res33.x)) 

  # print("model34")
  # res34 = least_squares(res_function, x34, args=(cov_inv, model34))
  # plot_fit(res34, cov_matrix, model34, directory, GL_min, GL_max, ext=10, alpha=res34.x[0], lambduh=res34.x[4], incl_K1=True)
  # chisq34 = chisq_calc(res34.x, cov_inv, model34)
  # dof = g_s_cut.shape[0] - len(res34.x)
  # p34 = chisq_pvalue(dof, chisq34)
  # print(f"chisq = {chisq34}")
  # print(f"chisq/dof = {chisq34 / dof}")
  # print(f"pvalue = {p34}")
  # print("\n")
  # numpy.save(f"{directory}model34_best_fit_params.npy", numpy.array(res34.x)) 


  # print("model35")
  # res35 = least_squares(res_function, x35, args=(cov_inv, model35))
  # plot_fit(res35, cov_matrix, model35, directory, GL_min, GL_max, ext=10, alpha=res35.x[0], lambduh=res35.x[4], incl_K1=True)
  # chisq35 = chisq_calc(res35.x, cov_inv, model35)
  # dof = g_s_cut.shape[0] - len(res35.x)
  # p35 = chisq_pvalue(dof, chisq35)
  # print(f"chisq = {chisq35}")
  # print(f"chisq/dof = {chisq35 / dof}")
  # print(f"pvalue = {p35}")
  # print("\n")
  # numpy.save(f"{directory}model35_best_fit_params.npy", numpy.array(res35.x)) 


  # print("model36")
  # res36 = least_squares(res_function, x36, args=(cov_inv, model36))
  # plot_fit(res36, cov_matrix, model36, directory, GL_min, GL_max, ext=10, alpha=res36.x[0], lambduh=res36.x[4], incl_K1=True)
  # chisq36 = chisq_calc(res36.x, cov_inv, model36)
  # dof = g_s_cut.shape[0] - len(res36.x)
  # p36 = chisq_pvalue(dof, chisq36)
  # print(f"chisq = {chisq36}")
  # print(f"chisq/dof = {chisq36 / dof}")
  # print(f"pvalue = {p36}")
  # print("\n")
  # numpy.save(f"{directory}model36_best_fit_params.npy", numpy.array(res36.x)) 


  # print("model37")
  # res37 = least_squares(res_function, x37, args=(cov_inv, model37))
  # plot_fit(res37, cov_matrix, model37, directory, GL_min, GL_max, ext=10, alpha=res37.x[0], lambduh=res37.x[4], incl_K1=True)
  # chisq37 = chisq_calc(res37.x, cov_inv, model37)
  # dof = g_s_cut.shape[0] - len(res37.x)
  # p37 = chisq_pvalue(dof, chisq37)
  # print(f"chisq = {chisq37}")
  # print(f"chisq/dof = {chisq37 / dof}")
  # print(f"pvalue = {p37}")
  # print("\n")
  # numpy.save(f"{directory}model37_best_fit_params.npy", numpy.array(res37.x))


  # res40 = least_squares(res_function, x40, args=(cov_inv, model40))
  # plot_fit(res40, cov_matrix, model40, directory, GL_min, GL_max, ext=10, alpha=res40.x[0], lambduh=res40.x[4], incl_K1=True)
  # chisq40 = chisq_calc(res40.x, cov_inv, model40)
  # dof = g_s_cut.shape[0] - len(res40.x)
  # p40 = chisq_pvalue(dof, chisq40)
  # print(f"chisq = {chisq40}")
  # print(f"chisq/dof = {chisq40 / dof}")
  # print(f"pvalue = {p40}")
  # print("\n")
  # numpy.save(f"{directory}model40_best_fit_params.npy", numpy.array(res40.x))

  # res41 = least_squares(res_function, x41, args=(cov_inv, model41))
  # plot_fit(res41, cov_matrix, model41, directory, GL_min, GL_max, ext=10, alpha=res41.x[0], lambduh=res41.x[4], incl_K1=True)
  # chisq41 = chisq_calc(res41.x, cov_inv, model41)
  # dof = g_s_cut.shape[0] - len(res41.x)
  # p41 = chisq_pvalue(dof, chisq41)
  # print(f"chisq = {chisq41}")
  # print(f"chisq/dof = {chisq41 / dof}")
  # print(f"pvalue = {p41}")
  # print("\n")
  # numpy.save(f"{directory}model41_best_fit_params.npy", numpy.array(res41.x))

  # To investigate the model with priors feed these into the res_function
  # call this model model 81

  # prior_values = {"omega": 0.782}
  # prior_sigmas = {"omega": 0.0013}

  # kwargs = {"prior": True, "prior_values": prior_values, "prior_sigmas": prior_sigmas}
  # res81 = least_squares(res_function, x8, args=(cov_inv, model81), method='lm', kwargs=kwargs)
  # plot_fit(res81, cov_matrix, model81, directory, GL_min, GL_max, ext=10, alpha=res81.x[0], lambduh=res81.x[4], incl_K1=True)
  # chisq81 = chisq_calc(res81.x, cov_inv, model81, **kwargs)
  # dof = g_s_cut.shape[0] - len(res81.x)
  # p81 = chisq_pvalue(dof, chisq81)
  # print(f"chisq = {chisq81}")
  # print(f"chisq/dof = {chisq81 / dof}")
  # print(f"pvalue = {p81}")
  # numpy.save(f"{directory}model81_best_fit_params.npy", numpy.array(res81.x))

  # prior_values = {"omega": 0.782, "nu": 0.7073}
  # prior_sigmas = {"omega": 0.0013, "nu": 0.0035}

  # kwargs = {"prior": True, "prior_values": prior_values, "prior_sigmas": prior_sigmas}
  # res82 = least_squares(res_function, x8, args=(cov_inv, model82), method='lm', kwargs=kwargs)
  # plot_fit(res82, cov_matrix, model82, directory, GL_min, GL_max, ext=10, alpha=res82.x[0], lambduh=res82.x[4], incl_K1=True)
  # chisq82 = chisq_calc(res82.x, cov_inv, model82, **kwargs)
  # dof = g_s_cut.shape[0] - len(res82.x)
  # p82 = chisq_pvalue(dof, chisq82)
  # print(f"chisq = {chisq82}")
  # print(f"chisq/dof = {chisq82 / dof}")
  # print(f"pvalue = {p82}")
  # numpy.save(f"{directory}model82_best_fit_params.npy", numpy.array(res82.x))


  # res102 = least_squares(res_function, x8, args=(cov_inv, model102), method='lm', kwargs=kwargs)
  # plot_fit(res102, cov_matrix, model102, directory, GL_min, GL_max, ext=10, alpha=res102.x[0], lambduh=res102.x[4], incl_K1=True)
  # chisq102 = chisq_calc(res102.x, cov_inv, model102, **kwargs)
  # dof = g_s_cut.shape[0] - len(res102.x)
  # p102 = chisq_pvalue(dof, chisq102)
  # print(f"chisq = {chisq102}")
  # print(f"chisq/dof = {chisq102 / dof}")
  # print(f"pvalue = {p102}")
  # numpy.save(f"{directory}model102_best_fit_params.npy", numpy.array(res102.x))


  #  Take a look at the 5 parameter no correction to scaling model
  # res18 = least_squares(res_function, x18, args=(cov2_inv, model18), method='lm', 
  #             kwargs={"m_s": m_s_cut2, "N_s": N_s_cut2, "g_s": g_s_cut2, "L_s": L_s_cut2, "Bbar_s": Bbar_s_cut2})
  # plot_fit(res18, cov_matrix2, model18, directory, GL_min2, GL_max, ext=10, alpha=res18.x[0], lambduh=res18.x[3], incl_K1=True, 
  #         m_s_cut=m_s_cut2, N_s_cut=N_s_cut2, g_s_cut=g_s_cut2, L_s_cut=L_s_cut2, Bbar_s_cut=Bbar_s_cut2)
  # pdb.set_trace()
  # # Gives 1.70277772 for f1 :)
  # chisq18 = chisq_calc(res18.x, cov2_inv, model18)
  # dof = g_s_cut.shape[0] - len(res18.x)
  # p18 = chisq_pvalue(dof, chisq18)
  # print(f"chisq = {chisq18}")
  # print(f"chisq/dof = {chisq18 / dof}")
  # print(f"pvalue = {p18}")
  # numpy.save(f"{directory}model18_best_fit_params.npy", numpy.array(res18.x))

  # kwargs = {"prior": True, "prior_values": prior_values, "prior_sigmas": prior_sigmas}
  # res84 = least_squares(res_function, x8, args=(cov_inv, model84), method='lm', kwargs=kwargs)
  # plot_fit(res84, cov_matrix, model84, directory, GL_min, GL_max, ext=10, alpha=res84.x[0], lambduh=res84.x[4], incl_K1=True)
  # chisq84 = chisq_calc(res84.x, cov_inv, model84, **kwargs)
  # dof = g_s_cut.shape[0] - len(res84.x)
  # p84 = chisq_pvalue(dof, chisq84)
  # print(f"chisq = {chisq84}")
  # print(f"chisq/dof = {chisq84 / dof}")
  # print(f"pvalue = {p84}")
  # numpy.save(f"{directory}model84_best_fit_params.npy", numpy.array(res84.x))
  # Investigate model 9
  # res9 = least_squares(res_function, x9, args=(cov_inv, model9), method='lm')
  # plot_fit(res9, cov_matrix, model9, directory, GL_min, GL_max, ext=10, alpha=res9.x[0], lambduh=res9.x[4], incl_K1=True)
  # chisq9 = chisq_calc(res9.x, cov_inv, model9)
  # dof = g_s_cut.shape[0] - len(res9.x)
  # p9 = chisq_pvalue(dof, chisq9)
  # print(f"chisq = {chisq9}")
  # print(f"chisq/dof = {chisq9 / dof}")
  # print(f"pvalue = {p9}")
  # numpy.save(f"{directory}model9_best_fit_params.npy", numpy.array(res9.x))
  # pdb.set_trace()

  # res3 = least_squares(res_function, x3, args=(cov_inv, model3), method='lm')
  # plot_fit(res3, cov_matrix, model3, ext=10, alpha=res3.x[0], lambduh=res3.x[4], incl_K1=True)
  # chisq3 = chisq_calc(res3.x, cov_inv, model3)
  # p3 = chisq_pvalue(g_s_cut.shape[0] - len(res3.x), chisq3)

  # # res_new = [res3.x[1], res3.x[2] * res3.x[3], res3.x[5]]
  # # res = least_squares(res_function, res_new, args=(cov_inv_new, model3_diff_Bbar),
  # #                     kwargs={"m_s": m_s_new, "g_s": g_s_new, "L_s": L_s_new, "N_s": N_s_new, "Bbar_s": diff_Bbar})
  # # chisq1 = chisq_calc(res.x, cov_inv_new, model3_diff_Bbar, m_s=m_s_new, g_s=g_s_new, L_s=L_s_new, N_s=N_s_new, Bbar_s=diff_Bbar)
  # # plot_fit_diff_Bbar(res, cov_matrix_new, model3_diff_Bbar)
  
  # pdb.set_trace()

  # res4 = least_squares(res_function, x4, args=(cov_inv, model4), method='lm')
  # plot_fit(res4, cov_matrix, model4, ext=10, alpha=res4.x[0], lambduh=res4.x[4], incl_K1=True)
  # chisq4 = chisq_calc(res4.x, cov_inv, model4)
  # p4 = chisq_pvalue(g_s_cut.shape[0] - len(res4.x), chisq4)

  # for model in [model2, model1_small, model2_small]:
  #   res = least_squares(res_function, x_dict[model], args=(cov_inv, model), method='lm')
  #   plot_fit(res, cov_matrix, model, ext=10, alpha=res.x[0])
