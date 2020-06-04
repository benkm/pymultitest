import json
import sys
import numpy
from numpy import log, exp, pi
import scipy.stats, scipy
# import pymultinest
import matplotlib.pyplot as plt
import pdb
import datetime
import time
from scipy.special import gammaincc
from scipy.optimize import minimize

today = datetime.date.fromtimestamp(time.time())

# Where the results are saved
datafile = f'./model_output_{today.year}_{today.month}_{today.day}/'

x = numpy.linspace(0, 1, 400)
ydata = None # loaded below

# Input data
GL_min = 12.8
GL_max = 76.8

g_s = [0.1] * 9 + [0.2] * 12 + [0.3] * 14 + [0.5] * 12 + [0.6] * 10

Bbar_s = [0.888] * 5 + [0.900] * 4 + [0.888] * 7 + [0.900] * 5 + [0.888] * 7 + [0.900] * 7 \
       + [0.888] * 6 + [0.900] * 6 + [0.888] * 5 + [0.900] * 5

L_s = [8, 16, 48, 64, 128, 16, 48, 64, 128] + [8, 16, 32, 48, 64, 96, 128, 32, 48, 64, 96, 128] \
    + [8, 16, 32, 48, 64, 96, 128] * 2 + [8, 16, 32, 48, 64, 128] * 2 + [32, 48, 64, 96, 128] * 2

# plt.plot(g_s)
# plt.plot(Bbar_s)
# plt.plot(numpy.array(L_s) / 100)
# plt.show()

mdata = [-0.03036, -0.030562, -0.031017, -0.0310944, -0.0312200, -0.033111, -0.031499, -0.0314031, -0.0313369] \
      + [-0.05923, -0.060453, -0.061218, -0.061678, -0.061815, -0.062090, -0.062105, -0.062518, -0.062395, -0.062262, -0.062299, -0.062258] \
      + [-0.08776, -0.089819, -0.091466, -0.091932, -0.092273, -0.092527, -0.092694, -0.100209, -0.094301, -0.093032, -0.092749, -0.092812, -0.092812, -0.092917] \
      + [-0.14386, -0.148445, -0.150971, -0.152069, -0.152384, -0.153055, -0.15999, -0.154149, -0.153083, -0.153155, -0.153153, -0.153310] \
      + [-0.180687, -0.181470, -0.182244, -0.182704, -0.182882, -0.182835, -0.18275, -0.183016, -0.183175, -0.183220]

N_s = [2] * (9 + 12 + 14 + 12 + 10)

mPT_1loop_s = [-0.031591] * 9 + [-0.063182] * 12 + [-0.094774] * 14 + [-0.157956] * 12 + [-0.189548] * 10

# Fit a straight line to the mPT_1loop
curve = numpy.polyfit(g_s, mPT_1loop_s, 1)

def mPT_1loop(g):
  return curve[1] + g * curve[0]

sigma_s = [0.00010, 0.000033, 0.000018, 0.0000099, 0.0000077, 0.000029, 0.000014, 0.0000087, 0.0000053, 0.00013, 0.000056, 0.000041, 0.000043, 0.000041, 0.000033, 0.000038, 0.000033, 0.000032, 0.000028, 0.000021, 0.000015, 0.00010, 0.000090, 0.000044, 0.000041, 0.000032, 0.000021, 0.000033, 0.000088, 0.000058, 0.000030, 0.000035, 0.000020, 0.000023, 0.000024, 0.00014, 0.000088, 0.000063, 0.000063, 0.000063, 0.000026, 0.00011, 0.000060, 0.000046, 0.000054, 0.000059, 0.000026, 0.000092, 0.000096, 0.000063, 0.000059, 0.000049, 0.000065, 0.00013, 0.000046, 0.000050, 0.000034]

g_s = numpy.array(g_s)
N_s = numpy.array(N_s)
L_s = numpy.array(L_s)
Bbar_s = numpy.array(Bbar_s)
mdata = numpy.array(mdata)
mPT_1loop_s = numpy.array(mPT_1loop_s)
sigma_s = numpy.array(sigma_s)


def K1(g, N):
  return numpy.log((g / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


def K2(L, N):
  return numpy.log(((1 / L) / (4 * numpy.pi * N))) * ((1 - (6 / N ** 2) + (18 / N ** 4)) / (4 * numpy.pi) ** 2)


K1_s = K1(g_s, N_s)
K2_s = K2(L_s, N_s)

size = len(g_s)

assert len(Bbar_s) == size
assert len(N_s) == size
assert len(L_s) == size
assert len(K1_s) == size
assert len(K2_s) == size
assert len(mPT_1loop_s) == size
assert len(mdata) == size
assert len(sigma_s) == size

data = g_s, Bbar_s, N_s, L_s, K1_s, K2_s, mPT_1loop_s, mdata, sigma_s


# Used to alter the range of data being considered
def cut(GL_min, GL_max, data):
  g_s, Bbar_s, N_s, L_s, K1_s, K2_s, mPT_1loop_s, mdata, sigma_s = data

  GL_s = g_s * L_s

  keep = numpy.logical_and(GL_s >= GL_min * (1 - 10 ** -10), 
                                          GL_s <= GL_max * (1 + 10 ** -10))

  g_s_cut = g_s[keep]
  Bbar_s_cut = Bbar_s[keep]
  N_s_cut = N_s[keep]
  L_s_cut = L_s[keep]
  K1_s_cut = K1_s[keep]
  K2_s_cut = K2_s[keep]
  mPT_1loop_s_cut = mPT_1loop_s[keep]
  mdata_cut = mdata[keep]
  sigma_s_cut = sigma_s[keep]

  data_cut = g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, K1_s_cut, K2_s_cut, mPT_1loop_s_cut, mdata_cut, sigma_s_cut

  return data_cut


# Model 1 using g as a regulator
def model1(mPT_1loop_s, g_s, L_s, Bbar_s, K1_s, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop_s + g_s ** 2 * (alpha + (g_s * L_s) ** (-1 / nu) * ((f1 * Bbar_s) / (1 + c * (g_s * L_s) ** -omega) - 1) * f0 - lambduh * K1_s)


# Model 2 using (1 / L) as a regulator
def model2(mPT_1loop_s, g_s, L_s, Bbar_s, K2_s, alpha, c, f0, f1, lambduh, nu, omega):
  return mPT_1loop_s + g_s ** 2 * (alpha + (g_s * L_s) ** (-1 / nu) * ((f1 * Bbar_s) / (1 + c * (g_s * L_s) ** -omega) - 1) * f0 - lambduh * K2_s)


# Test out Andreas' best fit parameters
g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, K1_s_cut, K2_s_cut, mPT_1loop_s_cut, mdata_cut, sigma_s_cut = cut(GL_min, GL_max, data)

# Best fit params
alpha_fit = 0.0018  # Got 0.00183
c_fit = 0.024 # Got -0.0602
f0_fit = -64.3 # Got -64.2
f1_fit = 1.1131 # Got 1.102
lambduh_fit = 1.057 # Got 1.056
nu_fit = 0.677 # Got 0.602
omega_fit = 0.800  # 0.798

# Caculate the residuals between the model and the data
predictions = model1(mPT_1loop_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, K1_s_cut,
                     alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit, omega_fit)

residuals = mdata_cut - predictions

chisq = numpy.sum((residuals / sigma_s_cut) ** 2)

plt.plot((residuals / sigma_s_cut) ** 2)
plt.show()

def chisq_pvalue(k, x):
  "k is the rank, x is the chi-sq value"
  return gammaincc(k / 2, x / 2)


rank = 7  # 7 free parameters fitted
pvalue = chisq_pvalue(rank, chisq)

# Plot data for diagnosis
plt.errorbar(g_s_cut * L_s_cut, mdata_cut / g_s_cut, yerr=sigma_s_cut / g_s_cut, ls='', label='data')
plt.scatter(g_s_cut * L_s_cut, predictions / g_s_cut, color='r', label='prediction')
# for i in range(len(g_s_cut)):
#   plt.arrow(g_s_cut[i] * L_s_cut[i], mdata_cut[i] / g_s_cut[i], 0, (predictions[i] - mdata_cut[i]) / g_s_cut[i])

plt.xlabel("gL")
plt.ylabel("value / g")
plt.legend()
plt.close()

# Do my own chisq minimization
def chisq_calc(x):
  alpha, c, f0, f1, lambduh, nu, omega = x
  predictions = model1(mPT_1loop_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, K1_s_cut,
                     alpha, c, f0, f1, lambduh, nu, omega)

  residuals = mdata_cut - predictions

  chisq = numpy.sum((residuals / sigma_s_cut) ** 2)

  return chisq


res = minimize(chisq_calc, [alpha_fit, c_fit, f0_fit, f1_fit, lambduh_fit, nu_fit, omega_fit], method='CG')

chisq = chisq_calc(res.x)
pvalue = chisq_pvalue(7, chisq_calc(res.x))

alpha_fit2, c_fit2, f0_fit2, f1_fit2, lambduh_fit2, nu_fit2, omega_fit2 = res.x

predictions = model1(mPT_1loop_s_cut, g_s_cut, L_s_cut, Bbar_s_cut, K1_s_cut,
                     alpha_fit2, c_fit2, f0_fit2, f1_fit2, lambduh_fit2, nu_fit2, omega_fit2)

plt.errorbar(g_s_cut * L_s_cut, mdata_cut / g_s_cut, yerr=sigma_s_cut / g_s_cut, ls='', label='data')
plt.scatter(g_s_cut * L_s_cut, predictions / g_s_cut, color='r', label='prediction')
# for i in range(len(g_s_cut)):
#   plt.arrow(g_s_cut[i] * L_s_cut[i], mdata_cut[i] / g_s_cut[i], 0, (predictions[i] - mdata_cut[i]) / g_s_cut[i])

plt.xlabel("gL")
plt.ylabel("value / g")
plt.legend()
pdb.set_trace()
plt.close()


residuals = mdata_cut - predictions


# Make a continuous plot of the model predictions
def predictions_smooth(g, L, Bbar, N=2):
  return model1(mPT_1loop(g), g, L, Bbar, K1(g, N),
                alpha_fit2, c_fit2, f0_fit2, f1_fit2, lambduh_fit2, nu_fit2, omega_fit2)


for Bbar in [0.888, 0.9]:
  for L in [32, 48, 64, 96, 128]:
    g_range = numpy.linspace(GL_min / L, GL_max / L, 100)
    plt.plot(g_range * L, predictions_smooth(g_range, L, Bbar) / g_range, label=f'L={L}, Bbar={Bbar}')

plt.errorbar(g_s_cut * L_s_cut, mdata_cut / g_s_cut, yerr=sigma_s_cut / g_s_cut, ls='', label='data')

plt.legend()

pdb.set_trace()


# Use uniform priors
def prior(cube, ndim, nparams):
  cube[0] = cube[0] * 20 - 10 # Prior on alpha
  cube[1] = cube[1] * 20 - 10 # Prior on c
  cube[2] = cube[2] * 20 - 10 # Prior on f0
  cube[3] = cube[3] * 20 - 10 # Prior on f1
  cube[4] = cube[4] * 20 - 10 # Prior on Lambda
  cube[5] = cube[5] * 20  # Prior on nu (must be bigger than 0)
  cube[6] = cube[6] * 20  # Prior on omega (must be bigger than 0)


def loglike1(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu, omega = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6]
  ymodel = model1(mPT_1loop_s, g_s, L_s, Bbar_s, K1_s, alpha, c, f0, f1, lambduh, nu, omega)
  loglikelihood = numpy.sum(-0.5 * ((ymodel - mdata) / sigma_s) ** 2)
  return loglikelihood


def loglike2(cube, ndim, nparams):
  alpha, c, f0, f1, lambduh, nu, omega = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6]
  ymodel = model2(mPT_1loop_s, g_s, L_s, Bbar_s, K2_s, alpha, c, f0, f1, lambduh, nu, omega)
  loglikelihood = numpy.sum(-0.5 * ((ymodel - mdata) / sigma_s) ** 2)
  return loglikelihood

# number of dimensions our problem has
parameters = ["alpha", "c", "a", "b", "lambduh", "nu", "omega"]
n_params = len(parameters)

# run MultiNest
pymultinest.run(loglike1, prior, n_params, outputfiles_basename=datafile + '1', resume=False, verbose=True)
json.dump(parameters, open(datafile + 'params.json', 'w')) # save parameter names

# plot the distribution of a posteriori possible models
plt.figure()
x = g_s * L_s
plt.plot(x, mdata / g_s, '+ ', color='red', label='data')
run1 = pymultinest.Analyzer(outputfiles_basename=datafile + '1', n_params=n_params)
for (alpha, c, f0, f1, lambduh, nu, omega) in run1.get_equal_weighted_posterior()[::100,:-1]:
  plt.plot(x, model1(mPT_1loop_s, g_s, L_s, Bbar_s, K1_s, alpha, c, f0, f1, lambduh, nu, omega) / g_s, '-', color='blue', alpha=0.3, label='data')

plt.savefig(datafile + 'posterior1.pdf')
plt.close()

a_lnZ = run1.get_stats()['global evidence']
print()
print('************************')
print('MAIN RESULT: Evidence Z ')
print('************************')
print(f'log Z for model 1 = {(a_lnZ / log(10)):.1f}\n')

# Repeat again for model 2

# run MultiNest
pymultinest.run(loglike2, prior, n_params, outputfiles_basename=datafile + '2', resume=False, verbose=True)
json.dump(parameters, open(datafile + 'params.json', 'w')) # save parameter names

# plot the distribution of a posteriori possible models
plt.figure()
x = g_s * L_s
plt.plot(x, mdata / g_s, '+ ', color='red', label='data')
run2 = pymultinest.Analyzer(outputfiles_basename=datafile + '2', n_params=n_params)
for (alpha, c, f0, f1, lambduh, nu, omega) in run2.get_equal_weighted_posterior()[::100,:-1]:
  plt.plot(x, model2(mPT_1loop_s, g_s, L_s, Bbar_s, K2_s, alpha, c, f0, f1, lambduh, nu, omega) / g_s, '-', color='blue', alpha=0.3, label='data')

plt.savefig(datafile + 'posterior2.pdf')
plt.close()

a2_lnZ = run2.get_stats()['global evidence']
print()
print('************************')
print('MAIN RESULT: Evidence Z ')
print('************************')
print(f'log Z for model 2 = {(a_lnZ / log(10)):.1f}\n')
print(f"ln(Bayesian Evidence) : {a2_lnZ - a_lnZ}")

