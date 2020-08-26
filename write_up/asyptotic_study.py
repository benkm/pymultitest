from chisq_fit3 import *
from tqdm import tqdm

alpha_range = [-0.1, 0.1]
c_range = [-10, 10]
f1_range = [-10 ** 10, 10 ** 10]
lambduh_range = [0, 2]
nu_range = [0.5, 0.9]
omega_range = [0, 2]


x28 = [-0.016, -0.97, 10, -1.1, 0.98, 0.713, 0.0018]

c_s = []
omega_s = []
f0_s = []
f1_s = []

lim_range = 10 * 2 ** numpy.linspace(0, 5, 50)

for lim in tqdm(lim_range):
  f0_range = [-lim, lim]

  bounds = ([alpha_range[0], c_range[0], f0_range[0], f1_range[0], lambduh_range[0], nu_range[0], omega_range[0]],
    [alpha_range[1], c_range[1], f0_range[1], f1_range[1], lambduh_range[1], nu_range[1], omega_range[1]])

  res28 = least_squares(res_function, x28, bounds=bounds, args=(cov_inv, model28), method='dogbox')

  c_s.append(res28.x[1])
  omega_s.append(res28.x[6])
  f0_s.append(res28.x[2])
  f1_s.append(res28.x[3])

c_s = numpy.array(c_s)
omega_s = numpy.array(omega_s)
f0_s = numpy.array(f0_s)
f1_s = numpy.array(f1_s)

plt.plot(numpy.log(lim_range), numpy.log(c_s + 1))
plt.plot(numpy.log(lim_range), numpy.log(omega_s))
plt.plot(numpy.log(lim_range), numpy.log(f0_s))
plt.plot(numpy.log(lim_range), numpy.log(-f1_s))


# Plot our constants
c_primes = c_s + 1
alpha1 = 1 / (f1_s * c_primes)
alpha2 = omega_s / (f1_s * c_primes ** 2)
alpha3 = f0_s / f1_s

# Very nice - they are indeeed contsant! alpha2 asymptotes to it's preffered
# value


