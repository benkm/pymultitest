import numpy
import matplotlib.pyplot as plt
import pdb
import datetime
import time
today = datetime.date.fromtimestamp(time.time())
from chisq_fit import *

directory = f'posterior_data/2020_6_16/'
data = numpy.load(f"{directory}posterior_data1.npy")
data3 = numpy.load(f"{directory}posterior_data3.npy")


# Extract the points with small omega
# data2 = data[data[:, 6] > -0.01]
# data3 = data2[data2[:, 7] > -20]


def plot_data(data, param_names):
  r = (len(param_names) + 1) // 2
  fig, axes = plt.subplots(2, r)

  # There are 7 parameters
  for i in range(len(param_names)):
    axes[i // r, i % r].scatter(data[:, i], data[:, 7])
    axes[i // r, i % r].set_xlim(min(data[:, i]), max(data[:, i]))
    axes[i // r, i % r].set_xlim(min(data[:, i]), max(data[:, i]))
    axes[i // r, i % r].set_xlabel(param_names[i])
    axes[i // r, i % r].set_ylabel('log(L)')

  plt.savefig(f"{directory}posterior_plot1.png")
  plt.show()
  plt.close()


plot_data(data, ['alpha', 'c', 'f0', 'f1', 'lambduh', 'nu', 'omega'])
plot_data(data3, ['alpha', 'c', 'f0', 'f1', 'lambduh', 'nu'])

# # Describes how broad a prior is used
# for model in ['1', '2']:
#   for prior_size in [3, 10, 100]:
#     directory = f'posterior_data/{today.year}_{today.month}_{today.day}/'

#     data = numpy.load(f"{directory}posterior_data{model}_size{prior_size}.npy")

#     fig, axes = plt.subplots(2, 4)
    
#     param_names = ['alpha', 'c', 'f0', 'f1', 'lambduh', 'nu', 'omega']

#     # There are 7 parameters
#     for i in range(7):
#       axes[i // 4, i % 4].scatter(data[:, i], data[:, 7])
#       axes[i // 4, i % 4].set_xlim(min(data[:, i]), max(data[:, i]))
#       axes[i // 4, i % 4].set_xlabel(param_names[i])
#       axes[i // 4, i % 4].set_ylabel('log(L)')

#     plt.savefig(f"{directory}posterior_plot{model}_size{prior_size}")
#     plt.show()
#     plt.close()


# # Describes how broad a prior is used
# for model in ['1_small', '2_small']:
#   for prior_size in [10]:
#     directory = f'posterior_data/{today.year}_{today.month}_{today.day}/'
#     data = numpy.load(f"{directory}posterior_data{model}_size{prior_size}.npy")

#     fig, axes = plt.subplots(2, 3)
    
#     param_names = ['alpha', 'f0', 'f1', 'lambduh', 'nu']

#     # There are 7 parameters
#     for i in range(5):
#       axes[i // 3, i % 3].scatter(data[:, i], data[:, 7])
#       axes[i // 3, i % 3].set_xlim(min(data[:, i]), max(data[:, i]))
#       axes[i // 3, i % 3].set_xlabel(param_names[i])
#       axes[i // 3, i % 3].set_ylabel('log(L)')

#     plt.savefig(f"{directory}posterior_plot{model}_size{prior_size}")
#     plt.show()
#     plt.close()
