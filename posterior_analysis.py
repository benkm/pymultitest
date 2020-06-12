import numpy
import matplotlib.pyplot as plt
import pdb
import datetime
import time
today = datetime.date.fromtimestamp(time.time())



# Describes how broad a prior is used
for prior_size in [3, 10, 100]:

  directory = f'posterior_data/{today.year}_{today.month}_{today.day}/'
  data1 = numpy.load(f"{directory}posterior_data1_size{prior_size}.npy")
  data2 = numpy.load(f"{directory}posterior_data2_size{prior_size}.npy")

  fig, axes = plt.subplots(2, 4)

  param_names = ['alpha', 'c', 'f0', 'f1', 'lambduh', 'nu', 'omega']

  # There are 7 parameters
  for i in range(7):
    axes[i // 4, i % 4].scatter(data1[:, i], data1[:, 7])
    axes[i // 4, i % 4].set_xlim(min(data1[:, i]), max(data1[:, i]))
    axes[i // 4, i % 4].set_xlabel(param_names[i])
    axes[i // 4, i % 4].set_ylabel('log(L)')

  plt.savefig(f"{directory}posterior_plot_size{prior_size}")
  plt.show()
  plt.close()

  fig, axes = plt.subplots(2, 4)

  # There are 7 parameters
  for i in range(7):
    axes[i // 4, i % 4].scatter(data2[:, i], data2[:, 7])
    axes[i // 4, i % 4].set_xlim(min(data2[:, i]), max(data2[:, i]))
    axes[i // 4, i % 4].set_xlabel(param_names[i])
    axes[i // 4, i % 4].set_ylabel('log(L)')

    
  plt.savefig(f"{directory}posterior_plot_size{prior_size}")
  plt.show()
