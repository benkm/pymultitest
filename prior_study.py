import numpy
import pickle
import pdb
import os
import time, datetime
import pandas as pd
from chisq_fit import model27
from sklearn.linear_model import LinearRegression
from statsmodels.regression import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt

today = datetime.date.fromtimestamp(time.time())
directory = f'posterior_data/{today.year}_{today.month}_{2}/'

GL_min = 8
GL_max = 76.8
n_live_points = 1000

results_full = numpy.zeros((3 ** 7, 9))

columns = list(model27.__code__.co_varnames[4:]) + ['E1', 'E2']
# df = pd.DataFrame(columns=columns)

for i1 in range(3):
  for i2 in range(3):
    for i3 in range(3):
      for i4 in range(3):
        for i5 in range(3):
          for i6 in range(3):
            for i7 in range(3):
              filename1 = f"{directory}model27_{i1}_{i2}_{i3}_{i4}_{i5}_{i6}_{i7}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_points{n_live_points}_analysis_small.pcl"
              
              try:
                E1, delta_E1, sigma_1_range1, sigma_2_range1, median1 = pickle.load(open(filename1, "rb"))
              except:
                E1, delta_E1, sigma_1_range1, sigma_2_range1, median1 = numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan

              filename2 = f"{directory}model28_{i1}_{i2}_{i3}_{i4}_{i5}_{i6}_{i7}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_points{n_live_points}_analysis_small.pcl"

              try:
                E2, delta_E2, sigma_2_range2, sigma_2_range2, median2 = pickle.load(open(filename2, "rb"))
              except:
                E2, delta_E2, sigma_2_range2, sigma_2_range2, median2 = numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan

              results_full[numpy.ravel_multi_index([i1, i2, i3, i4, i5, i6, i7], [3, ] * 7)] = numpy.array([i1, i2, i3, i4, i5, i6, i7, E1, E2])

              # z = pd.DataFrame([[i1, i2, i3, i4, i5, i6, i7, E1, E2]], columns=columns)
              # df = pd.concat([z, df], ignore_index=True)


# Remove nanvalues
results1 = results_full[~numpy.isnan(results_full[:, -2])]
y1 = results1[:, -2]

results2 = results_full[~numpy.isnan(results_full[:, -1])]
y2 = results2[:, -1]

x1 = numpy.log(2 ** results1[:, :-2])
x1 = sm.add_constant(x1)

x2 = numpy.log(2 ** results2[:, :-2])
x2 = sm.add_constant(x2)

# Let's see if there is a general pattern depending on the size of the priors
model1 = linear_model.OLS(y1, x1)
model2 = linear_model.OLS(y2, x2)

result1 = model1.fit()
result2 = model2.fit()


print(result1.summary())
print(result2.summary())


# Find the "best" prior for each model
results_full[:, -1] = numpy.where(numpy.isnan(results_full[:, -1]), -numpy.inf, results_full[:, -1])
results_full[:, -2] = numpy.where(numpy.isnan(results_full[:, -2]), -numpy.inf, results_full[:, -2])

y1_full = results_full[:, -2]
y2_full = results_full[:, -1]
z_full = numpy.sum(numpy.log(2 ** results_full[:, :-2]), axis=1)

# Use a factor of 0.95, so that statistical fluctuations don't dominate
best1 = numpy.unravel_index(numpy.argmax(z_full * 0.95 + y1_full), (3, ) * 7)
best2 = numpy.unravel_index(numpy.argmax(z_full * 0.95 + y2_full), (3, ) * 7)

# Get the evidence at these locations
y1_full[numpy.ravel_multi_index(best1, (3, ) * 7)]
z_full[numpy.ravel_multi_index(best1, (3, ) * 7)]

y2_full[numpy.ravel_multi_index(best2, (3, ) * 7)]
z_full[numpy.ravel_multi_index(best2, (3, ) * 7)]

plt.scatter(z_full, y1_full, color='r', label='model1', marker='x')
plt.scatter(z_full, y2_full, color='b', label='model2', marker='x')

plt.xlabel("log(Volume)")
plt.ylabel("log(Evidence)")
plt.legend()

graph_dir = f"graphs/{today.year}_{today.month}_{today.day}"

if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

plt.savefig(f"{graph_dir}/prior_E_graph.png")
plt.close()

plt.scatter(z_full, y1_full + z_full, color='r', label='model1', marker='x')
plt.scatter(z_full, y2_full + z_full, color='b', label='model2', marker='x')

plt.xlabel("log(Volume)")
plt.ylabel("log(Evidence)")
plt.legend()

graph_dir = f"graphs/{today.year}_{today.month}_{today.day}"

if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

plt.savefig(f"{graph_dir}/prior_E_graph_corrected.png")

plt.close()

# Calculate the prior volume for each prior
Base_volume = 0.1 * 1 * 2 * 1 * 1 * 0.2 * 1
Volume = numpy.product(2 ** results_full[:, :-2], axis=1) * Base_volume

plt.scatter(Volume, y1_full + numpy.log(Volume), color='r', label='log(g)', marker='x')
plt.scatter(Volume, y2_full + numpy.log(Volume), color='b', label='log(1/L)', marker='x')

plt.xscale('log')
plt.xlabel("Prior Volume")
plt.ylabel("log(Evidence) + log(Volume)")
plt.legend()

plt.savefig(f"{graph_dir}/prior_E_graph_Vol_corrected.png")
plt.close()


# Graph of the Evidence, e.g. Delta(log(E))
plt.scatter(Volume, y1_full - y2_full, color='k', marker='x')
plt.fill_between([min(Volume) / 2, max(Volume) * 2], [0, 0], [1, 1], color='g', alpha=0.2, label='Insignificant 0 < x < 1')
plt.fill_between([min(Volume) / 2, max(Volume) * 2], [1, 1], [2.5, 2.5], color='r', alpha=0.2, label='Significant 1 < x < 2.5')
plt.fill_between([min(Volume) / 2, max(Volume) * 2], [2.5, 2.5], [max(y1_full - y2_full) * 1.1, max(y1_full - y2_full) * 1.1], color='b', alpha=0.2, label='Very Significant, x > 2.5')

plt.xscale('log')
plt.xlabel("Prior Volume")
plt.xlim((min(Volume) / 2, max(Volume) * 2))
plt.ylim((0, max(y1_full - y2_full) * 1.1))

plt.ylabel("log(E(model1)) - log(E(model2))")
plt.legend()
plt.show()

plt.savefig(f"{graph_dir}/prior_E_diff.png")

pdb.set_trace()

# Plot some bar charts to for each variable
# for j in range(7):
  #   N = 3
  #   ind = numpy.arange(N)  # the x locations for the groups
  #   width = 0.4       # the width of the bars

  #   fig = plt.figure()
  #   ax = fig.add_subplot(111)

  #   values1 = [100 + numpy.mean(results1[results1[:, j] == i][:, -2]) for i in range(3)]
  #   std1 = [numpy.std(results1[results1[:, j] == i][:, -2]) for i in range(3)]
  #   rects1 = ax.bar(ind, values1, width, color='r', yerr=std1, bottom=-100)

  #   values2 = [100 + numpy.mean(results2[results2[:, j] == i][:, -1]) for i in range(3)]
  #   std2 = [numpy.std(results2[results2[:, j] == i][:, -1]) for i in range(3)]
  #   rects2 = ax.bar(ind + width, values2, width, color='g', yerr=std2, bottom=-100)

  #   ax.set_ylabel('log(E)')
  #   ax.set_xticks(ind + width)
  #   ax.set_xticklabels(('0', '1', '2'))
  #   ax.legend((rects1[0], rects2[0]), ('model1', 'model2'))

  #   def autolabel(rects):
  #     for rect in rects:
  #       h = rect.get_height()
  #       ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
  #               ha='center', va='bottom')

  #   autolabel(rects1)
  #   autolabel(rects2)

  #   plt.title(f'{columns[j]}')

  #   plt.show()


# results2 = results2[results2[:, 1] == 2]
# results2 = results2[results2[:, 0] == 1]
# results1 = results1[results1[:, 0] == 2]


# For each prior zoom in on the error bands to look for significant results
for j in range(7):
  values = [(i - 1) * numpy.log(2) - numpy.mean(results1[:, -2]) + numpy.mean(results1[results1[:, j] == i][:, -2]) for i in range(3)]
  std = numpy.array([numpy.std(results1[results1[:, j] == i][:, -2]) for i in range(3)])
  count = numpy.array([numpy.sum(results1[:, j] == i) for i in range(3)])
  plt.bar([0, 1, 2], values, color='r', yerr=std / numpy.sqrt(count), bottom=numpy.mean(results1[:, -2]))

  plt.title(f'model1 : {columns[j]}')

  plt.show()

  values = [(i - 1) * numpy.log(2) - numpy.mean(results2[:, -1]) + numpy.mean(results2[results2[:, j] == i][:, -1]) for i in range(3)]
  std = numpy.array([numpy.std(results2[results2[:, j] == i][:, -1]) for i in range(3)])
  count = numpy.array([numpy.sum(results2[:, j] == i) for i in range(3)])
  plt.bar([0, 1, 2], values, color='r', yerr=std / numpy.sqrt(count), bottom=numpy.mean(results2[:, -1]))

  plt.title(f'model2 : {columns[j]}')

  plt.show()


# To catch interacting variables, plot paried bar charts
for j1 in range(7):
  for j2 in range(j1 + 1, 7):
    N = 9
    ind = numpy.arange(N)  # the x locations for the groups
    width = 0.4       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    values1 = [100 + numpy.mean(results1[numpy.logical_and(results1[:, j1] == i//3, results1[:, j2] == i%3)][:, -2]) for i in range(9)]
    std1 = [numpy.std(results1[numpy.logical_and(results1[:, j1] == i//3, results1[:, j2] == i%3)][:, -2]) for i in range(9)]
    rects1 = ax.bar(ind, values1, width, color='r', yerr=std1)

    values2 = [100 + numpy.mean(results2[numpy.logical_and(results2[:, j1] == i//3, results2[:, j2] == i%3)][:, -1]) for i in range(9)]
    std2 = [numpy.std(results2[numpy.logical_and(results2[:, j1] == i//3, results2[:, j2] == i%3)][:, -1]) for i in range(9)]
    rects2 = ax.bar(ind + width, values2, width, color='g', yerr=std2)

    ax.set_ylabel('log(E)')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((f'{i // 3}, {i % 3}' for i in range(9)))
    ax.legend((rects1[0], rects2[0]), ('model1', 'model2'))

    def autolabel(rects):
      for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, '%d' % int(h),
                ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.title(f'{columns[j1]}, {columns[j2]}')

    plt.show()

##### REFERENCE ##### PRIORS USED HERE
# Half ranged  === 0
alpha_range_small = [-0.05, 0.05]
c_range_small = [-0.5, 0.5]
f0_range_small = [-1, 1]
f1_range_small = [-0.5, 0.5]
lambduh_range_small = [0.5, 1.5]
nu_range_small = [0.6, 0.8]
omega_range_small = [0, 1]

# Normal ranged === 1
alpha_range = [-0.1, 0.1]
c_range = [-1, 1]
f0_range = [-2, 2]
f1_range = [-1, 1]
lambduh_range = [0, 2]
nu_range = [0.5, 0.9]
omega_range = [0, 2]

# Double ranged === 2
alpha_range_large = [-0.2, 0.2]
c_range_large = [-2, 2]
f0_range_large = [-4, 4]
f1_range_large = [-2, 2]
lambduh_range_large = [-1, 3]
nu_range_large = [0.3, 1.1]
omega_range_large = [0, 4]
