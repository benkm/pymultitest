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
