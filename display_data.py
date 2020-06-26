import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt4Agg')
import os
from scanf import scanf
from chisq_fit import *

points = 1000 # n_live_points in multilevel
model1 = model23
model2 = model24

GL_min_ratio = 2

# Where the saved data is
directory = "posterior_data/2020_6_25/"

# Let's start of by plotting the parameter estimates
files = os.popen(f'ls {directory}')
files = [x[:-1] for x in files]

data1 = {}
data2 = {}

for x in [(data1, model1), (data2, model2)]:
  data, model = x
  data["GL_mins"] = []
  data["GL_maxs"] = []
  data["logE"] = []
  data["deltaE"] = []

  for param in model.__code__.co_varnames[4:]:
    data[param] = []
    data[param + "_lower"] = []
    data[param + "_upper"] = []
    data[param + "_lower2"] = []
    data[param + "_upper2"] = []


  for filename in files:
    if filename[-12:] == "analysis.pcl":
      model_name, GL_min, GL_max = scanf(f"%s_GLmin%f_GLmax%f_points{points}_analysis.pcl", filename)
    
      if GL_max > GL_min * GL_min_ratio and model_name == model.__name__:
        data["GL_mins"].append(GL_min)
        data["GL_maxs"].append(GL_max)

        filedata = pickle.load(open(f"{directory}{filename}", "rb"))

        E, delta_E, sigma_1_range, sigma_2_range, posterior_data, median = filedata

        data["logE"].append(E)
        data["deltaE"].append(delta_E)

        for i, param in enumerate(model.__code__.co_varnames[4:]):
          data[param].append(median[i])
          data[param + "_lower"].append(sigma_1_range[i][0])
          data[param + "_upper"].append(sigma_1_range[i][1])
          data[param + "_lower2"].append(sigma_2_range[i][0])
          data[param + "_upper2"].append(sigma_2_range[i][1])


def plot_params(model, param, data, pvalue=False):
  # Load in the best fit data
  pvalues, fit_params, bounds = pickle.load(open(f"posterior_data/{today.year}_{today.month}_{today.day}/{model.__name__}_best_fit_data.pcl", "rb"))

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  surf = ax.plot_trisurf(data["GL_mins"], data["GL_maxs"], data[param], cmap=plt.cm.jet, linewidth=0.01, zorder=2)

  # plot lines with lower and higher zorder, respectively
  for i in range(len(data["GL_mins"])):
      ax.plot((data["GL_mins"][i], data["GL_mins"][i]), (data["GL_maxs"][i], data["GL_maxs"][i]), (data[param + "_lower"][i], data[param][i]), c='k', zorder=1)
      ax.plot((data["GL_mins"][i], data["GL_mins"][i]), (data["GL_maxs"][i], data["GL_maxs"][i]), (data[param][i], data[param + "_upper"][i]), c='k', zorder=3)

      # 2 sigma error bands
      ax.plot((data["GL_mins"][i], data["GL_mins"][i]), (data["GL_maxs"][i], data["GL_maxs"][i]), (data[param + "_lower2"][i], data[param + "_lower"][i]), c='grey', zorder=1)
      ax.plot((data["GL_mins"][i], data["GL_mins"][i]), (data["GL_maxs"][i], data["GL_maxs"][i]), (data[param + "_upper"][i], data[param + "_upper2"][i]), c='grey', zorder=3)

      fit_param = []
      GL_mins_best_fit = []
      GL_maxs_best_fit = []
      sizes = []
      
      # Best fit params
      for i, GL_min in enumerate(data["GL_mins"]):
        GL_max = data["GL_maxs"][i]

        if (f"{GL_min:.1f}", f"{GL_max:.1f}", param) in fit_params:
          GL_mins_best_fit.append(GL_min)
          GL_maxs_best_fit.append(GL_max)
          fit_param.append(fit_params[(f"{GL_min:.1f}", f"{GL_max:.1f}", param)])
          sizes.append(int(pvalues[(f"{GL_min:.1f}", f"{GL_max:.1f}")] * 20))

      if pvalue:
        ax.scatter(GL_mins_best_fit, GL_maxs_best_fit, fit_param, color='r', s=sizes)


  ax.set_xlabel("GL_min")
  ax.set_ylabel("GL_max")
  ax.set_zlabel(param)

  ax.invert_xaxis()

  plt.show()


def plot_pvalues(model, data, alpha=0.001):
  # Load in the best fit data
  pvalues, fit_params, bounds = pickle.load(open(f"posterior_data/{today.year}_{today.month}_{today.day}/{model.__name__}_best_fit_data.pcl", "rb"))

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  pvalue = []
  GL_mins_best_fit = []
  GL_maxs_best_fit = []
  
  # Best fit params
  for i, GL_min in enumerate(data["GL_mins"]):
    GL_max = data["GL_maxs"][i]

    if (f"{GL_min:.1f}", f"{GL_max:.1f}", param) in fit_params:
      if pvalues[(f"{GL_min:.1f}", f"{GL_max:.1f}")] > alpha:
        GL_mins_best_fit.append(GL_min)
        GL_maxs_best_fit.append(GL_max)
        pvalue.append(pvalues[(f"{GL_min:.1f}", f"{GL_max:.1f}")])

  ax.scatter(GL_mins_best_fit, GL_maxs_best_fit, pvalue, cmap=plt.cm.jet)
  # surf = ax.plot_trisurf(GL_mins_best_fit, GL_maxs_best_fit, pvalue, cmap=plt.cm.jet, linewidth=0.01, zorder=2)

  ax.set_xlabel("GL_min")
  ax.set_ylabel("GL_max")
  ax.set_zlabel("p-value")

  ax.invert_xaxis()

  plt.show()


def look_at_fit_data(model):
  pvalues, fit_params, bounds = pickle.load(open(f"posterior_data/{today.year}_{today.month}_{today.day}/{model.__name__}_best_fit_data.pcl", "rb"))

  pdb.set_trace()


# Plot the Bayesian Evidence
def Plot_E():
  E_diff = []
  E_diff_delta = []
  GL_min_diff = []
  GL_max_diff = []

  for i, GL_min in enumerate(data1["GL_mins"]):
    GL_max = data1["GL_maxs"][i]

    # pdb.set_trace()
    if GL_min in data2["GL_mins"] and GL_max in data2["GL_maxs"]:
      set_min = set(numpy.argwhere(numpy.array(data2["GL_mins"]) == GL_min)[:, 0])
      set_max = set(numpy.argwhere(numpy.array(data2["GL_maxs"]) == GL_max)[:, 0])
      index2 = list(set_min.intersection(set_max))[0]

      E_diff.append(data1["logE"][i] - data2["logE"][index2])
      E_diff_delta.append(numpy.sqrt(data1["deltaE"][i] ** 2 + data2["deltaE"][index2] ** 2))
      GL_min_diff.append(GL_min)
      GL_max_diff.append(GL_max)

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  surf = ax.plot_trisurf(GL_min_diff, GL_max_diff, E_diff, cmap=plt.cm.jet, linewidth=0.01, zorder=2)

  # plot lines with lower and higher zorder, respectively
  for i in range(len(GL_min_diff)):
      ax.plot((GL_min_diff[i], GL_min_diff[i]), (GL_max_diff[i], GL_max_diff[i]), (E_diff[i] - E_diff_delta[i], E_diff[i]), c='k', zorder=1)
      ax.plot((GL_min_diff[i], GL_min_diff[i]), (GL_max_diff[i], GL_max_diff[i]), (E_diff[i], E_diff[i] + E_diff_delta[i]), c='k', zorder=3)

  ax.set_xlabel("GL_min")
  ax.set_ylabel("GL_max")
  ax.set_zlabel("log(E)")

  ax.invert_xaxis()

  plt.show()