import pickle
from chisq_fit import *


def plot_data(data, basename, model):
  E, delta_E, sigma_1_range, sigma_2_range, posterior_data = data

  # pdb.set_trace()

  # Get the parameter names of the model from the function
  # Ignore the first 4 parameters because they aren't fitting parameters
  param_names = model.__code__.co_varnames[4:]

  r = (len(param_names) + 1) // 2
  fig, axes = plt.subplots(2, r)

  res_x = numpy.load(f"best_fit_graphs/{today.year}_{today.month}_{today.day}/{model.__name__}_best_fit_params.npy")

  for i in range(len(param_names)):
    axes[i // r, i % r].scatter(posterior_data[:, i], posterior_data[:, -1], marker='o', s=3)
    axes[i // r, i % r].set_xlim(min(posterior_data[:, i]), max(posterior_data[:, i]))
    axes[i // r, i % r].set_xlim(min(posterior_data[:, i]), max(posterior_data[:, i]))
    axes[i // r, i % r].set_xlabel(param_names[i], fontsize=6)

    if i % r == 0:
      axes[i // r, i % r].set_ylabel('log(L)', fontsize=6)
    else:
      axes[i // r, i % r].set_yticklabels([])

    # Add lines for the 1 sigma and 2 sigma range
    axes[i // r, i % r].plot([sigma_1_range[i][0], sigma_1_range[i][0]],
                            [min(posterior_data[:, -1]), max(posterior_data[:, -1])],
                              color='k', lw='1')

    axes[i // r, i % r].plot([sigma_1_range[i][1], sigma_1_range[i][1]],
                            [min(posterior_data[:, -1]), max(posterior_data[:, -1])],
                              color='k', lw='1')

    axes[i // r, i % r].plot([sigma_2_range[i][0], sigma_2_range[i][0]],
                            [min(posterior_data[:, -1]), max(posterior_data[:, -1])],
                              color='r', lw='1')

    axes[i // r, i % r].plot([sigma_2_range[i][1], sigma_2_range[i][1]],
                            [min(posterior_data[:, -1]), max(posterior_data[:, -1])],
                              color='r', lw='1')

    axes[i // r, i % r].tick_params(axis='x', labelsize=6)
    axes[i // r, i % r].tick_params(axis='y', labelsize=6)

    # Add the best fit values to the plot
    axes[i // r, i % r].scatter([res_x[i]], [- 0.5 * chisq_calc(res_x, cov_inv, model)], color='purple', s=10)

  plt.savefig(f"{basename}_posterior_plot.png", size=(20, 12), dpi=600)
  plt.show()
  plt.close()


directory = f'posterior_data/2020_6_18/'

data_dict = {}
# for model in [model6, model7]:
#   basename = f"{directory}{model.__name__}_GLmin{GL_min}_GLmax{GL_max}"

#   data = pickle.load(open(f"{basename}_analysis.pcl", "rb"))
#   plot_data(data, basename, model)
  
#   data_dict[model.__name__] = data


# E6, E7 = data_dict["model6"][0], data_dict["model7"][0]
# deltaE_6, deltaE_7 = data_dict["model6"][1], data_dict["model7"][1]
# print(f"Global Evidence : {E6 - E7}")
# print(f"Error = {numpy.sqrt(deltaE_6 ** 2 + deltaE_7 ** 2)}")

for model in [model8]:
  basename = f"{directory}{model.__name__}_GLmin{GL_min}_GLmax{GL_max}"

  data = pickle.load(open(f"{basename}_analysis.pcl", "rb"))
  plot_data(data, basename, model)
  
  data_dict[model.__name__] = data
