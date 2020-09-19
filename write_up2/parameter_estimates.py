from chisq_functions import *

N = 4
model = param_8g
GL_max = 76.8
no_samples = 40
points = 800
prior_name = "update_1"

Bbar_s = ["0.420", "0.440", "0.460", "0.480", "0.500"]

Bbar_list = []
for i in range(len(Bbar_s)):
  for j in range(i + 1, len(Bbar_s)):
    Bbar_list.append([Bbar_s[i], Bbar_s[j]])

in_directory = 'output_data/chisq_fit/'
in_directory2 = 'output_data/GL_min_bayes/'

out_directory = 'graphs/parameter_estimates/'


# GL_mins = pickle.load(open(f"{in_directory}GL_mins_{model.__name__}_GL_min{GL_min:.1f}_N{N}_GLmax{GL_max:.1f}.pcl", "rb"))
GL_mins = numpy.array([4.8, 6.4, 8, 9.6, 12.8, 14.4, 16, 19.2, 24, 25.6, 28.8, 32])

pvalues = pickle.load(open(f"{in_directory}pvalues_{model.__name__}_N{N}_GLmax{GL_max:.1f}_prior{prior_name}.pcl", "rb"))
param_estimates = pickle.load(open(f"{in_directory}param_estimates_{model.__name__}_N{N}_GLmax{GL_max:.1f}_prior{prior_name}.pcl", "rb"))

for i, param in enumerate(param_8g.__code__.co_varnames[4:]):
  fig, axes = plt.subplots(5, 2)
  fig.tight_layout()

  for j in range(len(Bbar_list)):
    ax = axes[j % 5, j // 5]

    Bbar_1, Bbar_2 = Bbar_list[j]

    for k in range(no_samples):
      for GL_min in GL_mins:
        sigma_range = pickle.load(open(f"{in_directory2}{model.__name__}{Bbar_1}_{Bbar_2}_{k}_prior_N{N}_GLmin{GL_min}_GLmax{GL_max}_p{points}_analysis_small.pcl", "rb"))[2][i]

        p = pvalues[(Bbar_1, Bbar_2)][GL_min]
        param_value = param_estimates[(Bbar_1, Bbar_2)][GL_min][i]

        ax.fill_between([GL_min - 0.2, GL_min + 0.2], [sigma_range[0], sigma_range[0]], [sigma_range[1], sigma_range[1]], alpha=0.01, color='k')

        if (p > 0.05 and p < 0.95):
          ax.scatter(GL_min, param_value, alpha=p, color='r', s=2)

    ax.tick_params(direction='in')
    if j % 5 == 4:
      ax.set_xlabel("GL_min")

    else:
      ax.set_xticklabels([])

    ax.set_title(f"Bbar1 = {Bbar_1}, BBar2 = {Bbar_2}")

  fig.suptitle(param)

  plt.savefig(f"{out_directory}{model.__name__}_N{N}_{param}_points{points}_prior{prior_name}.png", dpi=500, figsize=(10, 8))

  plt.show()
