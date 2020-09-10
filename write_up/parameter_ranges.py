from chisq_fit3 import *

model = model27
N = 4
points = 3000
no_samples = 30
day = 6
directory = f"posterior_data/{today.year}_{9}_{day}/"
GL_max = 76.8
tag = ""

GL_mins = pickle.load(open(f"{directory}evidence_GL_mins_points{points}_samples{no_samples}.pcl", "rb"))

sigma_1s = []
median_s = []
sigma_2s = []

for GL_min in GL_mins:
  analysis_small = pickle.load(open(f"{directory}{model.__name__}{tag}_N{N}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{points}_analysis_small.pcl", "rb"))
  sigma_1s.append(analysis_small[2])
  median_s.append(analysis_small[4])

param_names = model.__code__.co_varnames[4:]

sigma_1 = {}
mid = {}

for i, param in enumerate(param_names):
  mid[param] = [median_s[j][i] for j in range(len(median_s))]

  sigma_1[param] = numpy.array([sigma_1s[j][i] for j in range(len(sigma_1s))]).T

  sigma_1[param][0] = mid[param] - sigma_1[param][0]
  sigma_1[param][1] = sigma_1[param][1] - mid[param]

# Plot it
for param in param_names:
  plt.errorbar(GL_mins, mid[param], yerr=sigma_1[param])
  plt.show()

# model27_N4_GLmin6.4_GLmax76.8_p3000_analysis_small.pcl