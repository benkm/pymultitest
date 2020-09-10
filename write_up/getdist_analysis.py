from getdist import loadMCSamples, plots_edit
import matplotlib.pyplot as plt
import matplotlib
import time
import datetime
import os
import pdb
from inspect import getmembers
import pickle
from chisq_fit3 import model27, model28

matplotlib.use('TkAgg')


today = datetime.date.fromtimestamp(time.time())
day = f'{today.year}_{today.month}_{10}'


N = 2
GL_min = "8.0"
GL_max = 76.8
points = 10000
directory = f"graphs/{day}/"
tag = "parameter_range_run"
# directory = f"graphs/{today.year}_6_23/"


for model in [model27, model28]:
  if not os.path.isdir(directory):
      os.makedirs(directory)

  samples = loadMCSamples(f'posterior_data/{day}/{model.__name__}{tag}_N{N}_GLmin{GL_min}_GLmax{GL_max}_p{points}')
  # samples = loadMCSamples(f'posterior_data/{day}/{model.__name__}small_except_c_omega_N{N}_GLmin{GL_min}_GLmax{GL_max}_p{points}')
  g = plots_edit.get_subplot_plotter()

  g.triangle_plot(samples, filled=True)
  # plt.savefig(f"{directory}2D_posterior_{model.__name__}.png")

  # plt.savefig(f"special_graphs/2D_posterior_{model.__name__}_edit.png")

  # fig = plt.gcf()

  # fig2, axis = plt.subplots()

  # fig.axes[0].scatter([0], [0])

  analysis_small = pickle.load(open(f"posterior_data/{day}/{model.__name__}{tag}_N{N}_GLmin{GL_min}_GLmax{GL_max}_p{points}_analysis_small.pcl", "rb"))

  for i, param in enumerate(model.__code__.co_varnames[4:]):
    print(f"1 sigma range for {param} : {analysis_small[2][i]}")

  plt.show()
