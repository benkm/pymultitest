from getdist import loadMCSamples, plots_edit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
import datetime
import os
import pdb
from inspect import getmembers


today = datetime.date.fromtimestamp(time.time())


GL_min = "8.0"
GL_max = 76.8
points = 5000
directory = f"graphs/{today.year}_{today.month}_{today.day}/"
# directory = f"graphs/{today.year}_6_23/"


for model in ["model27", "model28"]:
  if not os.path.isdir(directory):
      os.makedirs(directory)

  samples = loadMCSamples(f'posterior_data/{today.year}_{today.month}_{today.day}/{model}_GLmin{GL_min}_GLmax{GL_max}_p{points}')
  g = plots_edit.get_subplot_plotter()

  g.triangle_plot(samples, filled=True)
  # plt.savefig(f"{directory}2D_posterior_{model}.png")

  # fig = plt.gcf()

  # fig2, axis = plt.subplots()

  # fig.axes[0].scatter([0], [0])

  plt.show()
