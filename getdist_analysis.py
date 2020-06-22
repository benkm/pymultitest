from getdist import loadMCSamples, plots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
import datetime
import os


today = datetime.date.fromtimestamp(time.time())


GL_min = 8
GL_max = 76.8
directory = f"graphs/{today.year}_{today.month}_{today.day}/"

for model in ["model82", "model102"]:
  if not os.path.isdir(directory):
      os.makedirs(directory)

  samples = loadMCSamples(f'posterior_data/2020_6_22/{model}_GLmin{GL_min}_GLmax{GL_max}')
  g = plots.get_subplot_plotter()

  g.triangle_plot(samples, filled=True)
  plt.savefig(f"{directory}2D_posterior_{model}.png")

  plt.show()
