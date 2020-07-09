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
directory = f'posterior_data/{today.year}_{today.month}_{today.day}/'

GL_min = 8
GL_max = 76.8
n_live_points = 1000

prior_num = 51
prior_range = numpy.linspace(-5, 5, prior_num)

best_fit = 1.064
EFT = 1
size = abs(best_fit - EFT)

results = numpy.zeros(51)

for i, prior in enumerate(prior_range):
  filename = f"{directory}model27_{prior:.1f}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_points{n_live_points}_analysis_small.pcl"
  
  E, delta_E, sigma_1_range, sigma_2_range, median = pickle.load(open(filename, "rb"))

  results[i] = E

without = -51.1
plt.plot([prior_range[0], prior_range[-1]], [without, without], ls="--", color='r')
plt.scatter(prior_range, results)
plt.show()

