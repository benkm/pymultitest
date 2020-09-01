## TO INVESTIGATE THE 5 PARAMETER MODEL

from bayesian_2 import *
from tqdm import tqdm

# PARAMETERS
model1 = model27
model2 = model28
points = 2000
no_samples = 30

if __name__ == "__main__":
  # Where the results are saved
  directory = f'model_output_{today.year}_{today.month}_{today.day}/'

  alpha_range = [-0.1, 0.1]
  c_range = [-2, 2]
  f0_range = [-4, 4]
  f1_range = [-2, 2]
  lambduh_range = [0, 2]
  nu_range = [0.5, 0.9]
  omega_range = [0, 2]

  if model1 is model50 and model2 is model51:
    prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
  
  if model1 is model27 and model2 is model28:
    prior_range = [alpha_range, c_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]

  n_params = len(prior_range)

  # Because the model has n parameters we need at least (n+1) data points
  GL_maxi = numpy.around(numpy.sort(g_s * L_s)[-(n_params + 1)], 1)
  GL_max = 76.8

  GL_mins = numpy.sort(list(set(numpy.around(g_s * L_s, 1))))
  GL_mins = GL_mins[GL_mins <= GL_maxi]

  # Apply a lower-bound GL_min, as we aren't particularly interested in results
  # below a certain threshold
  GL_mini = 5
  GL_mins = GL_mins[GL_mins >= GL_mini]

  results1 = numpy.zeros((len(GL_mins), no_samples))
  results2 = numpy.zeros((len(GL_mins), no_samples))

  def f0(GL_min):
    results_piece1 = numpy.zeros(no_samples)
    results_piece2 = numpy.zeros(no_samples)

    for i in tqdm(range(no_samples)):
      analysis1 = run_pymultinest(prior_range, model1, GL_min, GL_max, n_params, directory, n_live_points=points, sampling_efficiency=0.3, clean_files=False, return_analysis_small=True)
      analysis2 = run_pymultinest(prior_range, model2, GL_min, GL_max, n_params, directory, n_live_points=points, sampling_efficiency=0.3, clean_files=False, return_analysis_small=True)

      # stats1 = analysis1.get_stats()
      # stats2 = analysis2.get_stats()

      # E1 = stats1['global evidence']
      # E2 = stats2['global evidence']

      results_piece1[i] = analysis1[0]
      results_piece2[i] = analysis2[0]

    return numpy.array([results_piece1, results_piece2])

  # for j, GL_min in enumerate(GL_mins):
  #   results = f0(GL_min)
  #   results1[j] = results[0]
  #   results2[j] = results[1]

  p = Pool(4)
  results = numpy.array(p.map(f0, GL_mins, chunksize=1))
  p.close()
  
  results1 = results[:, 0, :]
  results2 = results[:, 1, :]

  pickle.dump(results1, open(f"{directory}evidence_{model1.__name__}_points{points}_samples{no_samples}.pcl", "wb"))
  pickle.dump(results2, open(f"{directory}evidence_{model2.__name__}_points{points}_samples{no_samples}.pcl", "wb"))
  pickle.dump(GL_mins, open(f"{directory}evidence_GL_mins_points{points}_samples{no_samples}.pcl", "wb"))
