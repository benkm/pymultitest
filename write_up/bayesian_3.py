## TO INVESTIGATE THE 5 PARAMETER MODEL

from bayesian_2 import *
from tqdm import tqdm

if __name__ == "__main__":
  # Where the results are saved
  directory = f'model_output_{today.year}_{today.month}_{today.day}/'

  alpha_range = [-0.2, 0.2]
  f0_range = [-4, 4]
  f1_range = [-2, 2]
  lambduh_range = [-1, 3]
  nu_range = [0.3, 1.1]

  prior_range = [alpha_range, f0_range, f1_range, lambduh_range, nu_range]
  n_params = len(prior_range)

  # Because the model has 5 parameters we need at least 6 data points
  GL_maxi = numpy.around(numpy.sort(g_s * L_s)[-6], 1)
  GL_max = 76.8

  points = 300
  no_samples = 15
  GL_mins = numpy.sort(list(set(numpy.around(g_s * L_s, 1))))
  GL_mins = GL_mins[GL_mins <= GL_maxi]

  results = numpy.zeros((len(GL_mins), no_samples))

  def f0(GL_min):
    results_piece = numpy.zeros(no_samples)

    for i in tqdm(range(no_samples)):
      analysis1 = run_pymultinest(prior_range, model50, GL_min, GL_max, n_params, directory, n_live_points=points, sampling_efficiency=0.3, clean_files=True, return_analysis_small=True)
      analysis2 = run_pymultinest(prior_range, model51, GL_min, GL_max, n_params, directory, n_live_points=points, sampling_efficiency=0.3, clean_files=True, return_analysis_small=True)

      # stats1 = analysis1.get_stats()
      # stats2 = analysis2.get_stats()

      # E1 = stats1['global evidence']
      # E2 = stats2['global evidence']

      E1 = analysis1[0]
      E2 = analysis2[0]

      results_piece[i] = E1 - E2

    return results_piece

  for j, GL_min in enumerate(GL_mins):
    results[j] = f0(GL_min)

  # p = Pool(4)
  # results = p.map(f0, GL_mins, chunksize=1)
  # p.close()

  pickle.dump(results, open(f"{directory}evidence_results_points{points}_samples{no_samples}.pcl", "wb"))
  pickle.dump(GL_mins, open(f"{directory}evidence_GL_mins_points{points}_samples{no_samples}.pcl", "wb"))

