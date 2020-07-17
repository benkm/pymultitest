from chisq_fit import *
import pymultinest
import os
import pickle
from multiprocessing import Pool, current_process
from copy import copy
from time import sleep

today = datetime.date.fromtimestamp(time.time())


# Function that returns likelihood functions
def likelihood_maker(n_params, cov_inv, model, **kwargs):
  def loglike(cube, ndim, nparams):
    params = []
    for i in range(n_params):
      params.append(cube[i])
    
    chisq = chisq_calc(params, cov_inv, model, **kwargs)

    return -0.5 * chisq

  return loglike


def prior_maker(prior_range):
  def prior(cube, ndim, nparams):
    for i in range(len(prior_range)):
      cube[i] = cube[i] * (prior_range[i][1] - prior_range[i][0]) + prior_range[i][0]

  return prior


def run_pymultinest(prior_range, model, GL_min, GL_max, n_params, directory,
                    n_live_points=400, likelihood_kwargs={}, INS=False, clean_files=False):

  if not os.path.isdir(directory):
    os.makedirs(directory)

  g_s_cut, Bbar_s_cut, N_s_cut, L_s_cut, samples_cut, m_s_cut = cut(GL_min, GL_max, g_s, Bbar_s, N_s, L_s, samples, m_s)

  cov_matrix, different_ensemble = cov_matrix_calc(samples_cut, m_s_cut,
                                N_s_cut=N_s_cut, g_s_cut=g_s_cut, L_s_cut=L_s_cut)

  cov_1_2 = numpy.linalg.cholesky(cov_matrix)
  cov_inv = numpy.linalg.inv(cov_1_2)

  likelihood_kwargs["g_s"] = g_s_cut
  likelihood_kwargs["Bbar_s"] = Bbar_s_cut
  likelihood_kwargs["N_s"] = N_s_cut
  likelihood_kwargs["L_s"] = L_s_cut
  likelihood_kwargs["m_s"] = m_s_cut

  likelihood_function = likelihood_maker(n_params, cov_inv, model, **likelihood_kwargs)

  prior = prior_maker(prior_range)

  basename = f"{directory}{model.__name__}_GLmin{GL_min:.1f}_GLmax{GL_max:.1f}_p{n_live_points}"

  # Get the parameter names of the model from the function
  # Ignore the first 4 parameters because they aren't fitting parameters
  param_names = model.__code__.co_varnames[4:]

  pymultinest.run(likelihood_function, prior, n_params, outputfiles_basename=basename, resume=False,
                  n_live_points=n_live_points, sampling_efficiency=0.3, # Reccomended for Bayesian evidence,
                  evidence_tolerance=0.1, importance_nested_sampling=INS
                  )
  # save parameter names
  f = open(basename + '.paramnames', 'w')
  for i in range(len(param_names)):
    f.write(f"{param_names[i]}\n")

  f.close()

  # Save the prior ranges
  f = open(basename + '.ranges', 'w')
  for i in range(len(param_names)):
    f.write(f"{param_names[i]} {prior_range[i][0]} {prior_range[i][1]}\n")
  
  f.close()
  # Also save as a json
  json.dump(param_names, open(f'{basename}params.json', 'w'))

  analysis = pymultinest.Analyzer(outputfiles_basename=basename, n_params=n_params)

  stats = analysis.get_stats()

  E, delta_E = stats['global evidence'], stats['global evidence error']

  sigma_1_range = [analysis.get_stats()['marginals'][i]['1sigma'] for i in range(n_params)]
  sigma_2_range = [analysis.get_stats()['marginals'][i]['2sigma'] for i in range(n_params)]
  median = [analysis.get_stats()['marginals'][i]['median'] for i in range(n_params)]

  posterior_data = analysis.get_equal_weighted_posterior()
  
  analysis_data = [E, delta_E, sigma_1_range, sigma_2_range, posterior_data, median]

  if not clean_files:
    pickle.dump(analysis_data, open(f"{basename}_analysis.pcl", "wb"))

  # Make a cut down version for the purpose of quicker transfer
  analysis_small = [E, delta_E, sigma_1_range, sigma_2_range, median]

  print(f"{current_process()}: saving {basename}_analysis_small.pcl")
  pickle.dump(analysis_small, open(f"{basename}_analysis_small.pcl", "wb"))

  if clean_files:
    # Remove the remaining saved files to conserve disk space
    print(f"Removing files : {basename}*")
    os.popen(f'rm {basename}ev.dat')
    os.popen(f'rm {basename}live.points')
    os.popen(f'rm {basename}.paramnames')
    os.popen(f'rm {basename}params.json')
    os.popen(f'rm {basename}phys_live.points')
    os.popen(f'rm {basename}post_equal_weights.dat')
    os.popen(f'rm {basename}post_separate.dat')
    os.popen(f'rm {basename}.ranges')
    os.popen(f'rm {basename}resume.dat')
    os.popen(f'rm {basename}stats.dat')
    os.popen(f'rm {basename}summary.txt')
    os.popen(f'rm {basename}.txt')

  return analysis


if __name__ == "__main__":
  # Where the results are saved
  directory = f'{today.year}_{today.month}_{today.day}/'

  # Default prior range
  alpha_range = [-0.1, 0.1]
  c_range = [-1, 1]
  f0_range = [-2, 2]
  f1_range = [-0.25, 0.25]
  lambduh_range = [0.5, 1.5]
  nu_range = [0.6, 0.8]
  omega_range = [0, 2]

  prior_range = [alpha_range, c_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]
  n_params = len(prior_range)

  ## Standard fit
  GL_min = 8
  GL_max = 76.8
  points = 1000

  # analysis1 = run_pymultinest(prior_range, model27, GL_min, GL_max, n_params, directory, n_live_points=points)
  # analysis2 = run_pymultinest(prior_range, model28, GL_min, GL_max, n_params, directory, n_live_points=points)

  ## GL_min, GL_max study
  # directory = f'model_output_{today.year}_{today.month}_{today.day}/'

  # points = 3000
  # # GL_min = 0.8
  # # GL_max = 28
  # # analysis1 = run_pymultinest(prior_range, model23, GL_min, GL_max, n_params, directory, n_live_points=points)
  # # analysis2 = run_pymultinest(prior_range, model24, GL_min, GL_max, n_params, directory, n_live_points=points)

  
  # for GL_min in numpy.sort(list(set(g_s * L_s))):
  #   for GL_max in numpy.sort(list(set(g_s * L_s))):
  #     if GL_max > GL_min:
  #       print(f"Analysing for GL_min = {GL_min}, GL_max = {GL_max}")
  #       analysis1 = run_pymultinest(prior_range, model23, GL_min, GL_max, n_params, directory, n_live_points=points)
  #       analysis2 = run_pymultinest(prior_range, model24, GL_min, GL_max, n_params, directory, n_live_points=points)

  #     else:
  #       continue

  ## Prior range study
  GL_min = 8
  GL_max = 76.8
  points = 8000

  # Half ranged
  # alpha_range_small = [-0.05, 0.05]
  c_range_small = [-0.5, 0.5]
  f0_range_small = [-1, 1]
  f1_range_small = [-0.125, 0.125]
  # lambduh_range_small = [0.5, 1.5]
  nu_range_small = [0.65, 0.75]
  omega_range_small = [0, 1]

  # Double ranged
  # alpha_range_large = [-0.2, 0.2]
  c_range_large = [-2, 2]
  f0_range_large = [-4, 4]
  f1_range_large = [-0.5, 0.5]
  # lambduh_range_large = [-1, 3]
  # nu_range_large = [0.5, 0.9]
  omega_range_large = [0, 4]

  # alpha_ranges = [alpha_range_small, alpha_range, alpha_range_large]
  c_ranges = [c_range_small, c_range, c_range_large]
  f0_ranges = [f0_range_small, f0_range, f0_range_large]
  f1_ranges = [f1_range_small, f1_range, f1_range_large]
  # lambduh_ranges = [lambduh_range_small, lambduh_range, lambduh_range_large]
  # nu_ranges = [nu_range_small, nu_range, nu_range_large]
  omega_ranges = [omega_range_small, omega_range, omega_range_large]

  # Create all possible combinations of ranges
  priors = []
  names = []  # Model names to keep track
  # for i1 in range(len(alpha_ranges)):
  for i2 in range(len(c_ranges)):
    for i3 in range(len(f0_ranges)):
      for i4 in range(len(f1_ranges)):
        # for i5 in range(len(lambduh_ranges)):
        # for i6 in range(len(nu_ranges)):
        for i7 in range(len(omega_ranges)):
          priors.append([alpha_range, c_ranges[i2], f0_ranges[i3],
                f1_ranges[i4], lambduh_range, nu_range, omega_ranges[i7]])
          names.append(f"1_{i2}_{i3}_{i4}_1_1_{i7}")

  def run(i):
    print(f"{current_process()}: i = {i}")
    prior = priors[i]
    name = names[i]
    print(f"{current_process()}: prior = {prior}")
    print(f"{current_process()}: name = {name}")

    def model1(*args):
      return model27(*args)

    def model2(*args):
      return model28(*args)

    model1.__name__ = f"model27_{name}"
    model2.__name__ = f"model28_{name}"

    # print(prior)
    # print(name)
    # print(model1.__name__)

    # analysis1 = run_pymultinest(prior, model1, GL_min, GL_max, n_params, directory, n_live_points=points, clean_files=True)
    # analysis2 = run_pymultinest(prior, model2, GL_min, GL_max, n_params, directory, n_live_points=points, clean_files=True)

  # pdb.set_trace()

  # p.map(run, range(len(priors)), chunksize=1)

  ## Random prior range study
  GL_min = 8
  GL_max = 76.8
  points = 1000

  ## Removing a parameter test
  prior_num = 51
  prior_range = numpy.linspace(-5, 5, prior_num)

  EFT_s = {'alpha': 0, 'c': 0, 'f0': 0.657, 'f1': -0.0380, 'lambduh': 1, 'nu': 0.71, 'omega': 0.71}
  best_fits = {'alpha': 0.0014, 'c': -0.134, 'f0': 0.608, 'f1': -0.0602, 'lambduh': 1.064, 'nu': 0.6844, 'omega': 0.454}

  prior = [alpha_range, c_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]

  # for i, param in enumerate(['alpha', 'c', 'f0', 'f1', 'lambduh', 'nu', 'omega']):
  for i, param in enumerate(['nu']):
    def run2(prior_size):
      def model1(*args):
        return model27(*args)

      size = abs(best_fits[param] - EFT_s[param])

      print(f"Running {param} for prior_size = numpy.exp({prior_size:.1f})")
      model1.__name__ = f"model27_{param}_{EFT_s[param]}_{prior_size:.1f}"

      variable_prior = [EFT_s[param] - size * 0.5 * numpy.exp(prior_size), EFT_s[param] + size * 0.5 * numpy.exp(prior_size)]

      # prior_copy = copy(prior)
      # prior_copy[i] = variable_prior
      prior_copy = [alpha_range, c_range, f0_range, f1_range, lambduh_range, variable_prior, omega_range]

      # analysis1 = run_pymultinest(prior_copy, model1, GL_min, GL_max, n_params, directory, n_live_points=points, clean_files=True)

      return None


    # p = Pool() 
    # p.map(run2, prior_range, chunksize=1)
    # p.close()

  # Now also run the model without lambduh as a variable to compare
  # prior = [alpha_range, c_range, f0_range, f1_range, nu_range, omega_range]
  # analysis = run_pymultinest(prior, model35, GL_min, GL_max, 6, directory, n_live_points=points, clean_files=True)


  # for i in range(3 ** 7 - 6, 3 ** 7):
  #   run(i)

  # alpha_range = [-0.1, 0.1]
  # c_range = [-2, 2]
  # f0_range = [-20, 20]
  # f1_range = [-5, 5]
  # lambduh_range = [-10, 10]
  # nu_range = [0.5, 0.9]
  # omega_range = [-2, 2]

  # prior_values = {"omega": 0.782, "nu": 0.7073}
  # prior_sigmas = {"omega": 0.0013, "nu": 0.0035}
  # prior_range = [alpha_range, c_range, f0_range, f1_range, lambduh_range, nu_range, omega_range]
  # n_params = len(prior_range)

  # kwargs = {"prior": True, "prior_values": prior_values, "prior_sigmas": prior_sigmas,
  #           "m_s": m_s_cut, "N_s": N_s_cut, "g_s": g_s_cut, "L_s": L_s_cut, "Bbar_s": Bbar_s_cut}

  # analysis1 = run_pymultinest(prior_range, model82, GL_min, GL_max, n_params,
  #                            directory, likelihood_kwargs=kwargs, n_live_points=points)
  # analysis2 = run_pymultinest(prior_range, model102, GL_min, GL_max, n_params,
  #                            directory, likelihood_kwargs=kwargs, n_live_points=points)


  ## Follow Andreas suggestion - use no. sigma as the units of volume
  GL_min = 8
  GL_max = 76.8

  def get_prior_range_from_sigma(no_sigma):
    central_values = [0.00139776, -0.13428683, 0.60781802, -0.06022454, 1.06381208,
                      0.68442289, 0.45382393]

    sigma_sizes = [0.001201552008911135, 0.13898290215125328, 0.09694044410239777,
                   0.010912615653831864, 0.07759036879052572, 0.01227074127553418,
                   0.8113394466285749]
    
    prior_range = numpy.zeros((7, 2))
    for i in range(7):
      prior_range[i][0] = central_values[i] - sigma_sizes[i] * 10 ** no_sigma[i] / 2
      prior_range[i][1] = central_values[i] + sigma_sizes[i] * 10 ** no_sigma[i] / 2

    # nu and omega need a lower limit of zero
    if prior_range[5][0] < 0:
      x = -prior_range[5][0]
      prior_range[5][0] = 0
      prior_range[5][1] = prior_range[5][1] + x

    if prior_range[6][0] < 0:
      x = -prior_range[6][0]
      prior_range[6][0] = 0
      prior_range[6][1] = prior_range[6][1] + x
    
    return prior_range

  no_points = [500, 1000, 2000, 5000, 10000, 20000]
  no_samples = 100

  for points in no_points:
    def run3(i):
      # def model1(*args):
      #   return model27(*args)

      # def model2(*args):
      #   return model28(*args)

      # no_sigma = numpy.random.rand(7) * 2

      # prior_range = get_prior_range_from_sigma(no_sigma)

      # name = f"{no_sigma[0]:.3f}_{no_sigma[1]:.3f}_{no_sigma[2]:.3f}_{no_sigma[3]:.3f}_{no_sigma[4]:.3f}_{no_sigma[5]:.3f}_{no_sigma[6]:.3f}"

      # model1.__name__ = f"27_{name}"
      # model2.__name__ = f"28_{name}"

      # run_pymultinest(prior_range, model1, GL_min, GL_max, 7, directory, n_live_points=points, clean_files=True)
      # run_pymultinest(prior_range, model2, GL_min, GL_max, 7, directory, n_live_points=points, clean_files=True)
      
      def model1(*args):
        return model27(*args)

      def model2(*args):
        return model28(*args)

      # Also run a systematic sized prior which is uniform in all directions
      no_sigma = numpy.array([i / 50, ] * 7)

      prior_range = get_prior_range_from_sigma(no_sigma)

      name = f"{no_sigma[0]:.3f}_{no_sigma[1]:.3f}_{no_sigma[2]:.3f}_{no_sigma[3]:.3f}_{no_sigma[4]:.3f}_{no_sigma[5]:.3f}_{no_sigma[6]:.3f}"

      def model1(*args):
        return model27(*args)

      def model2(*args):
        return model28(*args)

      model1.__name__ = f"27_{name}"
      model2.__name__ = f"28_{name}"

      run_pymultinest(prior_range, model1, GL_min, GL_max, 7, directory, n_live_points=points, clean_files=True)
      run_pymultinest(prior_range, model2, GL_min, GL_max, 7, directory, n_live_points=points, clean_files=True)

    p = Pool()
    p.map(run3, range(no_samples), chunksize=1)
    p.close()


