# Arguments:
# $1 : if "y" then will pull data from my-PC
# $2 : if "y" then will push scripts to my-PC
# $3 : labels the model for pulling data
# $4 : PC being pushed to/from

DAY=$(echo $(date +%d) | sed 's/^0*//')
MONTH=$(echo $(date +%m) | sed 's/^0*//')
YEAR=$(echo 20$(date +%y) | sed 's/^0*//')

MODEL=$3

# Pull in results
if [ "$1" == "y" ]; then
  if [ "$4" == "my-PC" ]; then
    scp my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up2/output_data/GL_min_bayes/results* my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up2/output_data/GL_min_bayes/GL* ./output_data/GL_min_bayes
    scp my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up2/output_data/evidence/$3* ./output_data/evidence/
  fi

  if [ "$4" == "iridis" ]; then
    rsync -rv iridis:/home/bkm1n18/pymultinest/write_up2/output_data/GL_min_bayes/results* ./output_data/GL_min_bayes/
    rsync -rv iridis:/home/bkm1n18/pymultinest/write_up2/output_data/GL_min_bayes/${3}*analysis_small.pcl ./output_data/GL_min_bayes/
  fi
fi

# Push changes to my-PC
if [ "$2" == "y" ]; then
  if [ "$4" == "my-PC" ]; then
    scp output_data/chisq_fit/*.pcl my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up2/output_data/chisq_fit/
    scp GL_min_bayes4.py worker_script.sh best_fit_param_estimates.py point_evidence.py chisq_functions.py chisq_fit.py bayes_functions.py evidence.py GL_min_bayes.py GL_min_bayes2.py GL_min_bayes3.py $4:/local/scratch/bkm1n18/phd/pymultinest/write_up2/
  fi

  if [ "$4" == "iridis" ]; then
    scp boss_script.sh worker_script.sh time_mpi.sh mpi_test.py GL_run_me.sh point_evidence.py chisq_functions.py chisq_fit.py bayes_functions.py evidence.py GL_min_bayes.py GL_min_bayes2.py GL_min_bayes3.py GL_min_bayes4.py GL_min_bayes_script.sh GL_min_bayes_script4.sh $4:/home/bkm1n18/pymultinest/write_up2/
  fi
fi
