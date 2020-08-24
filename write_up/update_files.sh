# Arguments:
# $1 : if "y" then will pull data from my-PC
# $2 : if "y" then will push scripts to my-PC
# $3 : labels the model for pulling data

DAY=$(echo $(date +%d) | sed 's/^0*//')
MONTH=$(echo $(date +%m) | sed 's/^0*//')
YEAR=$(echo 20$(date +%y) | sed 's/^0*//')

MODEL=$3

# Make directory to contain the MULTINEST DATA
mkdir posterior_data/${YEAR}_${MONTH}_${DAY}/

# Make directory to contain best fit graphs
mkdir best_fit_graphs/${YEAR}_${MONTH}_${DAY}/

# Pull the data from my-PC
if [ "$1" == "y" ]; then
  # rsnyc posterior_data/${YEAR}_${MONTH}_${DAY}/ my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.txt my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.json my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.paramnames my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.ranges my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.pcl   posterior_data/${YEAR}_${MONTH}_${DAY}/
  scp my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.txt my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.json my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.paramnames my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.ranges my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/$MODEL*.pcl   posterior_data/${YEAR}_${MONTH}_${DAY}/
fi

# Pull all analysis_small files from relevent folder
if [ "$1" == "y" ]; then
  scp my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/model_output_${YEAR}_${MONTH}_${DAY}/*_analysis_small.pcl posterior_data/${YEAR}_${MONTH}_${DAY}/
fi

# Push changes to my-PC
if [ "$2" == "y" ]; then
  scp Ben2.pcl bayesian_2.py chisq_fit3.py my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up/
fi
