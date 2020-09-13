# Arguments:
# $1 : if "y" then will pull data from my-PC
# $2 : if "y" then will push scripts to my-PC
# $3 : labels the model for pulling data

DAY=$(echo $(date +%d) | sed 's/^0*//')
MONTH=$(echo $(date +%m) | sed 's/^0*//')
YEAR=$(echo 20$(date +%y) | sed 's/^0*//')

MODEL=$3

# Push changes to my-PC
if [ "$2" == "y" ]; then
  scp point_evidence.py chisq_functions.py chisq_fit.py bayes_functions.py evidence.py GL_min_bayes.py my-PC:/local/scratch/bkm1n18/phd/pymultinest/write_up2/
fi
