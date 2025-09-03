#!/bin/bash -l
#SBATCH -c 64
#SBATCH --mail-type=BEGIN,END
#SBATCH -t 5:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pymc523
echo "$(which python)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# command line argument to use prior predictive or parameter recovery
# Remember to adjust the -c and -t arguments accordingly
prior_predictive=false

cd ../
echo "Env activated, starting python job"

if [ "$prior_predictive" = true ]; then
    python -m model.scripts.ceilingmodel --prior_predictive true
else
    for i in {0..64}; do
        python -m model.scripts.ceilingmodel --jobindex $i --prior_predictive false &
        echo "Submitted job $i"
    done
fi

# wait for all jobs to finish
wait
echo "All jobs finished"
