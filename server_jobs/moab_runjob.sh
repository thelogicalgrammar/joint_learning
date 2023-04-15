echo "Importing anaconda module if not already available"
module load anaconda3/4.3.0 || true

echo "Deactivating environment if one was activated"
# source deactivate

echo "Activating joint_learning environment"
source activate pymc_env

echo "Submitting job..."
qsub -l 'walltime=70:00:00,mem=8gb' ${PWD}/moab_runmain.sh
echo "Submitted job!"
