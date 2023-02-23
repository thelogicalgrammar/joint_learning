echo "Importing anaconda module if not already available"
module load anaconda2/4.3.0 || true

echo "Deactivating environment if one was activated"
source conda deactivate

echo "Activating joint_learning environment"
source conda activate joint_learning

cd ../
echo "Submitting job..."
echo 'python -m server_jobs.main --cores 1 --tune 1 --draws 1' | qsub -l 'walltime=70:00:00,mem=8gb'

echo "Submitted job!"
