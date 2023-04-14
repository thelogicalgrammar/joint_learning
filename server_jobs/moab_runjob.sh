echo "Importing anaconda module if not already available"
module load anaconda3/5.0.0 || true

echo "Deactivating environment if one was activated"
source conda deactivate

echo "Activating joint_learning environment"
source activate joint_learning

cd ../model/scripts
echo "Submitting job..."
echo 'python -m server_jobs.main --cores 4 --tune 1 --draws 1' | qsub -l 'walltime=70:00:00,mem=8gb'

echo "Submitted job!"
