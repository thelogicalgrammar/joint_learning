cd /project/3017068.03/fausto/joint_learning/

# print current directory
echo "Current directory:"
pwd

python -m server_jobs.main --cores 1 --tune 1 --draws 1
