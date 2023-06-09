cd /project/3017068.03/fausto/joint_learning/

# print current directory
echo "Current directory:"
pwd

# Call the main script with unbuffered output 
# so that the output is printed to the log file
# immediately
python -u -m server_jobs.main --cores 4 --tune 1000 --draws 1000
