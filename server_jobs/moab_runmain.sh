cd /project/3017068.03/fausto/joint_learning/

# print current directory
echo "Current directory:"
pwd

# Get job id and create a unique file name
OUTPUTFILE="server_jobs/output_$PBS_JOBID.txt"

# Call the main script with unbuffered output 
# so that the output is printed to the log file
# immediately
python -u -m server_jobs.main --cores 4 --tune 1000 --draws 1000 | tee $OUTPUTFILE
