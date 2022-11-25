# Server stuff

This folder contains the files to submit the fitting job to the SLURM scheduler used in many servers.

## Files

- `__init__`: Used to make this folder into a package
- `main.py`: Calls the script we care about
- `runjob.sh`: The SLURM job file.

## How to run

Steps:
1. Update the repository 
	- Easiest if you've changed something: delete old repo and create new copy with `git clone https://github.com/thelogicalgrammar/joint_learning`.
1. Check that the repository (the folder `joint_learning`) is in the same folder as the folder containing the virtual environment (`virtualenv`)
1. Put the `data.csv` file in the main folder of the repo (`joint_learning`).
1. Access the server and navigate to the `server_jobs` folder.
1. Submit the job with the command `sbatch runjob.sh`.
