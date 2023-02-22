# Server stuff

This folder contains the files to submit the fitting job to the SLURM or the MOAB schedulers.

## Files

- `__init__`: Used to make this folder into a package
- `main.py`: Calls the script we care about
- `slurm_runjob.sh`: The SLURM job file.
- `moab_runjob.sh`: The MOAB job file.

## How to run on server

Steps:
1. Update the repository 
	- Easiest if you've changed something on the server: 
	- delete old repo and create new copy with `git clone https://github.com/thelogicalgrammar/joint_learning`.
1. Check that the repository (the folder `joint_learning`) is in the same folder as the folder containing the virtual environment (`virtualenv`)
1. Put the `data.csv` file in the main folder of the repo (`joint_learning`).
1. Access the server and navigate to the `server_jobs` folder.
1. Next step depends on server:
	1. If you are using SLURM: Submit the job with the command `sbatch runjob.sh`.
	1. If you are using MOAB: Submit the job with the command `sh moab_runjob.sh`.
