#!/bin/bash
#SBATCH -n 32
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fausto.carcassi@gmail.com
#SBATCH -t 99:00:00

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
source ../../virtualenv/bin/activate

cd ../
# launch 8 jobs in parallel, each using 4 cores
for n in {1..8}; do
	python -m server_jobs.main --outputfile_append "$n" &
done
