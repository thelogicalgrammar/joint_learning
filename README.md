# Learning what comes first

## Project structure

- `param_recovery`, `results`, `server_jobs`: Various scripts and simulation results, explained in the READMEs or in a file below.

All the modelling scripts are in the `model` folder, which contains:
- `bayesian.ipynb`: A simple Bayesian model of the experiment.
- `ceiling_model.ipybn': The ceiling model, with analysis and plots.
- `logistic_model`: Logistic statistical model of the data.
- `figures`: Contains some of the modelling and data visualizations.
- `scripts`: Contains files with most of the code. Specifically:
	- `data_functions.py`: Functions to deal with the experimental data, e.g., get it, wrangle it, etc.
	- `model_free_analysis.py`: Functions to run the logistic regression on experimental data.
	- `plotting_functions.py`: Functions to plot the data and modelling results.
	- `simulation_functions.py`: Functions to simulate fake datasets from the weight model and idealized Bayesian model.

The folder `server_jobs` contains files to run the weight model fit on a SLURM scheduler.
