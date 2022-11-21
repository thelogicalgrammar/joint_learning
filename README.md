# Joint learning

## Project structure

All the modelling is in the `model` folder. The `model` folder contains:
- `basicmodel.ipynb`: The main file to see the work in the project. Contains plots and explanations.
- `figures`: Contains some of the modelling and data visualizations.
- `scripts`: Contains files with most of the code. Specifically:
	- `data_functions.py`: Functions to deal with the experimental data, e.g., get it, wrangle it, etc.
	- `model_free_analysis.py`: Functions to run the logistic regression on experimental data.
	- `plotting_functions.py`: Functions to plot the data and modelling results.
	- `simulation_functions.py`: Functions to simulate fake datasets from the weight model and idealized Bayesian model.
	- `weight_model_fit`: Functions to fit the weight model to experimental data.

## How to fit weight model

Following libraries are needed:
- Pandas
- Numpy
- Scipy
- Pymc3 or pymc v.4 
- Arviz 

The first step is to change the file `model/scripts/weight_model_fit.py` to tell it where the data is located. Go to the end of the file and change the needed bit:

```python
if __name__=='__main__':
    
    get_and_fit_data(
        participant_exclusion=True, 
        method='hmc',
        fit_kwargs={
            'draws': 1000,
            'tune': 1000,
            'chains': 4
        },
        model_kwargs={
            'hierarchicallearningweights': False,
            'softmax_choice': True
        },
        datapath= GIVE RELATIVE PATH TO DATA FILE
    )
```

Since the project is organized as a package, please go to the 'model' folder and run:

```bash
python -m scripts.weight_model_fit
```

This should start the fitting!


 
