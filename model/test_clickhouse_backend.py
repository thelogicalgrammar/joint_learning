import arviz
import numpy as np
import pymc as pm
import mcbackend as mcb
from clickhouse_driver import Client


def define_simple_model():
    seconds = np.linspace(0, 5)
    observations = np.random.normal(0.5 + np.random.uniform(size=3)[:, None] * seconds[None, :])
    with pm.Model(
        coords={
            "condition": ["A", "B", "C"],
        }
    ) as pmodel:
        x = pm.ConstantData("seconds", seconds, dims="time")
        a = pm.Normal("scalar")
        b = pm.Uniform("vector", dims="condition")
        pm.Deterministic("matrix", a + b[:, None] * x[None, :], dims=("condition", "time"))
        obs = pm.MutableData("obs", observations, dims=("condition", "time"))
        pm.Normal("L", pmodel["matrix"], observed=obs, dims=("condition", "time"))
        
    return pmodel


if __name__=='__main__':
    
    simple_model = define_simple_model()
    
    ch_client = Client("localhost")
    # Check that it is defined properly
    print(ch_client.execute('SHOW DATABASES'))
    backend = mcb.ClickHouseBackend(ch_client)
    
    # backend = mcb.NumPyBackend()
    
    with simple_model:
        trace = pm.sample(
            trace=backend,
            tune=100,
            draws=100,
            cores=3,
            chains=3,
            discard_tuned_samples=False,
        )
        
    print(trace)