import logging
import multiprocessing
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import pickle
import os
os.environ['NUMEXPR_MAX_THREADS'] = '255'
os.environ['NUMEXPR_NUM_THREADS'] = '255'
import chaospy as cp
import numpy as np
from cicemok.sensitivity import get_sa_from_experiment, get_sampling_from_experiment
from cicemok.comsol import setup_comsol_worker
from cicemok.configuration import (
    ComsolConfiguration,
    CurrentConfigurations,
    SensitivityConfiguration,
)

logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.getcwd(), "sensitivity.log"),
    encoding="utf-8",
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    distribution = cp.J(
        cp.Uniform(1e-17, 1e-13),
        cp.Uniform(1e-17, 1e-13),
        cp.Uniform(1e-12, 1e-7),
        cp.Uniform(1e-12, 1e-7),
        cp.Uniform(0.001, 0.1),
        cp.Uniform(0.001, 0.1),
        cp.Uniform(0.1, 300.0),
        cp.Uniform(0.1, 300.0),
        cp.Uniform(5.0, 15.0),
        cp.Uniform(5.0, 15.0),
        cp.Uniform(1.0, 6.0),
        cp.Uniform(1.0, 6.0),
        cp.Uniform(1.0, 6.0),
    )

    experiment = np.array([[0.0, -2.0]])
    current_cfg = CurrentConfigurations(isoc=1.0, texp=5000.0*0.5, experiment=experiment)
    comsol_cfg = ComsolConfiguration(
        names=["Ds_p", "Ds_n", "k_p", "k_n", "Rct_n", "Rct_p", "sigma_n", "sigma_p", "R_n", "R_p", "tau_n", "tau_p", "tau_s"],
        filename="../../../lg_basic.mph",
        expression=["liion.phis0_ec1", "-t*I_app*0.000277778"],
        unit=["m^2/s", "m^2/s", "m/s", "m/s", "mohm*m^2", "mohm*m^2", "S/m", "S/m", "um", "um", "1", "1", "1"],
        database="Study 1//Solution 1",
        evname="C_rate",
        isocname="isoc",
        experiment=current_cfg,
    )


    npool = 200
    ncores = 1
    ninterp = 10
    exclude = 3.95

    lnsamples = [16384]
    sens_cfg = SensitivityConfiguration(distribution=distribution, rule='sobol')
    distribution_r = cp.J(
        *[cp.Uniform(-1, 1) for _ in range(distribution.lower.shape[0])]
    )
    results_mean = []
    results_var = []

    with open("./sobol_n16384.pkl", "rb") as file:
        sobol = pickle.load(file) 
    with open("./evaluations_n16384.pkl", "rb") as file:
        evals = pickle.load(file) 
        x = evals[0]
        evals = evals[1]
    with open("./samples_n16384.pkl", "rb") as file:
        samples = pickle.load(file) 

    for i, xs in enumerate(x):
        npevals = np.stack(evals)
        data = (npevals[:16384, i],)
        res = bootstrap(data, np.mean, confidence_level=0.95)

        mean = np.mean(npevals[:16384, i])
        results_mean.append((mean, res))

        res = bootstrap(data, np.var, confidence_level=0.95)

        mean = np.var(npevals[:16384, i], ddof=1)
        results_var.append((mean, res))

    with open("./statistics_data_vs_voltage.pkl", "wb") as file:
        pickle.dump([x, results_mean, results_var], file)

if __name__ == "__main__":
    main()
