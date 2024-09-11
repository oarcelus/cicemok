import logging
import pickle
import os
import chaospy as cp
import numpy as np
from cicemok.sensitivity import get_sa_from_experiment
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
        cp.Uniform(1e-18, 1e-14),
        cp.Uniform(1e-18, 1e-14),
        cp.Uniform(1e-13, 1e-9),
        cp.Uniform(1e-13, 1e-9),
    )

    experiment = np.array([[0.0, -1.0]])
    current_cfg = CurrentConfigurations(isoc=1.0, texp=5000.0, experiment=experiment)
    comsol_cfg = ComsolConfiguration(
        names=["Ds_p", "Ds_n", "k_p", "k_n"],
        filename="./comsol_models/lg_basic.mph",
        expression=["liion.phis0_ec1", "-t*I_app*0.000277778"],
        unit=["m^2/s", "m^2/s", "m/s", "m/s"],
        database="Study 1//Solution 1",
        evname="C_rate",
        isocname="isoc",
        experiment=current_cfg,
    )
    sens_cfg = SensitivityConfiguration(distribution=distribution, order=1)

    npool = 4
    ncores = 1
    nsamples = 4

    samples_r, x, ys, polyno, fourier, surrogate, sobol, sobol_2, sobol_t = (
        get_sa_from_experiment(
            npool=npool,
            ncores=ncores,
            nsamples=nsamples,
            experiment=comsol_cfg,
            config=sens_cfg,
        )
    )

    # Save samples for the current iteration
    with open("samples.pkl", "wb") as file:
        pickle.dump(samples_r, file)
    # Save evals for the current iteration
    with open("evaluations.pkl", "wb") as file:
        pickle.dump([x, ys], file)
    # Save surrogate for the current iteration
    with open("surrogate.pkl", "wb") as file:
        pickle.dump([x, surrogate], file)

    if sobol_t:
        # Save sobol indices of the full time series
        with open("sobol_total.pkl", "wb") as file:
            pickle.dump(sobol_t, file)

    if sobol_2:
        # Save sobol indices of the full time series
        with open("sobol_interaction.pkl", "wb") as file:
            pickle.dump(sobol_2, file)

    # Save sobol indices of the full time series
    with open("sobol.pkl", "wb") as file:
        pickle.dump(sobol, file)

if __name__ == "__main__":
    main()
