import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[v] = "16"
os.environ.pop("NUMEXPR_MAX_THREADS", None)

import logging
import multiprocessing
import pickle
import os
import matplotlib.pyplot as plt

import numpy as np
import pybamm
from UQpy.distributions import Uniform, JointIndependent
from cicemok.sensitivity import get_sa_from_experiment, get_sampling_from_experiment
from cicemok.pybammrun import setup_pybamm_worker
from cicemok.configuration import (
    PybammConfiguration,
    SensitivityConfiguration,
    CurrentConfigurations
)

import logging
import os
import pickle

import dill
from cicemok.configuration import (
    SensitivityConfiguration,
)
from cicemok.sensitivity import (
    get_sa_from_experiment,
    precompute_polynomial_bases,
)
from UQpy.distributions import JointIndependent, Uniform

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
    distribution = JointIndependent(
        [
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
        ]
    )

    # repetitions = 20
    lnsamples = [50, 100, 200, 500, 1000, 2000]
    repetitions = (3, 4)
    # lnsamples = [2000]
    # repetitions = (16, 17)
    # lnsamples = [2000]

    sens_cfg = SensitivityConfiguration(
        distribution=distribution,
        cross_truncation=1.0,
        sobol_total=True,
        sobol_second=True,
        idtargets=[10, 11, 12, 13, 14]
    )
    for i in range(repetitions[0], repetitions[1]):
        for j, nsamples in enumerate(lnsamples):
            with open(f"../../reference_evals/samples_r_n{nsamples}_reps{i}.pkl", "rb") as file:
                samples_r = pickle.load(file)

            sens_cfg.order = 5
            precomputed_polynomials = precompute_polynomial_bases(samples_r, sens_cfg)
            sens_cfg.precomputed_poly = precomputed_polynomials

            # Save evals for the current iteration
            with open(f"../../reference_evals/evaluations_n{nsamples}_reps{i}.pkl", "rb") as file:
                vals = pickle.load(file)
                x = vals[0]
                ys = vals[1]

            fourier, surrogate, alpha, sobol, sobol_2, sobol_t = (
                get_sa_from_experiment(
                    samples=samples_r,
                    y=ys,
                    problem=None,
                    config=sens_cfg,
                    method="pq-omp-loo",
                )
            )

            # Save samples for the current iteration
            # Save surrogate for the current iteration
            targets = [k for k, val in enumerate(surrogate) if val is not None]

            if any(surr is None for surr in surrogate):
                for k in targets:
                    with open(f"surrogate_n{nsamples}_target{k}_reps{i}.pkl", "wb") as file:
                        dill.dump([x, surrogate[k]], file)
                    # Save surrogate for the current iteration
                    with open(f"alpha_n{nsamples}_target{k}_reps{i}.pkl", "wb") as file:
                        pickle.dump([x, alpha[k]], file)
                    # Save surrogate for the current iteration
                    with open(f"fourier_n{nsamples}_target{k}_reps{i}.pkl", "wb") as file:
                        pickle.dump([x, fourier[k]], file)
            else:
                with open(f"surrogate_n{nsamples}_alltarget_reps{i}.pkl", "wb") as file:
                    dill.dump([x, surrogate], file)
                # Save surrogate for the current iteration
                with open(f"alpha_n{nsamples}_alltarget_reps{i}.pkl", "wb") as file:
                    pickle.dump([x, alpha], file)
                # Save surrogate for the current iteration
                with open(f"fourier_n{nsamples}_alltarget_reps{i}.pkl", "wb") as file:
                    pickle.dump([x, fourier], file)

                if sobol_t:
                    # Save sobol indices of the full time series
                    with open(f"sobol_total_n{nsamples}_alltarget_reps{i}.pkl", "wb") as file:
                        pickle.dump(sobol_t, file)

                if sobol_2:
                    # Save sobol indices of the full time series
                    with open(
                        f"sobol_interaction_n{nsamples}_alltarget_reps{i}.pkl", "wb"
                    ) as file:
                        pickle.dump(sobol_2, file)

                # Save sobol indices of the full time series
                with open(f"sobol_n{nsamples}_alltarget_reps{i}.pkl", "wb") as file:
                    pickle.dump(sobol, file)


if __name__ == "__main__":
    main()
