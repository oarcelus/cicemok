import logging
import multiprocessing
import pickle
import os
import numpy as np
from cicemok.sensitivity import (
    get_sa_from_experiment,
    pq_loo_init_worker,
    precompute_polynomial_bases,
)
from cicemok.comsol import setup_comsol_worker
from cicemok.configuration import (
    ComsolConfiguration,
    CurrentConfigurations,
    SensitivityConfiguration,
)

from UQpy.distributions import Uniform, JointIndependent

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
            Uniform(1e-16, (1e-13 - 1e-16)),
            Uniform(1e-15, (1e-13 - 1e-15)),
            Uniform(1e-12, (1e-7 - 1e-12)),
            Uniform(1e-12, (1e-7 - 1e-12)),
            Uniform(0.0001, (0.001 - 0.0001)),
            Uniform(0.0001, (0.001 - 0.0001)),
            Uniform(0.001, (0.01 - 0.001)),
            Uniform(0.001, (0.01 - 0.001)),
            Uniform(0.01, (30.0 - 0.01)),
            Uniform(0.1, (400.0 - 0.1)),
            Uniform(1.0, (6.0 - 1.0)),
            Uniform(1.0, (6.0 - 1.0)),
            Uniform(0.02, (10.0 - 0.02)),
            Uniform(0.02, (10.0 - 0.02)),
            Uniform(800, (1600 - 800)),
        ]
    )

    # Start Computing Processes for COMSOL
    # lorder = [2, 3, 4, 5]
    # lnsamples = [
    #     [20, 50, 100, 400],
    #     [200, 500, 2000],
    #     [1000, 5000],
    #     [15000],
    # ]

    nprocesses = 5
    lorder = [4]
    lnsamples = [
        [5000],
    ]

    sens_cfg = SensitivityConfiguration(distribution=distribution)
    for i, order in enumerate(lorder):
        for j, nsamples in enumerate(lnsamples[i]):
            with open(f"samples_r_n{nsamples}_o{order}.pkl", "rb") as file:
                samples_r = pickle.load(file)

            with open(f"samples_q_n{nsamples}_o{order}.pkl", "rb") as file:
                samples_q = pickle.load(file)

            # Save evals for the current iteration
            with open(f"evaluations_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                x = vals[0]
                ys = vals[1]

            sens_cfg.order = 7
            precomputed_polynomials = precompute_polynomial_bases(samples_q, sens_cfg)
            with multiprocessing.Pool(
                processes=nprocesses, initializer=pq_loo_init_worker, initargs=(precomputed_polynomials,)
            ) as pool:
                fourier, surrogate, alpha, sobol, sobol_2, sobol_t = (
                    get_sa_from_experiment(
                        samples=samples_q,
                        y=ys,
                        problem=None,
                        config=sens_cfg,
                        method="pq-omp-loo",
                        pool=pool,
                    )
                )

            # Save samples for the current iteration
            # Save surrogate for the current iteration
            with open(f"surrogate_n{nsamples}_o{order}.pkl", "wb") as file:
                dill.dump([x, surrogate], file)
            # Save surrogate for the current iteration
            with open(f"alpha_n{nsamples}_o{order}.pkl", "wb") as file:
                pickle.dump([x, alpha], file)
            # Save surrogate for the current iteration
            with open(f"fourier_n{nsamples}_o{order}.pkl", "wb") as file:
                pickle.dump([x, fourier], file)

            if sobol_t:
                # Save sobol indices of the full time series
                with open(f"sobol_total_n{nsamples}_o{order}.pkl", "wb") as file:
                    pickle.dump(sobol_t, file)

            if sobol_2:
                # Save sobol indices of the full time series
                with open(f"sobol_interaction_n{nsamples}_o{order}.pkl", "wb") as file:
                    pickle.dump(sobol_2, file)

            # Save sobol indices of the full time series
            with open(f"sobol_n{nsamples}_o{order}.pkl", "wb") as file:
                pickle.dump(sobol, file)


if __name__ == "__main__":
    main()
