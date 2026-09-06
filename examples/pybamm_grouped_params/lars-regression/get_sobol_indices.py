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
    get_analytical_sobol,
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
    lorder = [2, 3, 4, 5]
    lnsamples = [
        [20, 50, 100, 400],
        [200, 500, 2000],
        [1000, 5000],
        [15000],
    ]

    sens_cfg = SensitivityConfiguration(distribution=distribution, cross_truncation=0.875)
    for i, order in enumerate(lorder):
        for j, nsamples in enumerate(lnsamples[i]):
            with open(f"fourier_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
            with open(f"./alpha_n{nsamples}_o{order}.pkl", "rb") as file:
                alpha = pickle.load(file)

                fourier = vals[1]
                alphas = alpha[1]

                sobol_t, sobol_2, sobol = get_analytical_sobol(fourier, alphas, sens_cfg)

                if sobol_t:
                    # Save sobol indices of the full time series
                    with open(f"sobol_total_n{nsamples}_o{order}_alltarget.pkl", "wb") as file:
                        pickle.dump(sobol_t, file)

                if sobol_2:
                    # Save sobol indices of the full time series
                    with open(
                        f"sobol_interaction_n{nsamples}_o{order}_alltarget.pkl", "wb"
                    ) as file:
                        pickle.dump(sobol_2, file)

                # Save sobol indices of the full time series
                with open(f"sobol_n{nsamples}_o{order}_alltarget.pkl", "wb") as file:
                    pickle.dump(sobol, file)


if __name__ == "__main__":
    main()
