import logging
import multiprocessing
import pickle
import os
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
        cp.Uniform(1e-16, 1e-13),
        cp.Uniform(1e-15, 1e-13),
        cp.Uniform(1e-12, 1e-7),
        cp.Uniform(1e-12, 1e-7),
        cp.Uniform(0.0001, 0.001),
        cp.Uniform(0.0001, 0.001),
        cp.Uniform(0.001, 0.01),
        cp.Uniform(0.001, 0.01),
        cp.Uniform(0.01, 30.0),
        cp.Uniform(0.1, 400.0),
        cp.Uniform(1.0, 6.0),
        cp.Uniform(1.0, 6.0),
        cp.Uniform(0.02, 10.0),
        cp.Uniform(0.02, 10.0),
        cp.Uniform(800, 1600),
    )

    experiment = np.array([[0.0, -1.0]])
    current_cfg = CurrentConfigurations(
        isoc=1.0, texp=5000.0 * 1.0, experiment=experiment
    )
    comsol_cfg = ComsolConfiguration(
        names=[
            "Ds_p",
            "Ds_n",
            "k_p",
            "k_n",
            "Rc_p",
            "Rc_n",
            "Rfilm_p",
            "Rfilm_n",
            "sigma_p",
            "sigma_n",
            "tau_p",
            "tau_n",
            "Cdl_p",
            "Cdl_n",
            "cl_0",
        ],
        filename="../../../lg_basic.mph",
        expression=["liion.phis0_ec1", "-t*I_app*0.000277778"],
        unit=[
            "m^2/s",
            "m^2/s",
            "m/s",
            "m/s",
            "ohm*m^2",
            "ohm*m^2",
            "ohm*m^2",
            "ohm*m^2",
            "S/m",
            "S/m",
            "1",
            "1",
            "F/m^2",
            "F/m^2",
            "mol/m^3",
        ],
        database="Study 1//Solution 1",
        evname="C_rate",
        isocname="isoc",
        experiment=current_cfg,
    )

    npool = 200
    ncores = 1
    ninterp = 10
    exclude = 3.95
    # Start Computing Processes for COMSOL
    init_event = multiprocessing.Event()
    pool = multiprocessing.Pool(
        processes=npool,
        initializer=setup_comsol_worker,
        initargs=(ncores, comsol_cfg, init_event),
    )
    try:
        init_event.wait()

        lorder = [2, 3, 4, 5, 6]
        lnsamples = [
            [20, 50, 100, 400],
            [100, 200, 500, 2000],
            [500, 1000, 2000, 5000],
            [2000, 5000, 15000, 30000],
            [10000, 20000, 50000, 100000],
        ]

        sens_cfg = SensitivityConfiguration(distribution=distribution)
        distribution_r = cp.J(
            *[cp.Uniform(-1, 1) for _ in range(distribution.lower.shape[0])]
        )
        for i, nsamples_all in enumerate(lnsamples):
            sens_cfg.distribution = distribution
            sampling = [get_sampling_from_experiment(
                npool, ncores, nsamples, ninterp, comsol_cfg, sens_cfg, exclude, pool
            ) for nsamples in nsamples_all]
            sens_cfg.distribution = distribution_r
            for j, sample in enumerate(sampling):
                samples_r = sample[0]
                x = sample[1]
                ys = sample[2]
                weights = sample[3]
                problem = sample[4]
                order = lorder[i]
                nsamples = nsamples_all[j]

                sens_cfg.order = order

                fourier, surrogate, alpha, sobol, sobol_2, sobol_t = (
                    get_sa_from_experiment(
                        samples=samples_r,
                        y=ys,
                        weights=weights,
                        problem=problem,
                        config=sens_cfg,
                    )
                )

                # Save samples for the current iteration
                with open(f"samples_n{nsamples}_o{order}.pkl", "wb") as file:
                    pickle.dump(samples_r, file)
                # Save evals for the current iteration
                with open(f"evaluations_n{nsamples}_o{order}.pkl", "wb") as file:
                    pickle.dump([x, ys], file)
                # Save surrogate for the current iteration
                with open(f"surrogate_n{nsamples}_o{order}.pkl", "wb") as file:
                    pickle.dump([x, surrogate], file)
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
                    with open(
                        f"sobol_interaction_n{nsamples}_o{order}.pkl", "wb"
                    ) as file:
                        pickle.dump(sobol_2, file)

                # Save sobol indices of the full time series
                with open(f"sobol_n{nsamples}_o{order}.pkl", "wb") as file:
                    pickle.dump(sobol, file)
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
