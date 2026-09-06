import os 
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
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
    distribution = JointIndependent(
        [
            Uniform(1e-16, (1e-13 - 1e-16)),
            Uniform(1e-16, (1e-13 - 1e-16)),
            Uniform(1e-12, (1e-7 - 1e-12)),
            Uniform(1e-12, (1e-7 - 1e-12)),
            Uniform(0.01, (30.0 - 0.01)),
            Uniform(0.1, (400.0 - 0.1)),
            Uniform(0.02, (10.0 - 0.02)),
            Uniform(0.02, (10.0 - 0.02)),
            Uniform(800, (1600 - 800)),
        ]
    )

    experiment = [("Discharge at C/1 for 1 hour or until 1.5 V")]
    pybamm_cfg = PybammConfiguration(
        names=[
            "Positive particle diffusivity [m2.s-1]",
            "Negative particle diffusivity [m2.s-1]",
            "Positive electrode reaction rate constant [m.s-1]",
            "Negative electrode reaction rate constant [m.s-1]",
            "Positive electrode conductivity [S.m-1]",
            "Negative electrode conductivity [S.m-1]",
            "Positive electrode double-layer capacity [F.m-2]",
            "Negative electrode double-layer capacity [F.m-2]",
            "Initial concentration in electrolyte [mol.m-3]",
        ],
        expression=["Voltage [V]", "Throughput capacity [A.h]"],
        sto=[0.005578964859308404, 0.941527124154424, 0.6278040986814413, 0.48121843603383746],
        experiment=experiment,
        isoc=1.0,
        modeltype="DFN",
        solver_safety=True, 
        parameter_set="cell_hakadi_cice24",
    )

    npool = 10
    ninterp = 10
    exclude = 4.0

    ctx = multiprocessing.get_context("spawn")
    # Start Computing Processes for COMSOL
    init_event = ctx.Event()
    pool = ctx.Pool(
        processes=npool,
        initializer=setup_pybamm_worker,
        initargs=(pybamm_cfg, init_event),
        maxtasksperchild=100,
    )
    try:
        init_event.wait()

        lnsamples = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2*2048]
        sens_cfg = SensitivityConfiguration(
            distribution=distribution, rule="sobol", sobol_second=True, sobol_total=True
        )
        for nsamples in lnsamples:
            sens_cfg.distribution = distribution
            samples_r, samples_q, x, ys, problem = get_sampling_from_experiment(
                nsamples,
                ninterp,
                pybamm_cfg,
                sens_cfg,
                exclude,
                pool,
                method="salib",
            )

            sobol = get_sa_from_experiment(
                samples=samples_r,
                y=ys,
                problem=problem,
                config=sens_cfg,
                method="salib",
            )

            with open(f"sobol_n{nsamples}.pkl", "wb") as file:
                pickle.dump(sobol, file)
            # Save samples for the current iteration
            with open(f"samples_n{nsamples}.pkl", "wb") as file:
                pickle.dump(samples_r, file)
            ## Save evals for the current iteration
            with open(f"evaluations_n{nsamples}.pkl", "wb") as file:
                pickle.dump([x, ys], file)
            ## Save surrogate for the current iteration
            # with open(f"surrogate_n{nsamples}_o{order}.pkl", "wb") as file:
            #    pickle.dump([x, surrogate], file)
            ## Save surrogate for the current iteration
            # with open(f"alpha_n{nsamples}_o{order}.pkl", "wb") as file:
            #    pickle.dump([x, alpha], file)
            ## Save surrogate for the current iteration
            # with open(f"fourier_n{nsamples}_o{order}.pkl", "wb") as file:
            #    pickle.dump([x, fourier], file)

            # if sobol_t:
            #    # Save sobol indices of the full time series
            #    with open(f"sobol_total_n{nsamples}_o{order}.pkl", "wb") as file:
            #        pickle.dump(sobol_t, file)

            # if sobol_2:
            #    # Save sobol indices of the full time series
            #    with open(f"sobol_interaction_n{nsamples}_o{order}.pkl", "wb") as file:
            #        pickle.dump(sobol_2, file)

            ## Save sobol indices of the full time series
            # with open(f"sobol_n{nsamples}_o{order}.pkl", "wb") as file:
            #    pickle.dump(sobol, file)

    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
