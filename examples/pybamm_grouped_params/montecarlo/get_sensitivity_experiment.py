import os

for v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
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
from cicemok.grouped_dfn_dl import GroupedDFNDoubleLayer
from cicemok.pybammrun import setup_pybamm_worker
from cicemok.configuration import (
    PybammConfiguration,
    SensitivityConfiguration,
    ExperimentType,
    PybammExperimentalConfigurations,
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
    alpha1 = 5.0
    alpha = 2.0
    distribution = JointIndependent(
        [
            Uniform(20 / alpha, 20 * alpha),
            Uniform(2 / alpha, 2 * alpha),
            Uniform(3e-4 / alpha, 3e-4 * alpha), # Negative film resistance
            Uniform(200 / alpha, 200 * alpha),
            Uniform(2e5 / alpha, 2e5 * alpha),
            Uniform(1e-4 / alpha1, 1e-4 * alpha1),
            Uniform(1e-3 / alpha1, 1e-3 * alpha1),
            Uniform(250 / alpha, 250 * alpha),
            Uniform(250 / alpha, 250 * alpha),
            Uniform(2500 / alpha, 2500 * alpha), # kappa_hat_sep
            Uniform(0.6 / alpha, 0.6 * alpha), 
            Uniform(0.6 / alpha, 0.6 * alpha), 
            Uniform(-1e-4 / alpha, (-1e-4 / alpha + 1e-4 * alpha)), # kappa_hat_D
            Uniform(1e-4 / alpha,  1e-4 * alpha), # psi_hat
            Uniform(0.1 / alpha, 0.1 * alpha),
            Uniform(0.1 / alpha, 0.1 * alpha),
            Uniform(2e-2 / alpha, 2e-2 * alpha),
        ]
    )

    xinterp = np.linspace(4.2, 2.5, 20)
    experiment = pybamm.Experiment([("Discharge at C/2 for 2 hours or until 2.5 V")])

    current = PybammExperimentalConfigurations(
        texp=[0, 3600.0], isoc=1.0, current=experiment, cutoff=(2.5, 4.2), experiment=ExperimentType.EXPERIMENT
    )
    pybamm_cfg = PybammConfiguration(
        names=[
            "Positive electrode lumped reaction rate constant [A]",
            "Negative electrode lumped reaction rate constant [A]",
            "Negative electrode lumped film resistance [Ohm]",
            "Positive electrode lumped solid conductivity [S]",
            "Negative electrode lumped solid conductivity [S]",
            "Positive electrode lumped solid diffusivity [s-1]",
            "Negative electrode lumped solid diffusivity [s-1]",
            "Positive electrode lumped electrolyte conductivity [S]",
            "Negative electrode lumped electrolyte conductivity [S]",
            "Separator lumped electrolyte conductivity [S]",
            "Positive electrode lumped double layer capacitance [F]",
            "Negative electrode lumped double layer capacitance [F]",
            ["Positive electrolyte lumped constant for transport and thermodynamic factor [V.K-1]",
             "Negative electrolyte lumped constant for transport and thermodynamic factor [V.K-1]",
             "Separator electrolyte lumped constant for transport and thermodynamic factor [V.K-1]"],
            ["Positive electrolyte scale ratio between lumped diffusivity and conductivity [V.K-1]",
             "Negative electrolyte scale ratio between lumped diffusivity and conductivity [V.K-1]",
             "Separator electrolyte scale ratio between lumped diffusivity and conductivity [V.K-1]"],
            "Positive electrode lumped quantity of electrolyte concentration [A.h]",
            "Negative electrode lumped quantity of electrolyte concentration [A.h]",
            "Separator lumped quantity of electrolyte concentration [A.h]",
        ],
        expression=["Voltage [V]", "Throughput capacity [A.h]"],
        xinterp=xinterp,
        conditions=current,
        modeltype="GroupedDFNDoubleLayer",
        solver_safety=False,
        parameter_set="Chen2020",
    )

    npool = 30

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

        lnsamples = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 2*4096, 2*2*4096, 2*2*2*4096]
        sens_cfg = SensitivityConfiguration(
            distribution=distribution, rule="sobol", sobol_second=True, sobol_total=True
        )
        for nsamples in lnsamples:
            sens_cfg.distribution = distribution
            samples_r, samples_q, x, ys, problem = get_sampling_from_experiment(
                nsamples,
                pybamm_cfg,
                sens_cfg,
                pool,
                method="salib",
            )

            ys1 = [y.flatten() for y in ys]

            sobol = get_sa_from_experiment(
                samples=samples_r,
                y=ys1,
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
