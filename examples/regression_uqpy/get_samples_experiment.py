import logging
import multiprocessing
import pickle
import os
import numpy as np
from cicemok.sensitivity import get_sa_from_experiment, get_sampling_from_experiment
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
    nsamples = 500000

    # Start Computing Processes for COMSOL
    init_event = multiprocessing.Event()
    pool = multiprocessing.Pool(
        processes=npool,
        initializer=setup_comsol_worker,
        initargs=(ncores, comsol_cfg, init_event),
    )
    try:
        init_event.wait()

        sens_cfg = SensitivityConfiguration(distribution=distribution)
        samples_r, samples_q, x, ys, _ = get_sampling_from_experiment(
            npool, ncores, nsamples, ninterp, comsol_cfg, sens_cfg, exclude, pool
        )

        # Save samples for the current iteration
        with open(f"samples_r_n{nsamples}_reference.pkl", "wb") as file:
            pickle.dump(samples_r, file)
        with open(f"samples_q_n{nsamples}_reference.pkl", "wb") as file:
            pickle.dump(samples_q, file)
        # Save evals for the current iteration
        with open(f"evaluations_n{nsamples}_reference.pkl", "wb") as file:
            pickle.dump([x, ys], file)

    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
