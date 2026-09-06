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
    CurrentConfigurations,
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
    with open("./alpha_n500_target4_reps7.pkl", "rb") as file:
        val = pickle.load(file)

    print(val[1].shape)


if __name__ == "__main__":
    main()
