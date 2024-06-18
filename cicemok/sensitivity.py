from functools import partial
import os

import chaospy as cp
import gstools as gs
import mph
import numpy as np
from sklearn.linear_model import LarsCV
import matplotlib.pyplot as plt

from cicemok import comsol
from cicemok.configuration import (
    EvaluationConfiguration,
    ComsolConfiguration,
    SensitivityConfiguration,
)


def generate_polynomials(config: SensitivityConfiguration, normed: bool = False):
    return cp.generate_expansion(config.order, config.distribution, normed=normed)


def curate_none_evaluations(
    evaluations: list[np.ndarray | None], samples: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    idxs = [i for i, val in enumerate(evaluations) if val is None]

    if len(idxs) == len(evaluations):
        raise ValueError("All samples failed.")

    if float(len(idxs)) / len(evaluations) > 0.2:
        raise ValueError("More than 20% of runs failed.")

    count = 0
    while idxs:
        closeidx = [
            np.argsort(np.linalg.norm(samples.T - samples.T[idx], axis=1))[1 + count]
            for idx in idxs
        ]

        for i, idx in enumerate(idxs):
            evaluations[idx] = evaluations[closeidx[i]]

        idxs = [i for i, val in enumerate(evaluations) if val is None]
        count += 1

    assert all(isinstance(result, np.ndarray) for result in evaluations)
    curated_evaluations: list[np.ndarray] = evaluations  # type: ignore
    time = curated_evaluations[0][:, 0]
    evals = [result[:, 1] for result in curated_evaluations]

    return time, evals


def curate_cutoff_evaluations(
    evaluations: list[np.ndarray], samples: np.ndarray
) -> list[np.ndarray]:
    maxim = max([val.shape[0] for val in evaluations])
    coincidence = [i for i, val in enumerate(evaluations) if val.shape[0] == maxim]
    idxs = [i for i, val in enumerate(evaluations) if val.shape[0] < maxim]

    if len(coincidence) < 2:
        raise ValueError("All samples reached cutoff ahead of time.")

    if float(len(idxs)) / len(evaluations) > 0.2:
        raise ValueError("More than 20% of runs failed.")

    count = 0
    while idxs:
        closeidx = [
            np.argsort(np.linalg.norm(samples.T - samples.T[idx], axis=1))[1 + count]
            for idx in idxs
        ]

        for i, idx in enumerate(idxs):
            evaluations[idx] = evaluations[closeidx[i]]

        idxs = [i for i, val in enumerate(evaluations) if len(val) < maxim]
        count += 1

    assert all(isinstance(result, np.ndarray) for result in evaluations)
    assert all(len(val) == maxim for val in evaluations)

    return evaluations


def evaluate_models_pool(pool, samples: np.ndarray, config: ComsolConfiguration):
    samples_pool = [sample for sample in samples.T]

    func = partial(comsol.comsol_worker_pool, config=config)
    results = pool.map(func, samples_pool)

    return results


def get_sobol(
    polyno, samples, evals: list[np.ndarray], config: SensitivityConfiguration
) -> tuple[np.ndarray, np.ndarray]:
   
    # Uses Least Squares regression to fit fourier coefficients of polynomials
    surrogate = cp.fit_regression(polyno, samples, evals)
    sobol = cp.Sens_m(surrogate, config.distribution)

    return (sobol, surrogate)


def get_sobol_pck(
    polyno, samples, evals: list[np.ndarray], config: SensitivityConfiguration
) -> tuple[np.ndarray, np.ndarray]:

    # Normalize sample data within bounds
    up = config.distribution.upper
    lo = config.distribution.lower
    _samples = (samples.T - lo) / (up - lo)

    # Fit variogram of the evaluations to a Gaussian Covariance Model
    bin, gamma = gs.vario_estimate((var for var in _samples.T), evals)
    gauss = gs.Gaussian(dim=samples.shape[0])
    gauss.fit_variogram(bin, gamma)

    ax = gauss.plot(x_max=max(bin))
    ax.scatter(bin, gamma)

    # Fit evaluations using angular regression model
    lars = LarsCV(fit_intercept=False, max_iter=1000)
    surrogate, coeffs = cp.fit_regression(polyno, samples, evals, model=lars, retall=1)
    
    # Reduce polynomial pool by eliminating 0 fourier coefficients
    _polyno = polyno[coeffs != 0]

    # Fit variogram 

    model = gs.Gaussian(dim=samples.shape[0], var=variance)
    sobol = cp.Sens_m(surrogate, config.distribution)

    return (sobol, surrogate)
