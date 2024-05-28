import multiprocessing
from functools import partial

import chaospy as cp
import mph
import numpy as np

from cicemok import comsol
from cicemok.configuration import (
    EvaluationConfiguration,
    SensitivityConfiguration,
    ComsolConfiguration,
)


def generate_polynomials(config: SensitivityConfiguration):
    return cp.generate_expansion(config.order, config.distribution)


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


def evaluate_models(model: mph.Model, config: SensitivityConfiguration):
    polyno = generate_polynomials(config)
    samples = config.distribution.sample(polyno.shape[0], rule=config.rule)
    results = [
        comsol.run_comsol_model(sample, model, config.config) for sample in samples.T
    ]

    return polyno, samples, results


def evaluate_models_pool(pool, polyno, config: SensitivityConfiguration):
    samples = config.distribution.sample(polyno.shape[0], rule=config.rule)
    samples_pool = [sample for sample in samples.T]

    func = partial(comsol_worker_pool, config=config.config)
    results = pool.map(func, samples_pool)

    return samples, results


def evaluate_ps_pool(pool, config: SensitivityConfiguration):
    samples, weights = cp.generate_quadrature(config.order, config.distribution, rule=config.rule, sparse=True)
    samples_pool = [sample for sample in samples.T]

    func = partial(comsol_worker_pool, config=config.config)
    results = pool.map(func, samples_pool)

    return samples, weights, results


def evaluate_mc_pool(pool, nsample: int, config: EvaluationConfiguration):
    samples = config.distribution.sample(nsample, rule=config.rule)
    samples_pool = [sample for sample in samples.T]

    func = partial(comsol_worker_pool, config=config.config)
    results = pool.map(func, samples_pool)

    return samples, results


def setup_comsol_worker(
    ncores: int, config: ComsolConfiguration, event: multiprocessing.Event
):
    global model

    client = comsol.start_client(cores=ncores)
    model = client.load(config.filename)
    event.set()


def comsol_worker_pool(sample: np.ndarray, config: ComsolConfiguration):
    global model

    model = comsol.set_configuration(model, config)
    result = comsol.run_comsol_model(sample, model, config)

    return result


def get_sobol_ps(
    polyno, samples, weights, evals: list[np.ndarray], config: SensitivityConfiguration
) -> tuple[np.ndarray, np.ndarray]:

    surrogate = cp.fit_quadrature(polyno, samples, weights, evals)
    sobol = cp.Sens_m(surrogate, config.distribution)

    return (sobol, surrogate)

def get_sobol(
    polyno, samples, evals: list[np.ndarray], config: SensitivityConfiguration
) -> tuple[np.ndarray, np.ndarray]:

    surrogate = cp.fit_regression(polyno, samples, evals)
    sobol = cp.Sens_m(surrogate, config.distribution)

    return (sobol, surrogate)
