import chaospy as cp
import mph
import numpy as np
from cicemok.configuration import SensitivityConfiguration

from cicemok import comsol


def generate_polynomials(config: SensitivityConfiguration):
    return cp.generate_expansion(config.order, config.distribution)


def curate_evaluations(evaluations: list[np.ndarray | None], samples: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    idxs = [i for i, val in enumerate(evaluations) if val is None]
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
     

def evaluate_models(model: mph.Model, config: SensitivityConfiguration) -> tuple[np.ndarray, list[np.ndarray]]:
    global polyno
    global samples

    polyno = generate_polynomials(config)
    samples = config.distribution.sample(polyno.shape[0], rule=config.rule)
    results = [comsol.run_comsol_model(sample, model, config.config) for sample in samples.T]

    return curate_evaluations(results, samples)


def get_sobol(evals: list[np.ndarray], config: SensitivityConfiguration) -> tuple[np.ndarray, np.ndarray]:
    global polyno
    global sample 

    surrogate = cp.fit_regression(polyno, samples, evals)
    sobol = cp.Sens_m(surrogate, config.distribution)
   
    assert isinstance(surrogate, np.ndarray)
    return (sobol, surrogate)

