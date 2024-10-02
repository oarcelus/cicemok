import copy
import logging
import multiprocessing
import os
import pickle
from datetime import datetime
from functools import partial

import chaospy as cp
import gstools as gs
import matplotlib.pyplot as plt
import mph
import numpoly
import numpy as np
from scipy import interpolate
from sklearn.linear_model import Lars, LarsCV, LassoLars, LassoLarsCV
from sklearn.model_selection import LeaveOneOut

from cicemok import comsol
from cicemok.configuration import (
    ComsolConfiguration,
    SensitivityConfiguration,
)


def generate_polynomials(config: SensitivityConfiguration):
    stop = config.order + 1
    dimension = config.distribution.lower.shape[0]
    trunc = config.cross_truncation

    alpha = cp.glexindex(
        start=0,
        stop=stop,
        dimensions=dimension,
        cross_truncation=trunc,
        graded=True,
        reverse=True,
    ).T

    polynomials = generate_expansion_from_alpha(alpha, config)
    return alpha, polynomials


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


def pce(samples, evals: list[np.ndarray], config: SensitivityConfiguration):
    alpha, polyno = generate_polynomials(config)
    surrogate, fourier = cp.fit_regression(polyno, samples, evals, retall=True)

    return (
        np.array([alpha] * len(surrogate)),
        numpoly.aspolynomial([polyno] * len(surrogate)),
        fourier,
        surrogate,
    )


def pce_spectral(
    samples, weights, evals: list[np.ndarray], config: SensitivityConfiguration
):
    alpha, polyno = generate_polynomials(config)
    surrogate, fourier = cp.fit_quadrature(polyno, samples, weights, evals, retall=True)

    return (
        np.array([alpha] * len(surrogate)),
        numpoly.aspolynomial([polyno] * len(surrogate)),
        fourier,
        surrogate,
    )


def lars_pq_pce(
    samples,
    evals: list[np.ndarray],
    config: SensitivityConfiguration,
):
    """
    samples: Experimental design. (nsamples, nfeatures)
    evals: Model evaluations. (nsamples, ntargets)
    config: SensitivityConfiguration class
    """
    # Standardize response data (this is so that there is no numerical issues with LARS pathing)
    evaluations = np.asarray(evals)

    # Problem dimensions
    n = evaluations.shape[0]
    ntrgt = evaluations.shape[1]

    results = {}
    for q in np.linspace(0.5, 1, 5):
        # Initialize arrays to check overfitting and save optimal surrogates
        cverrors = np.full(ntrgt, np.inf, dtype=float)
        counter = np.zeros(ntrgt, dtype=int)
        activeidx = np.arange(ntrgt)
        fourier = [0] * ntrgt
        surrogates = [0] * ntrgt
        polyno = [0] * ntrgt
        alphas = [0] * ntrgt
        for p in range(config.minorder, config.order + 1):
            alpha, polynomials = generate_polynomials(config)
            poly_evals = polynomials(*samples).T

            ATA = poly_evals.T @ poly_evals
            I_ATA = np.linalg.inv(ATA)
            h = (poly_evals @ I_ATA @ poly_evals.T).diagonal()
            hi = 1.0 - h

            # Correction factor
            tpn = (
                float(n)
                * (1.0 + np.trace(I_ATA))
                / (float(n) - float(len(polynomials)))
            )
            lars = Lars(fit_intercept=False, n_nonzero_coefs=polynomials.shape[0])
            lars.fit(poly_evals, evaluations[:, activeidx])

            for i, coeffs in enumerate(lars.coef_path_):
                surrogate = numpoly.sum(polynomials * coeffs.T, axis=1)

                idx = activeidx[i]

                # LeaveOneOut cross validation
                residual = (evaluations[:, idx] - surrogate(*samples)) / hi
                errloo = np.mean(residual**2, axis=1)
                var = np.var(evaluations[:, idx], ddof=1)
                eloo = tpn * errloo / var

                # Select and gather best LAR model
                # Accept only if error is lower, and keep a counter for overfitting.
                id = np.argmin(eloo)
                if cverrors[idx] < eloo[id]:
                    counter[idx] += 1
                else:
                    if counter[idx] > 0:
                        counter[idx] -= 1

                    fourier[idx] = coeffs[:, id]
                    surrogates[idx] = surrogate[id]
                    alphas[idx] = alpha
                    polyno[idx] = polynomials
                    cverrors[idx] = eloo[id]

            # Check active targets
            activeidx = np.where(counter != 2)[0]         
            if activeidx.size == 0:
                logging.info(f"LARS: Norm: {q} Order: {p} Mean-ELOO: {np.mean(cverrors)}")
                break

        results[q] = {
            "cverror": cverrors,
            "fourier": fourier,
            "surrogates": surrogates,
            "alphas": alphas,
            "polyno": polyno,
        }

    qerrors = np.array([val["cverror"] for key, val in results.items()])
    qfourier = np.array([val["fourier"] for key, val in results.items()])
    qsurrogates = numpoly.aspolynomial(
        [val["surrogates"] for key, val in results.items()]
    )
    qalphas = np.array([val["alphas"] for key, val in results.items()])
    qpolyno = numpoly.aspolynomial([val["polyno"] for key, val in results.items()])

    idxmin = np.argmin(qerrors, axis=0)
    jdxmin = np.arange(idxmin.shape[0])

    minfourier = qfourier[idxmin, jdxmin, :]
    minsurrogates = qsurrogates[idxmin, jdxmin]
    minalphas = qalphas[idxmin, jdxmin, :, :]
    minpolynomials = qpolyno[idxmin, jdxmin, :]

    return minalphas, minpolynomials, minfourier.T, minsurrogates


def sp_fn_pce(samples, evals: list[np.ndarray], config: SensitivityConfiguration):
    # Standardize response data (this is so that there is no numerical issues with LARS pathing)
    evaluations = np.asarray(evals)
    yhat = np.mean(evaluations, axis=0)
    evaluations_ = evaluations - yhat
    varY = np.var(evaluations, axis=0)
    evaluations_ = evaluations_ / varY

    # Problem dimensions
    dimension = len(config.distribution)
    n = len(evals)

    alpha = cp.glexindex(
        start=0,
        stop=4,
        dimensions=dimension,
        cross_truncation=0.9,
        graded=True,
    ).T

    polynomials = generate_expansion_from_alpha(alpha, config)
    poly_evals = polynomials(*samples).T

    coeff = subspace_pursuit(1, poly_evals, evaluations_)
    print(coeff)


def subspace_pursuit(K, X, y):
    """subspace_pursuit
    K: Approximate bound on signal sparsity such that K >= s
    X: (nsamples, nfeatures) shapes measurement matrix
    y: (nsamples, ntargets) or (nsamples, ) measurements"""

    uhat = np.zeros((X.shape[1], y.shape[1]))
    max_iter = X.shape[1]
    W = np.eye(max_iter)

    # Initial estimateo
    for i in range(y.shape[1]):
        x = np.zeros(max_iter)
        corr = np.abs(X.T @ y[:, i])
        s0 = np.sort(corr)[::-1]
        idk = np.nonzero(corr >= s0[K])[0]

        x[idk] = np.linalg.pinv(X[:, idk]) @ y[:, i]
        ur0 = y[:, i] - X @ x

        iter = 0
        while True:
            corr = np.abs(X.T @ y[:, i])
            s0 = np.sort(corr)[::-1]
            idk2 = np.nonzero(corr >= s0[K])[0]
            idk2 = np.union1d(idk, idk2)

            x = np.zeros(max_iter)
            x[idk2] = np.linalg.pinv(X[:, idk2]) @ y[:, i]

            # Updated support estimation
            idk0 = idk
            s0 = np.sort(np.abs(x))[::-1]
            idk = np.nonzero(np.abs(x) >= s0[K])[0]

            # Update residual
            x = np.zeros(max_iter)
            x[idk] = np.linalg.pinv(X[:, idk]) @ y[:, i]
            ur = y[:, i] - X @ x

            # Break conditions
            iter += 1
            urhat = np.linalg.norm(ur)
            ur0hat = np.linalg.norm(ur0)
            if urhat >= ur0hat:
                idk = idk0
                ur0 = ur
                break

            if iter == max_iter:
                ur0 = ur
                break

        slice = uhat[:, i]
        np.put(slice, idk, np.linalg.pinv(X[:, idk]) @ y[:, i])
        uhat[:, i] = slice
        uhat[:, i] = W @ uhat[:, i]

    return uhat


def generate_expansion_from_alpha(alpha, config):
    qs = cp.variable(len(config.distribution))
    polyno = cp.prod(
        [
            cp.generate_expansion(config.order, config.distribution[idx], normed=True)[
                alpha[idx]
            ](**{"q0": qs[idx]})
            for idx in range(len(config.distribution))
        ],
        axis=0,
    )

    return polyno


def get_analytical_sobol(fouriers, alphas, config: SensitivityConfiguration):
    dimension = config.distribution.lower.shape[0]
    d_hat = np.sum(fouriers[1:, :] ** 2, axis=0)

    sens_t_hat = None
    if config.sobol_total:
        sens_t_hat = np.empty((dimension, d_hat.shape[0]))
        for idx in range(dimension):
            index = alphas[:, idx, :] > 0
            sens_t_hat[idx, :] = np.sum((fouriers * index.T) ** 2, axis=0) / d_hat

    sens_m2_hat = None
    if config.sobol_second:
        sens_m2_hat = np.empty((dimension, dimension, d_hat.shape[0]))
        for idx in range(dimension):
            for jdx in range(dimension):
                index = (
                    (idx != jdx)
                    & (alphas[:, idx, :] > 0)
                    & (alphas[:, jdx, :] > 0)
                    & (alphas.sum(0) == alphas[:, idx, :] + alphas[:, jdx, :])
                )
                sens_m2_hat[idx, jdx, :] = np.sum(fouriers * index.T**2, axis=0) / d_hat

    sens_m_hat = np.empty((dimension, d_hat.shape[0]))
    for idx in range(dimension):
        index = (alphas[:, idx, :] > 0) & (alphas.sum(1) == alphas[:, idx, :])
        sens_m_hat[idx, :] = np.sum((fouriers * index.T) ** 2, axis=0) / d_hat

    return sens_t_hat, sens_m2_hat, sens_m_hat


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
    lars = Lars(fit_intercept=False, max_iter=1000)
    surrogate, coeffs = cp.fit_regression(polyno, samples, evals, model=lars, retall=1)

    # Reduce polynomial pool by eliminating 0 fourier coefficients
    _polyno = polyno[coeffs != 0]

    # Fit variogram
    model = gs.Gaussian(dim=samples.shape[0], var=variance)
    sobol = cp.Sens_m(surrogate, config.distribution)

    return (sobol, surrogate)


def get_sampling_from_experiment(
    npool: int,
    ncores: int,
    nsamples: int,  # If project = True this is the order of the quadrature
    ninterp: int,
    experiment: ComsolConfiguration,
    config: SensitivityConfiguration,
    exclude: float,
    project: bool = False,
):
    # Start Computing Processes for COMSOL
    init_event = multiprocessing.Event()
    pool = multiprocessing.Pool(
        processes=npool,
        initializer=comsol.setup_comsol_worker,
        initargs=(ncores, experiment, init_event),
    )
    try:
        init_event.wait()
        distribution_q = config.distribution
        distribution_r = cp.J(
            *[cp.Uniform(-1, 1) for _ in range(distribution_q.lower.shape[0])]
        )

        if not project:
            samples_r = distribution_r.sample(nsamples, rule=config.rule)
            samples_q = distribution_q.inv(distribution_r.fwd(samples_r))
            weights = None
        else:
            samples_r, weights = cp.generate_quadrature(
                nsamples, distribution_r, rule="clenshaw_curtis", sparse=True
            )
            samples_q = distribution_q.inv(distribution_r.fwd(samples_r))

        evals = evaluate_models_pool(pool, samples_q, experiment)
        nevb = len(evals)

        # INVERT MIN MAX FUNCTION FOR INCREASING VOLTAGE VALUES (CHARGE)
        xinit = min([v[0, 0] for v in evals if v is not None and v[0, 0] > exclude])
        xfin = max([v[-1, 0] for v in evals if v is not None])

        f = [
            (
                interpolate.interp1d(
                    v[:, 0], v[:, 1], assume_sorted=False, fill_value="extrapolate"
                )
                if v is not None and v[0, 0] > exclude
                else None
            )
            for v in evals
        ]

        x = np.linspace(xinit, xfin, ninterp)
        ys = [interp(x) if interp is not None else None for interp in f]

        evals = [np.column_stack((x, y)) if y is not None else None for y in ys]
        neva = len(evals)

        if project and (nevb != neva):
            raise ValueError(
                "You are trying to the spectral projection for failed evaluations in quadrature points"
            )

        x, ys = curate_none_evaluations(evals, samples_q)

    finally:
        pool.close()
        pool.join()

    return samples_r, x, ys, weights


def get_sa_from_experiment(
    npool: int,
    ncores: int,
    nsamples: int,
    ninterp: int,
    experiment: ComsolConfiguration,
    config: SensitivityConfiguration,
    exclude: float,
    project: bool = False,
    method: str = "pce",
):
    distribution_q = config.distribution
    distribution_r = cp.J(
        *[cp.Uniform(-1, 1) for _ in range(distribution_q.lower.shape[0])]
    )
    samples_r, x, ys, weights = get_sampling_from_experiment(
        npool, ncores, nsamples, ninterp, experiment, config, exclude, project
    )

    assert (project and weights is not None) or (not project and weights is None)

    sens_cfg_copy = copy.deepcopy(config)
    sens_cfg_copy.distribution = distribution_r

    if not project:
        if method == "pce":
            alpha, polyno, fourier, surrogate = pce(samples_r, ys, sens_cfg_copy)
        elif method == "lars":
            alpha, polyno, fourier, surrogate = lars_pq_pce(samples_r, ys, sens_cfg_copy)
        else:
            raise ValueError("method variable must be 'pce' or 'lars'")
    else:
        alpha, polyno, fourier, surrogate = pce_spectral(
            samples_r, weights, ys, sens_cfg_copy
        )

    sobol_t, sobol_2, sobol = get_analytical_sobol(fourier, alpha, sens_cfg_copy)

    return samples_r, x, ys, polyno, fourier, surrogate, sobol, sobol_2, sobol_t
