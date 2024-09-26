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
    return cp.generate_expansion(
        config.order,
        config.distribution,
        normed=config.normed,
        retall=config.retall,
        cross_truncation=config.cross_truncation,
    )


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
    polyno = generate_polynomials(config)
    surrogate, fourier = cp.fit_regression(polyno, samples, evals, retall=True)

    return polyno, fourier, surrogate


def pce_spectral(samples, weights, evals: list[np.ndarray], config: SensitivityConfiguration):
    polyno = generate_polynomials(config)
    surrogate, fourier = cp.fit_quadrature(polyno, samples, weights, evals, retall=True)

    return polyno, fourier, surrogate


def larscv_pq_pce(
    samples,
    evals: list[np.ndarray],
    config: SensitivityConfiguration,
):
    """
    samples: Experimental design. (nsamples, nfeatures)
    evals: Model evaluations. (nsamples, ntargets)
    config: SensitivityConfiguration class
    """
    evaluations = np.asarray(evals)
    yhat = np.mean(evaluations, axis=0)
    evaluations_ = evaluations - yhat
    empvar = 1.0 / (evaluations.shape[1] - 1) * np.sum(evaluations_**2, axis=0)
    dimension = len(config.distribution)
    n = len(evals)
    for p in range(config.minorder, config.order + 1):
        for q in np.arange(0.5, 1, 0.1):
            alpha = cp.glexindex(
                start=0,
                stop=p + 1,
                dimensions=dimension,
                cross_truncation=q,
                graded=True,
            )

            polynomials = generate_expansion_from_alpha(alpha.T, config)
            poly_evals = polynomials(*samples).T

            # Experimental matrix diagonal
            invATA = np.linalg.inv(np.matmul(poly_evals.T, poly_evals))
            h = np.matmul(np.matmul(poly_evals, invATA), poly_evals.T).diagonal()
            hi = 1.0 - h

            # Correction factor
            cemp = 1.0 / n * np.matmul(poly_evals.T, poly_evals)
            tpn = (
                float(n)
                / (float(n) - float(len(polynomials)))
                * (1.0 + np.trace(np.linalg.inv(cemp)) / float(n))
            )

            loo = LeaveOneOut()
            larscv = LarsCV(fit_intercept=False, cv=loo, n_jobs=-1)
            for i in range(n):
                larscv.fit(poly_evals, evaluations[:, i])
                print(larscv.mse_path_)
                alpha_ = alpha[larscv.coef_ != 0]
                polynomials_ = generate_expansion_from_alpha(alpha_.T, config)

                surrogate, coef = cp.fit_regression(
                    polynomials_, samples, evaluations[:, i], retall=True
                )
                hi = 1.0 - h
                residual = (evaluations[:, i] - surrogate(*samples)) / hi
                errloo = np.mean(residual**2)

                eloo = errloo / empvar[i]


def lars_pq_pce(
    samples,
    evals: list[np.ndarray],
    config: SensitivityConfiguration,
):
    """
    samples: Experimental design. (nsamples, nfeatures)
    evals: Model evaluations. (nsamples, ntargets)
    config: SensitivityConfiguration class
    naive: False if LOO error is computed on the LAR path of each **ntarget** data point. True if only one point is computed
    """
    # Standardize response data (this is so that there is no numerical issues with LARS pathing)
    evaluations = np.asarray(evals)
    yhat = np.mean(evaluations, axis=0)
    evaluations_ = evaluations - yhat
    varY = np.var(evaluations, axis=0)
    evaluations_ = evaluations_ / varY

    # Problem dimensions
    dimension = len(config.distribution)
    n = len(evals)

    pq = []
    pqeloo = []
    pqcoeff = []
    pqsurr = []
    for p in range(config.minorder, config.order + 1):
        for q in np.arange(0.5, 1, 0.1):
            alpha = cp.glexindex(
                start=0,
                stop=p + 1,
                dimensions=dimension,
                cross_truncation=q,
                graded=True,
            ).T

            polynomials = generate_expansion_from_alpha(alpha, config)
            poly_evals = polynomials(*samples).T

            # Experimental matrix diagonal
            invATA = np.linalg.inv(np.matmul(poly_evals.T, poly_evals))
            h = np.matmul(np.matmul(poly_evals, invATA), poly_evals.T).diagonal()
            hi = 1.0 - h

            # Correction factor
            cemp = 1.0 / n * np.matmul(poly_evals.T, poly_evals)
            tpn = (
                float(n)
                / (float(n) - float(len(polynomials)))
                * (1.0 + np.trace(np.linalg.inv(cemp)) / float(n))
            )

            lars = Lars(fit_intercept=False, n_nonzero_coefs=n - 1)
            lars.fit(poly_evals, evaluations_)

            coeffs = np.asarray(lars.coef_path_)[:, :, 1:]
            surrogates = numpoly.aspolynomial(
                [
                    numpoly.sum(polynomials * coeffs[:, :, idx], -1)
                    for idx in range(coeffs.shape[2])
                ]
            )

            # LeaveOneOut cross validation
            residual = (evaluations_.T - surrogates(*samples)) / hi
            errloo = np.mean(residual**2, axis=2)
            eloo = tpn * errloo
            neloo = (
                eloo / eloo[0, :]
            )  # I have to do this to normalize to 1, else numbers are HUGE (why not in UQLab?)

            mineloo = np.min(neloo, axis=0)
            idmin = np.argmin(neloo, axis=0)
            idall = np.arange(idmin.shape[0])

            surrogate_mins = varY * surrogates[idmin, idall] + yhat

            pq.append([p, q])
            pqeloo.append(mineloo)
            mincoeff = np.asarray([coeffs[i, :, idmin[i]] for i in idall]).T
            mincoeff *= varY
            mincoeff[0, :] += yhat
            pqcoeff.append(mincoeff)
            pqsurr.append(surrogate_mins)

    pqeloo = np.asarray(pqeloo)
    mineloo = np.min(pqeloo, axis=0)
    idxmin = np.argmin(pqeloo, axis=0)

    surrmin = numpoly.aspolynomial([pqsurr[idx] for idx in idxmin])
    pqmin = np.asarray([pq[idx] for idx in idxmin])
    pqcoeffmin = [pqcoeff[idx].T for idx in idxmin]

    return surrmin, mineloo, pqcoeffmin, pqmin


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


def get_analytical_sobol(fourier, config: SensitivityConfiguration):
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

    d_hat = np.sum(fourier[1:] ** 2, axis=0)

    sens_t_hat = None
    if config.sobol_total:
        sens_t_hat = np.empty((dimension, d_hat.shape[0]))
        for idx in range(dimension):
            index = alpha[idx, :] > 0
            sens_t_hat[idx, :] = np.sum(fourier[index] ** 2, axis=0) / d_hat

    sens_m2_hat = None
    if config.sobol_second:
        sens_m2_hat = np.empty((dimension, dimension, d_hat.shape[0]))
        for idx in range(dimension):
            for jdx in range(dimension):
                index = (
                    (idx != jdx)
                    & (alpha[idx, :] > 0)
                    & (alpha[jdx, :] > 0)
                    & (alpha.sum(0) == alpha[idx, :] + alpha[jdx, :])
                )
                sens_m2_hat[idx, jdx, :] = np.sum(fourier[index] ** 2, axis=0) / d_hat

    sens_m_hat = np.empty((dimension, d_hat.shape[0]))
    for idx in range(dimension):
        index = (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :])
        sens_m_hat[idx] = np.sum(fourier[index] ** 2, axis=0) / d_hat

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
    nsamples: int,
    ninterp: int,
    experiment: ComsolConfiguration,
    config: SensitivityConfiguration,
    exclude: float,
    project: bool,
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
                config.order, distribution_r, rule="clenshaw_curtis", sparse=True
            )
            samples_q = distribution_q.inv(distribution_r.fwd(samples_r))

        logging.info("Starting Evaluations of Samples")
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
    project: bool,
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

    logging.info("SUCCESS: All samples computed")

    if not project:
        logging.info("SURROGATE: START -> Fitting regression PCE")
        polyno, fourier, surrogate = pce(samples_r, ys, sens_cfg_copy)
    else:
        logging.info("SURROGATE: START -> Fitting quadrature PCE")
        polyno, fourier, surrogate = pce_spectral(samples_r, weights, ys, sens_cfg_copy)
    logging.info("SURROGATE: Done")

    logging.info("SOBOL: START")
    sobol_t, sobol_2, sobol = get_analytical_sobol(fourier, sens_cfg_copy)
    logging.info("SOBOL: END")

    return samples_r, x, ys, polyno, fourier, surrogate, sobol, sobol_2, sobol_t
