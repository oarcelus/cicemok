import copy
import math
import logging
import multiprocessing
import os
import pickle
import time
from typing import Callable
from functools import partial

import chaospy as cp
import matplotlib.pyplot as plt
import mph
import numpoly
import numpy as np
from SALib import ProblemSpec
from SALib.analyze import sobol as analyzer
from SALib.sample import sobol as sampler
from scipy import interpolate
from sklearn.linear_model import Lars, OrthogonalMatchingPursuit

from cicemok import comsol
from cicemok.configuration import (
    ComsolConfiguration,
    SensitivityConfiguration,
)


def count_number_coeffs(p: int, d: int, q: float):
    if abs(q - 1.0) < 1e-9:
        count = math.comb(p + d, p)
    else:
        alpha = cp.glexindex(
            start=0,
            stop=p + 1,
            dimensions=d,
            cross_truncation=q,
        )
        count = alpha.shape[0]

    return count


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

    polynomials = generate_expansion_from_alpha(alpha, config.distribution)

    return alpha, polynomials


def generate_expansion_from_alpha(
    alpha,
    dist,
):
    """
    Modified version of cp.expansion.stieltjes to directly accept lexicographical indexing

    """
    order = np.max(np.sum(alpha, axis=0))
    (
        _,
        polynomials,
        norms,
    ) = cp.stieltjes(np.max(order), dist)
    polynomials = numpoly.true_divide(numpoly.polynomial(polynomials), np.sqrt(norms))

    polynomials = polynomials.reshape((len(dist), np.max(order) + 1))

    order = np.array(order)
    if len(dist) > 1:
        polynomials = numpoly.prod(
            cp.polynomial([poly[idx] for poly, idx in zip(polynomials, alpha)]),
            0,
        )
    else:
        polynomials = polynomials.flatten()

    return polynomials


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
    times = curated_evaluations[0][:, 0]
    evals = [result[:, 1] for result in curated_evaluations]

    return times, evals


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
        fourier.T,
        surrogate,
    )


def pce_spectral(
    samples, weights, evals: list[np.ndarray], config: SensitivityConfiguration
):
    alpha, polyno = generate_polynomials(config)
    surrogate, fourier = cp.fit_quadrature(polyno, samples, weights, evals, retall=True)

    return (
        np.array([alpha] * len(surrogate)),
        fourier.T,
        surrogate,
    )


def pq_loo_cv(
    samples, evals: list[np.ndarray], config: SensitivityConfiguration, method: Callable
):
    """
    This algorithm uses Hybrid-LARS for sparse signal regression with Leave-One-Out Error CV.
    It uses PQ-basis adaptivity for best model selection, minimizing the Leave-One-Out Error again.
    We run all combinations of PQ, we will add a break if CV error goes up later.

    samples: Experimental design. (nsamples, nfeatures)
    evals: Model evaluations. (nsamples, ntargets)
    config: SensitivityConfiguration class
    """
    # Standardize response data (this is so that there is no numerical issues with LARS pathing)
    evaluations = np.asarray(evals)

    # Problem dimensions
    n = evaluations.shape[0]
    ntrgt = evaluations.shape[1]

    logging.info(f"REGRESSION: nsamples {n} ntargets: {ntrgt}")
    logging.info("REGRESSION: Generating PQ basis")
    polynomials, alphas = generate_pq_basis(config)
    logging.info("REGRESSION: PQ basis loaded")

    activeidx = np.arange(ntrgt)
    if config.early_stop:
        errors = [np.inf] * ntrgt
        fourier = [0] * ntrgt
        polyno = [0] * ntrgt
        alphas_ = [0] * ntrgt
        counter = np.zeros(ntrgt, dtype=int)
    else:
        errors = {k: [0] * ntrgt for k in polynomials.keys()}
        fourier = {k: [0] * ntrgt for k in polynomials.keys()}
        polyno = {k: [0] * ntrgt for k in polynomials.keys()}
        alphas_ = {k: [0] * ntrgt for k in polynomials.keys()}

    for qpc, polynomial in polynomials.items():
        poly_evals = polynomial(*samples).T
        # Loop over targets
        for i in activeidx:
            cveloo, uhat = method(poly_evals, evaluations[:, i])

            idnonzero = uhat != 0

            if config.early_stop:
                if cveloo < errors[i]:
                    if counter[i] > 0:
                        counter[i] -= 1

                    errors[i] = cveloo
                    fourier[i] = uhat
                    polyno[i] = polynomial * idnonzero
                    alphas_[i] = alphas[qpc] * idnonzero
                else:
                    counter[i] += 1

            else:
                errors[qpc][i] = cveloo
                fourier[qpc][i] = uhat
                polyno[qpc][i] = polynomial * idnonzero
                alphas_[qpc][i] = alphas[qpc] * idnonzero

        if config.early_stop:
            activeidx = np.where(counter != 2)[0]

            logging.info(
                f"REGRESSION: Norm: {qpc[0]} Order: {qpc[1]} Mean-ELOO: {np.mean(errors):.4f} Cardinality: {np.mean(np.count_nonzero(polyno))} - {qpc[2]}"
            )
            if activeidx.size == 0:
                break
        else:
            logging.info(
                f"REGRESSION: Norm: {qpc[0]} Order: {qpc[1]} Mean-ELOO: {np.mean(errors[qpc]):.4f} Cardinality: {np.mean([np.count_nonzero(pol) for pol in polyno[qpc]])} - {qpc[2]}"
            )

    if config.early_stop:
        minfourier = fourier
        minpolynomials = polyno
        minalphas = alphas_
        minsurrogates = numpoly.aspolynomial(
            [
                numpoly.sum(poly * fouri)
                for poly, fouri in zip(minpolynomials, minfourier)
            ]
        )
    else:
        min_qpc = [min(errors, key=lambda k: errors[k][i]) for i in range(ntrgt)]
        minfourier = [fourier[k][i] for i, k in enumerate(min_qpc)]
        minpolynomials = [polyno[k][i] for i, k in enumerate(min_qpc)]
        minalphas = [alphas_[k][i] for i, k in enumerate(min_qpc)]
        minsurrogates = numpoly.aspolynomial(
            [
                numpoly.sum(poly * fouri)
                for poly, fouri in zip(minpolynomials, minfourier)
            ]
        )

    return minalphas, minfourier, minsurrogates


def fn_loo_cv(
    samples, evals: list[np.ndarray], config: SensitivityConfiguration, method: Callable
):
    """
    This algorithm uses LARS for sparse signal regression with Leave-One-Out Error CV.
    It uses Forward Neighbors for the best model selection, also minimizing the Leave-One-Out Error
    We follow J. Jakeman's paper 2015

    samples: Experimental design. (nsamples, nfeatures)
    evals: Model evaluations. (nsamples, ntargets)
    config: SensitivityConfiguration class
    """

    evaluations = np.asarray(evals)

    # Problem dimensions
    n = evaluations.shape[0]
    ntrgt = evaluations.shape[1]

    # Initialize basis to have q=1 and a cardinality closest to 10*n
    alpha, polynomial = generate_fn_basis(None, 10 * n, config)
    poly_evals = polynomial(*samples).T

    # Loop over targets
    eloos = [np.inf] * ntrgt
    polynomials = [polynomial] * ntrgt
    fouriers = [0] * ntrgt
    surrogates = [0] * ntrgt
    alphas = [alpha] * ntrgt
    for i in range(ntrgt):
        cvelook, uhatk = method(poly_evals, evaluations[:, i])
        alphak = alphas[i] * uhatk
        polynomialk = polynomials[i] * uhatk
        while cvelook < eloos[i]:
            eloos[i] = cvelook
            polynomials[i] = polynomialk
            alphas[i] = alphak
            fouriers[i] = uhatk
            surrogates[i] = numpoly.sum(polynomialk * uhatk)

            cvelook = np.inf
            alphakt = alphak[:, uhatk != 0]
            for t in range(3):
                new_alpha = generate_fn_basis(alphakt, 1, config)
                alphakt = np.hstack((alphakt, new_alpha))
                polynomialkt = generate_expansion_from_alpha(
                    alphakt, config.distribution, "ttr"
                )
                poly_evalkt = polynomialkt(*samples).T
                cvelookt, uhatkt = method(poly_evalkt, evaluations[:, i])

                if cvelookt < cvelook:
                    cvelook = cvelookt
                    uhatk = uhatkt
                    alphak = alphakt
                    polynomialk = polynomialkt

    pass


def lars_loo_cv(X, y):
    """
    X: (nsamples, nfeatures) shapes measurement matrix
    y: (nsamples, ) measurements

    This method performs Cross-Validation of Subspace-Pursuit sparsity hyperparameters based on Leave-One-Out Error
    """
    N = X.shape[0]

    eloo_min = np.inf
    uhat_min = np.zeros(X.shape[1], dtype=float)
    count_eloo = 0

    lars = Lars(fit_intercept=False, n_nonzero_coefs=N - 1)
    lars.fit(X, y)
    for uhat in lars.coef_path_.T[1:]:
        idnonzero = uhat != 0
        eloo = get_eloo(X[:, idnonzero], uhat[idnonzero], y)

        if eloo_min - eloo < -1e-8:
            count_eloo += 1
        else:
            if count_eloo > 0:
                count_eloo -= 1

            uhat_min = uhat

        if count_eloo == 2:
            break

    idnonzero = uhat_min != 0
    uhat_min[idnonzero] = np.linalg.lstsq(X[:, idnonzero], y, rcond=None)[0]
    eloo_min = get_eloo(X[:, idnonzero], uhat_min[idnonzero], y)

    return eloo_min, uhat_min


def omp_loo_cv(X, y):
    """
    X: (nsamples, nfeatures) shapes measurement matrix
    y: (nsamples, ) measurements

    This method performs Cross-Validation of Subspace-Pursuit sparsity hyperparameters based on Leave-One-Out Error
    """
    N = X.shape[0]
    P = X.shape[1]

    eloo_min = np.inf
    uhat_min = np.zeros(X.shape[1], dtype=float)
    count_eloo = 0

    for K in range(min(N - 1, P)):
        omp = OrthogonalMatchingPursuit(fit_intercept=False, n_nonzero_coefs=K)
        omp.fit(X, y)
        uhat = omp.coef_
        idnonzero = uhat != 0
        eloo = get_eloo(X[:, idnonzero], uhat[idnonzero], y)

        if eloo_min - eloo < -1e-8:
            count_eloo += 1
        else:
            if count_eloo > 0:
                count_eloo -= 1

            eloo_min = eloo
            uhat_min = uhat

        if count_eloo == 2:
            break

    return eloo_min, uhat_min


def sp_loo_cv(X, y):
    """
    X: (nsamples, nfeatures) shapes measurement matrix
    y: (nsamples, ) measurements

    This method performs Cross-Validation of Subspace-Pursuit sparsity hyperparameters based on Leave-One-Out Error
    """

    N = X.shape[0]
    P = X.shape[1]

    eloo_min = np.inf
    uhat_min = np.zeros(X.shape[1], dtype=float)
    count_eloo = 0
    for K in range(min(int(N / 2), int(P / 2)) + 1):
        uhat = subspace_pursuit(K, X, y)

        idnonzero = uhat != 0
        eloo = get_eloo(X[:, idnonzero], uhat[idnonzero], y)

        if eloo_min - eloo < -1e-8:
            count_eloo += 1
        else:
            if count_eloo > 0:
                count_eloo -= 1

            eloo_min = eloo
            uhat_min = uhat

        if count_eloo == 2:
            break

    return eloo_min, uhat_min


def subspace_pursuit(K, X, y):
    """subspace_pursuit
    K: Approximate bound on signal sparsity such that K >= s
    X: (nsamples, nfeatures) shapes measurement matrix
    y: (nsamples, ) measurements

    This method is an own implementation of what is included in Diaz 2018, uses LOO CV with OLS feeting
    """

    uhat = np.zeros(X.shape[1], dtype=float)
    max_iter = X.shape[1]

    # Initial estimation
    x = np.zeros(max_iter)
    corr = np.abs(X.T @ y)
    s0 = np.sort(corr)[::-1]
    idk = np.nonzero(corr >= s0[K])[0]

    x[idk] = np.linalg.lstsq(X[:, idk], y, rcond=None)[0]
    ur0 = y - X @ x

    iter = 0
    while True:
        corr = np.abs(X.T @ ur0)
        s0 = np.sort(corr)[::-1]
        idk2 = np.nonzero(corr >= s0[K])[0]
        idk2 = np.union1d(idk, idk2)

        x = np.zeros(max_iter)
        x[idk2] = np.linalg.lstsq(X[:, idk2], y, rcond=None)[0]

        # Updated support estimation
        idk0 = idk
        s0 = np.sort(np.abs(x))[::-1]
        idk = np.nonzero(np.abs(x) >= s0[K])[0]

        # Update residual
        x = np.zeros(max_iter)
        x[idk] = np.linalg.lstsq(X[:, idk], y, rcond=None)[0]
        ur = y - X @ x

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

    uhat[idk] = np.linalg.lstsq(X[:, idk], y, rcond=None)[0]

    return uhat


def get_eloo(X, x, y):
    """
    X: observation matrix (nsamples, nfeatures)
    x: OLS fitting coefficients (nfeatures,)
    y: model evaluations (nsamples,)
    """
    n = y.shape[0]
    # Correction factors
    ATA = X.T @ X
    I_ATA = np.linalg.inv(ATA)
    h = (X @ I_ATA @ X.T).diagonal()
    hi = 1.0 - h

    tpn = float(n) * (1.0 + np.trace(I_ATA)) / (float(n) - float(X.shape[1]))

    # Leave One Out Error
    residual = (y - X @ x) / hi
    errloo = np.mean(residual**2)
    var = np.var(y, ddof=1)
    eloo = tpn * errloo / var

    return eloo


def generate_fn_basis(alpha, N: int, config: SensitivityConfiguration):
    fn_config = copy.deepcopy(config)
    fn_config.cross_truncation = 1.0
    d = fn_config.distribution.lower.shape[0]

    if alpha is None:
        combinations = [abs(count_number_coeffs(p, d, 1.0) - N) for p in range(1, 20)]
        order = combinations.index(min(combinations))
        fn_config.order = order
        alpha, polyno = generate_polynomials(fn_config)

        return alpha, polyno

    II = np.eye(d, d, dtype=int)

    # Calculate all possible forward neighboughrs
    I_NEW = II[np.newaxis, :, :]
    FWD = (I_NEW + alpha.T[:, np.newaxis, :]).reshape(d * alpha.shape[1], d)

    # Get all unique elements that are not already in alpha
    FWD = np.unique(FWD, axis=0)
    mask = ~np.any(np.all(FWD[:, np.newaxis] == alpha.T, axis=2), axis=1)
    FWD_UQ = FWD[mask]

    BWD = FWD_UQ[:, np.newaxis, :] - I_NEW

    comparison = BWD[:, :, np.newaxis, :] == alpha.T[np.newaxis, :, :]
    contained = np.any(np.all(comparison, axis=-1), axis=-1)
    idx = np.where(np.all(contained, axis=1))[0]

    return FWD_UQ[idx].T


def generate_pq_basis(config: SensitivityConfiguration):
    if "polynomials_pq.pkl" not in os.listdir(os.getcwd()):
        # Create all expansions and eliminate duplicates first
        config_loop = copy.deepcopy(config)
        d = config_loop.distribution.lower.shape[0]
        alphas_pq = {}
        polynomials_pq = {}
        for q in np.linspace(0.5, 1, 5):
            for p in range(config.minorder, config.order + 1):
                if count_number_coeffs(p, d, q) > config_loop.max_size:
                    logging.info(
                        f"LARS: Polynomial of p: {p} d: {d} q: {q} is too big > {config_loop.max_size}"
                    )
                    break

                config_loop.order = p
                config_loop.cross_truncation = q
                alpha, polynomials = generate_polynomials(config_loop)
                polynomials_pq[(q, p, len(polynomials))] = polynomials
                alphas_pq[(q, p, len(polynomials))] = alpha

        uq_polynomials_pq = {}
        for key, value in polynomials_pq.items():
            if not any(
                (value == v).all() if len(value) == len(v) else False
                for v in uq_polynomials_pq.values()
            ):
                uq_polynomials_pq[key] = value

        uq_sorted_polynomials_pq = dict(
            sorted(uq_polynomials_pq.items(), key=lambda qpc: qpc[0][2])
        )

        with open("polynomials_pq.pkl", "wb") as file:
            pickle.dump(uq_sorted_polynomials_pq, file)
        with open("alphas_pq.pkl", "wb") as file:
            pickle.dump(alphas_pq, file)

    else:
        with open("polynomials_pq.pkl", "rb") as file:
            uq_sorted_polynomials_pq = pickle.load(file)
        with open("alphas_pq.pkl", "rb") as file:
            alphas_pq = pickle.load(file)

    return uq_sorted_polynomials_pq, alphas_pq


def get_analytical_mean(fouriers):
    return np.array(fouriers)[:, 0]

def get_analytical_variance(fouriers, alphas):
    return np.sum(np.array(fouriers)[:, 1:] ** 2, axis=1)

def get_analytical_sobol(fouriers, alphas, config: SensitivityConfiguration):
    dimension = config.distribution.lower.shape[0]
    ntrgt = len(fouriers)
    d_hat = np.array([np.sum(fourier**2) for fourier in fouriers])

    sens_t_hat = None
    if config.sobol_total:
        sens_t_hat = np.empty((dimension, d_hat.shape[0]))
        for idx in range(dimension):
            index = [alpha[idx, :] > 0 for alpha in alphas]
            sens_t_hat[idx, :] = [
                np.sum((fouriers[i] * index[i]) ** 2) / d_hat[i] for i in range(ntrgt)
            ]

    sens_m2_hat = None
    if config.sobol_second:
        sens_m2_hat = np.empty((dimension, dimension, d_hat.shape[0]))
        for idx in range(dimension):
            for jdx in range(dimension):
                index = [
                    (idx != jdx)
                    & (alpha[idx, :] > 0)
                    & (alpha[jdx, :] > 0)
                    & (alpha.sum(0) == alpha[idx, :] + alpha[jdx, :])
                    for alpha in alphas
                ]
                sens_m2_hat[idx, jdx, :] = [
                    np.sum((fouriers[i] * index[i]) ** 2) / d_hat[i]
                    for i in range(ntrgt)
                ]

    sens_m_hat = np.empty((dimension, d_hat.shape[0]))
    for idx in range(dimension):
        index = [
            (alpha[idx, :] > 0) & (alpha.sum(0) == alpha[idx, :]) for alpha in alphas
        ]
        sens_m_hat[idx, :] = [
            np.sum((fouriers[i] * index[i]) ** 2) / d_hat[i] for i in range(ntrgt)
        ]

    return sens_t_hat, sens_m2_hat, sens_m_hat


def get_sampling_from_experiment(
    nsamples: int,  # If project = True this is the order of the quadrature
    ninterp: int,
    experiment: ComsolConfiguration,
    config: SensitivityConfiguration,
    exclude: float,
    pool,
    method: str = "pce",
):
    distribution_q = config.distribution
    distribution_r = cp.J(
        *[cp.Uniform(-1, 1) for _ in range(distribution_q.lower.shape[0])]
    )

    if method == "project":
        samples_r, weights = cp.generate_quadrature(
            nsamples, distribution_r, rule=config.rule, sparse=True
        )
        samples_q = distribution_q.inv(distribution_r.fwd(samples_r))
        problem = None
    elif method == "salib":
        if config.rule != "sobol":
            raise ValueError(
                "'rule' in ComsolConfiguration must be 'sobol' if method = 'salib'"
            )

        sp = {
            "names": experiment.names,
            "bounds": [[-1.0, 1.0]] * distribution_r.lower.shape[0],
            "num_vars": len(experiment.names),
        }

        samples_r = sampler.sample(sp, nsamples, calc_second_order=config.sobol_second)
        samples_q = distribution_q.inv(distribution_r.fwd(samples_r.T))
        weights = None
        problem = sp
    else:
        samples_r = distribution_r.sample(nsamples, rule=config.rule)
        samples_q = distribution_q.inv(distribution_r.fwd(samples_r))
        weights = None
        problem = None

    evals = evaluate_models_pool(pool, samples_q, experiment)
    nevb = len(evals)

    # INVERT MIN MAX FUNCTION FOR INCREASING VOLTAGE VALUES (CHARGE)
    check_nones = any(v is None for v in evals)
    logging.info(f"EVALUATIONS: Finished {len(evals)} evaluations, Nones -> {check_nones}")
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

    if method == "project" and (nevb != neva):
        raise ValueError(
            "You are trying to the spectral projection for failed evaluations in quadrature points"
        )
    if method == "salib" and (nevb != neva):
        raise ValueError(
            "You are trying to use method from SALib with a sobol sampler for failed evaluations"
        )

    x, ys = curate_none_evaluations(evals, samples_q)

    return samples_r, x, ys, weights, problem


def get_sa_from_experiment(
    samples,
    y,
    weights,
    problem,
    config: SensitivityConfiguration,
    method: str = "pce",
):
    assert (method == "project" and weights is not None) or (
        method != "project" and weights is None
    )
    assert (method == "salib" and problem is not None) or (
        method != "salib" and problem is None
    )

    if method == "pce":
        alpha, fourier, surrogate = pce(samples, y, config)
    elif method == "pq-lars-loo":
        alpha, fourier, surrogate = pq_loo_cv(samples, y, config, lars_loo_cv)
    elif method == "pq-omp-loo":
        alpha, fourier, surrogate = pq_loo_cv(samples, y, config, omp_loo_cv)
    elif method == "pq-sp-loo":
        alpha, fourier, surrogate = pq_loo_cv(samples, y, config, sp_loo_cv)
    elif method == "fn-lars-loo":
        alpha, fourier, surrogate = fn_loo_cv(samples, y, config, lars_loo_cv)
    elif method == "project":
        alpha, fourier, surrogate = pce_spectral(samples, weights, y, config)
    elif method == "salib":
        if config.rule != "sobol":
            raise ValueError(
                "'rule' in ComsolConfiguration must be 'sobol' if method = 'salib'"
            )
        data = np.array(y).T
        sobol = [
            analyzer.analyze(
                problem,
                d,
                print_to_console=False,
                calc_second_order=config.sobol_second,
            )
            for d in data
        ]
    else:
        raise ValueError(
            "method variable must be 'pce', 'project', 'salib', 'pq-lars-loo', 'pq-sp-loo', or 'fn-lars-loo'"
        )

    if method == "salib":
        return sobol
    else:
        sobol_t, sobol_2, sobol = get_analytical_sobol(fourier, alpha, config)
        return fourier, surrogate, alpha, sobol, sobol_2, sobol_t
