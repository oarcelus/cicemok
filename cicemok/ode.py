import logging
import multiprocessing
import os
import pickle
import queue

import matplotlib.pyplot as plt
import chaospy as cp
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger

from cicemok import comsol, ode, sensitivity
from cicemok.configuration import (
    ComsolConfiguration,
    ExperimentConfiguration,
    EvaluationConfiguration,
    SensitivityConfiguration,
)


logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.getcwd(), "ode.log"),
    encoding="utf-8",
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate_experiment(config: ExperimentConfiguration) -> np.ndarray:
    assert (
        isinstance(config.texp, float)
        and isinstance(config.dt, float)
        and isinstance(config.rate, float)
        and isinstance(config.rmax, float)
        and isinstance(config.dynamics, float)
    )

    n = int(config.texp * config.dynamics / (2.0 * config.dt)) + 1
    gamma = cp.Gamma(config.rate, config.rate / config.rmax)
    samples = gamma.sample(n)
    samples = np.clip(samples, config.minrate, config.maxrate, out=samples)
    chdch = np.random.choice([-1, 1], size=n)

    samples = samples * chdch

    dt = config.texp / n
    time = np.arange(n) * dt

    results = np.column_stack((time, samples))

    return results


def postprocess_experiment(
    samples: np.ndarray, config: ExperimentConfiguration
) -> np.ndarray:
    """Bounds samples to configuration limits
    1) Modify Crate if socinit is below socmin or above socmax and steps is discharging / charging
    2) Modify Crate if trajectory reaches to socmin or socmax"""
    assert (
        isinstance(config.texp, float)
        and isinstance(config.isoc, float)
        and isinstance(config.minsoc, float)
        and isinstance(config.maxsoc, float)
    )

    dt = config.texp / samples.shape[0] / 3600.0

    time = samples[:, 0]
    samples = samples[:, 1]

    # 1)
    soc = config.isoc
    if soc < config.minsoc and samples[0] < 0:
        val = (config.minsoc - soc + 0.01) / dt
        samples[0] = val
    elif soc > config.maxsoc and samples[0] > 0:
        val = (config.maxsoc - soc - 0.01) / dt
        samples[0] = val

    # 2)
    for i, sample in enumerate(samples.copy()):
        if soc + sample * dt < config.minsoc:
            val = (config.minsoc - soc) / dt
            samples[i] = val
        elif soc + sample * dt > config.maxsoc:
            val = (config.maxsoc - soc) / dt
            samples[i] = val

        soc += samples[i] * dt

    results = np.column_stack((time, samples))

    return results


def total_shannon_entropy(sobol: np.ndarray) -> float:
    avg = np.mean(sobol, axis=1)

    return 1.0 / np.sum(avg * np.log(avg))


def time_shannon_entropy(sobol: np.ndarray) -> np.ndarray:
    return np.sum(np.sum(sobol * np.log(sobol), axis=0))


def parameter_dependency(sobol: np.ndarray) -> np.ndarray:
    return np.sum(1.0 - np.sum(sobol, axis=0))


def init_pool_parallel_optimization(
    np: int = 1, npool: int = 1, ncores: int = 1, **kwargs
):
    assert "idxs" in kwargs
    assert np <= len(kwargs["idxs"])

    jobs = multiprocessing.Queue()
    for idx in kwargs["idxs"]:
        jobs.put(idx)

    for _ in range(np):
        process = multiprocessing.Process(
            target=lambda x, y, z: parallel_pool_worker(x, y, z, **kwargs),
            args=(npool, ncores, jobs),
        )
        process.start()


def parallel_pool_worker(
    npool: int,
    ncores: int,
    jobs: multiprocessing.Queue,
    dynamics: tuple[float, float],
    isoc: tuple[float, float],
    rate: tuple[float, float],
    texp: tuple[float, float],
    rmax: float,
    dt: float,
    minsoc: float,
    maxsoc: float,
    minrate: float,
    maxrate: float,
    filename: str,
    names: list[str],
    idxs: list[int],
    expression: list[str],
    units: list[str],
    database: str,
    evname: str,
    isocname: str,
    order: int,
    distribution: cp.J,
    method: str,
    rule: str,
    kind: str,
    kappa: float,
    kappa_decay: float,
    kappa_decay_delay: int,
    init_points: int,
    n_iter: int,
    log_name: str,
    threads_config: str,
):
    global experiment_cfg
    global pool
    global comsol_cfg
    global sens_cfg
    global idx
    global bo_iter

    import mkl
    mkl.set_num_threads(npool*ncores) 

    if method == "point_collocation":
        pool_experiment_optimization = pool_experiment_optimization_pc
    elif method == "pseudo_spectral":
        pool_experiment_optimization = pool_experiment_optimization_ps
    elif method == "pck":
        pool_experiment_optimization = pool_experiment_optimization_pck
    else:
        raise ValueError("Method not implemented")

    # Build Experiment
    experiment_cfg = ExperimentConfiguration(
        rmax=rmax,
        dt=dt,
        minsoc=minsoc,
        maxsoc=maxsoc,
        minrate=minrate,
        maxrate=maxrate,
    )

    # Build Comsol Config
    comsol_cfg = ComsolConfiguration(
        names=names,
        filename=filename,
        expression=expression,
        unit=units,
        database=database,
        evname=evname,
        isocname=isocname,
    )

    # Build Sensititvity Config
    sens_cfg = SensitivityConfiguration(
        order=order, distribution=distribution, rule=rule, config=comsol_cfg
    )

    # Start Computing Processes for COMSOL
    init_event = multiprocessing.Event()
    pool = multiprocessing.Pool(
        processes=npool,
        initializer=comsol.setup_comsol_worker,
        initargs=(ncores, comsol_cfg, init_event),
    )
    try:
        init_event.wait()

        # Set Bayessian Optimizer for each target parameter
        bounds = {"dynamics": dynamics, "isoc": isoc, "rate": rate, "texp": texp}
        while True:
            try:
                idx = jobs.get(block=False)
            except queue.Empty:
                break

            assert idx in idxs

            bo_iter = 0
            optimizer = BayesianOptimization(
                f=pool_experiment_optimization, pbounds=bounds, verbose=2
            )
            acquisition = UtilityFunction(
                kind=kind, kappa=kappa, kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay
            )
            logger = JSONLogger(path=f"{log_name}_param_{idx}")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize(
                init_points=init_points, n_iter=n_iter, acquisition_function=acquisition
            )

    finally:
        pool.close()
        pool.join()


def init_experiment_optimization(
    dynamics: tuple[float, float],
    isoc: tuple[float, float],
    rate: tuple[float, float],
    texp: tuple[float, float],
    rmax: float,
    dt: float,
    minsoc: float,
    maxsoc: float,
    minrate: float,
    maxrate: float,
    filename: str,
    names: list[str],
    idxs: list[int],
    expression: list[str],
    units: list[str],
    database: str,
    evname: str,
    isocname: str,
    order: int,
    distribution: cp.J,
    rule: str,
    kind: str,
    kappa: float,
    kappa_decay: float,
    kappa_decay_delay: int,
    init_points: int,
    n_iter: int,
    log_name: str,
):
    global experiment_cfg
    global model
    global comsol_cfg
    global sens_cfg
    global idx
    global bo_iter

    # Start Comsol Client
    client = comsol.start_client()

    # Open COMSOL model
    model = comsol.load_model(client, filename)

    # Build Experiment
    experiment_cfg = ExperimentConfiguration(
        rmax=rmax,
        dt=dt,
        minsoc=minsoc,
        maxsoc=maxsoc,
        minrate=minrate,
        maxrate=maxrate,
    )

    # Build Comsol Config
    comsol_cfg = ComsolConfiguration(
        names=names,
        filename=filename,
        expression=expression,
        unit=units,
        database=database,
        evname=evname,
        isocname=isocname,
    )

    # Build Sensititvity Config
    sens_cfg = SensitivityConfiguration(
        order=order, distribution=distribution, rule=rule, config=comsol_cfg
    )

    # Set Bayessian Optimizer for each target parameter
    bounds = {"dynamics": dynamics, "isoc": isoc, "rate": rate, "texp": texp}
    for idx in idxs:
        bo_iter = 0
        optimizer = BayesianOptimization(
            f=experiment_optimization, pbounds=bounds, verbose=2
        )
        acquisition = UtilityFunction(
            kind=kind, kappa=kappa, kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay
        )
        logger = JSONLogger(path=f"{log_name}_param_{idx}")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(
            init_points=init_points, n_iter=n_iter, acquisition_function=acquisition
        )


def pool_experiment_optimization_ps( 
    dynamics: float,
    isoc: float,
    rate: float,
    texp: float,
):
    global experiment_cfg
    global pool
    global comsol_cfg
    global sens_cfg
    global idx
    global bo_iter

    experiment_cfg.dynamics = dynamics
    experiment_cfg.isoc = isoc
    experiment_cfg.rate = rate
    experiment_cfg.texp = texp

    experiment = ode.generate_experiment(experiment_cfg)
    experiment_cfg.experiment = ode.postprocess_experiment(experiment, experiment_cfg)
    comsol_cfg.experiment = experiment_cfg
    sens_cfg.config = comsol_cfg

    polyno = sensitivity.generate_polynomials(sens_cfg)
    samples, weights, evals = sensitivity.evaluate_ps_pool(pool, sens_cfg)
    try:
        time, evaluations = sensitivity.curate_none_evaluations(evals, samples)
        evaluations = sensitivity.curate_cutoff_evaluations(evaluations, samples)
    except ValueError as error:
        logging.error(f"Parameter: {idx} -> BO Loop: {bo_iter} ({error})")
        bo_iter += 1
        return 0.0

    logging.info(f"SUCCESS: Parameter: {idx} -> BO Loop: {bo_iter}")
    logging.info(f"SURROGATE: START -> Parameter: {idx} -> BO Loop: {bo_iter}")
    sobol, surrogate = sensitivity.get_sobol_ps(polyno, samples, weights, evaluations, sens_cfg)
    logging.info(f"SURROGATE: END -> Parameter: {idx} -> BO Loop: {bo_iter}")

    # Save surrogate for the current iteration
    with open(f"surrogate_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump([time, surrogate], file)

    # Save experiment for the current iteration
    with open(f"experiment_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(experiment_cfg, file)

    # Save sobol indices of the full time series
    with open(f"sobol_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(sobol, file)

    bo_iter += 1
    return np.mean(sobol[idx, :])


def pool_experiment_optimization_pc( 
    dynamics: float,
    isoc: float,
    rate: float,
    texp: float,
):
    global experiment_cfg
    global pool
    global comsol_cfg
    global sens_cfg
    global idx
    global bo_iter

    experiment_cfg.dynamics = dynamics
    experiment_cfg.isoc = isoc
    experiment_cfg.rate = rate
    experiment_cfg.texp = texp

    experiment = ode.generate_experiment(experiment_cfg)
    experiment_cfg.experiment = ode.postprocess_experiment(experiment, experiment_cfg)
    comsol_cfg.experiment = experiment_cfg
    sens_cfg.config = comsol_cfg

    polyno = sensitivity.generate_polynomials(sens_cfg)
    samples, evals = sensitivity.evaluate_models_pool(pool, polyno, sens_cfg)
    try:
        time, evaluations = sensitivity.curate_none_evaluations(evals, samples)
        evaluations = sensitivity.curate_cutoff_evaluations(evaluations, samples)
    except ValueError as error:
        logging.error(f"Parameter: {idx} -> BO Loop: {bo_iter} ({error})")
        bo_iter += 1
        return 0.0

    logging.info(f"SUCCESS: Parameter: {idx} -> BO Loop: {bo_iter}")
    logging.info(f"SURROGATE: START -> Parameter: {idx} -> BO Loop: {bo_iter}")
    sobol, surrogate = sensitivity.get_sobol(polyno, samples, evaluations, sens_cfg)
    logging.info(f"SURROGATE: END -> Parameter: {idx} -> BO Loop: {bo_iter}")

    # Save surrogate for the current iteration
    with open(f"surrogate_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump([time, surrogate], file)

    # Save experiment for the current iteration
    with open(f"experiment_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(experiment_cfg, file)

    # Save sobol indices of the full time series
    with open(f"sobol_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(sobol, file)

    bo_iter += 1
    return np.mean(sobol[idx, :])


def pool_experiment_optimization_pck( 
    dynamics: float,
    isoc: float,
    rate: float,
    texp: float,
):
    global experiment_cfg
    global pool
    global comsol_cfg
    global sens_cfg
    global idx
    global bo_iter

    experiment_cfg.dynamics = dynamics
    experiment_cfg.isoc = isoc
    experiment_cfg.rate = rate
    experiment_cfg.texp = texp

    experiment = ode.generate_experiment(experiment_cfg)
    experiment_cfg.experiment = ode.postprocess_experiment(experiment, experiment_cfg)
    comsol_cfg.experiment = experiment_cfg
    sens_cfg.config = comsol_cfg

    polyno = sensitivity.generate_polynomials(sens_cfg, normed=True)
    samples, evals = sensitivity.evaluate_models_pool(pool, polyno, sens_cfg)
    try:
        time, evaluations = sensitivity.curate_none_evaluations(evals, samples)
        evaluations = sensitivity.curate_cutoff_evaluations(evaluations, samples)
    except ValueError as error:
        logging.error(f"Parameter: {idx} -> BO Loop: {bo_iter} ({error})")
        bo_iter += 1
        return 0.0

    logging.info(f"SUCCESS: Parameter: {idx} -> BO Loop: {bo_iter}")
    logging.info(f"SURROGATE: START -> Parameter: {idx} -> BO Loop: {bo_iter}")
    sobol, surrogate = sensitivity.get_sobol_pck(polyno, samples, evaluations, sens_cfg)
    logging.info(f"SURROGATE: END -> Parameter: {idx} -> BO Loop: {bo_iter}")

    # Save surrogate for the current iteration
    with open(f"surrogate_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump([time, surrogate], file)

    # Save experiment for the current iteration
    with open(f"experiment_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(experiment_cfg, file)

    # Save sobol indices of the full time series
    with open(f"sobol_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(sobol, file)

    bo_iter += 1
    return np.mean(sobol[idx, :])


def experiment_optimization(
    dynamics: float,
    isoc: float,
    rate: float,
    texp: float,
):
    global experiment_cfg
    global model
    global comsol_cfg
    global sens_cfg
    global idx
    global bo_iter

    experiment_cfg.dynamics = dynamics
    experiment_cfg.isoc = isoc
    experiment_cfg.rate = rate
    experiment_cfg.texp = texp

    experiment = ode.generate_experiment(experiment_cfg)
    experiment_cfg.experiment = ode.postprocess_experiment(experiment, experiment_cfg)
    comsol_cfg.experiment = experiment_cfg
    sens_cfg.config = comsol_cfg

    model = comsol.set_configuration(model, comsol_cfg)
    polyno, samples, results = sensitivity.evaluate_models(model, sens_cfg)

    try:
        time, evaluations = sensitivity.curate_none_evaluations(results, samples)
        evaluations = sensitivity.curate_cutoff_evaluations(evaluations, samples)
    except ValueError as error:
        logging.error(f"Paramer: {idx} -> BO Loop: {bo_iter} ({error})")
        bo_iter += 1
        return 0.0

    logging.info(f"SUCCESS: Paramer: {idx} -> BO Loop: {bo_iter}")
    logging.info(f"SURROGATE: START -> Parameter: {idx} -> BO Loop: {bo_iter}")
    sobol, surrogate = sensitivity.get_sobol(polyno, samples, evaluations, sens_cfg)
    logging.info(f"SURROGATE: END -> Parameter: {idx} -> BO Loop: {bo_iter}")

    # Save surrogate for the current iteration
    with open(f"surrogate_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump([time, surrogate], file)

    # Save experiment for the current iteration
    with open(f"experiment_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(experiment_cfg, file)

    # Save sobol indices of the full time series
    with open(f"sobol_param{idx}_boiter{bo_iter}.pkl", "wb") as file:
        pickle.dump(sobol, file)

    bo_iter += 1
    return np.mean(sobol[idx, :])


def get_surrogate_samples(filenames_surrogate: list[str], **kwargs):
    assert len(filenames_surrogate) == len(kwargs["filenames_experiment"]) # Must correspond

    samples, results = run_parallel_mc_samples(**kwargs)
    data = []
    for i, filename_surrogate in enumerate(filenames_surrogate):
        with open(filename_surrogate, "rb") as file:
            values: list = pickle.load(file)

        time_surrogate = values[0]
        volt_surrogate = values[1]

        try:
            time, evaluations = sensitivity.curate_none_evaluations(results[i], samples[i])
            evaluations = sensitivity.curate_cutoff_evaluations(evaluations, samples[i])
        except ValueError as error:
            logging.error(f"{error}")
            return

        volt_surrogate = [volt_surrogate(*sample) for sample in samples[i].T]
        volt_surrogate = [
            np.interp(time, time_surrogate, volt) for volt in volt_surrogate
        ]

        data.append([time, evaluations, volt_surrogate])
    
    # Save sobol indices of the full time series
    with open("surrogate_evaluations.pkl", "wb") as file:
        pickle.dump(data, file)

    return data


def run_parallel_mc_samples(
    npool: int,
    ncores: int,
    nsamples: int,
    filenames_experiment: list[str],
    filename_comsol: str,
    names: list[str],
    idxs: list[int],
    expression: list[str],
    units: list[str],
    database: str,
    evname: str,
    isocname: str,
    distribution: cp.J,
    rule: str,
):
    # Build Comsol Config
    comsol_cfg = ComsolConfiguration(
        names=names,
        filename=filename_comsol,
        expression=expression,
        unit=units,
        database=database,
        evname=evname,
        isocname=isocname,
    )

    # Start Computing Processes for COMSOL
    init_event = multiprocessing.Event()
    pool = multiprocessing.Pool(
        processes=npool,
        initializer=comsol.setup_comsol_worker,
        initargs=(ncores, comsol_cfg, init_event),
    )

    try:
        init_event.wait()
        sample_params = []
        evals_params = []
        for filename_experiment in filenames_experiment:
            with open(filename_experiment, "rb") as file:
                experiment_cfg: ExperimentConfiguration = pickle.load(file)

            comsol_cfg.experiment = experiment_cfg

            # Build Sensititvity Config
            eval_cfg = EvaluationConfiguration(
                distribution=distribution, rule=rule, config=comsol_cfg
            )

            logging.info(
                f"EVALUATE: MC Samples -> START -> Experiment: {filename_experiment}"
            )
            samples, evals = sensitivity.evaluate_mc_pool(pool, nsamples, eval_cfg)
            logging.info("EVALUATE: MC Samples -> END")
            sample_params.append(samples)
            evals_params.append(evals)

    finally:
        pool.close()
        pool.join()

    return sample_params, evals_params
