import dataclasses
import json
import logging
import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import mph
import numpy as np
import pybobyqa
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from cicemok import comsol, sensitivity
from cicemok.configuration import (
    ComsolConfiguration,
    CurrentConfigurations,
    ExperimentConfiguration,
)

_history = []


class ComsolCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.xhist = []
        self.fhist = []

    def notify(self, algorithm):
        x = algorithm.pop.get("X")
        f = algorithm.pop.get("F")

        idf = np.argmin(f)

        self.xhist.append(x[idf, :])
        self.fhist.append(f[idf])

        logging.info(
            f"PYMOO: Generation {algorithm.n_gen} -> X: {x[idf, :]} -> F: {f[idf]}"
        )


class ComsolProblem(Problem):
    def __init__(
        self,
        experiments: list[np.ndarray],
        configs: list[ComsolConfiguration],
        model: mph.Model | None = None,
        pool=None,
        n_var=-1,
        n_obj=1,
        n_ieq_constr=0,
        n_eq_constr=0,
        xl=None,
        xu=None,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            n_eq_constr=n_eq_constr,
            xl=xl,
            xu=xu,
        )

        if pool is None:
            self._pool = None
        else:
            self._pool = pool

        self._experiments = experiments
        self._configs = configs
        self._model = model

    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs,
    ):
        objectives = []
        for idx, (experiment, config) in enumerate(
            zip(self._experiments, self._configs)
        ):
            if self._pool is None:
                assert self._model is not None
                results = [
                    comsol.run_comsol_model(sample, self._model, config) for sample in x
                ]
            else:
                func = partial(comsol.comsol_worker_pool, config=config)
                results = self._pool.map(func, x)

            try:
                time, evaluations = sensitivity.curate_none_evaluations(results, x)
                evaluations = sensitivity.curate_cutoff_evaluations(evaluations, x)
                experiment = np.interp(time, experiment[:, 0], experiment[:, 1])

                lstsq = [
                    np.sum((evaluation - experiment) ** 2.0)
                    for evaluation in evaluations
                ]  # type: ignore
                objectives.append(lstsq)

            except ValueError as error:
                logging.error(f"{error}")
                raise ValueError(
                    "FATAL ERROR: We cannot continue too many evaluations failed in the given iteration"
                )
            finally:
                out["F"] = np.column_stack(objectives)


def find_closest_in_history(input: np.ndarray, scale: float) -> float:
    if not _history:
        return np.inf

    iter_history = iter(_history)
    distance = np.inf
    f = 1e10
    try:
        while True:
            distance, x, f = next(
                (np.linalg.norm(input - x), x, f)
                for x, f in iter_history
                if np.linalg.norm(input - x) < distance
            )
    except StopIteration:
        assert isinstance(f, float)
        return f * scale


def objective_pybobyqa(
    input: np.ndarray,
    experiments: list[np.ndarray],
    model: mph.Model,
    configs: list[ComsolConfiguration],
    surrogates: list | None = None,
    usehistory: bool = False,
) -> float | None:
    lstsq = 0.0

    if surrogates is None:
        assert all(len(val) == len(experiments) for val in [experiments, configs])
    else:
        assert all(
            len(val) == len(experiments) for val in [experiments, configs, surrogates]
        )

    for idx, (experiment, config) in enumerate(zip(experiments, configs)):
        model = comsol.set_configuration(model, config)
        result = comsol.run_comsol_model(input, model, config)

        if result is not None:
            logging.info("PYBOBYQA: COMSOL run successfuly")
        else:
            logging.info("PYBOBYQA: COMSOL Failed -> Checking options")

        if result is None and surrogates is None:
            if usehistory:
                logging.info("PYBOBYQA: Using history to estimate closest points")
                return find_closest_in_history(input, 10.0)
            else:
                raise ValueError(
                    "No Surrogates nor Optimization history provided, end this job now!"
                )

        if result is None and surrogates is not None:
            if surrogates[idx] is None:
                if usehistory:
                    logging.info("PYBOBYQA: Using history to estimate closest points")
                    return find_closest_in_history(input, 10.0)
                else:
                    raise ValueError(
                        "No Surrogates nor Optimization history provided, end this job now!"
                    )
            else:
                logging.info("PYBOBYQA: Using Surrogate to handle crash")
                time = surrogates[idx][0]
                evaluation = surrogates[idx][1](*input)
                result = np.column_stack((time, evaluation))

        assert isinstance(result, np.ndarray)

        time = result[:, 0]
        evaluation = result[:, 1]
        experiment = np.interp(time, experiment[:, 0], experiment[:, 1])

        lstsq += np.sum((evaluation - experiment) ** 2.0)  # type: ignore

    _history.append((input, lstsq))

    return lstsq  # type: ignore


def get_most_sensitive(log_name: str) -> list[tuple]:
    logfiles = [
        files
        for files in os.listdir(os.getcwd())
        if log_name in files and ".json" in files and os.path.isfile(files)
    ]
    indexfiles = [int(Path(file).stem.split("_")[-1]) for file in logfiles]

    indexes = []
    for i, log in zip(indexfiles, logfiles):
        targets = []
        texp = []
        isoc = []
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                    iteration = json.loads(iteration)
                    targets.append(iteration["target"])
                    texp.append(iteration["params"]["texp"])
                    isoc.append(iteration["params"]["isoc"])
                except StopIteration:
                    break

        targets = np.array(targets)
        ibo = np.argmax(targets)
        indexes.append(
            (
                i,
                ibo,
                texp[ibo],
                isoc[ibo],
                np.amax(targets),
            )
        )
        indexes = sorted(indexes, key=lambda x: x[-1])

    return indexes


def check_noe(sobol: np.ndarray, estimated: list = []) -> int | None:
    means = np.mean(sobol, axis=1)
    means = [(i, mean) for i, mean in enumerate(means)]

    sortmean = iter(sorted(means, key=lambda x: x[-1], reverse=True))
    try:
        idx1, mean1 = next(val for val in sortmean if val[0] not in estimated)
        idx2, mean2 = next(val for val in sortmean if val[0] not in estimated)

        if mean2 / mean1 > 0.9:
            return idx2
        else:
            return None
    except StopIteration:
        return None


def optimize_parameters_multi_obj(
    npool: int,
    ncores: int,
    n_gen: int,
    pop_size: int,
    loads: list[np.ndarray],
    experiments: list[np.ndarray],
    isocs: list[float],
    texps: list[float],
    bounds: tuple[np.ndarray, np.ndarray],
    filename_comsol: str,
    names: list[str],
    expression: list[str],
    units: list[str],
    database: str,
    evname: str,
    isocname: str,
):
    logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(os.getcwd(), "optimize.log"),
        encoding="utf-8",
        force=True,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    assert all(len(var) == len(loads) for var in [loads, experiments, isocs, texps])
    assert all(len(var) == len(names) for var in bounds)

    comsol_cfg = ComsolConfiguration(
        names=names,
        filename=filename_comsol,
        expression=expression,
        unit=units,
        database=database,
        evname=evname,
        isocname=isocname,
    )

    if npool > 1:
        init_event = multiprocessing.Event()
        pool = multiprocessing.Pool(
            processes=npool,
            initializer=comsol.setup_comsol_worker,
            initargs=(ncores, comsol_cfg, init_event),
        )
        init_event.wait()
        model = None
    else:
        client = comsol.start_client(cores=ncores)
        model = client.load(filename_comsol)

    cfgs = []
    for load, texp, isoc in zip(loads, texps, isocs):
        current_cfg = CurrentConfigurations(isoc=isoc, texp=texp, experiment=load)
        tmp_cfg = dataclasses.replace(comsol_cfg)
        tmp_cfg.experiment = current_cfg
        cfgs.append(tmp_cfg)

    if npool > 1:
        problem = ComsolProblem(
            experiments,
            cfgs,
            None,
            pool,  # type: ignore
            n_var=len(names),
            n_obj=len(experiments),
            xl=bounds[0],
            xu=bounds[1],
        )
    else:
        problem = ComsolProblem(
            experiments,
            cfgs,
            model,
            None,
            n_var=len(names),
            n_obj=len(experiments),
            xl=bounds[0],
            xu=bounds[1],
        )

    termination = get_termination("n_gen", n_gen)
    callback = ComsolCallback()
    algorithm = PSO(pop_size=pop_size)
    soln = minimize(
        problem,
        algorithm,
        termination,
        seed=3,
        callback=callback,
        verbose=False,
        experiments=experiments,
        configs=cfgs,
        model=model,
    )

    x = soln.X

    if npool > 1:
        func = partial(comsol.set_model_parameters_pool, config=comsol_cfg)
        pool.map(func, [x for _ in range(npool)])  # type: ignore
    else:
        model = comsol.set_model_parameters(x, model, comsol_cfg)  # type: ignore
        model.save("./test.mph")

def optimize_parameters_ode(
    ncores: int,
    input0: np.ndarray,
    experiment: dict[int, np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    filename_comsol: str,
    names: list[str],
    idxs: list[int],
    expression: list[str],
    units: list[str],
    database: str,
    evname: str,
    isocname: str,
    log_name: str,
    with_surrogate: bool,
    global_opt: bool,
    use_restarts: bool,
    rhoend: float = 1e-8,
    slowiter: float = 1e-8,
    maxfun: int = 100,
    usehistory: bool = False,
    use_pso: bool = False,
    pop_size: int = 25,
    npool: int = 1,
    n_gen: int = 1,
):
    logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(os.getcwd(), "optimize.log"),
        encoding="utf-8",
        force=True,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input0_cfg = np.copy(input0)
    bounds_cfg = bounds
    names_cfg = names.copy()
    idxs_cfg = idxs.copy()
    expression_cfg = expression.copy()
    units_cfg = units.copy()

    indexes = get_most_sensitive(log_name)
    if use_pso and npool > 1:
        comsol_cfg = ComsolConfiguration(
            names=names_cfg,
            filename=filename_comsol,
            expression=expression_cfg,
            unit=units_cfg,
            database=database,
            evname=evname,
            isocname=isocname,
        )

        init_event = multiprocessing.Event()
        pool = multiprocessing.Pool(
            processes=npool,
            initializer=comsol.setup_comsol_worker,
            initargs=(ncores, comsol_cfg, init_event),
        )
        init_event.wait()
        model = None
    else:
        client = comsol.start_client(cores=ncores)
        model = client.load(filename_comsol)

    estimated = []
    while indexes:
        idx = indexes.pop()
        cfgs = []
        experiments = []
        surrs = []

        file_ode = f"experiment_param{idx[0]}_boiter{idx[1]}.pkl"
        file_surr = f"surrogate_param{idx[0]}_boiter{idx[1]}.pkl"
        file_sobol = f"sobol_param{idx[0]}_boiter{idx[1]}.pkl"
        with open(file_ode, "rb") as f:
            ode: ExperimentConfiguration = pickle.load(f)
        with open(file_surr, "rb") as f:
            surrogate: list = pickle.load(f)
        with open(file_sobol, "rb") as f:
            sobol: np.ndarray = pickle.load(f)

        comsol_cfg_ode = ComsolConfiguration(
            names=names_cfg,
            filename=filename_comsol,
            expression=expression_cfg,
            unit=units_cfg,
            database=database,
            evname=evname,
            isocname=isocname,
        )

        ode_cfg = CurrentConfigurations(
            isoc=idx[3], texp=idx[2], experiment=ode.experiment
        )
        comsol_cfg_ode.experiment = ode_cfg
        cfgs.append(comsol_cfg_ode)
        experiments.append(experiment[idx[0]].T)
        if with_surrogate:
            surrs.append(surrogate)
        else:
            surrs.append(None)

        val = check_noe(sobol, estimated)
        if val is not None:
            idx_noe = next(index for index in indexes if index[0] == val)
            file_noe = f"experiment_param{idx_noe[0]}_boiter{idx_noe[1]}.pkl"
            file_noe_surr = f"surrogate_param{idx_noe[0]}_boiter{idx_noe[1]}.pkl"
            with open(file_noe, "rb") as f:
                noe: ExperimentConfiguration = pickle.load(f)
            with open(file_noe_surr, "rb") as f:
                noe_surr: list = pickle.load(f)

            noe_cfg = CurrentConfigurations(
                isoc=idx_noe[3], texp=idx_noe[2], experiment=noe.experiment
            )
            comsol_cfg_noe = dataclasses.replace(comsol_cfg_ode)
            comsol_cfg_noe.experiment = noe_cfg
            cfgs.append(comsol_cfg_noe)
            experiments.append(experiment[idx_noe[0]].T)
            if with_surrogate:
                surrs.append(noe_surr)
            else:
                surrs.append(None)

        if use_pso:
            if npool > 1:
                problem = ComsolProblem(
                    experiments,
                    cfgs,
                    None,
                    pool,  # type: ignore
                    n_var=len(names_cfg),
                    n_obj=len(experiments),
                    xl=bounds_cfg[0],
                    xu=bounds_cfg[1],
                )
            else:
                problem = ComsolProblem(
                    experiments,
                    cfgs,
                    model,
                    None,
                    n_var=len(names_cfg),
                    n_obj=len(experiments),
                    xl=bounds_cfg[0],
                    xu=bounds_cfg[1],
                )

            termination = get_termination("n_gen", n_gen)
            callback = ComsolCallback()
            algorithm = PSO(pop_size=pop_size)
            soln = minimize(
                problem,
                algorithm,
                termination,
                seed=3,
                callback=callback,
                verbose=False,
                experiments=experiments,
                configs=cfgs,
                model=model,
            )

            x = soln.X
            f = soln.F
        else:
            soln = pybobyqa.solve(
                objective_pybobyqa,
                input0_cfg,
                args=(experiments, model, cfgs, surrs, usehistory),  # type: ignore
                bounds=bounds_cfg,
                do_logging=True,
                rhoend=rhoend,
                seek_global_minimum=global_opt,
                maxfun=maxfun,
                scaling_within_bounds=True,
                user_params={
                    "restarts.use_restarts": use_restarts,
                    "slow.thresh_for_slow": slowiter,
                },
            )

            x = soln.x
            f = soln.f

        if use_pso and npool > 1:
            func = partial(comsol.set_model_parameters_pool, config=comsol_cfg_ode)
            pool.map(func, [x for _ in range(npool)])  # type: ignore
        else:
            model = comsol.set_model_parameters(x, model, comsol_cfg_ode)  # type: ignore
            model.save("./test.mph")

        name = names_cfg.pop(idxs_cfg.index(idx[0]))
        units_cfg.pop(idxs_cfg.index(idx[0]))
        idxs_cfg.pop(idxs_cfg.index(idx[0]))
        input0_cfg = input0[idxs_cfg]
        bounds_cfg = tuple([var[idxs_cfg] for var in bounds])
        estimated.append(idx[0])
        logging.info(
            f"ESTIMATED: Parameters {' '.join([str(est) for est in estimated])} -> Last Parameter {name} -> X: {x} F: {f}"
        )


