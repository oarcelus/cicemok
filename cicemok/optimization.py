import chaospy as cp
import logging
import pickle
import json
from pathlib import Path
import os
import numpy as np
import pybobyqa
import mph
import matplotlib.pyplot as plt

from cicemok import comsol
from cicemok.configuration import (
    ComsolConfiguration,
    ExperimentConfiguration,
    CurrentConfigurations,
)


def objective_function(
    input: np.ndarray,
    experiments: list[np.ndarray],
    model: mph.Model,
    configs: list[ComsolConfiguration],
    surrogates: list | None = None,
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
            logging.info("SUCCESS: Successful COMSOL evaluations")
        else:
            logging.info("CRASH: COMSOL Failed")

        if result is None and surrogates is None:
            return np.inf

        if result is None and surrogates is not None:
            if surrogates[idx] is None:
                return np.inf
            else:
                time = surrogates[idx][0]
                evaluation = surrogates[idx][1](*input)
                result = np.column_stack((time, evaluation))

        assert isinstance(result, np.ndarray)

        time = result[:, 0]
        evaluation = result[:, 1]
        experiment = np.interp(time, experiment[:, 0], experiment[:, 1])

        lstsq += np.sum((evaluation - experiment) ** 2.0)  # type: ignore

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

        if mean1 / mean2 > 0.9:
            return idx2
        else:
            return None
    except StopIteration:
        return None


def optimize_parameters_ode(
    ncores: int,
    input0: np.ndarray,
    experiment: dict[int, np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    filename_comsol: str,
    names: list[str],
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

    indexes = get_most_sensitive(log_name)
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

        print(ode)
        comsol_cfg = ComsolConfiguration(
            names=names,
            filename=filename_comsol,
            expression=expression,
            unit=units,
            database=database,
            evname=evname,
            isocname=isocname,
        )

        ode_cfg = CurrentConfigurations(
            isoc=idx[3], texp=idx[2], experiment=ode.experiment
        )
        comsol_cfg.experiment = ode_cfg
        cfgs.append(comsol_cfg)
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

            ode_cfg = CurrentConfigurations(
                isoc=idx_noe[3], texp=idx_noe[2], experiment=noe.experiment
            )
            comsol_cfg.experiment = ode_cfg
            cfgs.append(comsol_cfg)
            experiments.append(experiment[idx_noe[0]].T)
            if with_surrogate:
                surrs.append(noe_surr)
            else:
                surrs.append(None)

        soln = pybobyqa.solve(
            objective_function,
            input0,
            args=(experiments, model, cfgs, surrs),
            bounds=tuple(bounds),
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

        model = comsol.set_model_parameters(soln.x, model, comsol_cfg)

        name = names.pop(idx[0])
        expression.pop(idx[0])
        units.pop(idx[0])
        estimated.append(idx[0])
        logging.info(
            f"ESTIMATED: Parameters {' '.join([str(est) for est in estimated])} -> Last Parameter {name} -> X: {soln.x} F: {soln.f}"
        )
