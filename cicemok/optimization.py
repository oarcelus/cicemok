import chaospy as cp
import pickle
import json
from pathlib import Path
import os
import numpy as np
import dfols
import mph

from cicemok import comsol
from cicemok.configuration import ComsolConfiguration, CurrentConfigurations


def objective_function(
    input: np.ndarray,
    experiment: np.ndarray,
    model: mph.Model,
    config: ComsolConfiguration,
) -> float:
    model = comsol.set_configuration(model, config)
    result = comsol.run_comsol_model(input, model, config)
    if result is None:
        raise ValueError("COMSOL evaluation is None")

    time = result[:, 0]
    evaluation = result[:, 1]
    experiment = np.interp(time, experiment[:, 0], experiment[:, 1])

    return np.sum((evaluation - experiment) ** 2.0) # type: ignore


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


def check_noe(sobol: np.ndarray, index: int) -> int | None:
    means = np.mean(sobol, axis=1)
    assert np.argmax(means) == index

    idx2 = np.argsort(means)[-2]

    mean1 = means[index]
    mean2 = np.sort(means)[-2]

    if mean1 / mean2 > 0.9:
        return idx2
    else:
        return None


def optimize_parameters_ode(
    ncores: int,
    input0: np.ndarray,
    experiment: dict[int, np.ndarray],
    bounds: list[np.ndarray],
    filename_comsol: str,
    names: list[str],
    expression: list[str],
    units: list[str],
    database: str,
    evname: str,
    isocname: str,
    log_name: str,
    experiment_name: str,
):
    indexes = get_most_sensitive(log_name)
    client = comsol.start_client(cores=ncores)
    model = client.load(filename_comsol)
    while names:
        idx = indexes[-1]
        cfgs = []

        file_ode = f"experiment_param{idx[0]}_boiter{idx[1]}.pkl"
        file_surr = f"surrogate_param{idx[0]}_boiter{idx[1]}.pkl"
        file_sobol = f"sobol_param{idx[0]}_boiter{idx[1]}.pkl"
        with open(file_ode, "rb") as f:
            ode: np.ndarray = pickle.load(f)
        with open(file_surr, "rb") as f:
            surrogate: list = pickle.load(f)
        with open(file_sobol, "rb") as f:
            sobol: np.ndarray = pickle.load(f)

        comsol_cfg = ComsolConfiguration(
            names=names,
            filename=filename_comsol,
            expression=expression,
            unit=units,
            database=database,
            evname=evname,
            isocname=isocname,
        )

        ode_cfg = CurrentConfigurations(isoc=idx[3], texp=idx[2], experiment=ode)
        comsol_cfg.experiment = ode_cfg
        cfgs.append(comsol_cfg)

        val = check_noe(sobol, idx[0])
        if val is not None:
            idx_noe = next(index for index in indexes if index[0] == val) 
            file_noe = f"experiment_param{idx_noe[0]}_boiter{idx_noe[1]}.pkl"
            file_noe_surr = f"surrogate_param{idx_noe[0]}_boiter{idx_noe[1]}.pkl"
            with open(file_noe, "rb") as f:
                noe: np.ndarray = pickle.load(f)
            with open(file_noe_surr, "rb") as f:
                noe_surr: list = pickle.load(f)

            comsol_cfg = ComsolConfiguration(
                names=names,
                filename=filename_comsol,
                expression=expression,
                unit=units,
                database=database,
                evname=evname,
                isocname=isocname,
            )

            ode_cfg = CurrentConfigurations(isoc=idx_noe[3], texp=idx_noe[2], experiment=noe)
            comsol_cfg.experiment = ode_cfg
            cfgs.append(comsol_cfg)

        dfols.solve(
            objective_function,
            input0,
            args=(experiment, model, comsol_cfg),
            bounds=tuple(bounds),
        )
