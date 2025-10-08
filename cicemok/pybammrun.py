import multiprocessing
from collections.abc import Iterable
from typing import Union, Any
import os

import logging
import matplotlib.pyplot as plt
import pybamm
import numpy as np
from scipy import interpolate

from cicemok.configuration import PybammConfiguration


def load_model(model_name: str = "DFN") -> pybamm.BaseModel:
    """Load a PyBaMM model. Defaults to DFN."""
    model_map = {
        "DFN": pybamm.lithium_ion.DFN(),
        "SPM": pybamm.lithium_ion.SPM(),
        "SPMe": pybamm.lithium_ion.SPMe(),
    }
    return model_map.get(model_name, pybamm.lithium_ion.DFN())


def set_soc(
    parameter_values: pybamm.ParameterValues, config: PybammConfiguration
) -> pybamm.ParameterValues:
    """
    Set initial concentrations in the negative and positive electrodes
    based on state of charge (SOC) and update the parameter values dict.

    Parameters:
    - parameter_values: pybamm.ParameterValues
    - isoc: float, initial SOC in [0, 1]
    """
    # Max concentrations
    csmax_n = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
    csmax_p = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]

    # Stoichiometries at 0% and 100% SOC
    soln0 = config.sto[0]
    solp0 = config.sto[1]
    soln100 = config.sto[2]
    solp100 = config.sto[3]

    # Convert stoichiometry to concentrations
    cs0_n = csmax_n * soln0
    cs100_n = csmax_n * soln100
    cs0_p = csmax_p * solp0
    cs100_p = csmax_p * solp100

    # Interpolate based on SOC
    csinit_n = cs0_n + config.conditions.isoc * (cs100_n - cs0_n)
    csinit_p = cs100_p + (1 - config.conditions.isoc) * (cs0_p - cs100_p)

    return csinit_n, csinit_p


def set_input_parameters(
    parameters: pybamm.ParameterValues, config: PybammConfiguration
) -> pybamm.ParameterValues:
    """Set model parameters."""
    for key in config.names:
        parameters[key] = "[input]"

    parameters["Initial concentration in negative electrode [mol.m-3]"] = "[input]"
    parameters["Initial concentration in positive electrode [mol.m-3]"] = "[input]"

    if config.experiment is None:
        parameters["Current function [A]"] = "[input]"

    return parameters


def set_model_parameters(
    input: Iterable[Any], simulation: pybamm.Simulation, config: PybammConfiguration
) -> dict[str, float]:
    """Set model parameters."""
    assert len(config.names) == len(input)

    if isinstance(input, np.ndarray):
        input = input.tolist()

    # Parameters to analyze
    parameters: dict[str, float] = {k: v for k, v in zip(config.names, input)}

    # Conditions
    aux_parameters = simulation.parameter_values
    csinit_n, csinit_p = set_soc(aux_parameters, config)

    parameters["Initial concentration in negative electrode [mol.m-3]"] = csinit_n
    parameters["Initial concentration in positive electrode [mol.m-3]"] = csinit_p

    if config.experiment is None:
        if isinstance(config.conditions.experiment, np.ndarray):
            # Space for drive-cycles or interpolates
            pass

        parameters["Current function [A]"] = config.conditions.experiment

    return parameters


def run_pybamm_model(
    input: np.ndarray, simulation: pybamm.Simulation, config: PybammConfiguration
) -> np.ndarray | None:
    parameters = set_model_parameters(input, simulation, config)
    try:
        if config.experiment is None:
            solution = simulation.solve(config.conditions.texp, inputs=parameters)
        else:
            solution = simulation.solve(inputs=parameters)

        result: list = [solution[name].entries for name in config.expression]
        res = np.array(result).T
        f = interpolate.interp1d(
            res[:, 0], res[:, 1], assume_sorted=False, fill_value="extrapolate"
        )

        x = config.xinterp
        y = f(x)

        res = np.column_stack((x, y))

        return res
    except Exception:
        return None


def setup(config: PybammConfiguration):
    model = load_model(config.modeltype)
    params = pybamm.ParameterValues(config.parameter_set)

    params = set_input_parameters(params, config)
    if config.solver_safety:
        solver = pybamm.CasadiSolver(mode="safe")
    else:
        solver = pybamm.IDAKLUSolver()

    if config.experiment is not None:
        sim = pybamm.Simulation(
            model, solver=solver, experiment=config.experiment, parameter_values=params
        )
    else:
        sim = pybamm.Simulation(model, solver=solver, parameter_values=params)

    return sim


def setup_pybamm_worker(config: PybammConfiguration, event: multiprocessing.Event):
    global sim
    sim = setup(config)
    event.set()


def pybamm_worker_pool(sample: Iterable[Any], config: PybammConfiguration):
    global sim

    result = run_pybamm_model(sample, sim, config)

    return result
