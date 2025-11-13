import multiprocessing
from collections.abc import Iterable
from typing import Union, Any
import os

import logging
import matplotlib.pyplot as plt
import pybamm
import pybammeis
import numpy as np
from scipy import interpolate

from cicemok.configuration import PybammConfiguration, ExperimentType


def load_model(config: PybammConfiguration) -> pybamm.BaseModel:
    """Load a PyBaMM model. Defaults to DFN."""

    if config.options is None:
        model_map = {
            "DFN": pybamm.lithium_ion.DFN(),
            "SPM": pybamm.lithium_ion.SPM(),
            "SPMe": pybamm.lithium_ion.SPMe(),
        }
    else:
        model_map = {
            "DFN": pybamm.lithium_ion.DFN(config.options),
            "SPM": pybamm.lithium_ion.SPM(config.options),
            "SPMe": pybamm.lithium_ion.SPMe(config.options),
        }

    return model_map.get(config.modeltype, pybamm.lithium_ion.DFN())


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
    parameters["Lower voltage cut-off [V]"] = "[input]"
    parameters["Upper voltage cut-off [V]"] = "[input]"

    if config.conditions.temp is not None and config.conditions.init_temp is not None:
        parameters["Ambient temperature [K]"] = "[input]"
        parameters["Initial temperature [K]"] = "[input]"

    if config.conditions.experiment == ExperimentType.EXPERIMENT:
        return parameters
    elif config.conditions.experiment == ExperimentType.EIS:
        return parameters
    elif config.conditions.experiment == ExperimentType.PROFILE:
        assert isinstance(config.conditions.current, pybamm.Interpolant)
        parameters["Current function [A]"] = config.conditions.current
        return parameters
    elif config.conditions.experiment == ExperimentType.CC:
        assert isinstance(config.conditions.current, float)
        parameters["Current function [A]"] = (
            "[input]"  # Only condition for now which can loop over currents at solve time
        )
        return parameters
    else:
        raise ValueError("ExperimentType enum value not specified")

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
    parameters["Lower voltage cut-off [V]"] = config.conditions.cutoff[0]
    parameters["Upper voltage cut-off [V]"] = config.conditions.cutoff[1]

    if config.conditions.temp is not None and config.conditions.init_temp is not None:
        parameters["Ambient temperature [K]"] = config.conditions.temp
        parameters["Initial temperature [K]"] = config.conditions.init_temp

    if config.conditions.experiment == ExperimentType.CC:
        parameters["Current function [A]"] = config.experiment.current

    return parameters


def run_pybamm_model(
    input: np.ndarray, simulation: pybamm.Simulation, config: PybammConfiguration
) -> np.ndarray | None:
    parameters = set_model_parameters(input, simulation, config)
    try:
        if config.conditions.experiment in (ExperimentType.CC, ExperimentType.EIS):
            solution = simulation.solve(config.conditions.texp, inputs=parameters)
        else:
            solution = simulation.solve(inputs=parameters)

        if config.conditions.experiment == ExperimentType.EIS:
            result: list = [
                config.conditions.texp,
                solution.real,
                solution.imag,
            ]
        else:
            result: list = [solution[name].entries for name in config.expression]

        res = np.array(result).T
        f = interpolate.interp1d(
            res[:, 0], res[:, 1:], assume_sorted=False, axis=0, fill_value="extrapolate"
        )

        x = config.xinterp
        y = f(x)

        res = np.column_stack((x, y))
        return res
    except Exception as e:
        print(e)
        return None


def setup(config: PybammConfiguration):
    model = load_model(config)
    params = pybamm.ParameterValues(config.parameter_set)
    params = set_input_parameters(params, config)

    if config.solver_safety:
        solver = pybamm.CasadiSolver(mode="safe")
    else:
        solver = pybamm.IDAKLUSolver()

    if config.conditions.experiment == ExperimentType.EXPERIMENT:
        assert isinstance(config.conditions.current, pybamm.Experiment)
        sim = pybamm.Simulation(
            model,
            solver=solver,
            experiment=config.conditions.current,
            parameter_values=params,
        )
    elif config.conditions.experiment == ExperimentType.EIS:
        sim = pybammeis.EISSimulation(model, parameter_values=params)
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
