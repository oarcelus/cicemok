import multiprocessing
from typing import Union

import matplotlib.pyplot as plt
import pybamm
import numpy as np

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
    csinit_n = cs0_n + config.isoc * (cs100_n - cs0_n)
    csinit_p = cs100_p + (1 - config.isoc) * (cs0_p - cs100_p)

    # Set initial concentrations
    parameter_values["Initial concentration in negative electrode [mol.m-3]"] = csinit_n
    parameter_values["Initial concentration in positive electrode [mol.m-3]"] = csinit_p

    return parameter_values


def set_model_parameters(
    input: np.ndarray, config: PybammConfiguration
) -> dict[str, float]:
    """Set model parameters."""
    assert len(config.names) == len(input)
    parameters: dict[str, float] = {k: v for k, v in zip(config.names, input.tolist())}
    return parameters


def run_pybamm_model(
    input: np.ndarray, simulation: pybamm.Simulation, config: PybammConfiguration
) -> np.ndarray | None:
    parameters = set_model_parameters(input, config)
    try:
        solution = simulation.solve(inputs=parameters)
        result: list = [solution[name].entries for name in config.expression]

        return np.array(result)
    except Exception:
        return None


def setup_pybamm_worker(config: PybammConfiguration, event: multiprocessing.Event):
    global sim

    model = load_model(config.modeltype)
    params = pybamm.ParameterValues(config.parameter_set)
    model = set_soc(params, config)
    experiment = pybamm.Experiment(config.experiment)
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
    event.set()


# def set_model_parameters_pool(input: np.ndarray, config: ComsolConfiguration):
#     global model
#
#     model = set_model_parameters(input, model, config)


def pybamm_worker_pool(sample: np.ndarray, config: PybammConfiguration):
    global sim

    result = run_pybamm_model(sample, sim, config)

    return result
