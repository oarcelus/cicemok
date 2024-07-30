import numpy as np
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt
from typing import Literal


def objective(
    x: list[float],
    p_ocp: np.ndarray,
    n_ocp: np.ndarray,
    cell_ocv: np.ndarray,
    c_host_n: float,
    c_host_p: float,
    mode: str,
    method: str,
):
    yhost_p = x[0]
    yhost_n = x[1]
    ylitot = x[2]
    soln0 = x[3]

    soc = cell_ocv[:, 0] / cell_ocv[-1, 0]

    if mode == "discharge":
        soc = 1 - soc

    soln = soln0 + soc/yhost_n 
    solp = ylitot/yhost_p - soln*yhost_n/yhost_p

    ocpn = np.interp(soln, n_ocp[:, 0], n_ocp[:, 1])
    ocpp = np.interp(solp, p_ocp[:, 0], p_ocp[:, 1])

    ocvsim = ocpp - ocpn

    return np.sum((cell_ocv[:, 1] - ocvsim) ** 2)


def balance_electrodes(
    p_ocp: np.ndarray,
    n_ocp: np.ndarray,
    cell_ocv: np.ndarray,
    c_host_p: float,
    c_host_n: float,
    c_dch_max_n_cyl: float,
    mode: Literal["charge", "discharge"], 
    method: str = "BFGS",
):
    """
    p_ocp: (N, 2) numpy array with SoL vs Voltage
    n_ocp: (N', 2) numpy array with SoL vs Voltage
    cell_ocv: (N'', 2) numpy array with Capacity vs Voltage in full-cell
    c_host_p: Positive host capacity scaled to the 'cell_ocv' level
    c_host_n: Negative host capacity scaled to the 'cell_ocv' level
    c_dch_max_n_cyl: Available capacity of Li ions between positive and negative
    """
    yhost_p_init = c_host_p / cell_ocv[-1, 0]
    yhost_n_init = c_host_n / cell_ocv[-1, 0]
    ylitot_init = c_dch_max_n_cyl / cell_ocv[-1, 0]
    soln0_init = 0.0 

    obj = partial(
        objective,
        p_ocp=p_ocp,
        n_ocp=n_ocp,
        cell_ocv=cell_ocv,
        c_host_n=c_host_n,
        c_host_p=c_host_p,
        mode=mode,
        method=method,
    )

    res = minimize(obj, [yhost_p_init, yhost_n_init, ylitot_init, soln0_init], method=method)

    return res
