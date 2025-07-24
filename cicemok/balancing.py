import numpy as np
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt
from typing import Literal
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination


class BalancingProblem(Problem):
    def __init__(
        self,
        p_ocp: np.ndarray,
        n_ocp: np.ndarray,
        cell_ocv: np.ndarray,
        c_host_n: float,
        c_host_p: float,
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

        self._p_ocp = p_ocp
        self._n_ocp = n_ocp
        self._cell_ocv = cell_ocv
        self._c_host_n = c_host_n
        self._c_host_p = c_host_p


    def _evaluate(
        self,
        x,
        out,
        *args,
        **kwargs,
    ):

        objectives = []
        for xin in x:
            soln0 = xin[0]
            solp0 = xin[1]

            q = self._cell_ocv[-1, 0]

            soln100 = soln0 + q/self._c_host_n
            solp100 = solp0 - q/self._c_host_p

            soln = np.linspace(soln100, soln0, self._cell_ocv.shape[0])
            solp = np.linspace(solp100, solp0, self._cell_ocv.shape[0])

            ocpn = np.interp(soln, self._n_ocp[:, 0], self._n_ocp[:, 1])
            ocpp = np.interp(solp, self._p_ocp[:, 0], self._p_ocp[:, 1])

            ocvsim = ocpp - ocpn
            objectives.append(np.sum((self._cell_ocv[:, 1] - ocvsim) ** 2)
)

        out["F"] = objectives

def objective_q_fixed(
    x: list[float],
    p_ocp: np.ndarray,
    n_ocp: np.ndarray,
    cell_ocv: np.ndarray,
    c_host_n: float,
    c_host_p: float,
):
    soln0 = x[0]
    solp0 = x[1]

    q = cell_ocv[-1, 0]

    soln100 = soln0 + q/c_host_n
    solp100 = solp0 - q/c_host_p

    soln = np.linspace(soln100, soln0, cell_ocv.shape[0])
    solp = np.linspace(solp100, solp0, cell_ocv.shape[0])

    ocpn = np.interp(soln, n_ocp[:, 0], n_ocp[:, 1])
    ocpp = np.interp(solp, p_ocp[:, 0], p_ocp[:, 1])

    ocvsim = ocpp - ocpn

    return np.sum((cell_ocv[:, 1] - ocvsim) ** 2)


def objective_adimensional(
    x: list[float],
    p_ocp: np.ndarray,
    n_ocp: np.ndarray,
    cell_ocv: np.ndarray,
    c_host_n: float,
    c_host_p: float,
    mode: str,
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


def balance_electrodes_q(
    p_ocp: np.ndarray,
    n_ocp: np.ndarray,
    cell_ocv: np.ndarray,
    c_host_p: float,
    c_host_n: float,
    bounds: list[tuple],
    method: str = "BFGS",
):
    """
    p_ocp: (N, 2) numpy array with SoL vs Voltage
    n_ocp: (N', 2) numpy array with SoL vs Voltage
    cell_ocv: (N'', 2) numpy array with Capacity vs Voltage in full-cell
    c_host_p: Positive host capacity scaled to the 'cell_ocv' level
    c_host_n: Negative host capacity scaled to the 'cell_ocv' level
    """
    if method != "PSO":
        soln0_init = 0.0 
        solp0_init = 1.0 

        obj = partial(
            objective_q_fixed,
            p_ocp=p_ocp,
            n_ocp=n_ocp,
            cell_ocv=cell_ocv,
            c_host_n=c_host_n,
            c_host_p=c_host_p,
        )

        res = minimize(obj, [soln0_init, solp0_init], method=method, bounds=bounds)
        res = res.x

    else:
        problem = BalancingProblem(
            p_ocp,
            n_ocp,
            cell_ocv,
            c_host_p,
            c_host_n,
            n_var=2,
            n_obj=1,
            xl=tuple([b[0] for b in bounds]),
            xu=tuple([b[1] for b in bounds]),
        )

        termination = get_termination("n_gen", 500)
        algorithm = PSO(pop_size=100)
        soln = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=3,
            verbose=False,
        )

        res = soln.X

    return res


def balance_electrodes(
    p_ocp: np.ndarray,
    n_ocp: np.ndarray,
    cell_ocv: np.ndarray,
    c_host_p: float,
    c_host_n: float,
    c_max: float,
    mode: Literal["charge", "discharge"], 
    method: str = "BFGS",
):
    """
    p_ocp: (N, 2) numpy array with SoL vs Voltage
    n_ocp: (N', 2) numpy array with SoL vs Voltage
    cell_ocv: (N'', 2) numpy array with Capacity vs Voltage in full-cell
    c_host_p: Positive host capacity scaled to the 'cell_ocv' level
    c_host_n: Negative host capacity scaled to the 'cell_ocv' level
    c_max: Available capacity of Li ions between positive and negative (depend on chg or dchg)
    """
    yhost_p_init = c_host_p / cell_ocv[-1, 0]
    yhost_n_init = c_host_n / cell_ocv[-1, 0]
    ylitot_init = c_max / cell_ocv[-1, 0]
    soln0_init = 0.0 

    obj = partial(
        objective_adimensional,
        p_ocp=p_ocp,
        n_ocp=n_ocp,
        cell_ocv=cell_ocv,
        c_host_n=c_host_n,
        c_host_p=c_host_p,
        mode=mode,
    )

    res = minimize(obj, [yhost_p_init, yhost_n_init, ylitot_init, soln0_init], method=method)
    res = res.x

    return res
