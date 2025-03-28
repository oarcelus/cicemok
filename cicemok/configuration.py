import dataclasses
from dataclasses import field
import numpy as np
import chaospy as cp
import UQpy as uq
from typing import Union


@dataclasses.dataclass
class CurrentConfigurations:
    isoc: float = 0.0
    texp: float = 0.0
    experiment: np.ndarray | None = None


@dataclasses.dataclass
class ExperimentConfiguration(CurrentConfigurations):
    rmax: float = 0.0 
    dt: float = 0.0
    minsoc: float = 0.0
    maxsoc: float = 0.0
    minrate: float = 0.0
    maxrate: float = 0.0
    dynamics: float = 0.0
    rate: float = 0.0


@dataclasses.dataclass
class ComsolConfiguration:
    names: list[str]
    filename: str
    expression: list[str]
    unit: list[str]
    database: str
    evname: str
    isocname: str 
    experiment: CurrentConfigurations | None = None


@dataclasses.dataclass
class SensitivityConfiguration:
    distribution: Union[cp.J, uq.distributions.JointIndependent]
    order: int = 1
    rule: str = "latin_hypercube"
    minorder: int = 1
    cross_truncation: float = 1.0
    retall: bool = False
    normed: bool = True
    sobol_total: bool = False
    sobol_second: bool = False
    max_size: int = 25000
    early_stop: bool = False
    idtargets: list = field(default_factory=list)
    precomputed_poly: dict = field(default_factory=dict)
    
