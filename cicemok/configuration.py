import dataclasses
import numpy as np
import chaospy as cp


@dataclasses.dataclass
class CurrentConfigurations:
    isoc: float
    texp: float
    experiment: np.ndarray | None = None


@dataclasses.dataclass
class ExperimentConfiguration(CurrentConfigurations):
    rmax: float | None = None 
    dt: float | None = None
    minsoc: float | None = None
    maxsoc: float | None = None
    minrate: float | None = None
    maxrate: float | None = None
    dynamics: float = 0.0
    rate: float = 0.0
    isoc: float = 0.0
    texp: float = 0.0
    experiment: np.ndarray | None = None


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
class EvaluationConfiguration:
    distribution: cp.J
    rule: str
    config: ComsolConfiguration


@dataclasses.dataclass
class SensitivityConfiguration(EvaluationConfiguration):
    order: int
    
