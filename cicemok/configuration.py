import dataclasses
import numpy as np
import chaospy as cp

@dataclasses.dataclass
class ExperimentConfiguration:
    rmax: float
    dt: float
    minsoc: float
    maxsoc: float
    minrate: float
    maxrate: float
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
    experiment: ExperimentConfiguration | None = None


@dataclasses.dataclass
class EvaluationConfiguration:
    distribution: cp.J
    rule: str
    config: ComsolConfiguration


@dataclasses.dataclass
class SensitivityConfiguration(EvaluationConfiguration):
    order: int
    
