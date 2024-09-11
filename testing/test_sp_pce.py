import pickle
import numpy as np
import chaospy as cp
from cicemok import sensitivity
from cicemok.configuration import SensitivityConfiguration


with open("./evaluations0_boiter0.pkl", "rb") as f:
    evals = pickle.load(f)

with open("./samples0_boiter0.pkl", "rb") as f:
    samples = pickle.load(f)

distribution = cp.J(
        cp.Uniform(-1, 1),
        cp.Uniform(-1, 1),
        cp.Uniform(-1, 1),
        cp.Uniform(-1, 1),
        # cp.Uniform(1.0, 10.0),
        # cp.Uniform(0.1, 4.0),
    )

sens_cfg = SensitivityConfiguration(distribution=distribution, minorder=1, order=5)

sensitivity.sp_fn_pce(samples, evals[1], sens_cfg)

