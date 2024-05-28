import chaospy as cp
import numpy as np
import pickle
from cicemok.optimization import optimize_parameters_ode
import matplotlib.pyplot as plt


def main():
    params = np.array([5e-15, 3e-15, 5e-11, 3e-11])
    params0 = np.array([1e-15, 1e-15, 1e-10, 1e-10])
    surrogates = [
        "./surrogate_param0_boiter1.pkl",
        "./surrogate_param1_boiter1.pkl",
        "./surrogate_param2_boiter0.pkl",
        "./surrogate_param3_boiter0.pkl",
    ]

    lower = np.array([1e-16, 1e-16, 1e-11, 1e-11])
    upper = np.array([1e-14, 1e-14, 1e-9, 1e-9])
    bounds = (lower, upper)

    models = []
    for file in surrogates:
        with open(file, "rb") as f:
            models.append(pickle.load(f))

    time = [model[0] for model in models]
    data = [model[1](*params) for model in models]
    experiment = dict()
    for i, (x, curve) in enumerate(zip(time, data)):
        experiment[i] = np.array([x, curve])

    optimize_parameters_ode(
        ncores=1,
        input0=params0,
        experiment=experiment,
        bounds=bounds,
        filename_comsol="./comsol_models/nib_withsoc.mph",
        names=["Ds_p", "Ds_n", "k_p", "k_n"],  # , "R_n", "R_p"],
        idxs=[0, 1, 2, 3],
        expression=["t", "liion.phis0_ec1"],
        units=["m^2/s", "m^2/s", "m/s", "m/s"],  # , "um", "um"],
        database="Study 1//Solution 1",
        evname="C_rate",
        isocname="isoc",
        log_name="nib_log",
        with_surrogate=False,
        global_opt=True,
        use_restarts=True,
        rhoend=1e-3,
        slowiter=1e-8,
        maxfun=5,
        usehistory=True,
        use_pso=True,
        pop_size=4,
        npool=4,
        n_gen=2,
    )


if __name__ == "__main__":
    main()
