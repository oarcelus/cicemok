import chaospy as cp
import numpy as np
import pickle
from cicemok.optimization import optimize_parameters_ode, get_most_sensitive
import matplotlib.pyplot as plt

def main():
    params = [1e-17, 1e-15, 1e-12, 1e-10]
    surrogates = [
        "./surrogate_param0_boiter1.pkl",
        "./surrogate_param1_boiter1.pkl",
        "./surrogate_param2_boiter0.pkl",
        "./surrogate_param3_boiter0.pkl",
    ]

    models = []
    for file in surrogates:
        with open(file, "rb") as f:
            models.append(pickle.load(f))

    time = [model[0] for model in models]
    data = [model[1](*params) for model in models]
    experiment = dict()
    for i, (x, curve) in enumerate(zip(time, data)):
        experiment[i] = np.array([x, curve])

    print(experiment)

if __name__ == "__main__":
    main()
