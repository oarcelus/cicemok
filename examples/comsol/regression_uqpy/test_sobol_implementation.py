import pickle
import dill
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from cicemok import sensitivity
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from UQpy.sensitivity import PceSensitivity
from UQpy.surrogates import PolynomialChaosExpansion, PolynomialBasis
from UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression import LeastSquareRegression

def main():
    names=[
        r"$D_s^p$",
        r"$D_s^n$",
        r"$\kappa_p$",
        r"$\kappa_n$",
        r"$R_c^p$",
        r"$R_c^n$",
        r"$R_{film}^p$",
        r"$R_{film}^n$",
        r"$\sigma_p$",
        r"$\sigma_n$",
        r"$\tau_p$",
        r"$\tau_n$",
        r"$C_{dl}^p$",
        r"$C_{dl}^n$",
        r"$c_l^0$",
    ]

    lorder = [2, 3, 4, 5, 6]
    lnsamples = [
        [20, 50, 100, 400],
        [100, 200, 500, 2000],
        [500, 1000, 2000, 5000],
        [2000, 5000, 15000, 30000],
        [10000, 20000, 50000, 100000],
    ]

    for k, name in enumerate(names):
        for i, order in enumerate(lorder):
            err = []
            for j, nsamples in enumerate(lnsamples[i]):
                with open(f"./sobol_n{nsamples}_o{order}.pkl", "rb") as file:
                    vals = pickle.load(file)

                with open(f"./surrogate_n{nsamples}_o{order}.pkl", "rb") as file:
                    surr = dill.load(file)

                
                
                print(surr[1].multi_index_set)
                reg = LeastSquareRegression()
                expans = PolynomialChaosExpansion(surr[1], reg)
                trusobol = vals[k, :]


                
    plt.show()

if __name__ == "__main__":
    main()

