import pickle
import dill
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from UQpy.surrogates import PolynomialChaosExpansion, TotalDegreeBasis

from UQpy.distributions import Uniform, JointIndependent
from UQpy.transformations import Nataf

def main():
    distribution = JointIndependent(
        [
            Uniform(1e-16, (1e-13 - 1e-16)),
            Uniform(1e-15, (1e-13 - 1e-15)),
            Uniform(1e-12, (1e-7 - 1e-12)),
            Uniform(1e-12, (1e-7 - 1e-12)),
            Uniform(0.0001, (0.001 - 0.0001)),
            Uniform(0.0001, (0.001 - 0.0001)),
            Uniform(0.001, (0.01 - 0.001)),
            Uniform(0.001, (0.01 - 0.001)),
            Uniform(0.01, (30.0 - 0.01)),
            Uniform(0.1, (400.0 - 0.1)),
            Uniform(1.0, (6.0 - 1.0)),
            Uniform(1.0, (6.0 - 1.0)),
            Uniform(0.02, (10.0 - 0.02)),
            Uniform(0.02, (10.0 - 0.02)),
            Uniform(800, (1600 - 800)),
        ]
    )
    distribution_1 = JointIndependent(
        [
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
            Uniform(-1, 2),
        ]
    )
    poly = TotalDegreeBasis(distribution, 1)
    
    X = distribution.rvs(10)
    nat = Nataf(distribution, X)
    print(nat.samples_z)

    M = poly.evaluate_basis(X)
    for i in range(M.shape[1]):
        for j in range(M.shape[1]):
            if i <= j:
                res = np.sum(M[:, i] * M[:, j] / 1000000)

                print(i, j, res)

if __name__ == "__main__":
    main()

