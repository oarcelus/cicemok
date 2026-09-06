import pickle
import dill
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from UQpy.surrogates import PolynomialChaosExpansion, TotalDegreeBasis

from UQpy.distributions import Uniform, JointIndependent

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

    poly = TotalDegreeBasis(distribution_1, 2)
    
    distribution_r = cp.J(
        *[cp.Uniform(-1, 1) for _ in range(15)]
    )
    # poly = cp.generate_expansion(1, distribution_r)
    X, Y = cp.generate_quadrature(1, distribution_r)
    X = X.T
    print(poly.evaluate_basis(X))
    # for i, p1 in enumerate(poly):
    #     for j, p2 in enumerate(poly):
    #         res = np.sum(p1(*X) * p2(*X) * Y)
    #         print(f"Inner product of P_{i} and P_{j}: {res:.5e}")
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i <= j:
                res = np.sum(X[:, i] * X[:, j] * Y)

                print(i, j, res)

if __name__ == "__main__":
    main()

