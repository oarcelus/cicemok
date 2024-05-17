import chaospy as cp


distribution = cp.J(
    cp.Uniform(1e-18, 1e-14),
    cp.Uniform(1e-18, 1e-14),
    cp.Uniform(1e-13, 1e-9),
    cp.Uniform(1e-13, 1e-9),
    cp.Uniform(1.0, 10.0),
    cp.Uniform(0.1, 4.0),
)

samples, weight = cp.generate_quadrature(4, distribution, sparse=True)

print(samples.shape)
