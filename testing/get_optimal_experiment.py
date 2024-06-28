import chaospy as cp
from cicemok.ode import (
    init_pool_parallel_optimization,
)


def main():
    distribution = cp.J(
        cp.Uniform(1e-18, 1e-14),
        cp.Uniform(1e-18, 1e-14),
        cp.Uniform(1e-13, 1e-9),
        cp.Uniform(1e-13, 1e-9),
        # cp.Uniform(1.0, 10.0),
        # cp.Uniform(0.1, 4.0),
    )

    init_pool_parallel_optimization(
        np=2,
        npool=2,
        ncores=1,
        dynamics=(0.0, 1.0),
        isoc=(0.0, 1.0),
        rate=(0.5, 3.0),
        texp=(200.0, 300.0),
        rmax=3.0,
        dt=1.0,
        minsoc=0.05,
        maxsoc=0.95,
        minrate=0.02,
        maxrate=2.0,
        filename="./comsol_models/nib_withsoc.mph",
        names=["Ds_p", "Ds_n", "k_p", "k_n"],  # , "R_n", "R_p"],
        idxs=[0, 1, 2, 3],
        expression=["t", "liion.phis0_ec1"],
        units=["m^2/s", "m^2/s", "m/s", "m/s"],  # , "um", "um"],
        database="Study 1//Solution 1",
        evname="C_rate",
        isocname="isoc",
        order=1,
        distribution=distribution,
        nsample=10,
        gpce=True,
        rule="latin_hypercube",
        kind="ucb",
        kappa=10.0,
        kappa_decay=1.0,
        kappa_decay_delay=0,
        init_points=1,
        n_iter=1,
        log_name="nib_log",
    )


if __name__ == "__main__":
    main()
