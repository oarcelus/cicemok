import chaospy as cp
from cicemok.ode import init_experiment_optimization, init_parallel_optimization


def main():
    distribution = cp.J(
        cp.Uniform(1.2e-19, 1.2e-16),
        cp.Uniform(0.01, 100.0),
        cp.Uniform(0.1, 10.0),
        cp.Uniform(50.0, 500.0),
    )

    init_parallel_optimization(
        np=4,
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
        filename="./comsol_models/CP_01000_goodfit_H1H2.mph",
        names=["D_LMO", "sigma_LMO", "i0_ref_LMO", "rpLMO"],
        expression=["t", "liion.phis0_ec1"],
        units=["m^2/s", "S/m", "A/m^2", "nm"],
        database="Study 1//Solution 1",
        iappname="Iapp",
        i1Cname= "I_1C",
        isocname="socinit_LMO",
        order=1,
        distribution=distribution,
        rule="latin_hypercube",
        kind="ucb",
        kappa_decay=1.0,
        kappa_decay_delay=0,
        init_points=2,
        n_iter=5,
        log_name="test",
    )


if __name__ == "__main__":
    main()

