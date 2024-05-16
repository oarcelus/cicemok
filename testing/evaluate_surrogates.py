import chaospy as cp
from cicemok.ode import (
    get_surrogate_samples,
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

    _ = get_surrogate_samples(
        filenames_surrogate=["./surrogate_param0_boiter0.pkl"],
        npool=4,
        ncores=1,
        nsamples=8,
        filenames_experiment=["./experiment_param0_boiter0.pkl"],
        filename_comsol="./comsol_models/nib_withsoc.mph",
        names=["Ds_p", "Ds_n", "k_p", "k_n"],  # , "R_n", "R_p"],
        idxs=[0, 1, 2, 3],
        expression=["t", "liion.phis0_ec1"],
        units=["m^2/s", "m^2/s", "m/s", "m/s"],  # , "um", "um"],
        database="Study 1//Solution 1",
        evname="C_rate",
        isocname="isoc",
        distribution=distribution,
        rule="random",
    )


if __name__ == "__main__":
    main()
