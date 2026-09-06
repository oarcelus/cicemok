import pickle
import multiprocessing
import dill
import numpy as np
import matplotlib.pyplot as plt


def main():
    repetitions = 20
    lnsamples = [50, 100, 200, 500, 1000, 2000]

    with open("../reference_evals/ref_evaluations_n100000.pkl", "rb") as file:
        refevals = pickle.load(file)
    with open("../reference_evals/ref_samples_r_n100000.pkl", "rb") as file:
        refsamples = pickle.load(file)

    npevals = np.array(refevals[1], dtype=float)
    err_mean = []
    for i in range(repetitions):
        for j, nsamples in enumerate(lnsamples):
            with open(f"./surrogate_n{nsamples}_reps{i}.pkl", "rb") as file:
                vals = dill.load(file)
                ysurr = vals[1]
            with open(f"./fourier_n{nsamples}_reps{i}.pkl", "rb") as file:
                vals = pickle.load(file)
                four = vals[1]

            erremp = []
            for k, (surr, fouri) in enumerate(zip(ysurr, four)):
                xnew = surr.evaluate_basis(refsamples)

                ypred = xnew.dot(fouri)
                erremp.append(np.mean((ypred - npevals[:, k]) ** 2))
                print("Target Loop", erremp)

            err_mean.append(np.mean(np.array(erremp)))
            print(f"Targets Mean for samples {nsamples}", err_mean)

            with open(
                f"omp_generalization_error_n{nsamples}_reps{i}.pkl", "wb"
            ) as file:
                pickle.dump(err_mean, file)


    # X = [item for sublist in lnsamples for item in sublist]
    # fig, ax = plt.subplots()
    # ax.set_ylabel("Test L2-error", fontsize=12)
    # ax.set_xlabel("Number of Samples", fontsize=12)
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    #
    # ax.plot(
    #     X,
    #     err_mean,
    #     "-",
    #     label="PCE",
    #     color="b",
    #     marker="s",
    #     linestyle="--",
    #     linewidth=2.0,
    # )
    #
    # ax.spines["top"].set_linewidth(2)
    # ax.spines["bottom"].set_linewidth(2)
    # ax.spines["left"].set_linewidth(2)
    # ax.spines["right"].set_linewidth(2)
    # ax.tick_params(axis="both", which="major", labelsize=12)
    # ax.legend(fontsize=10)
    # plt.savefig("error.png", dpi=300)
    # plt.show()
    #

if __name__ == "__main__":
    main()
