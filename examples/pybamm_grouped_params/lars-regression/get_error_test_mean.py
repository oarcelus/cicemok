import pickle
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from cicemok import sensitivity

def main():
    lorder = [2, 3, 4, 5]
    lnsamples = [
        [20, 50, 100, 400],
        [200, 500, 2000],
        [1000, 5000],
        [15000],
    ]

    with open("evaluations_n500000_reference.pkl", "rb") as file:
        refevals = pickle.load(file)

    refmean = np.mean(np.stack(refevals[1]), axis=0)
    err_mean = []
    for i, order in enumerate(lorder):
        for j, nsamples in enumerate(lnsamples[i]):
            with open(f"fourier_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                fourier = vals[1]
                mean = sensitivity.get_analytical_mean(fourier)
                
                error = np.mean(np.abs(mean-refmean)**2)
                
            err_mean.append(error)
            with open(f"lars_error_on_mean_n{nsamples}_o{order}.pkl", "wb") as file:
                pickle.dump(err_mean, file)

    
    # fig, ax = plt.subplots()
    # ax.set_ylabel("L2-error in the mean", fontsize=12)
    # ax.set_xlabel("Number of Samples", fontsize=12)
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    # cmap = plt.get_cmap("cool")
    # for i, means in enumerate(err_mean):
    #     index = float(i)/len(lorder)
    #     iter = lnsamples[i]
    #     order = lorder[i]
    #
    #     ax.plot(iter, means, "-", label=f"mean (order{order})", color=cmap(index), marker="s", linestyle="--", linewidth=2.0)
    #
    # ax.spines["top"].set_linewidth(2)
    # ax.spines["bottom"].set_linewidth(2)
    # ax.spines["left"].set_linewidth(2)
    # ax.spines["right"].set_linewidth(2)
    # ax.tick_params(axis="both", which="major", labelsize=12)
    # ax.legend(fontsize=10)
    # plt.show()
    #
if __name__ == "__main__":
    main()
