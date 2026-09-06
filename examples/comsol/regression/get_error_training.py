import pickle
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

def main():
    lorder = [2, 3, 4, 5]
    lnsamples = [2000, 5000, 10000, 20000, 50000]

    err_mean = []
    for order in lorder:
        temp1 = []
        for nsamples in lnsamples:
            with open(f"./surrevals_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                ysurreval = vals
            with open(f"./evaluations_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                yevaluations = np.array(vals[1], dtype=float)


            erremp = np.mean((ysurreval - yevaluations)**2, axis=0)
            temp1.append(np.mean(erremp))
                
        err_mean.append(temp1)

    fig, ax = plt.subplots()
    ax.set_ylabel("Empirical L2-error", fontsize=12)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_yscale("log")
    cmap = plt.get_cmap("cool")
    for i, means in enumerate(err_mean):
        index = float(i)/len(lorder)
        order = lorder[i]

        ax.plot(lnsamples, means, "-", label=f"mean (order{order})", color=cmap(index), marker="s", linestyle="--", linewidth=2.0)

    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10)
    plt.show()

if __name__ == "__main__":
    main()

