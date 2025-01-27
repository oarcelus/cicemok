import pickle
import numpoly as npoly
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

def main():
    distribution = cp.J(
        cp.Uniform(1e-16, 1e-13),
        cp.Uniform(1e-15, 1e-13),
        cp.Uniform(1e-12, 1e-7),
        cp.Uniform(1e-12, 1e-7),
        cp.Uniform(0.0001, 0.001),
        cp.Uniform(0.0001, 0.001),
        cp.Uniform(0.001, 0.01),
        cp.Uniform(0.001, 0.01),
        cp.Uniform(0.01, 30.0),
        cp.Uniform(0.1, 400.0),
        cp.Uniform(1.0, 6.0),
        cp.Uniform(1.0, 6.0),
        cp.Uniform(0.02, 10.0),
        cp.Uniform(0.02, 10.0),
        cp.Uniform(800, 1600),
    )

    lorder = [2, 3, 4, 5, 6]
    lnsamples = [
        [20, 50, 100, 400],
        [100, 200, 500, 2000],
        [500, 1000, 2000, 5000],
        [2000, 5000, 15000, 30000],
        [10000, 20000, 50000, 100000],
    ]

    with open("evaluations_n500000_reference.pkl", "rb") as file:
        refevals = pickle.load(file)
    with open("samples_n500000_reference.pkl", "rb") as file:
        refsamples = pickle.load(file)

    npevals = np.array(refevals[1], dtype=float)
    err_mean = []
    for i, order in enumerate(lorder):
        temp1 = []
        for j, nsamples in enumerate(lnsamples[i]):
            with open(f"surrogate_n{nsamples}_o{order}.pkl", "rb") as file:
                vals = pickle.load(file)
                ysurr = vals[1]

                surrevals = ysurr(*refsamples)
                erremp = np.mean((surrevals.T - npevals)**2, axis=0)
                print(erremp[-1])
                
                temp1.append(np.mean(erremp))
                
        err_mean.append(temp1)

        print(err_mean)

    
    fig, ax = plt.subplots()
    ax.set_ylabel("Test L2-error", fontsize=12)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_yscale("log")
    ax.set_xscale("log")
    cmap = plt.get_cmap("cool")
    for i, means in enumerate(err_mean):
        index = float(i)/len(lorder)
        iter = lnsamples[i]
        order = lorder[i]

        ax.plot(iter, means, "-", label=f"mean (order{order})", color=cmap(index), marker="s", linestyle="--", linewidth=2.0)

    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10)
    plt.show()

if __name__ == "__main__":
    main()

