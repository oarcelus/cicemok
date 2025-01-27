import pickle
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

def main():
    distribution = cp.J(
        cp.Uniform(1e-17, 1e-12),
        cp.Uniform(1e-17, 1e-12),
        cp.Uniform(1e-13, 1e-9),
        cp.Uniform(1e-13, 1e-9),
        cp.Uniform(0.001, 1.0),
        cp.Uniform(0.001, 1.0),
        cp.Uniform(0.01, 300.0),
        cp.Uniform(0.01, 300.0),
        cp.Uniform(5.0, 15.0),
        cp.Uniform(5.0, 15.0),
        cp.Uniform(1.0, 3.0),
        cp.Uniform(1.0, 3.0),
        cp.Uniform(1.0, 3.0),
    )


    with open("evaluations_n100000_reference.pkl", "rb") as file:
        refevals = pickle.load(file)
    with open("samples_n100000_reference.pkl", "rb") as file:
        refsamples = pickle.load(file)

    print(len(refevals[1]))
    refmean = np.mean(np.stack(refevals[1]), axis=0)
    refvar = np.var(np.stack(refevals[1]), axis=0)

    fig, ax = plt.subplots()

    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.set_xlabel("Capacity (Ah)", fontsize=12)
    col = float(50)/255
    for ev in refevals[1]:
        ax.plot(ev, refevals[0], color=(col, col, col, 0.05), linewidth=0.03)

    ax.plot(refmean, refevals[0], color="r", linewidth=2.0, label="Mean")

    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(fontsize=10)
    plt.show()
#    plt.savefig("model_evaluations.png", dpi=300)


if __name__ == "__main__":
    main()

