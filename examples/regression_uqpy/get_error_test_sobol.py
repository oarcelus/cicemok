import pickle
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
from cicemok import sensitivity
from matplotlib.ticker import MultipleLocator, ScalarFormatter

def main():
    names=[
        r"$D_s^p$",
        r"$D_s^n$",
        r"$\kappa_p$",
        r"$\kappa_n$",
        r"$R_c^p$",
        r"$R_c^n$",
        r"$R_{film}^p$",
        r"$R_{film}^n$",
        r"$\sigma_p$",
        r"$\sigma_n$",
        r"$\tau_p$",
        r"$\tau_n$",
        r"$C_{dl}^p$",
        r"$C_{dl}^n$",
        r"$c_l^0$",
    ]

    lorder = [2, 3, 4, 5, 6]
    lnsamples = [
        [20, 50, 100, 400],
        [100, 200, 500, 2000],
        [500, 1000, 2000, 5000],
        [2000, 5000, 15000, 30000],
        [10000, 20000, 50000, 100000],
    ]

    fig, ax = plt.subplots(3, 5)
    ax_flat = ax.flatten()
    fig.supylabel("L2-error in First Sobol Index", fontsize=14)
    cmap = plt.get_cmap("cool")

    s1s = {k: [] for k in names}
    s1sconfs = {k: [] for k in names}
    with open("./sobol_n32768.pkl", "rb") as file:
        sobol = pickle.load(file)

    for i, name in enumerate(names):
        for sens in sobol:
            s1s[name].append(sens["S1"][i])
            s1sconfs[name].append(sens["S1_conf"][i])

    
    for k, name in enumerate(names):
        for i, order in enumerate(lorder):
            err = []
            for j, nsamples in enumerate(lnsamples[i]):
                with open(f"./sobol_n{nsamples}_o{order}.pkl", "rb") as file:
                    vals = pickle.load(file)

                surrsobol = s1s[name]
                trusobol = vals[k, :]

                err.append(np.mean(np.abs(surrsobol-trusobol)**2))

            index = float(i)/len(lorder)
            ax_flat[k].set_yscale("log")
            ax_flat[k].set_xscale("log")
            ax_flat[k].plot(
                lnsamples[i],
                err,
                "-",
                color=cmap(index),
                marker="s",
                linestyle="--",
                linewidth=2.0,
            )

        ax_flat[k].legend([name])
        for ax in ax_flat:
            ax.tick_params(axis='both', which='both', width=2, labelsize=12)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
    
                
    plt.show()

if __name__ == "__main__":
    main()

