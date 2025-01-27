import logging
import multiprocessing
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import pickle
import os

os.environ["NUMEXPR_MAX_THREADS"] = "255"
os.environ["NUMEXPR_NUM_THREADS"] = "255"
import chaospy as cp
import numpy as np
from cicemok.sensitivity import get_sa_from_experiment, get_sampling_from_experiment
from cicemok.comsol import setup_comsol_worker
from cicemok.configuration import (
    ComsolConfiguration,
    CurrentConfigurations,
    SensitivityConfiguration,
)

logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.getcwd(), "sensitivity.log"),
    encoding="utf-8",
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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

    fig, ax = plt.subplots(3, 5)

    # with open("./statistics_data.pkl", "rb") as file:
    #     stats = pickle.load(file)
    #     lnsamples = stats[0]
    #     means = stats[1]
    #     vars = stats[2]
    #
    # meanarr = np.array([mean[0] for mean in means])
    # vararr = np.array([var[0] for var in vars])
    # mean_err = np.array([[mean[1].confidence_interval.low, mean[1].confidence_interval.high] for mean in means])
    # var_err = np.array([[var[1].confidence_interval.low, var[1].confidence_interval.high] for var in vars])
    # ax[0, 0].errorbar(
    #     lnsamples,
    #     [mean[0] for mean in means],
    #     yerr=np.abs(mean_err.T - meanarr),
    #     fmt='o',
    #     capsize=5,
    #     label="Mean"
    # )
    # ax[0, 0].legend()
    # ax[0, 0].set_xscale("log")
    # ax[0, 1].errorbar(
    #     lnsamples,
    #     [var[0] for var in vars],
    #     yerr=np.abs(var_err.T - vararr),
    #     fmt='o',
    #     color="r",
    #     capsize=5,
    #     label="Variance"
    # )
    # ax[0, 1].legend()
    # ax[0, 1].set_xscale("log")

#    plt.show()
    lnsamples = [256, 512, 1024, 2048, 4096, 8192, 16384, 2*16384]
    s1s = {k: [] for k in names}
    s1sconfs = {k: [] for k in names}
    for nsamples in lnsamples:
        with open(f"./sobol_n{nsamples}.pkl", "rb") as file:
            sobol = pickle.load(file)

        for i, (s1, s1_conf) in enumerate(zip(sobol[-1]["S1"], sobol[-1]["S1_conf"])):
            s1s[names[i]].append(s1)
            s1sconfs[names[i]].append(s1_conf)


    ax_flat = ax.flatten()
    cmap = plt.cm.tab20.colors
    for i, name in enumerate(names):
        ax_flat[i].errorbar(
            lnsamples,
            s1s[name],
            yerr=s1sconfs[name],
            fmt='o',
            capsize=5,
            label=name,
            color=cmap[i]
        )
        ax_flat[i].legend()
        ax_flat[i].set_xscale("log")
        ax_flat[i].ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
    
    for ax in ax_flat:
        ax.tick_params(axis='both', which='both', width=2, labelsize=12)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
    fig.supxlabel("Number of Samples", fontsize=14)
    plt.show()



if __name__ == "__main__":
    main()
