import logging
import multiprocessing
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
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
        r"$\sigma_p$",
        r"$\sigma_n$",
        r"$C_{dl}^p$",
        r"$C_{dl}^n$",
        r"$c_l^0$",
    ]


    fig, ax = plt.subplots(3, 3)
    with open("./evaluations_n32768.pkl", "rb") as file:
        evals = pickle.load(file) 
        lnsamples = evals[0]
        evals = evals[1]

    # with open("./statistics_data_vs_voltage.pkl", "rb") as file:
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
    #     [mean[0] for mean in means],
    #     lnsamples,
    #     xerr=np.abs(mean_err.T - meanarr),
    #     fmt='o',
    #     capsize=5,
    #     label="Mean"
    # )
    # ax[0, 0].legend()
    # ax[0, 1].errorbar(
    #     [var[0] for var in vars],
    #     lnsamples,
    #     xerr=np.abs(var_err.T - vararr),
    #     fmt='o',
    #     color="r",
    #     capsize=5,
    #     label="Variance"
    # )
    # ax[0, 1].legend()

#    plt.show()
    s1s = {k: [] for k in names}
    s1sconfs = {k: [] for k in names}
    for i, xs in enumerate(lnsamples):
        with open(f"./sobol_n32768.pkl", "rb") as file:
            sobol = pickle.load(file)

        for j, (s1, s1_conf) in enumerate(zip(sobol[i]["S1"], sobol[i]["S1_conf"])):
            s1s[names[j]].append(s1)
            s1sconfs[names[j]].append(s1_conf)


    ax_flat = ax.flatten()
    cmap = plt.cm.tab20.colors
    for i, name in enumerate(names):
        ax_flat[i].errorbar(
            s1s[name],
            lnsamples,
            xerr=s1sconfs[name],
            fmt='o',
            capsize=5,
            label=name,
            color=cmap[i]
        )
        ax_flat[i].legend()
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((0,0))
        # ax_flat[i+2].xaxis.set_major_formatter(formatter)
        # ax_flat[i+2].xaxis.get_offset_text().set_visible(False)
        # ax_max = max(ax_flat[i+2].get_xticks())
        ax_flat[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True)
        # exponent_axis = np.floor(np.log10(ax_max)).astype(int)
        # ax_flat[i+2].annotate(r'$\times$10$^{%i}$'%(exponent_axis), xy=(0.90, -0.08), xycoords='axes fraction', fontsize=10, fontweight="bold")
    
    for ax in ax_flat:
        ax.tick_params(axis='both', which='both', width=2, labelsize=12)
        ax.yaxis.set_major_locator(MultipleLocator(0.4))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
    # ax_flat[3].xaxis.set_major_locator(MultipleLocator(1.5e-1))
    # ax_flat[5].xaxis.set_major_locator(MultipleLocator(2e-2))
    # ax_flat[6].xaxis.set_major_locator(MultipleLocator(1e-2))
    # ax_flat[7].xaxis.set_major_locator(MultipleLocator(1e-2))
    # ax_flat[8].xaxis.set_major_locator(MultipleLocator(2e-3))
    # ax_flat[11].xaxis.set_major_locator(MultipleLocator(0.5e-1))
    # ax_flat[13].xaxis.set_major_locator(MultipleLocator(2e-1))
    fig.supylabel("Voltage (V)", fontsize=14)
    fig.subplots_adjust(hspace=0.3)
    plt.show()



if __name__ == "__main__":
    main()
