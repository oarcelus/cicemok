import logging
import multiprocessing
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import matplotlib.patches as mpatches
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

    fig, ax = plt.subplots(2, 2)

    with open("./evaluations_n32768.pkl", "rb") as file:
        evals = pickle.load(file) 
        lnsamples = evals[0]
        evals = evals[1]

    s1s = {k: [] for k in names}
    s1sconfs = {k: [] for k in names}
    sts = {k: [] for k in names}
    stsconfs = {k: [] for k in names}
    for i, xs in enumerate(lnsamples):
        with open("./sobol_n32768.pkl", "rb") as file:
            sobol = pickle.load(file)

        for j, (s1, s1_conf) in enumerate(zip(sobol[i]["S1"], sobol[i]["S1_conf"])):
            s1s[names[j]].append(s1)
            s1sconfs[names[j]].append(s1_conf)
        for j, (st, st_conf) in enumerate(zip(sobol[i]["ST"], sobol[i]["ST_conf"])):
            sts[names[j]].append(st)
            stsconfs[names[j]].append(st_conf)


    ax_flat = ax.flatten()
    cmap = plt.cm.tab20.colors
    voltages = [0, 3, 6, 9]
    bar_width = 0.25
    offset = 0.05 + bar_width / 2
    for i, volt in enumerate(voltages):
        for j, name in enumerate(names):
            ax_flat[i].bar(j - offset, s1s[name][volt], bar_width, color=cmap[j], alpha=0.5)
            ax_flat[i].bar(j + offset, sts[name][volt], bar_width, color=cmap[j], alpha=0.2, edgecolor=cmap[j], hatch="///")
            ax_flat[i].errorbar(
                j - offset,
                s1s[name][volt],
                yerr=s1sconfs[name][volt],
                fmt='o',
                capsize=5,
                color=cmap[j],
            )
            ax_flat[i].errorbar(
                j + offset,
                sts[name][volt],
                yerr=stsconfs[name][volt],
                fmt='o',
                capsize=5,
                color=cmap[j],
            )
        ax_flat[i].set_xticks(range(len(names)))  # Set tick positions
        ax_flat[i].set_xticklabels(names)  
        ax_flat[i].set_ylim(-0.02, 0.7)
        ax_flat[i].set_title(f"Voltage: {lnsamples[volt]:.2f} V")  
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((0,0))
        # ax_flat[i+2].xaxis.set_major_formatter(formatter)
        # ax_flat[i+2].xaxis.get_offset_text().set_visible(False)
        # ax_max = max(ax_flat[i+2].get_xticks())
        #ax_flat[i+2].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True)
        # exponent_axis = np.floor(np.log10(ax_max)).astype(int)
        # ax_flat[i+2].annotate(r'$\times$10$^{%i}$'%(exponent_axis), xy=(0.90, -0.08), xycoords='axes fraction', fontsize=10, fontweight="bold")
    
    non_hatched_patch = mpatches.Patch(alpha=0.5, facecolor="grey", edgecolor="black", label="First-order Sobol")
    hatched_patch = mpatches.Patch(alpha=0.2, facecolor="grey", edgecolor="black", hatch="///", label="Total-order Sobol")
    ax_flat[0].legend(handles=[non_hatched_patch, hatched_patch])
    for ax in ax_flat:
        ax.tick_params(axis='both', which='both', width=2, labelsize=12)
        #ax.yaxis.set_major_locator(MultipleLocator(0.4))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
    # ax_flat[3].xaxis.set_major_locator(MultipleLocator(1.5e-1))
    # ax_flat[5].xaxis.set_major_locator(MultipleLocator(2e-2))
    # ax_flat[6].xaxis.set_major_locator(MultipleLocator(1e-2))
    # ax_flat[7].xaxis.set_major_locator(MultipleLocator(1e-2))
    # ax_flat[8].xaxis.set_major_locator(MultipleLocator(2e-3))
    # ax_flat[11].xaxis.set_major_locator(MultipleLocator(0.5e-1))
    # ax_flat[13].xaxis.set_major_locator(MultipleLocator(2e-1))
    #fig.supylabel("Voltage (V)", fontsize=14)
    #fig.subplots_adjust(hspace=0.3)
    plt.show()



if __name__ == "__main__":
    main()
