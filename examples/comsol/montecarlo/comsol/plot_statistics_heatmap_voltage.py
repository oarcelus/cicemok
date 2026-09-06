import logging
import pandas as pd
import multiprocessing
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
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


    s2s = []
    s2sconfs = []
    for i, xs in enumerate(lnsamples):
        with open(f"./sobol_n32768.pkl", "rb") as file:
            sobol = pickle.load(file)

        s2 = np.nan_to_num(sobol[i]["S2"], nan=0.0)
        tmp = np.zeros(s2.shape) + s2 + s2.T
        for j in range(len(names)):
            for k in range(len(names)):
                s2s.append([names[j], names[k], xs, tmp[j, k]])
        s2_conf = np.nan_to_num(sobol[i]["S2_conf"], nan=0.0)
        for j in range(len(names)):
            for k in range(len(names)):
                s2sconfs.append([names[j], names[k], xs, tmp[j, k]])
    
    df2 = pd.DataFrame(s2s, columns=["Row", "Col", "Voltage", "Values"])
    df2_conf = pd.DataFrame(s2sconfs, columns=["Row", "Col", "Voltage", "Values"])
    voltages = [3, 6, 9]
    # voltages = [0]

    df2 = df2[df2["Voltage"].isin(lnsamples[voltages])]
    df2.loc[df2["Values"] < 0.001, "Values"] = 0.0

    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid", {"axes.edgecolor": "0.0", "xtick.bottom": True, "ytick.left": True})
    # ax_flat = ax.flatten()
    # for i, volt in enumerate(voltages):
    #     sns.heatmap(df2["Values"], ax=ax_flat[i], xticklabels=df2["Row"])
    max_val = df2["Values"].max() 
    min_val = df2["Values"].min() 
    g = sns.relplot(
        data=df2,
        x="Row",
        y="Col",
        hue="Values",
        size="Values",
        size_norm=(0.0, max_val),
        sizes=(0, 400),
        # palette=list(df2["Color"].values),
        palette="rainbow",
        col="Voltage",
        edgecolor="0.0",
        col_wrap=3,
        col_order=lnsamples[voltages],
    )
    g.set(xlabel="", ylabel="", aspect="equal")
    norm = plt.Normalize(0, max_val)
    sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
    sm.set_array([])

    g._legend.remove()
    colorbar = g.figure.colorbar(sm, ax=g.axes, fraction=0.03, pad=0.05)
    #g.despine(left=True, bottom=True)
    for i, (ax, voltage) in enumerate(zip(g.axes.flat, g.col_names)):
        ax.tick_params(labelsize=12)
        ax.set_title(f"Voltage: {voltage:.2f} V")
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_weight("bold")
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_visible(True)
            ax.spines[axis].set_alpha(1.0)
            ax.spines[axis].set_linewidth(2)

    # g.ax.margins(.02)
    # for label in g.ax.get_xticklabels():
    #     label.set_rotation(90)
    # ax_flat[i].set_xticks(range(len(names)))  # Set tick positions
    # ax_flat[i].set_xticklabels(names)
    # ax_flat[i].set_title(f"Voltage: {lnsamples[volt]:.2f} V")
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_powerlimits((0,0))
    # ax_flat[i+2].xaxis.set_major_formatter(formatter)
    # ax_flat[i+2].xaxis.get_offset_text().set_visible(False)
    # ax_max = max(ax_flat[i+2].get_xticks())
    # ax_flat[i+2].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True)
    # exponent_axis = np.floor(np.log10(ax_max)).astype(int)
    # ax_flat[i+2].annotate(r'$\times$10$^{%i}$'%(exponent_axis), xy=(0.90, -0.08), xycoords='axes fraction', fontsize=10, fontweight="bold")

    # non_hatched_patch = mpatches.Patch(
    #     alpha=0.5, facecolor="grey", edgecolor="black", label="First-order Sobol"
    # )
    # hatched_patch = mpatches.Patch(
    #     alpha=0.2,
    #     facecolor="grey",
    #     edgecolor="black",
    #     hatch="///",
    #     label="Total-order Sobol",
    # )
    # ax_flat[0].legend(handles=[non_hatched_patch, hatched_patch])
    # for ax in ax_flat:
    #     ax.tick_params(axis="both", which="both", width=2, labelsize=12)
    #     # ax.yaxis.set_major_locator(MultipleLocator(0.4))
    #     for axis in ["top", "bottom", "left", "right"]:
    #         ax.spines[axis].set_linewidth(2)
    # ax_flat[3].xaxis.set_major_locator(MultipleLocator(1.5e-1))
    # ax_flat[5].xaxis.set_major_locator(MultipleLocator(2e-2))
    # ax_flat[6].xaxis.set_major_locator(MultipleLocator(1e-2))
    # ax_flat[7].xaxis.set_major_locator(MultipleLocator(1e-2))
    # ax_flat[8].xaxis.set_major_locator(MultipleLocator(2e-3))
    # ax_flat[11].xaxis.set_major_locator(MultipleLocator(0.5e-1))
    # ax_flat[13].xaxis.set_major_locator(MultipleLocator(2e-1))
    # fig.supylabel("Voltage (V)", fontsize=14)
    # fig.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
