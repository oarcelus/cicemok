import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_ylabel("Test L2-error", fontsize=12)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_yscale("log")
    ax.set_xscale("log")

    repetitions = 20
    lnsamples = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    label = "Full-PCE"

    # Dictionary to store list of errors per nsample
    errors_by_n = {n: [] for n in lnsamples}

    # === Collect all error values ===
    for i in range(repetitions):
        for nsample in lnsamples:
            matches = [
                f
                for f in os.listdir(os.getcwd())
                if "pce_generalization" in f
                and f"_n{nsample}" in f
                and f"reps{i}" in f
            ]
            if not matches:
                continue

            with open(matches[0], "rb") as file:
                errors = pickle.load(file)

            errors_by_n[nsample].append(errors[-1])

    # === Prepare data ===
    data = [errors_by_n[n] for n in lnsamples]

    # Compute constant box width in *log10-space*
    log_positions = np.log10(lnsamples)
    box_width_log = 0.05  # width in log10 units
    box_widths_linear = []
    for p in log_positions:
        # width in linear space corresponding to ±0.5 * box_width_log around log10(p)
        left = 10 ** (p - box_width_log / 2)
        right = 10 ** (p + box_width_log / 2)
        width_linear = right - left
        box_widths_linear.append(width_linear)

    # === Plot boxplots with log-scaled widths ===
    box = ax.boxplot(
        data,
        positions=lnsamples,
        widths=box_widths_linear,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="#1f77b4", alpha=0.6, linewidth=1.5),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="gray", linewidth=1.2),
        capprops=dict(color="gray", linewidth=1.2),
    )

    # Median trend line
    ax.plot(
        lnsamples,
        [np.median(errors_by_n[n]) for n in lnsamples],
        "--",
        color="black",
        label=label,
        linewidth=1.5,
        markersize=5,
    )

    # Style
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2, which="both", linestyle="--")

    plt.tight_layout()
    plt.savefig("error_boxplot_logwidth.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()


