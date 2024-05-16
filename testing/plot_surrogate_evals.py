import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_error_histograms(data: list):
    num_plots = len(data)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    cmap = matplotlib.colormaps["Dark2"]
    colors = cmap(np.linspace(0, 1, num_plots))
    fig, axes = plt.subplots(num_rows, num_cols)

    for i, (data, color) in enumerate(zip(data, colors)):
        if num_rows == 1:
            ax = axes[i % num_cols]
        else:
            ax = axes[i // num_cols, i % num_cols]

        # Plot histogram with specified color
        ax.hist(data, bins=20, alpha=0.7, color=color, label=f"Data {i+1}")

        # Set labels and title
        ax.set_title(f"Data {i+1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

        # Add legend for each histogram
        ax.legend()

    for j in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()


def main():
    with open("surrogate_evaluations.pkl", "rb") as file:
        values: list = pickle.load(file)

    error_data = []
    for value in values:
        truths = value[1]
        surrogates = value[2]

        errors = []
        for truth, surrogate in zip(truths, surrogates):
            error = np.mean(np.abs(truth - surrogates) / truth) * 100.0
            errors.append(error)

        error_data.append(errors)

    plot_error_histograms(error_data)

if __name__ == "__main__":
    main()
