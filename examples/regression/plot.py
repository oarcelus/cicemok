import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Example data
parameters = ['Parameter 1', 'Parameter 2', 'Parameter 3', 'Parameter 4']
sobol_indices = np.array([0.3, 0.15, 0.05, 0.4])
confidence_intervals = np.array([0.05, 0.03, 0.02, 0.07])

# Example distribution samples for each parameter's Sobol index (from Monte Carlo or bootstrapping)
distributions = [
    np.random.normal(0.3, 0.05, 1000),
    np.random.normal(0.15, 0.03, 1000),
    np.random.normal(0.05, 0.02, 1000),
    np.random.normal(0.4, 0.07, 1000)
]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(parameters))

# Only add error bars without the main bars
ax.errorbar(x_pos, sobol_indices, yerr=confidence_intervals, fmt='o', color='black', capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(parameters)
ax.set_xlabel('Input Parameters')
ax.set_ylabel('Sobol Index')
ax.set_title('Sobol Indices with Confidence Intervals and Distribution Plots')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adding smooth distribution plots around each error bar
for i, (dist, sobol_index, conf_interval) in enumerate(zip(distributions, sobol_indices, confidence_intervals)):
    # Estimate the distribution using KDE for a smooth curve
    kde = gaussian_kde(dist)
    y_values = np.linspace(sobol_index - conf_interval, sobol_index + conf_interval, 100)
    x_values = kde(y_values) * 0.2  # Scale down for better fit within plot

    # Plot the distribution as a smooth line around the error bar
    ax.plot(x_pos[i] + x_values, y_values, color='green')
    ax.plot(x_pos[i] - x_values, y_values, color='green')  # Mirror on the other side for a symmetric plot

    # Expected value line (red dashed) in the center of the distribution
    ax.plot([x_pos[i] - 0.1, x_pos[i] + 0.1], [sobol_index, sobol_index], color='red', linestyle='--')

plt.show()

