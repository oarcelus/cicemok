import pickle
import numpy as np
import matplotlib.pyplot as plt

# ---- set your pickle filename here ----
fname = "evaluations_n100_reps0.pkl"
# --------------------------------------

# Load [x, ys]
with open(fname, "rb") as f:
    x, ys = pickle.load(f)

# ys is a list of arrays; stack into (n_curves, n_points)
Y = np.stack([np.asarray(y).squeeze() for y in ys], axis=0)

# x is the voltage grid (shared)
V = np.asarray(x).squeeze()
if V.ndim != 1:
    raise ValueError(f"x should be 1D (voltage grid). Got shape {V.shape}")

# Basic checks
if Y.shape[1] != V.size:
    raise ValueError(
        f"Mismatch: each y must have length {V.size} (len(x)). "
        f"Got stacked ys shape {Y.shape}"
    )

# Summary stats across curves
mean = np.mean(Y, axis=0)
p05, p95 = np.percentile(Y, [5, 95], axis=0)

# Plot: X = ys (e.g., capacity), Y = voltage
plt.figure()
for i in range(Y.shape[0]):
    plt.plot(Y[i], V, alpha=0.15, linewidth=1)

# Spread band + mean
plt.fill_betweenx(V, p05, p95, alpha=0.25, label="5–95%")
plt.plot(mean, V, linewidth=2, label="mean")

plt.xlabel("ys (e.g., capacity / x-output)")
plt.ylabel("Voltage [V]")
plt.title(fname)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


