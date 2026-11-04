"""
Vibration Field Visualizer for Frequency-Domain PINN Beam Inverse Solver

Loads predicted real and imaginary beam displacement components (ur, ui),
computes overall displacement magnitude, and produces publication-quality field plots.

FD-PINN_Optimized.py predicts on a grid of 101 beam positions (0..L) times 41
frequencies (0..fmax) and saves it flattened row by row (one row per frequency),
so the files are reshaped back to (frequency, position) here before plotting.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# Must match the prediction grid in FD-PINN_Optimized.py.
L = 10.0      # beam length
FMAX = 0.5    # highest frequency
NX = 101      # beam positions
NF = 41       # frequencies


def load_field(path: Path, nf: int = NF, nx: int = NX) -> np.ndarray:
    """A flattened prediction file as an (nf, nx) array: rows are frequencies."""
    values = np.loadtxt(path).ravel()
    if values.size != nf * nx:
        raise ValueError(f"{path.name} has {values.size} values, expected {nf} x {nx} = {nf * nx}")
    return values.reshape(nf, nx)


def visualize_beam_predictions(output_dir: Path = SCRIPT_DIR):
    ur_file = output_dir / "ur_pred_optimized.txt"
    ui_file = output_dir / "ui_pred_optimized.txt"

    if not ur_file.is_file() or not ui_file.is_file():
        print(f"Prediction files {ur_file.name} or {ui_file.name} not found.")
        return None

    ur = load_field(ur_file)
    ui = load_field(ui_file)
    x = np.linspace(0, L, NX)
    f = np.linspace(0, FMAX, NF)

    # Total displacement magnitude: |u| = sqrt(ur^2 + ui^2)
    magnitude = np.sqrt(ur**2 + ui**2)
    peak = int(np.argmax(magnitude.max(axis=1)))  # frequency with the largest response

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300)

    # Subplot 1: Real and Imaginary Components at the most excited frequency
    axes[0].plot(x, ur[peak], label="Real Component ($u_r$)", color="#1f77b4", linewidth=2)
    axes[0].plot(x, ui[peak], label="Imaginary Component ($u_i$)", color="#d62728", linewidth=2,
                 linestyle="--")
    axes[0].plot(x, magnitude[peak], label=r"Magnitude $|u|$", color="#2ca02c", linewidth=2)
    axes[0].set_xlabel("Beam Coordinate ($x$)", fontsize=12)
    axes[0].set_ylabel("Displacement", fontsize=12)
    axes[0].set_title(f"Predicted Complex Beam Deflection at f = {f[peak]:.4f}", fontsize=13,
                      fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Subplot 2: Vibration magnitude over position and frequency
    mesh = axes[1].pcolormesh(x, f, magnitude, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[1], label="Displacement Magnitude $|u|$")
    axes[1].set_xlabel("Beam Coordinate ($x$)", fontsize=12)
    axes[1].set_ylabel("Frequency ($f$)", fontsize=12)
    axes[1].set_title("Beam Vibration Amplitude over Position and Frequency", fontsize=13,
                      fontweight="bold")

    plt.tight_layout()
    output_png = output_dir / "beam_vibration_field.png"
    plt.savefig(output_png)
    plt.close(fig)
    print(f"Successfully generated beam vibration plot: {output_png}")
    return output_png


if __name__ == "__main__":
    visualize_beam_predictions()
