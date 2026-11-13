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


# Order of the loss terms in loss.dat, as compiled in FD-PINN_Optimized.py: the two PDE residuals,
# the two sensor-data fits (PointSetBC), then the four boundary conditions.
LOSS_TERMS = ["PDE (real)", "PDE (imag)", "sensor data (real)", "sensor data (imag)",
              "BC u_r = 0", "BC u_i = 0", "BC u_r'' = 0", "BC u_i'' = 0"]


def loss_breakdown(path: Path) -> dict:
    """Training loss per term at the first and last logged step of DeepXDE's loss.dat.

    Each row of loss.dat is: step, the weighted training loss of every term, then the same for
    the test set. Returns {"steps": (first, last), "terms": {name: (first, last)}, "total": (first, last)}.
    """
    rows = np.atleast_2d(np.loadtxt(path))
    n = len(LOSS_TERMS)
    if rows.shape[1] < 1 + n:
        raise ValueError(f"{path.name} has {rows.shape[1] - 1} loss columns, expected at least {n}")
    first, last = rows[0, 1:1 + n], rows[-1, 1:1 + n]
    return {
        "steps": (int(rows[0, 0]), int(rows[-1, 0])),
        "terms": {name: (float(a), float(b)) for name, a, b in zip(LOSS_TERMS, first, last)},
        "total": (float(first.sum()), float(last.sum())),
    }


def print_loss_breakdown(path: Path) -> None:
    info = loss_breakdown(path)
    s0, s1 = info["steps"]
    print(f"Training loss by term (step {s0} -> step {s1}):")
    for name, (a, b) in info["terms"].items():
        print(f"  {name:<20} {a:>12.4g} -> {b:.4g}")
    print(f"  {'total':<20} {info['total'][0]:>12.4g} -> {info['total'][1]:.4g}")


def sensor_fit(data_path: Path, ur: np.ndarray, ui: np.ndarray) -> dict:
    """How well the predicted field reproduces the sensor readings it was trained on.

    data_inverse.txt has one reading per row: position x, frequency f, u_r, u_i. Each is
    compared with the prediction at the same frequency row, interpolated along the beam.
    Returns the RMSE and the largest error of each component, and the number of readings.
    """
    data = np.atleast_2d(np.loadtxt(data_path))
    if data.shape[1] < 4:
        raise ValueError(f"{data_path.name} needs 4 columns (x, f, u_r, u_i), found {data.shape[1]}")
    nf, nx = ur.shape
    x = np.linspace(0, L, nx)
    f = np.linspace(0, FMAX, nf)
    rows = np.abs(f[:, None] - data[:, 1]).argmin(axis=0)
    off_grid = np.abs(f[rows] - data[:, 1]).max()
    if off_grid > 1e-9:
        raise ValueError(f"a sensor frequency is {off_grid:.3g} away from the prediction grid")
    pred_r = np.array([np.interp(xi, x, ur[r]) for xi, r in zip(data[:, 0], rows)])
    pred_i = np.array([np.interp(xi, x, ui[r]) for xi, r in zip(data[:, 0], rows)])
    err_r, err_i = pred_r - data[:, 2], pred_i - data[:, 3]
    return {
        "n": len(data),
        "rmse": (float(np.sqrt(np.mean(err_r ** 2))), float(np.sqrt(np.mean(err_i ** 2)))),
        "max_error": (float(np.abs(err_r).max()), float(np.abs(err_i).max())),
        "peak_reading": float(np.abs(data[:, 2:4]).max()),
    }


def print_sensor_fit(data_path: Path, output_dir: Path = SCRIPT_DIR) -> None:
    ur = load_field(output_dir / "ur_pred_optimized.txt")
    ui = load_field(output_dir / "ui_pred_optimized.txt")
    fit = sensor_fit(data_path, ur, ui)
    print(f"Fit to the {fit['n']} sensor readings (largest reading {fit['peak_reading']:.3g}):")
    for name, rmse, worst in zip(("u_r", "u_i"), fit["rmse"], fit["max_error"]):
        print(f"  {name}: RMSE {rmse:.3g}, worst {worst:.3g}")


def readings_at(data: np.ndarray, freq: float, tol: float = 1e-9) -> np.ndarray:
    """The sensor rows (x, f, u_r, u_i) taken at one frequency."""
    return data[np.abs(data[:, 1] - freq) <= tol]


def visualize_beam_predictions(output_dir: Path = SCRIPT_DIR, data_path: Path = None):
    """Plot the predicted field. When the sensor file is there (data_inverse.txt next to the
    predictions, or data_path), the readings are drawn on it too: where they were taken on the
    position-frequency map, and the measured u_r and u_i against the curves at the peak
    frequency, so the plot shows what the network was actually fitted to."""
    data_path = data_path or output_dir / "data_inverse.txt"
    data = np.atleast_2d(np.loadtxt(data_path)) if data_path.is_file() else None
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
    if data is not None:
        here = readings_at(data, f[peak])
        if len(here):
            axes[0].scatter(here[:, 0], here[:, 2], color="#1f77b4", edgecolor="k", s=50, zorder=5,
                            label="Measured $u_r$")
            axes[0].scatter(here[:, 0], here[:, 3], color="#d62728", edgecolor="k", s=50, zorder=5,
                            marker="s", label="Measured $u_i$")
    axes[0].set_xlabel("Beam Coordinate ($x$)", fontsize=12)
    axes[0].set_ylabel("Displacement", fontsize=12)
    axes[0].set_title(f"Predicted Complex Beam Deflection at f = {f[peak]:.4f}", fontsize=13,
                      fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Subplot 2: Vibration magnitude over position and frequency
    mesh = axes[1].pcolormesh(x, f, magnitude, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=axes[1], label="Displacement Magnitude $|u|$")
    if data is not None:
        axes[1].scatter(data[:, 0], data[:, 1], s=12, color="white", edgecolor="k", linewidth=0.5,
                        label=f"{len(data)} sensor readings")
        axes[1].legend(loc="upper right", frameon=True)
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
    if visualize_beam_predictions() and (SCRIPT_DIR / "data_inverse.txt").is_file():
        print_sensor_fit(SCRIPT_DIR / "data_inverse.txt")
    if (SCRIPT_DIR / "loss.dat").is_file():
        print_loss_breakdown(SCRIPT_DIR / "loss.dat")
