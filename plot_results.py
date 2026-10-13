"""
Vibration Field Visualizer for Frequency-Domain PINN Beam Inverse Solver

Loads predicted real and imaginary beam displacement components (ur, ui),
computes overall displacement magnitude, and produces publication-quality field plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def visualize_beam_predictions():
    ur_file = "ur_pred_optimized.txt"
    ui_file = "ui_pred_optimized.txt"

    if not os.path.exists(ur_file) or not os.path.exists(ui_file):
        print("Prediction files ur_pred_optimized.txt or ui_pred_optimized.txt not found.")
        return

    ur = np.loadtxt(ur_file)
    ui = np.loadtxt(ui_file)

    # Spatial coordinate along beam length normalized [0, 1]
    x = np.linspace(0, 1.0, len(ur))

    # Total displacement magnitude: |u| = sqrt(ur^2 + ui^2)
    magnitude = np.sqrt(ur**2 + ui**2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

    # Subplot 1: Real and Imaginary Components
    axes[0].plot(x, ur, label="Real Component ($u_r$)", color="#1f77b4", linewidth=2)
    axes[0].plot(x, ui, label="Imaginary Component ($u_i$)", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_ylabel("Displacement Component", fontsize=12)
    axes[0].set_title("Predicted Frequency-Domain Complex Beam Deflection", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", frameon=True)

    # Subplot 2: Total Vibration Magnitude
    axes[1].plot(x, magnitude, label=r"Total Magnitude $|u| = \sqrt{u_r^2 + u_i^2}$", color="#2ca02c", linewidth=2)
    axes[1].set_xlabel("Normalized Beam Coordinate ($x / L$)", fontsize=12)
    axes[1].set_ylabel("Displacement Magnitude", fontsize=12)
    axes[1].set_title("Resultant Beam Vibration Amplitude Profile", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    output_png = "beam_vibration_field.png"
    plt.savefig(output_png)
    print(f"Successfully generated beam vibration plot: {output_png}")


if __name__ == "__main__":
    visualize_beam_predictions()
