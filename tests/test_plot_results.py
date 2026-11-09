import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plot_results  # noqa: E402


def test_bundled_predictions_reshape_to_frequency_by_position():
    field = plot_results.load_field(plot_results.SCRIPT_DIR / "ur_pred_optimized.txt")
    assert field.shape == (plot_results.NF, plot_results.NX)


def test_reshape_keeps_rows_as_frequencies(tmp_path):
    # Same layout as FD-PINN_Optimized.py: meshgrid(x, f) flattened row by row.
    x = np.linspace(0, plot_results.L, plot_results.NX)
    f = np.linspace(0, plot_results.FMAX, plot_results.NF)
    X, F = np.meshgrid(x, f)
    np.savetxt(tmp_path / "f.txt", F.flatten())
    field = plot_results.load_field(tmp_path / "f.txt")
    assert np.allclose(field[:, 0], f) and np.allclose(field[7], f[7])


def test_wrong_size_is_rejected(tmp_path):
    np.savetxt(tmp_path / "bad.txt", np.zeros(10))
    with pytest.raises(ValueError, match="expected 41 x 101"):
        plot_results.load_field(tmp_path / "bad.txt")


def test_plot_is_written(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("ur_pred_optimized.txt", "ui_pred_optimized.txt"):
        np.savetxt(tmp_path / name, rng.normal(size=plot_results.NF * plot_results.NX))
    assert plot_results.visualize_beam_predictions(tmp_path).is_file()


def test_missing_files_return_none(tmp_path):
    assert plot_results.visualize_beam_predictions(tmp_path) is None


def test_loss_breakdown_matches_the_reported_final_loss():
    info = plot_results.loss_breakdown(plot_results.SCRIPT_DIR / "loss.dat")
    assert info["steps"] == (0, 3000)
    # The README quotes a final combined loss of 1.58e+02.
    assert info["total"][1] == pytest.approx(158, abs=1)
    assert info["total"][1] < info["total"][0]
    assert list(info["terms"]) == plot_results.LOSS_TERMS


def test_loss_breakdown_rejects_a_short_file(tmp_path):
    np.savetxt(tmp_path / "loss.dat", [[0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="expected at least 8"):
        plot_results.loss_breakdown(tmp_path / "loss.dat")
