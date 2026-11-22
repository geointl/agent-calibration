from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import pytest
import torch
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - matplotlib is optional for tests
    plt = None  # type: ignore

from src.norsand_model import NorSandModel

REF_DATA_DIR = Path(__file__).resolve().with_name("ref_data")
PLOT_SERIES = {
    "subplot_0_0": {
        "coords": (0, 0),
        "x": "eps1",
        "y": "q",
        "xlabel": "Axial strain (%)",
        "ylabel": "Deviatoric stress (kPa)",
    },
    "subplot_0_1": {
        "coords": (0, 1),
        "x": "p",
        "y": "q",
        "xlabel": "Mean effective stress (kPa)",
        "ylabel": "Deviatoric stress (kPa)",
    },
    "subplot_1_0": {
        "coords": (1, 0),
        "x": "eps1",
        "y": "epsV",
        "xlabel": "Axial strain (%)",
        "ylabel": "Volumetric strain (%)",
    },
    "subplot_1_1": {
        "coords": (1, 1),
        "x": "p",
        "y": "e",
        "xlabel": "Mean effective stress (kPa)",
        "ylabel": "Void ratio (-)",
    },
}
SIM_TO_REF_KEYS = {
    "eps1": "eps1",
    "epsV": "epsV",
    "p": "p",
    "q": "q",
    "e": "e",
}
CASES = [
    ("config_drained_loose_1.yaml", "drained_loose_1.tsv"),
    ("config_drained_dense_1.yaml", "drained_dense_1.tsv"),
    ("config_undrained_loose_1.yaml", "undrained_loose_1.tsv"),
    ("config_undrained_dense_1.yaml", "undrained_dense_1.tsv"),
]
ATOL = 1e-7
RTOL = 1e-6
SAVE_PLOTS = bool(os.getenv("NORSAND_SAVE_PLOTS"))  # set to save comparison plots
PLOT_OUTPUT_DIR = Path("logs/comparison_to_ref_data")


def _load_config(config_path: Path) -> Dict:
    """Return a configuration dictionary from YAML."""
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _flatten_series(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a history tensor to a contiguous 1-D CPU tensor."""
    return tensor.detach().cpu().reshape(-1)


def _load_reference_columns(tsv_path: Path, columns: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Load selected TSV columns into float64 tensors."""
    column_names = tuple(columns)
    selected = {column: [] for column in column_names}
    with tsv_path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            for column in column_names:
                selected[column].append(float(row[column]))
    return {column: torch.tensor(values, dtype=torch.float64) for column, values in selected.items()}


def _maybe_plot_curves(
    case_label: str,
    subplot_curves: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Optionally emit 2x2 comparison plots, gated behind env variables."""
    if not SAVE_PLOTS or plt is None:
        return
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for subplot, (sim_curve, ref_curve) in subplot_curves.items():
        metadata = PLOT_SERIES[subplot]
        row, col = metadata["coords"]
        axis = axes[row][col]
        axis.plot(ref_curve[:, 0], ref_curve[:, 1], label="reference", linestyle="--", linewidth=2)
        axis.plot(sim_curve[:, 0], sim_curve[:, 1], label="simulation", linestyle="-", linewidth=1.5)
        axis.set_xlabel(metadata["xlabel"])
        axis.set_ylabel(metadata["ylabel"])
        axis.set_title(subplot)
        axis.grid(True, linestyle=":", linewidth=0.5)
        axis.legend()
    fig.suptitle(f"Diff-NorSand vs reference — {case_label}", fontsize=14)
    fig.tight_layout()
    figure_path = PLOT_OUTPUT_DIR / f"{case_label.replace('.yaml', '')}_comparison.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


@pytest.mark.parametrize(
    ("config_name", "tsv_name"),
    CASES,
    ids=[
        "drained_loose_reference_curve",
        "drained_dense_reference_curve",
        "undrained_loose_reference_curve",
        "undrained_dense_reference_curve",
    ],
)
def test_simulation_curves_match_reference(config_name: str, tsv_name: str) -> None:
    """Ensure simulated histories overlap with reference data on key response plots."""
    config = _load_config(REF_DATA_DIR / config_name)
    model = NorSandModel(config)
    results = model.run()["results"]
    reference = _load_reference_columns(REF_DATA_DIR / tsv_name, SIM_TO_REF_KEYS.values())

    subplot_curves = {}
    for subplot, metadata in PLOT_SERIES.items():
        x_key, y_key = metadata["x"], metadata["y"]
        sim_curve = torch.stack((_flatten_series(results[x_key]), _flatten_series(results[y_key])), dim=1)
        ref_curve = torch.stack((reference[SIM_TO_REF_KEYS[x_key]], reference[SIM_TO_REF_KEYS[y_key]]), dim=1)
        torch.testing.assert_close(
            sim_curve,
            ref_curve,
            atol=ATOL,
            rtol=RTOL,
            msg=f"{subplot} deviates for case {config_name}",
        )
        subplot_curves[subplot] = (sim_curve, ref_curve)

    _maybe_plot_curves(config_name, subplot_curves)
    
if __name__ == "__main__":
    test_simulation_curves_match_reference("config_drained_loose_1.yaml", "drained_loose_1.tsv")
    test_simulation_curves_match_reference("config_drained_dense_1.yaml", "drained_dense_1.tsv")
    test_simulation_curves_match_reference("config_undrained_loose_1.yaml", "undrained_loose_1.tsv")
    test_simulation_curves_match_reference("config_undrained_dense_1.yaml", "undrained_dense_1.tsv")
