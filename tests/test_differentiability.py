from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch
import pytest

import src.norsand_model as nm


def _linspace(low: float, high: float, steps: int) -> torch.Tensor:
    """Return double-precision linspace tensor."""
    return torch.linspace(low, high, steps=steps, dtype=torch.float64)


def _batched_config(batch_size: int) -> dict:
    """Build config spanning parameter ranges for the given batch size."""
    cfg = deepcopy(nm.CONFIG)
    cfg["simulation"]["batch_size"] = batch_size
    mat = cfg["material_parameters"]
    mat["csl"]["Gamma"] = _linspace(0.7, 0.95, batch_size)
    mat["csl"]["lambda"] = _linspace(0.015, 0.035, batch_size)
    mat["plasticity"]["Mtc"] = _linspace(1.0, 1.8, batch_size)
    mat["plasticity"]["N"] = _linspace(0.2, 0.4, batch_size)
    mat["plasticity"]["chi_tc"] = _linspace(1.2, 2.2, batch_size)
    mat["plasticity"]["H0"] = _linspace(200.0, 300.0, batch_size)
    mat["plasticity"]["Hy"] = _linspace(40.0, 80.0, batch_size)
    mat["elasticity"]["G_at_p0"] = _linspace(60_000.0, 100_000.0, batch_size)
    mat["elasticity"]["Gexp"] = _linspace(0.3, 0.7, batch_size)
    mat["elasticity"]["nu"] = _linspace(0.2, 0.35, batch_size)
    mat["soil_state"]["e0"] = _linspace(0.64, 0.73, batch_size)
    mat["soil_state"]["sigM0"] = _linspace(80.0, 200.0, batch_size)
    mat["soil_state"]["eta0"] = _linspace(0.05, 0.25, batch_size)
    return cfg


def _collect_param_refs(model: nm.NorSandModel) -> Dict[str, torch.Tensor]:
    """Map parameter names to underlying tensors in the model."""
    refs: Dict[str, torch.Tensor] = {}
    for section, params in model.inputs["material_parameters"].items():
        for name, tensor in params.items():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                refs[f"{section}.{name}"] = tensor
    return refs


def _scalar_config_from_batch(batch_cfg: dict, sample_idx: int) -> dict:
    """Create a single-sample config by slicing a batched config at sample_idx."""
    cfg = deepcopy(batch_cfg)
    cfg["simulation"]["batch_size"] = None
    drainage = cfg["simulation"]["drainage"]
    if isinstance(drainage, (list, tuple)):
        cfg["simulation"]["drainage"] = drainage[sample_idx]

    for section in cfg["material_parameters"].values():
        for name, value in section.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                section[name] = value[sample_idx]
            elif isinstance(value, (list, tuple)) and value and not isinstance(value[0], str):
                section[name] = value[sample_idx]
    return cfg


@pytest.mark.parametrize("batch_size", (1, 3, 5))
def test_norsand_gradients_are_finite_across_ranges(batch_size: int) -> None:
    """Ensure gradients exist and remain finite across representative parameter ranges."""
    cfg = _batched_config(batch_size)
    model = nm.NorSandModel(cfg)
    param_refs = _collect_param_refs(model)

    results = model.run()["results"]
    terminal_state = results["p"][-1] + results["q"][-1]
    loss = terminal_state.sum()
    loss.backward()

    assert param_refs, "Expected at least one differentiable parameter."
    for name, tensor in param_refs.items():
        grad = tensor.grad
        assert grad is not None, f"{name} gradient missing."
        assert torch.isfinite(grad).all(), f"{name} has non-finite gradients."
        assert grad.shape == (batch_size,), f"{name} gradient wrong shape."


def test_mixed_drainage_batch_matches_independent_single_runs() -> None:
    """Mixed drained/undrained batches should match per-sample standalone simulations."""
    cfg = _batched_config(batch_size=2)
    cfg["simulation"]["drainage"] = ["drained", "undrained"]

    mixed_results = nm.NorSandModel(cfg).run()["results"]
    for idx in range(2):
        single_cfg = _scalar_config_from_batch(cfg, sample_idx=idx)
        single_results = nm.NorSandModel(single_cfg).run()["results"]
        for key, mixed_history in mixed_results.items():
            torch.testing.assert_close(
                mixed_history[:, idx],
                single_results[key],
                atol=1e-10,
                rtol=1e-9,
                msg=f"Mismatch for key={key}, sample={idx}",
            )


def test_mixed_drainage_batch_is_end_to_end_differentiable() -> None:
    """Gradients should remain finite when drainage differs across batch samples."""
    batch_size = 4
    cfg = _batched_config(batch_size=batch_size)
    cfg["simulation"]["drainage"] = ["drained", "undrained", "drained", "undrained"]
    model = nm.NorSandModel(cfg)
    param_refs = _collect_param_refs(model)

    results = model.run()["results"]
    terminal_state = results["p"][-1] + results["q"][-1] + 0.1 * results["epsV"][-1]
    loss = terminal_state.sum()
    loss.backward()

    assert param_refs, "Expected at least one differentiable parameter."
    for name, tensor in param_refs.items():
        grad = tensor.grad
        assert grad is not None, f"{name} gradient missing."
        assert torch.isfinite(grad).all(), f"{name} has non-finite gradients in mixed drainage mode."
        assert grad.shape == (batch_size,), f"{name} gradient wrong shape in mixed drainage mode."
