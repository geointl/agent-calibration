"""Jacobian/Fisher-based identifiability diagnostics.

- Compute the residual Jacobian J at the calibrated parameters
- Form the Fisher matrix F = J^T J
- Extract the weakest directions (smallest eigenvalues)
"""

from __future__ import annotations

from typing import Dict, List, Any

import os
import copy
import pickle

import torch


def _stack_residuals(
    predicted: Dict[str, torch.Tensor],
    ground_truth: Dict[str, torch.Tensor],
    loss_keys: List[str],
    data_scales: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Stack residual vector r(theta) for configured loss keys.

    Residuals are normalized by per-key std (from data_scales) so that the
    Fisher matrix is consistent with the std-normalized loss function.

    Notes:
    - Matches current calibration assumptions in `SolverBase.data_loss_function`:
      same independent variable and predicted length <= ground truth length.
    - Returns a 1D tensor of shape (m,).
    """
    parts: List[torch.Tensor] = []
    for key in loss_keys:
        if key not in predicted or key not in ground_truth:
            continue
        pred = predicted[key]
        tgt = ground_truth[key][: pred.shape[0]]
        residual = (pred - tgt).reshape(-1)
        if data_scales is not None and key in data_scales:
            residual = residual / data_scales[key]
        parts.append(residual)
    if not parts:
        raise ValueError(f"No overlapping keys between predicted and ground truth for loss_keys={loss_keys}")
    return torch.cat(parts, dim=0)


def compute_fisher_from_solver(solver) -> Dict[str, Any]:
    """Compute Fisher matrix F = J^T J from residual Jacobian at current params.

    Args:
        solver: Instance of `src.calibration.solver_base.SolverBase` (or subclass).

    Returns:
        Dictionary with keys:
          - param_names
          - loss_keys
          - available_data_keys
          - J (numpy-like via .cpu())
          - F
          - eigenvalues, eigenvectors (ascending eigenvalues)
          - weak_params_ranked (for smallest-eigenvalue direction)
    """
    device = solver.device
    dtype = torch.float64
    param_names = list(solver.material_params.keys())
    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("No optimizable parameters found; cannot compute Fisher/Jacobian.")

    loss_keys = solver.cfg["OptimizationSettings"].get("loss_keys") or list(solver.forward_model.config["simulation"]["output_keys"] or [])
    if not loss_keys:
        # Fallback: use all keys present in ground truth (stable for diagnostics)
        loss_keys = list(solver.ground_truth_data.keys())

    # Current theta in internal parameterization (normalized where configured)
    theta0 = torch.stack([solver.material_params[name] for name in param_names]).to(device=device, dtype=dtype)

    data_scales = getattr(solver, "data_scales", None)

    def residual_fn(theta_vec: torch.Tensor) -> torch.Tensor:
        predicted = solver.forward_with_theta(theta_vec, param_names=param_names)
        if getattr(solver, "use_interp_loss", False):
            # Same interpolated, std-normalized residuals as the calibration loss
            return solver.stacked_residuals(predicted)
        return _stack_residuals(predicted, solver.ground_truth_data, loss_keys=loss_keys, data_scales=data_scales)

    # Build Jacobian columns via JVP: J[:, j] = jvp(residual_fn, theta0, e_j)
    r0 = residual_fn(theta0)  # shape (m,) = (# output keys * # time steps) = (7400,)
    m = int(r0.numel())  # number of residuals
    J = torch.empty((m, n_params), dtype=dtype, device=device)
    for j in range(n_params):
        v = torch.zeros((n_params,), dtype=dtype, device=device)
        v[j] = 1.0
        _, jvp = torch.autograd.functional.jvp(residual_fn, (theta0,), (v,), create_graph=False, strict=False)
        J[:, j] = jvp

    F = J.transpose(0, 1) @ J
    # Symmetrize for numerical stability
    F = 0.5 * (F + F.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(F)

    # Weakest direction diagnostics (smallest eigenvalue)
    vmin = evecs[:, 0]
    contrib = (vmin.abs() / (vmin.abs().sum() + 1e-12)).detach().cpu().tolist()
    weak_params_ranked = sorted(zip(param_names, contrib), key=lambda x: x[1], reverse=True)

    return {
        "param_names": param_names,
        "loss_keys": loss_keys,
        "available_data_keys": list(solver.ground_truth_data.keys()),
        "J": J.detach().cpu(),
        "F": F.detach().cpu(),
        "eigenvalues": evals.detach().cpu(),
        "eigenvectors": evecs.detach().cpu(),
        "weak_params_ranked": weak_params_ranked,
    }


def write_identifiability_diagnostics(log_dir: str, diagnostics: Dict[str, Any]) -> str:
    """Write Jacobian/Fisher identifiability diagnostics to log_dir; return path."""
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "identifiability_diagnostics.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(diagnostics, f)
    return out_path



def screen_rho(norsand_cfg: Dict[str, Any], test: Dict[str, Any], param: str,
               bounds: List[float], channels: List[str], n: int = 30) -> Dict[str, float]:
    """Sensitivity screen of a proposed test (Eq. 7 of the paper).

    Runs the proposed test while sweeping `param` over `bounds`, with the other
    parameters fixed at their values in `norsand_cfg`, and returns
    rho = max spread / sigma per channel, where sigma is 2% of the predicted
    channel range at the current estimate.
    """
    from src.norsand_model import NorSandModel

    cfg = copy.deepcopy(norsand_cfg)
    cfg["simulation"].update(drainage=[test["drainage"]], batch_size=1)
    mp = cfg["material_parameters"]
    mp["soil_state"].update(e0=[test["e0"]], sigM0=[test["sigM0"]])
    section = next(s for s, d in mp.items() if param in d)

    def run(value=None):
        c = copy.deepcopy(cfg)
        if value is not None:
            c["material_parameters"][section][param] = value
        with torch.no_grad():
            out = NorSandModel(c).run()["results"]
        return {k: out[k][:, 0] for k in channels}

    base = run()
    sweeps = [run(float(v)) for v in torch.linspace(*bounds, n, dtype=torch.float64)]
    # Skip sweep points that leave the model's stable region (non-finite or blown-up responses).
    # shortcut: fixed 1e6 cap on |response| (kPa or strain), revisit if channels change units.
    sweeps = [r for r in sweeps if all(r[k].isfinite().all() and r[k].abs().max() < 1e6 for k in channels)]
    rho = {}
    for k in channels:
        a = torch.stack([r[k] for r in sweeps])
        sigma = 0.02 * (base[k].max() - base[k].min())
        rho[k] = float((a.max(0).values - a.min(0).values).max() / sigma)
    return rho
