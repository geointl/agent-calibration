"""Test-data loading and optimization monitoring."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_test_data(path):
    """Load one triaxial test as a DataFrame of response channels.

    Accepts a tab-separated table with a header row, or the JSON written by the
    report parser: {"drainage": ..., "sigM0": ..., "e0": ..., "data": {channel: [...]}}.
    Channels and units: eps1 (%), q (kPa), ue (kPa), epsV (%), p (kPa).
    """
    if str(path).lower().endswith(".json"):
        with open(path) as f:
            return pd.DataFrame(json.load(f)["data"]).astype(float)
    return pd.read_csv(path, sep="\t")


def print_optimization_status(history, current_iter, true_params, current_params):
    """Print current optimization status."""
    print("\n" + "=" * 60)
    print(f"Iteration {current_iter}")
    print("=" * 60)
    
    iter_data = history[current_iter]
    print(f"Data Loss: {iter_data['data_loss']:.6f}")
    if true_params:
        print(f"Model Loss: {iter_data['model_loss']:.6f}")
        print(f"\n{'Parameter':<15} {'Current':<15} {'True':<15} {'Error %':<10}")
        print("-" * 55)
        for name, true_val in true_params.items():
            curr_val = current_params[name].item()
            error_pct = abs(curr_val - true_val) / abs(true_val) * 100 if abs(true_val) > 1e-6 else abs(curr_val - true_val) * 100
            print(f"{name:<15} {curr_val:<15.4f} {true_val:<15.4f} {error_pct:<10.2f}")
    print("=" * 60 + "\n")


def plot_loss_history(history, save_path):
    """Plot loss history over iterations."""
    iters = sorted(history.keys())
    data_losses = [history[i]["data_loss"] for i in iters]
    model_losses = [history[i]["model_loss"] for i in iters]
    
    # Skip model loss plot if all zeros (no ground truth)
    has_model_loss = any(m > 0 for m in model_losses)
    
    if has_model_loss:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    
    ax1.plot(iters, data_losses, 'b-', linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Data Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    if has_model_loss:
        ax2.plot(iters, model_losses, 'r-', linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Parameter Loss')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_parameter_history(history, bounds, true_params, save_path):
    """Plot parameter evolution over iterations."""
    iters = sorted(history.keys())
    param_names = list(history[iters[0]]["material_params"].keys())
    n_params = len(param_names)
    
    n_cols = min(4, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, param_name in enumerate(param_names):
        ax = axes[idx // n_cols, idx % n_cols]
        values = [history[i]["material_params"][param_name].cpu().item() for i in iters]
        
        ax.plot(iters, values, 'b-', linewidth=1.5, label='Current')
        
        if param_name in bounds:
            b_min, b_max = bounds[param_name]
            ax.axhline(y=b_min, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=b_max, color='gray', linestyle='--', alpha=0.5)
        
        if param_name in true_params:
            ax.axhline(y=true_params[param_name], color='r', linestyle='-', alpha=0.7, label='True')
        
        ax.set_xlabel('Iteration')
        ax.set_title(param_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_params, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_stress_strain_comparison(predicted, ground_truth, save_path):
    """Plot predicted vs ground truth stress-strain curves with typical soil mechanics plots.
    
    Creates a 2x3 subplot showing: 
    (1) eps1-q, (2) p-q, (3) eps1-ue
    (4) eps1-epsV, (5) p-e.
    If ground_truth lacks certain fields, those subplots show predicted data only.
    
    Args:
        predicted: Dict with tensors ('eps1', 'q', 'p', 'epsV', 'e', 'ue') from model prediction.
        ground_truth: Dict with tensors from reference data (may have subset of keys),
            or a list of such dicts (one per test).
        save_path: Path to save the figure.
    """
    # Define the 5 typical soil mechanics plots
    plot_specs = [
        {"x": "eps1", "y": "q", "xlabel": "Axial Strain (%)", "ylabel": "Deviatoric Stress (kPa)", "pos": (0, 0)},
        {"x": "p", "y": "q", "xlabel": "Mean Stress (kPa)", "ylabel": "Deviatoric Stress (kPa)", "pos": (0, 1)},
        {"x": "eps1", "y": "ue", "xlabel": "Axial Strain (%)", "ylabel": "Excess Pore Pressure (kPa)", "pos": (0, 2)},
        {"x": "eps1", "y": "epsV", "xlabel": "Axial Strain (%)", "ylabel": "Volumetric Strain (%)", "pos": (1, 0)},
        {"x": "p", "y": "e", "xlabel": "Mean Stress (kPa)", "ylabel": "Void Ratio (-)", "pos": (1, 1), "xscale": "log"},
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Flatten axes if needed or handle 2D array
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    
    # Hide the last unused subplot (1, 2)
    axes[1, 2].set_visible(False)
    
    for spec in plot_specs:
        row, col = spec["pos"]
        ax = axes[row, col]
        x_key, y_key = spec["x"], spec["y"]
        
        # Check if predicted data has the required keys
        if x_key not in predicted or y_key not in predicted:
            ax.set_visible(False)
            continue
        
        pred_x = predicted[x_key].detach().cpu().numpy()
        pred_y = predicted[y_key].detach().cpu().numpy()
        
        # Plot ground truth if available
        for i, gt in enumerate(ground_truth if isinstance(ground_truth, list) else [ground_truth]):
            if x_key in gt and y_key in gt:
                gt_x = gt[x_key].detach().cpu().numpy()
                gt_y = gt[y_key].detach().cpu().numpy()
                ax.plot(gt_x, gt_y, 'k-', linewidth=2, label='Ground Truth' if i == 0 else None)
        
        # Always plot predicted
        ax.plot(pred_x, pred_y, 'r--', linewidth=1.5, label='Predicted')
        
        ax.set_xlabel(spec["xlabel"], fontsize=10)
        ax.set_ylabel(spec["ylabel"], fontsize=10)
        
        if "xscale" in spec:
            ax.set_xscale(spec["xscale"])
            
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
    
    fig.suptitle('Stress-Strain Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
