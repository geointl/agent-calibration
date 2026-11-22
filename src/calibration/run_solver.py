"""Entry point for running material parameter calibration."""

import sys
from pathlib import Path

# Add project root to path for direct script execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pickle
import yaml
import json
import copy
import torch
import argparse
import src.calibration.solvers as solvers


def main(cfg):
    """
    Run calibration optimization.
    
    Args:
        cfg (dict): Configuration dictionary containing 'OptimizationSettings' 
                    and 'NorSandConfig'.
    """
    device = torch.device('cpu')
    
    solver_type = cfg["OptimizationSettings"]["solver"]
    solver_classes = {
        "adam": solvers.AdamSolver,
        "lbfgsb": solvers.LBFGSBSolver,
    }
    solver = solver_classes[solver_type](cfg, device)
    
    results = solver.optimize()
    
    # Get final physical parameters
    final_params = solver.denormalize_parameters()
    
    # Create a copy of the original config to update
    final_config = copy.deepcopy(cfg)
    
    # Update the config with optimized parameters
    mat_params = final_config["NorSandConfig"]["material_parameters"]
    
    # Helper to convert tensor to python types
    def to_python(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            if v.numel() == 1:
                return float(v.item())
            else:
                return v.tolist()
        return float(v)

    # Update parameters in the config copy
    for section_name, section_params in mat_params.items():
        for param_name in section_params:
            if param_name in final_params:
                section_params[param_name] = to_python(final_params[param_name])

    log_dir = cfg["OptimizationSettings"]["log_dir"]
    # Save the exact config used for this run into the log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save calibrated parameters as JSON
    with open(f"{log_dir}/calibrated_params.json", "w") as f:
        json.dump(final_config, f, indent=2)
    print(f"Calibrated parameters saved to {log_dir}/calibrated_params.json")

    with open(f"{log_dir}/config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    with open(f"{log_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {log_dir}/results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, 
        default="data/synthetic/case01_undrained_loose_p100_calibration.yaml")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
