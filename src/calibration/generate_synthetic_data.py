import copy
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.norsand_model import NorSandModel
from src.calibration.utils import plot_stress_strain_comparison

# Default calibration template: full NorSandConfig + OptimizationSettings (agent / run_solver).
CALIBRATION_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "configs" / "synthetic_template.yaml"
)


# ==========================================
# CONFIGURATION
# ==========================================
GENERATION_CONFIG = {
    "output_dir": "data/synthetic",
    # YAML with NorSandConfig + OptimizationSettings; used to copy bounds and
    # guess/truth ratios for perturbed initial guesses in generated calib files.
    "calibration_template": str(CALIBRATION_TEMPLATE_PATH),
    
    # Base configuration for the NorSand model (matches NorSandConfig structure)
    "base_config": {
        "simulation": {
            "MaxStrainForSim": 0.22,
            "NumTheoPts": 3700,
            "drainage": "undrained",
            "batch_size": 1,
            "output_keys": None,  # Save all keys
        },
        "material_parameters": {
            "csl": {
                "Gamma": 0.95,
                "lambda": 0.050,
            },
            "plasticity": {
                "Mtc": 1.20,
                "N": 0.30,
                "chi_tc": 2.5,
                "H0": 250.0,
                "Hy": 10.0,
            },
            "elasticity": {
                "G_at_p0": 10_000.0,
                "Gexp": 0.0,
                "nu": 0.20,
            },
            "soil_state": {
                "e0": 0.685,
                "sigM0": 100.0,
                "eta0": 0.0,
            },
        },
    },

    # Agent evaluation matrix (initial triaxial scenarios): drainage, density
    # state, sigM0 (kPa), e0. psi0 in the paper is the implied state label; we
    # set e0 and sigM0 directly. Material block matches base_config (ground truth).
    #
    # Case 3 (undrained dense): steeper CSL (lambda) so ec catches e by max strain.
    # Cases 6-7 (drained): high chi_tc + moderate H0 so the image reaches the CSL.
    "scenarios": [
        {
            "name": "case01_undrained_loose_p100",
            "overrides": {
                "simulation": {"drainage": "undrained"},
                "material_parameters": {
                    "soil_state": {"sigM0": 100.0, "e0": 0.760},
                },
            },
        },
        {
            "name": "case03_undrained_dense_p100",
            "overrides": {
                "simulation": {"drainage": "undrained"},
                "material_parameters": {
                    "soil_state": {"sigM0": 100.0, "e0": 0.598},
                    # Steeper CSL so ec catches e before max strain (dense undrained).
                    "csl": {"lambda": 0.07},
                },
            },
        },
        {
            "name": "case06_drained_loose_p100",
            "overrides": {
                "simulation": {"drainage": "drained"},
                "material_parameters": {
                    "soil_state": {"sigM0": 100.0, "e0": 0.726},
                    "plasticity": {"chi_tc": 8.0, "H0": 100.0, "Hy": 6.0},
                },
            },
        },
        {
            "name": "case07_drained_dense_p100",
            "overrides": {
                "simulation": {"drainage": "drained"},
                "material_parameters": {
                    # Slightly looser than the old dense baseline; small chi_tc bump keeps
                    # end state near critical without the high-chi instability of very loose drained.
                    "soil_state": {"sigM0": 100.0, "e0": 0.650},
                    "csl": {"lambda": 0.05},
                    "plasticity": {"chi_tc": 8.0, "H0": 100.0, "Hy": 6.0},
                },
            },
        },
    ]
}

def update_config(base, overrides):
    """Recursively update dictionary."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base:
            update_config(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> dict:
    """Load a YAML file into a nested dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class _CompactListDumper(yaml.SafeDumper):
    """SafeDumper that writes sequences as [a, b] instead of block lists."""

def _represent_list_compact(dumper, data):
    return dumper.represent_sequence(
        yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
        data,
        flow_style=True,
    )


_CompactListDumper.add_representer(list, _represent_list_compact)


def _dump_yaml(obj: dict, path: str) -> None:
    """Write a dict to YAML; lists use inline flow style for a compact file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            obj,
            f,
            Dumper=_CompactListDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def _initial_guess_ratios_from_template(template: dict) -> dict:
    """Guess/truth ratio per optimized parameter from a calibration template.

    Used to build perturbed initial guesses for new scenarios while keeping the
    same relative difficulty as the template.

    Args:
        template: Full calibration dict with NorSandConfig and OptimizationSettings.

    Returns:
        Dict keyed by (section, param_name) -> float ratio.
    """
    norsand = template["NorSandConfig"]["material_parameters"]
    opt = template["OptimizationSettings"]["parameters"]
    ratios = {}
    for section, param_specs in opt.items():
        if section not in norsand:
            continue
        sec_guess = norsand[section]
        for pname, spec in param_specs.items():
            if not spec.get("optimize"):
                continue
            if pname not in sec_guess:
                continue
            gt = spec.get("ground_truth")
            guess = sec_guess[pname]
            if isinstance(gt, (int, float)) and abs(float(gt)) > 1e-30:
                ratios[(section, pname)] = float(guess) / float(gt)
            else:
                ratios[(section, pname)] = 1.0
    return ratios


def build_calibration_config(
    scenario_name: str,
    forward_config: dict,
    template_path: Path,
    data_path_relative: str,
) -> dict:
    """Assemble a full agent/run_solver YAML for one synthetic scenario.

    NorSandConfig.material_parameters: true values for fixed params; perturbed
    initial guesses for optimized params (same guess/truth ratios as template).
    OptimizationSettings.parameters: ground_truth set from forward_config truth.

    Args:
        scenario_name: Scenario key (used in default log_dir only).
        forward_config: Merged dict with keys simulation, material_parameters.
        template_path: Path to config.yaml (or compatible) calibration template.
        data_path_relative: data_path entry, e.g. data/synthetic/x.tsv.

    Returns:
        Full calibration config dict.
    """
    template = _load_yaml(template_path)
    ratios = _initial_guess_ratios_from_template(template)
    truth_mp = forward_config["material_parameters"]
    drainage = str(forward_config["simulation"]["drainage"]).lower()

    out = copy.deepcopy(template)
    tmpl_ns = template["NorSandConfig"]

    # Simulation: match synthetic run, single batch, template output_keys for ue/loss.
    out["NorSandConfig"]["simulation"] = copy.deepcopy(forward_config["simulation"])
    out["NorSandConfig"]["simulation"]["batch_size"] = 1
    out["NorSandConfig"]["simulation"]["output_keys"] = copy.deepcopy(
        tmpl_ns["simulation"]["output_keys"]
    )

    # Initial guesses: start from ground truth, then scale optimized params.
    mat = copy.deepcopy(truth_mp)
    for (section, pname), ratio in ratios.items():
        if section in mat and pname in mat[section]:
            mat[section][pname] = float(mat[section][pname]) * ratio
    out["NorSandConfig"]["material_parameters"] = mat

    opt = out["OptimizationSettings"]
    opt["data_path"] = data_path_relative
    opt["loss_keys"] = ["q", "epsV"] if drainage == "drained" else ["q", "ue"]

    # Ground truth for diagnostics / fixed params must match synthetic forward model.
    for section, param_specs in opt["parameters"].items():
        if section not in truth_mp:
            continue
        truth_sec = truth_mp[section]
        for pname, spec in param_specs.items():
            if pname in truth_sec:
                spec["ground_truth"] = copy.deepcopy(truth_sec[pname])

    opt["log_dir"] = f"logs/synthetic_{scenario_name}"
    return out


def generate_data():
    output_dir = GENERATION_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    template_path = Path(GENERATION_CONFIG["calibration_template"])

    base_config = GENERATION_CONFIG["base_config"]
    scenarios = GENERATION_CONFIG["scenarios"]

    print(f"Generating synthetic data for {len(scenarios)} scenarios...")

    for scenario in scenarios:
        # Create a deep copy of the base config
        current_config = copy.deepcopy(base_config)
        
        # Apply overrides
        overrides = scenario.get("overrides", {})
        update_config(current_config, overrides)
        
        # Ensure batch_size is 1
        current_config["simulation"]["batch_size"] = 1

        data_rel = f"{output_dir}/{scenario['name']}.tsv"
        calib = build_calibration_config(
            scenario["name"],
            current_config,
            template_path,
            data_rel,
        )
        calib_path = f"{output_dir}/{scenario['name']}_calibration.yaml"
        _dump_yaml(calib, calib_path)
        print(f"Saved calibration config: {calib_path}")

        # Run Model
        model = NorSandModel(current_config)
        results = model.run()["results"]
        
        # Convert to numpy/pandas
        data = {k: v.detach().squeeze().numpy() for k, v in results.items()}
        df = pd.DataFrame(data)
        
        # Save to TSV
        filename = f"{output_dir}/{scenario['name']}.tsv"
        df.to_csv(filename, sep='\t', index=False)
        print(f"Saved: {filename}")

        # Save plot
        plot_filename = f"{output_dir}/{scenario['name']}.png"
        plot_stress_strain_comparison(results, {}, plot_filename)
        print(f"Saved plot: {plot_filename}")

def generate_single_test(ground_truth_params, e0, sigM0, drainage, output_path,
                         max_strain=0.22, num_pts=3700):
    """Run forward model with ground truth params at given test conditions and save TSV.

    Args:
        ground_truth_params: Dict with material parameter ground truth values keyed by
            section, e.g. {"csl": {"Gamma": 0.95, ...}, "plasticity": {...}, ...}.
        e0: Initial void ratio for the test.
        sigM0: Initial mean effective stress (kPa).
        drainage: "drained" or "undrained".
        output_path: File path to write the TSV output.
        max_strain: Maximum axial strain for simulation.
        num_pts: Number of time steps.

    Returns:
        The output_path where the TSV was saved.
    """
    # Build a single-test forward model config from ground truth
    config = {
        "simulation": {
            "MaxStrainForSim": max_strain,
            "NumTheoPts": num_pts,
            "drainage": drainage,
            "batch_size": 1,
            "output_keys": None,
        },
        "material_parameters": copy.deepcopy(ground_truth_params),
    }
    # Override soil state with the requested test conditions
    config["material_parameters"]["soil_state"] = {
        "e0": e0, "sigM0": sigM0, "eta0": 0.0,
    }

    model = NorSandModel(config)
    results = model.run()["results"]

    # Save to TSV
    data = {k: v.detach().squeeze().numpy() for k, v in results.items()}
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Generated virtual test: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_data()
