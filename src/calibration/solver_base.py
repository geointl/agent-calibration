import torch
import numpy as np
import pandas as pd
import os
from src.norsand_model import NorSandModel as ForwardModel
from src.calibration import utils


class SolverBase(torch.nn.Module):
    """Base class for material parameter calibration solvers."""
    
    def __init__(self, cfg, device):
        """Initialize solver with configuration, forward model, and data."""
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.history = {}
        self.current_iter = 0
        
        self._parse_optimization_config()
        interp_cfg = self.cfg["OptimizationSettings"].get("loss_interpolation") or {}
        self.use_interp_loss = bool(interp_cfg.get("enabled", False))
        self.interp_x_key = interp_cfg.get("x_key", "eps1")
        self.forward_model = ForwardModel(self.cfg["NorSandConfig"], device=device)
        self.material_params = self._initialize_parameters()
        self.ground_truth_data = self._load_ground_truth_data()
        self.data_scales = self._compute_data_scales()
        if self.use_interp_loss:
            self._check_strain_coverage()
        self.optimizer = self.define_optimizer()
    
    def _parse_optimization_config(self):
        """Extract parameters to optimize, bounds, normalization_ranges, and ground truth from config."""
        self.params_to_optimize = []
        self.bounds = {}
        self.normalization_ranges = {}
        self.true_params = {}
        
        # Iterate through parameter groups (e.g., 'elasticity', 'state') in config
        for section, params in self.cfg["OptimizationSettings"]["parameters"].items():
            # Iterate through individual parameters within each group
            for param_name, settings in params.items():
                if settings["optimize"]:
                    self.params_to_optimize.append(param_name)
                    # Store bounds/normalization if defined for the parameter
                    if settings.get("bounds"):
                        self.bounds[param_name] = settings["bounds"]
                    if settings.get("normalization"):
                        self.normalization_ranges[param_name] = settings["normalization"]
                    if settings.get("ground_truth") is not None:
                        self.true_params[param_name] = settings["ground_truth"]
    
    def _get_initial_guess(self, param_name):
        """Get initial guess value for a parameter from config."""
        # Search for parameter in all material sections
        for section in self.cfg["NorSandConfig"]["material_parameters"].values():
            if param_name in section:
                return section[param_name]
    
    def _initialize_parameters(self):
        """Initialize optimizable parameters. Normalizes if normalization_ranges provided, else uses raw value."""
        params = torch.nn.ParameterDict()
        for param_name in self.params_to_optimize:
            value = self._get_initial_guess(param_name)
            
            # Apply min-max normalization if range is specified
            if param_name in self.normalization_ranges:
                norm_min, norm_max = self.normalization_ranges[param_name]
                value = (value - norm_min) / (norm_max - norm_min)
                
            params[param_name] = torch.nn.Parameter(
                torch.tensor(value, dtype=torch.float64, device=self.device)
            )
        return params
    
    def denormalize_parameters(self):
        """Convert parameters to physical scale. Returns raw value if no normalization_ranges."""
        denormalized = {}
        for param_name, param in self.material_params.items():
            # Reverse normalization to get physical value
            if param_name in self.normalization_ranges:
                norm_min, norm_max = self.normalization_ranges[param_name]
                denormalized[param_name] = param * (norm_max - norm_min) + norm_min
            else:
                denormalized[param_name] = param
        return denormalized

    def _denormalize_theta_vec(self, theta_vec: torch.Tensor, param_names: list[str]) -> dict[str, torch.Tensor]:
        """Denormalize a parameter vector (internal space) into physical parameters.

        Args:
            theta_vec: Tensor of shape (n_params,) in the solver's internal parameterization.
            param_names: Names corresponding to theta_vec entries.

        Returns:
            Dict mapping param name -> (possibly denormalized) tensor.
        """
        denorm: dict[str, torch.Tensor] = {}
        for i, name in enumerate(param_names):
            t = theta_vec[i]
            if name in self.normalization_ranges:
                norm_min, norm_max = self.normalization_ranges[name]
                denorm[name] = t * (norm_max - norm_min) + norm_min
            else:
                denorm[name] = t
        return denorm

    def forward_with_theta(self, theta_vec: torch.Tensor, param_names: list[str] | None = None) -> dict[str, torch.Tensor]:
        """Run forward simulation with an explicit parameter vector.

        This is used for identifiability diagnostics (Jacobian/Fisher) without
        mutating `self.material_params` in a way that breaks autograd.

        Args:
            theta_vec: Tensor of shape (n_params,) in internal parameterization.
            param_names: Parameter names in the same order. Defaults to the solver's ordering.

        Returns:
            Model outputs as a dict of tensors (same format as `forward()`).
        """
        if param_names is None:
            param_names = list(self.material_params.keys())
        denorm_params = self._denormalize_theta_vec(theta_vec, param_names)

        batch_size = self.forward_model.batch_size
        for section in self.forward_model.inputs["material_parameters"].values():
            for param_name in section:
                if param_name in denorm_params:
                    param = denorm_params[param_name]
                    if batch_size and param.ndim == 0:
                        param = param.expand(batch_size)
                    section[param_name] = param

        results = self.forward_model.run()["results"]
        return {k: v.squeeze(-1) if v.ndim > 1 else v for k, v in results.items()}
    
    def _load_ground_truth_data(self):
        """Load the observed tests (one TSV/JSON file per test).

        Returns a dict of concatenated channels (used for scales). With the
        interpolated loss the per-test curves are kept in
        ``self.ground_truth_tests``; they may have any length and strain grid.
        Otherwise the tests are stacked to (time, batch) and must lie on the
        simulation's own step grid.
        """
        data_paths = self.cfg["OptimizationSettings"]["data_path"]
        paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        sim = self.cfg["NorSandConfig"]["simulation"]
        if len(paths) != int(sim["batch_size"]):
            raise ValueError(f"{len(paths)} data file(s) but simulation.batch_size = {sim['batch_size']}")
        frames = [utils.load_test_data(path) for path in paths]

        def to_tensors(df):
            return {col: torch.tensor(df[col].values, dtype=torch.float64, device=self.device)
                    for col in df.columns}

        if self.use_interp_loss:
            for path, df in zip(paths, frames):
                if self.interp_x_key not in df:
                    raise ValueError(f"{path} lacks the interpolation key '{self.interp_x_key}'")
                x = df[self.interp_x_key]
                if not np.isfinite(x).all():
                    raise ValueError(f"{path}: '{self.interp_x_key}' contains missing or non-finite values")
                if 0 < x.max() < 1.0:
                    print(f"WARNING: {path}: max {self.interp_x_key} is {x.max():.3g}; "
                          "strains are expected in percent, not as fractions.")
            self.ground_truth_tests = [to_tensors(df) for df in frames]
            keys = set().union(*(t.keys() for t in self.ground_truth_tests))
            return {k: torch.cat([t[k] for t in self.ground_truth_tests if k in t]) for k in keys}

        n_pts = int(sim["NumTheoPts"])
        for path, df in zip(paths, frames):
            if len(df) != n_pts:
                raise ValueError(
                    f"{path} has {len(df)} rows but the simulation has {n_pts} steps. Step-aligned "
                    "comparison only suits data generated on the simulation's own step grid; set "
                    "OptimizationSettings.loss_interpolation.enabled: true for measured data."
                )
        tests = [to_tensors(df) for df in frames]
        if len(tests) == 1:
            return tests[0]  # (time,), matches forward() output after squeeze
        keys = [k for k in tests[0] if all(k in t for t in tests)]
        return {k: torch.stack([t[k] for t in tests], dim=1) for k in keys}  # (time, batch)

    def _check_strain_coverage(self):
        """Fail early if the simulation ends before a measured curve does.

        The axial strain reached by the simulation depends on the test and on
        the parameters, so it is checked at the initial guess; predictions past
        the simulated range would otherwise be held at their last value.
        """
        with torch.no_grad():
            x_pred = self.forward()[self.interp_x_key]
        for j, test in enumerate(self.ground_truth_tests):
            sim_end = float(x_pred[-1, j] if x_pred.ndim == 2 else x_pred[-1])
            obs_end = float(test[self.interp_x_key].max())
            if obs_end > sim_end + 1e-9:
                raise ValueError(
                    f"Test {j + 1}: measured {self.interp_x_key} extends to {obs_end:.2f} % but the "
                    f"simulation window ends at {sim_end:.2f} %; increase simulation.MaxStrainForSim."
                )

    def _compute_data_scales(self):
        """Compute per-key std from ground truth data for loss normalization.

        Each response (q, ue, epsV, ...) is scaled by its std so that
        residuals from different quantities contribute equally to the loss.
        Falls back to 1.0 for near-constant responses.
        """
        scales = {}
        for key, data in self.ground_truth_data.items():
            std = data[torch.isfinite(data)].std()
            scales[key] = std if std > 1e-10 else torch.tensor(1.0, dtype=data.dtype, device=self.device)
        return scales

    def forward(self):
        """Run forward simulation with current parameters."""
        denorm_params = self.denormalize_parameters()
        batch_size = self.forward_model.batch_size
        
        # Update model inputs with current optimized parameters
        for section in self.forward_model.inputs["material_parameters"].values():
            for param_name in section:
                if param_name in denorm_params:
                    param = denorm_params[param_name]
                    # Expand scalar params to match batch dimension if needed
                    if batch_size and param.ndim == 0:
                        param = param.expand(batch_size)
                    section[param_name] = param
        
        results = self.forward_model.run()["results"]
        return {k: v.squeeze(-1) if v.ndim > 1 else v for k, v in results.items()}

    def maybe_write_identifiability_diagnostics(self) -> None:
        """Optionally compute and write Jacobian/Fisher identifiability diagnostics."""
        id_cfg = self.cfg.get("OptimizationSettings", {}).get("identifiability_diagnostics", {}) or {}
        if not bool(id_cfg.get("enabled", False)):
            return

        from src.calibration.identifiability_diagnostics import (
            compute_fisher_from_solver,
            write_identifiability_diagnostics,
        )

        diagnostics = compute_fisher_from_solver(self)
        log_dir = self.cfg["OptimizationSettings"]["log_dir"]
        out_path = write_identifiability_diagnostics(log_dir, diagnostics)
        print(f"Identifiability diagnostics saved to {out_path}")
    
    def define_optimizer(self):
        """Abstract method to define the specific optimizer."""
        raise NotImplementedError
    
    @staticmethod
    def _torch_interp(x_new: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable piecewise-linear interpolation of y(x) at x_new.

        Assumes x is monotonically increasing. Queries outside the range of x
        are clamped to the end values (no extrapolation).
        """
        idx = torch.searchsorted(x.detach(), x_new.detach()).clamp(1, x.shape[0] - 1)
        x0, x1 = x[idx - 1], x[idx]
        dx = x1 - x0
        w = torch.where(dx > 0, (x_new - x0) / torch.where(dx > 0, dx, torch.ones_like(dx)),
                        torch.zeros_like(dx)).clamp(0.0, 1.0)
        return torch.lerp(y[idx - 1], y[idx], w)

    def _interp_residual_blocks(self, predicted_results):
        """Std-normalized residual blocks on the observed strain grids.

        For each test and loss key, the predicted response is interpolated from
        the simulation strain grid onto the observed strain grid of that test.
        Returns a list of 1-D residual tensors (one block per test/key pair).
        """
        loss_keys = self.cfg["OptimizationSettings"].get("loss_keys") or list(predicted_results.keys())
        x_key = self.interp_x_key
        blocks = []
        for j, test in enumerate(self.ground_truth_tests):
            x_pred = predicted_results[x_key]
            x_pred = x_pred[:, j].contiguous() if x_pred.ndim == 2 else x_pred
            x_obs = test[x_key]
            for key in loss_keys:
                if key not in predicted_results or key not in test:
                    continue
                y_pred = predicted_results[key]
                y_pred = y_pred[:, j].contiguous() if y_pred.ndim == 2 else y_pred
                y_obs = test[key]
                seen = torch.isfinite(y_obs)  # blank cells: channel not recorded at that point
                if not seen.any():
                    continue
                y_interp = self._torch_interp(x_obs[seen], x_pred, y_pred)
                y_obs = y_obs[seen]
                # Per-test std: a single-test run is unchanged (concat std ==
                # that test's std). Multi-test runs then do not let a large
                # dense epsV range drown the loose volumetric residuals.
                scale = y_obs.std()
                if float(scale) <= 1e-10:
                    scale = self.data_scales[key]
                blocks.append((y_interp - y_obs) / scale)
        if not blocks:
            raise ValueError("No overlapping keys between prediction and observed data")
        return blocks

    def stacked_residuals(self, predicted_results) -> torch.Tensor:
        """1-D residual vector consistent with the calibration loss (interp mode)."""
        if not self.use_interp_loss:
            raise RuntimeError("stacked_residuals is only defined for the interpolated loss")
        return torch.cat(self._interp_residual_blocks(predicted_results))

    def data_loss_function(self, predicted_results, option):
        """Compute loss between predicted and target sequences.

        Residuals are normalized by target std so that responses with different
        physical scales (e.g. q in kPa vs epsV as fraction) contribute equally.
        Uses 'loss_keys' from config to select which outputs to include in loss.
        Falls back to all predicted keys if 'loss_keys' is not specified.

        When OptimizationSettings.loss_interpolation.enabled is true, predictions
        are compared on each test's observed strain grid via differentiable
        linear interpolation; observed curves may then have any length and
        grid. Otherwise prediction and data are compared step by step, which
        only suits data generated on the simulation's own step grid.
        """
        if option == "mse" and self.use_interp_loss:
            blocks = self._interp_residual_blocks(predicted_results)
            return sum(b.pow(2).mean() for b in blocks) / len(blocks)
        if option == "mse":
            total_loss = 0.0
            num_keys = 0
            loss_keys = self.cfg["OptimizationSettings"].get("loss_keys", predicted_results.keys())
            for key in loss_keys:
                if key in predicted_results and key in self.ground_truth_data:
                    pred = predicted_results[key]
                    target = self.ground_truth_data[key][:len(pred)]
                    # Normalize by target std to equalize contribution across responses
                    scale = self.data_scales[key]
                    total_loss = total_loss + torch.nn.functional.mse_loss(pred / scale, target / scale)
                    num_keys += 1
            loss = total_loss / num_keys
        else:
            raise ValueError(f"Loss option '{option}' not supported.")

        return loss
        
    def model_loss_function(self):
        """Compute MSE between current and ground truth parameters (monitoring only)."""
        if not self.true_params:
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)
        
        denorm_params = self.denormalize_parameters()
        total_loss = 0.0
        for param_name, true_val in self.true_params.items():
            pred_val = denorm_params[param_name]
            total_loss = total_loss + (pred_val - true_val) ** 2
        return total_loss / len(self.true_params)
    
    def store_iteration(self, results, data_loss, model_loss):
        """Store results for current iteration."""
        denorm_params = self.denormalize_parameters()
        self.history[self.current_iter] = {
            "data_loss": data_loss.item(),
            "model_loss": model_loss.item(),
            "material_params": {name: param.clone().detach() for name, param in denorm_params.items()},
            "predictions": {k: v.clone().detach().cpu() for k, v in results.items()},
        }
    
    def optimization_monitoring(self, predicted_results):
        """Monitor optimization and save plots at specified intervals."""
        save_interval = self.cfg["OptimizationSettings"]["save_interval"]
        if self.current_iter % save_interval == 0:
            utils.print_optimization_status(self.history, self.current_iter, self.true_params, self.denormalize_parameters())
            
            monitor_dir = self.cfg["OptimizationSettings"]["log_dir"]
            os.makedirs(monitor_dir, exist_ok=True)
            utils.plot_loss_history(self.history, os.path.join(monitor_dir, "loss_history.png"))
            utils.plot_parameter_history(self.history, self.bounds, self.true_params, os.path.join(monitor_dir, "parameter_history.png"))
            utils.plot_stress_strain_comparison(
                predicted_results, getattr(self, "ground_truth_tests", self.ground_truth_data),
                os.path.join(monitor_dir, f"stress_strain_iter_{self.current_iter:04d}.png")
            )

    def _params_to_scipy(self):
        """Flatten PyTorch parameters to a single NumPy array for Scipy."""
        return torch.stack([p.detach() for p in self.material_params.values()]).cpu().numpy()

    def _scipy_to_params(self, x):
        """Update PyTorch parameters from Scipy's optimization vector."""
        for i, name in enumerate(self.material_params):
            self.material_params[name].data = torch.tensor(x[i], dtype=torch.float64, device=self.device)

    def _get_scipy_bounds(self, default_bound=(None, None), require_bounds=False):
        """Convert configuration bounds to Scipy format."""
        bounds = []
        for name in self.material_params:
            # Normalized parameters are always bound between [0, 1]
            if name in self.normalization_ranges:
                bounds.append((0.0, 1.0))
            # Use user-defined bounds if available
            elif name in self.bounds:
                bounds.append(tuple(self.bounds[name]))
            # Unbounded parameters (None indicates no bound in Scipy)
            else:
                if require_bounds:
                    raise ValueError(f"Optimization bounds required for parameter '{name}' with this solver.")
                bounds.append(default_bound)
        return bounds
