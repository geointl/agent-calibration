"""Differentiable NorSand (Diff-NorSand) triaxial simulation implemented in PyTorch."""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List

import torch
from torch import Tensor


CONFIG: Dict[str, Dict[str, Any]] = {
    "simulation": {
        "MaxStrainForSim": 0.22,
        "NumTheoPts": 3700,
        "drainage": "undrained",  # choices: "drained", "undrained"
        "batch_size": 3,  # None for scalar simulation, int for batched simulation
        # Available: eps1, epsV, p, q, e, pimg_over_p, pmax_over_p, psi, D, eta, ue
        "output_keys": None,  # None for all keys, or list of keys to save
    },
    "material_parameters": {
        "csl": {
            "Gamma": 0.85,
            "lambda": 0.025,
        },
        "plasticity": {
            "Mtc": 1.25,
            "N": 0.32,
            "chi_tc": 1.8,
            "H0": 250.0,
            "Hy": 60.0,
        },
        "elasticity": {
            "G_at_p0": 80_000.0,  # kPa
            "Gexp": 0.5,
            "nu": 0.25,
        },
        "soil_state": {
            "e0": 0.685,  # Void ratio corresponding to psi=-0.05 at p=100
            "sigM0": 100.0,  # kPa
            "eta0": 0.1,
        },
    },
}

TensorDict = Dict[str, Any]


class NorSandModel:
    """Semi-log NorSand constitutive model with PyTorch autograd support."""

    def __init__(
        self, config: Dict[str, Dict[str, Any]], dtype: torch.dtype = torch.float64, device: torch.device | None = None
    ) -> None:
        """Create a differentiable NorSand model.

        Args:
            config: Nested dictionary with simulation, CSL, plasticity, elasticity, and soil_state blocks.
            dtype: Torch dtype used for all calculations.
            device: Torch device. Defaults to CPU.
        """
        self.config = config
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self._batched_sections = ("csl", "plasticity", "elasticity", "soil_state")
        self.inputs = self._tensorize_inputs(config)
        sim_cfg = self.config.get("simulation")
        self.batch_size = sim_cfg.get("batch_size")
        if self.batch_size:
            self._expand_batch_inputs(self.inputs, int(self.batch_size))

    def _tensorize_value(self, value: Any) -> Any:
        """Convert numeric-like inputs to differentiable tensors."""
        if isinstance(value, torch.Tensor):
            # Respect existing tensors but align dtype/device with the model
            tensor = value.to(dtype=self.dtype, device=self.device)
        elif isinstance(value, Real):
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        elif isinstance(value, (list, tuple)):
            # Only convert numeric lists, return string lists as-is
            if value and isinstance(value[0], str):
                return value
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        elif hasattr(value, "__array__"):
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        else:
            return value

        if tensor.is_floating_point() and not tensor.requires_grad:
            # Ensure floating tensors participate in autograd by default
            tensor = tensor.clone().detach().requires_grad_()
        return tensor

    def _tensorize_inputs(self, config: Dict[str, Dict[str, Any]]) -> TensorDict:
        """Convert numeric inputs (scalars or batches) to differentiable tensors; assumes config has all sections."""
        sim_cfg = config["simulation"]
        material_cfg = config["material_parameters"]

        inputs: TensorDict = {
            "simulation": {key: self._tensorize_value(value) for key, value in sim_cfg.items()},
            "material_parameters": {},
        }
        for section, params in material_cfg.items():
            inputs["material_parameters"][section] = {key: self._tensorize_value(value) for key, value in params.items()}
        return inputs

    def _expand_batch_inputs(self, inputs: TensorDict, batch_size: int) -> None:
        """Broadcast scalar tensors across the detected batch size; assumes required sections exist."""
        # Validate drainage config length if it's a list
        drainage = self.config["simulation"].get("drainage")
        if isinstance(drainage, (list, tuple)) and len(drainage) != batch_size:
            raise ValueError(f"simulation.drainage expects length {batch_size}, got {len(drainage)}")

        mat_params = inputs["material_parameters"]
        for section in self._batched_sections:
            section_params = mat_params[section]
            for key, value in section_params.items():
                # Skip non-tensor entries (e.g., strings or metadata)
                if not isinstance(value, torch.Tensor):
                    continue
                if value.ndim == 0:
                    # Expand scalar tensors to a view with the requested batch size
                    section_params[key] = value.expand(batch_size)
                    continue
                if value.shape[0] != batch_size:
                    # Guard against malformed tensors that already have a wrong batch dim
                    msg = f"{section}.{key} expects leading batch dim {batch_size}, got {value.shape[0]}"
                    raise ValueError(msg)

    def _drained_mask(self, reference: Tensor) -> Tensor:
        """Build a bool mask that identifies drained samples in the current batch."""
        drainage_cfg = self.config["simulation"]["drainage"]
        if isinstance(drainage_cfg, str):
            is_drained = drainage_cfg.lower() == "drained"
            return torch.full_like(reference, fill_value=is_drained, dtype=torch.bool)

        modes = [mode.lower() == "drained" for mode in drainage_cfg]
        mask = torch.tensor(modes, dtype=torch.bool, device=self.device)
        if reference.ndim == 0:
            return mask.reshape(())
        return mask

    def run(self) -> Dict[str, Any]:
        """Execute the triaxial simulation for scalar or batched configurations."""
        results = self._simulate()
        return {"config": self.config, "results": results}

    def _simulate(self) -> Dict[str, Tensor]:
        """Core NorSand algorithm supporting batched states."""
        sim = self.inputs["simulation"]
        mat_params = self.inputs["material_parameters"]
        csl = mat_params["csl"]
        plasticity = mat_params["plasticity"]
        elasticity = mat_params["elasticity"]
        soil_state = mat_params["soil_state"]

        max_strain = sim["MaxStrainForSim"]
        num_pts = int(sim["NumTheoPts"])
        gamma = csl["Gamma"]
        lambda_ = csl["lambda"]

        Mtc = plasticity["Mtc"]
        N = plasticity["N"]
        chi_tc = plasticity["chi_tc"]
        H0 = plasticity["H0"]
        Hy = plasticity["Hy"]

        G_at_p0 = elasticity["G_at_p0"]
        Gexp = elasticity["Gexp"]
        nu = elasticity["nu"]

        e0 = soil_state["e0"]
        sigM0 = soil_state["sigM0"]
        eta0 = soil_state["eta0"]

        spacing_ratio = torch.exp(torch.tensor(1.0, dtype=self.dtype, device=self.device))
        delta_qp = torch.tensor(3.0, dtype=self.dtype, device=self.device)

        K0 = (1 - eta0 / 2) / (1 + eta0 / 2)
        sig1 = 3 * sigM0 / (1 + 2 * K0)
        sig3 = K0 * sig1
        sigQ = sig1 - sig3
        sigM = sigM0.clone()
        is_drained = self._drained_mask(sigM)
        all_drained = bool(torch.all(is_drained).item())
        all_undrained = bool(torch.all(~is_drained).item())
        sig3_0 = sigM - sigQ / 3.0
        eta = sigQ / sigM
        bulk_water = torch.full_like(sigM, 2_000_000.0)
        ep1 = torch.zeros_like(sigM)
        epV = torch.zeros_like(sigM)
        sample_ht = torch.ones_like(sigM)

        ec = self._semi_log_e_crit(sigM, gamma, lambda_)
        e = e0.clone()
        psi0 = e - ec
        psi = psi0.clone()
        ec_i = self._semi_log_e_crit(sigM / spacing_ratio, gamma, lambda_)
        psi_i = e - ec_i
        # e0 = e.clone()  # Already have e0
        inv_v0 = 1 / (1 + e0)

        Gmax = self._g_max(sigM, sigM0, G_at_p0, Gexp)
        K_over_G = (2 * (1 + nu)) / (3 * (1 - 2 * nu))
        Kmax = Gmax * K_over_G

        # Clamp denominator to prevent chi_i singularity when lambda*chi_tc/Mtc approaches 1.
        chi_i = chi_tc / torch.clamp(1 - lambda_ * chi_tc / Mtc, min=0.1)
        # Clamp Mi positive; _mobilized_friction_ratio can go negative for extreme psi_i.
        Mi = torch.clamp(self._mobilized_friction_ratio(Mtc, N, chi_i, psi_i), min=1e-4)
        # Cap both exponents (both ends) to prevent float64 overflow/underflow:
        # overflow → inf → NaN next step; underflow → 0 → pow(-2) = inf → NaN.
        pimg_over_p = torch.exp(torch.clamp(eta / Mi - 1, min=-50.0, max=50.0))
        pimx_over_p = torch.exp(torch.clamp(-chi_i * psi_i / Mi, max=50.0))

        bulk_pore_fluid = torch.where(is_drained, torch.zeros_like(bulk_water), bulk_water)

        histories = self._init_histories(sigM, sigQ, e, pimg_over_p, pimx_over_p, psi, Mi - eta, eta)

        depGp = max_strain / torch.tensor(num_pts - 1, dtype=self.dtype, device=self.device)

        for _ in range(1, num_pts):
            sigQ_old = sigQ
            sigM_old = sigM

            ec_i = self._semi_log_e_crit(sigM * pimg_over_p, gamma, lambda_)
            ec = self._semi_log_e_crit(sigM, gamma, lambda_)
            psi_i = e - ec_i
            psi = e - ec

            Gmax = self._g_max(sigM, sigM0, G_at_p0, Gexp)
            Kmax = Gmax * K_over_G
            Ktot = (bulk_pore_fluid * Kmax) / (bulk_pore_fluid + Kmax + 1e-12)

            Mi_old = Mi
            # Clamp Mi_tc to stay positive; prevents chi_i denominator from
            # going singular when large chi_tc / small Mtc / negative psi_i combine.
            Mi_tc = torch.clamp(self._mobilized_friction_ratio(Mtc, N, chi_i, psi_i), min=1e-4)
            Mi = Mi_tc
            # Clamp per-step relative change; unclamped, a sudden Mi collapse
            # makes dMi_over_Mi thousands-fold negative → eta spirals → pimg underflows.
            dMi_over_Mi = torch.clamp(1 - Mi_old / Mi, min=-2.0, max=2.0)
            Dp = Mi - eta
            depVp = Dp * depGp

            # Mirror the same clamp used at initialization (line ~208) to guard
            # against the denominator crossing zero during large-strain evolution.
            chi_i = chi_tc / torch.clamp(1 - lambda_ * chi_tc / Mi_tc, min=0.1)
            Dmin = chi_i * psi_i
            # Same exponent cap as at initialization to prevent float64 overflow.
            pimx_over_p = torch.exp(torch.clamp(-chi_i * psi_i / Mi_tc, max=50.0))

            dpimg_over_pimg = (
                self._hardening_modulus(psi, lambda_, H0, Hy) * pimg_over_p.pow(-2) * (pimx_over_p - pimg_over_p)
            ) * depGp

            if all_drained:
                eta_ratio = 1 + Mi / (delta_qp - eta)
                deta = (eta * dMi_over_Mi + Mi * dpimg_over_pimg) / eta_ratio
                # Clamp eta below delta_qp; overshoot makes delta_qp - eta negative,
                # causing dsigM to blow up exponentially across steps.
                eta = torch.clamp(eta + deta, max=delta_qp - 0.01)
                dsigM = sigM * deta / (delta_qp - eta)
                dsigQ = sigM * deta + eta * dsigM
                # Clamp sigM to stay positive (mirrors the undrained guard below).
                sigM = torch.clamp(sigM + dsigM, min=0.1)
                dsigM = sigM - sigM_old
                sigQ = sigQ + dsigQ
                pimg_over_p = torch.exp(torch.clamp(eta / Mi - 1, min=-50.0, max=50.0))
            elif all_undrained:
                dsigM = -depVp * Ktot
                sigM_candidate = sigM + dsigM
                sigM = torch.clamp(sigM_candidate, min=0.1)
                dsigM = sigM - sigM_old
                # Clamp pimg_over_p > 0 before log; large negative dpimg (softening
                # overshoot) can otherwise flip the sign and produce NaN.
                pimg_over_p = torch.clamp(
                    pimg_over_p * (1 + dpimg_over_pimg) * sigM_old / sigM, min=1e-6
                )
                eta = Mi * (1 + torch.log(pimg_over_p))
                sigQ = sigM * eta
                dsigQ = sigQ - sigQ_old
            else:
                # Mixed drainage batch: evaluate both updates and select sample-wise.
                eta_ratio_drained = 1 + Mi / (delta_qp - eta)
                deta_drained = (eta * dMi_over_Mi + Mi * dpimg_over_pimg) / eta_ratio_drained
                eta_drained = torch.clamp(eta + deta_drained, max=delta_qp - 0.01)
                dsigM_drained = sigM * deta_drained / (delta_qp - eta_drained)
                dsigQ_drained = sigM * deta_drained + eta_drained * dsigM_drained
                sigM_drained = torch.clamp(sigM + dsigM_drained, min=0.1)
                dsigM_drained = sigM_drained - sigM_old
                sigQ_drained = sigQ + dsigQ_drained
                pimg_drained = torch.exp(torch.clamp(eta_drained / Mi - 1, min=-50.0, max=50.0))

                dsigM_undrained = -depVp * Ktot
                sigM_candidate = sigM + dsigM_undrained
                sigM_undrained = torch.clamp(sigM_candidate, min=0.1)
                dsigM_undrained = sigM_undrained - sigM_old
                # Same pimg_over_p positivity guard as the all_undrained branch.
                pimg_undrained = torch.clamp(
                    pimg_over_p * (1 + dpimg_over_pimg) * sigM_old / sigM_undrained, min=1e-6
                )
                eta_undrained = Mi * (1 + torch.log(pimg_undrained))
                sigQ_undrained = sigM_undrained * eta_undrained
                dsigQ_undrained = sigQ_undrained - sigQ_old

                eta = torch.where(is_drained, eta_drained, eta_undrained)
                sigM = torch.where(is_drained, sigM_drained, sigM_undrained)
                sigQ = torch.where(is_drained, sigQ_drained, sigQ_undrained)
                dsigM = torch.where(is_drained, dsigM_drained, dsigM_undrained)
                dsigQ = torch.where(is_drained, dsigQ_drained, dsigQ_undrained)
                pimg_over_p = torch.where(is_drained, pimg_drained, pimg_undrained)

            depGe = dsigQ / (3 * Gmax)
            depVe = dsigM / Kmax

            depV = depVp + depVe
            epV = epV + depV
            depG = depGp + depGe
            dep1 = depG + depV / 3
            ep1 = ep1 + dep1
            e = e - (1 + e) * depV

            sample_ht = sample_ht * (1 - dep1)
            epV_eng = -(e - e0) * inv_v0
            ep1_eng = 1 - sample_ht

            histories["eps1"].append(100 * ep1_eng)
            histories["epsV"].append(100 * epV_eng)
            histories["p"].append(sigM)
            histories["q"].append(sigQ)
            histories["e"].append(e)
            histories["pimg_over_p"].append(pimg_over_p)
            histories["pmax_over_p"].append(pimx_over_p)
            histories["psi"].append(psi)
            histories["D"].append(Dp)
            histories["eta"].append(eta)

            sig3_curr = sigM - sigQ / 3.0
            ue = sig3_0 - sig3_curr
            histories["ue"].append(ue)

        all_results = {key: torch.stack(values) for key, values in histories.items()}
        output_keys = self.config["simulation"].get("output_keys")
        if output_keys is None:
            return all_results
        return {key: all_results[key] for key in output_keys}

    def _init_histories(
        self,
        sigM: Tensor,
        sigQ: Tensor,
        e: Tensor,
        pimg_over_p: Tensor,
        pimx_over_p: Tensor,
        psi: Tensor,
        Dp: Tensor,
        eta: Tensor,
    ) -> Dict[str, List[Tensor]]:
        """Initialize storage for simulation histories."""
        histories: Dict[str, List[Tensor]] = {
            key: [] for key in ("eps1", "epsV", "p", "q", "e", "pimg_over_p", "pmax_over_p", "psi", "D", "eta", "ue")
        }
        zero = torch.zeros_like(sigM)
        histories["eps1"].append(zero)
        histories["epsV"].append(zero.clone())
        histories["p"].append(sigM)
        histories["q"].append(sigQ)
        histories["e"].append(e)
        histories["pimg_over_p"].append(pimg_over_p)
        histories["pmax_over_p"].append(pimx_over_p)
        histories["psi"].append(psi)
        histories["D"].append(Dp)
        histories["eta"].append(eta)
        histories["ue"].append(zero.clone())
        return histories

    @staticmethod
    def _semi_log_e_crit(sigM: Tensor, gamma: Tensor, lambda_: Tensor) -> Tensor:
        """Critical void ratio for semi-log CSL."""
        return gamma - lambda_ * torch.log(sigM)

    @staticmethod
    def _g_max(sigM: Tensor, sigM0: Tensor, G_at_p0: Tensor, Gexp: Tensor) -> Tensor:
        """Elastic shear modulus."""
        return G_at_p0 * (sigM / sigM0).pow(Gexp)

    @staticmethod
    def _mobilized_friction_ratio(Mtc: Tensor, N: Tensor, chi_tc: Tensor, psi: Tensor) -> Tensor:
        """Triaxial compression critical friction ratio."""
        positive = psi > 0
        factor = torch.where(positive, 1 - chi_tc * N * psi / Mtc, 1 + chi_tc * N * psi / Mtc)
        return Mtc * factor

    @staticmethod
    def _hardening_modulus(psi: Tensor, lambda_: Tensor, H0: Tensor, Hy: Tensor) -> Tensor:
        """Plastic hardening modulus with lower bound."""
        H = H0 - Hy * psi
        Hmin = 0.5 / lambda_
        return torch.where(H < Hmin, Hmin, H)


def _tensor_to_python(tensor: Tensor) -> Any:
    """Convert (possibly batched) tensors to Python scalars or lists for printing."""
    data = tensor.detach().cpu()
    if data.numel() == 1:
        return data.item()
    return data.tolist()


if __name__ == "__main__":
    model = NorSandModel(CONFIG)
    output = model.run()
    print({k: _tensor_to_python(v[-1]) for k, v in output["results"].items()})
