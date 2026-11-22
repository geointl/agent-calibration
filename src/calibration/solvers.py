import torch
import numpy as np
from scipy.optimize import minimize
from src.calibration.solver_base import SolverBase


class AdamSolver(SolverBase):
    """Adam optimizer-based solver."""
    
    def define_optimizer(self):
        """Define the Adam optimizer for the model parameters."""
        return torch.optim.Adam(self.material_params.parameters(), lr=self.cfg["OptimizationSettings"]["learning_rate"])
        
    def optimize(self):
        """Run Adam optimization loop to minimize data loss."""
        n_iterations = self.cfg["OptimizationSettings"]["n_iterations"]
        
        for _ in range(n_iterations):
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            results = self.forward()
            data_loss = self.data_loss_function(
                results, self.cfg["OptimizationSettings"]["loss_function"])
            model_loss = self.model_loss_function()
            
            # Monitoring and logging
            self.store_iteration(results, data_loss, model_loss)
            self.optimization_monitoring(results)
            
            # Backpropagation and parameter update
            data_loss.backward()
            self.optimizer.step()
            
            print(f"Iteration {self.current_iter}, Loss: {data_loss.item():.6f}")
            self.current_iter += 1
                
        self.maybe_write_identifiability_diagnostics()
        return self.history


class LBFGSBSolver(SolverBase):
    """L-BFGS-B optimizer using scipy."""
    
    def define_optimizer(self):
        """No Torch optimizer needed as Scipy's L-BFGS-B is used."""
        return None
    
    def _objective_and_gradient(self, x):
        """Compute objective function and gradient for Scipy's L-BFGS-B."""
        # Sync PyTorch params with Scipy's current x
        self._scipy_to_params(x)
        for p in self.material_params.values():
            if p.grad is not None:
                p.grad.zero_()
        
        # Forward pass
        results = self.forward()
        data_loss = self.data_loss_function(
            results, self.cfg["OptimizationSettings"]["loss_function"])
        model_loss = self.model_loss_function()
        
        # Logging
        self.store_iteration(results, data_loss, model_loss)
        self.optimization_monitoring(results)
        self.current_iter += 1
        
        # Compute gradients
        data_loss.backward()
        grads = [p.grad.detach().cpu().numpy() for p in self.material_params.values()]
        
        print(f"Iteration {self.current_iter}, Loss: {data_loss.item():.6f}")
        loss, grads = data_loss.item(), np.array(grads)
        if not (np.isfinite(loss) and np.isfinite(grads).all()):
            # A trial step left the model's stable region (nan or inf): a large
            # finite loss makes the line search backtrack instead of ending the run.
            return 1e10, np.zeros_like(grads)
        return loss, grads

    def optimize(self):
        """Run L-BFGS-B optimization using Scipy."""
        lbfgsb_cfg = self.cfg["OptimizationSettings"].get("lbfgsb_settings", {})
        n_iterations = self.cfg["OptimizationSettings"]["n_iterations"]
        options = {
            'maxiter': n_iterations,
            'maxfun': lbfgsb_cfg.get("maxfun", int(n_iterations * 1.5)),
            'ftol': lbfgsb_cfg.get("ftol", 2.2e-9),
            'gtol': lbfgsb_cfg.get("gtol", 1e-5),
            'disp': True,
        }
        result = minimize(
            fun=self._objective_and_gradient,
            x0=self._params_to_scipy(),
            method='L-BFGS-B',
            jac=True,
            bounds=self._get_scipy_bounds(),
            options=options,
        )
        
        print(f"\nOptimization completed. Success: {result.success}, Final loss: {result.fun:.6f}")
        
        # Update parameters to the best found solution
        self._scipy_to_params(result.x)
        
        self.maybe_write_identifiability_diagnostics()
        return self.history
