"""
Data generation and utility functions for PINNs vs QPINNs benchmark
3D Heat Conduction Equation
"""

import numpy as np
import torch
from typing import Dict, Tuple
import warnings

from config import (
    L, T, alpha, sigma_0, nx, ny, nz, nt,
    initial_condition, boundary_condition, device
)


def generate_training_data() -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate training data for 3D heat conduction equation
    
    Returns:
        Dictionary containing:
        - 'initial': Initial condition points (x, y, z, t=0, u)
        - 'boundary': Boundary condition points (x, y, z, t, u=0)
        - 'interior': Interior collocation points (x, y, z, t)
        - 'trace_points': Points for density matrix evaluation (QPINN)
    """
    print("\nGenerating training data...")
    
    # Create grid
    x_vals = np.linspace(0, L, nx)
    y_vals = np.linspace(0, L, ny)
    z_vals = np.linspace(0, L, nz)
    t_vals = np.linspace(0, T, nt)
    
    training_data = {
        'initial': {'x': [], 'y': [], 'z': [], 't': [], 'u': []},
        'boundary': {'x': [], 'y': [], 'z': [], 't': [], 'u': []},
        'interior': {'x': [], 'y': [], 'z': [], 't': []},
        'trace_points': []
    }
    
    # 1. Initial condition points (t=0)
    print("  Generating initial condition points...")
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                training_data['initial']['x'].append(x)
                training_data['initial']['y'].append(y)
                training_data['initial']['z'].append(z)
                training_data['initial']['t'].append(0.0)
                training_data['initial']['u'].append(initial_condition(x, y, z))
    
    # 2. Boundary condition points
    print("  Generating boundary condition points...")
    for t in t_vals:
        # x = 0 and x = L faces
        for y in y_vals:
            for z in z_vals:
                # x = 0
                training_data['boundary']['x'].append(0.0)
                training_data['boundary']['y'].append(y)
                training_data['boundary']['z'].append(z)
                training_data['boundary']['t'].append(t)
                training_data['boundary']['u'].append(0.0)
                # x = L
                training_data['boundary']['x'].append(L)
                training_data['boundary']['y'].append(y)
                training_data['boundary']['z'].append(z)
                training_data['boundary']['t'].append(t)
                training_data['boundary']['u'].append(0.0)
        
        # y = 0 and y = L faces
        for x in x_vals:
            for z in z_vals:
                # y = 0
                training_data['boundary']['x'].append(x)
                training_data['boundary']['y'].append(0.0)
                training_data['boundary']['z'].append(z)
                training_data['boundary']['t'].append(t)
                training_data['boundary']['u'].append(0.0)
                # y = L
                training_data['boundary']['x'].append(x)
                training_data['boundary']['y'].append(L)
                training_data['boundary']['z'].append(z)
                training_data['boundary']['t'].append(t)
                training_data['boundary']['u'].append(0.0)
        
        # z = 0 and z = L faces
        for x in x_vals:
            for y in y_vals:
                # z = 0
                training_data['boundary']['x'].append(x)
                training_data['boundary']['y'].append(y)
                training_data['boundary']['z'].append(0.0)
                training_data['boundary']['t'].append(t)
                training_data['boundary']['u'].append(0.0)
                # z = L
                training_data['boundary']['x'].append(x)
                training_data['boundary']['y'].append(y)
                training_data['boundary']['z'].append(L)
                training_data['boundary']['t'].append(t)
                training_data['boundary']['u'].append(0.0)
    
    # 3. Interior collocation points
    print("  Generating interior collocation points...")
    interior_x = x_vals[1:-1]  # Exclude boundaries
    interior_y = y_vals[1:-1]
    interior_z = z_vals[1:-1]
    interior_t = t_vals[1:]    # Exclude t=0 (handled by initial condition)
    
    for t in interior_t:
        for x in interior_x:
            for y in interior_y:
                for z in interior_z:
                    training_data['interior']['x'].append(x)
                    training_data['interior']['y'].append(y)
                    training_data['interior']['z'].append(z)
                    training_data['interior']['t'].append(t)
    
    # 4. Trace points for QPINN (subset of interior points)
    print("  Generating trace points for QPINN...")
    from config import TrainingPoint
    n_trace = min(1000, len(training_data['interior']['x']))
    trace_indices = np.random.choice(
        len(training_data['interior']['x']), 
        size=n_trace, 
        replace=False
    )
    
    for idx in trace_indices:
        training_data['trace_points'].append(TrainingPoint(
            x=training_data['interior']['x'][idx],
            y=training_data['interior']['y'][idx],
            z=training_data['interior']['z'][idx],
            t=training_data['interior']['t'][idx],
            type='trace'
        ))
    
    # Convert to PyTorch tensors
    print("  Converting to tensors...")
    for key in ['initial', 'boundary', 'interior']:
        for coord in training_data[key]:
            training_data[key][coord] = torch.tensor(
                training_data[key][coord], 
                dtype=torch.float32, 
                device=device
            ).unsqueeze(1)
    
    print(f"\nTraining data generated:")
    print(f"  - Initial condition points: {len(training_data['initial']['x'])}")
    print(f"  - Boundary condition points: {len(training_data['boundary']['x'])}")
    print(f"  - Interior collocation points: {len(training_data['interior']['x'])}")
    print(f"  - Trace points (QPINN): {len(training_data['trace_points'])}")
    
    return training_data


def generate_test_data(n_points: int = 1000) -> Dict[str, torch.Tensor]:
    """
    Generate test data for model evaluation
    
    Args:
        n_points: Number of test points
        
    Returns:
        Dictionary containing test points and true values
    """
    print(f"\nGenerating {n_points} test points...")
    
    # Random points in the interior
    x_test = np.random.uniform(0.1 * L, 0.9 * L, n_points)
    y_test = np.random.uniform(0.1 * L, 0.9 * L, n_points)
    z_test = np.random.uniform(0.1 * L, 0.9 * L, n_points)
    t_test = np.random.uniform(0, T, n_points)
    
    # Compute true values using initial condition decay
    # (simplified analytical approximation)
    u_true = []
    for i in range(n_points):
        # Approximate solution using diffusion kernel
        u_init = initial_condition(x_test[i], y_test[i], z_test[i])
        # Simple exponential decay approximation
        decay = np.exp(-alpha * np.pi**2 * t_test[i] * 3 / L**2)
        u_true.append(u_init * decay)
    
    test_data = {
        'x': torch.tensor(x_test, dtype=torch.float32, device=device).unsqueeze(1),
        'y': torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1),
        'z': torch.tensor(z_test, dtype=torch.float32, device=device).unsqueeze(1),
        't': torch.tensor(t_test, dtype=torch.float32, device=device).unsqueeze(1),
        'u': torch.tensor(u_true, dtype=torch.float32, device=device).unsqueeze(1),
    }
    
    return test_data


def compute_pde_residual(
    model, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    z: torch.Tensor, 
    t: torch.Tensor,
    alpha_val: float = alpha
) -> torch.Tensor:
    """
    Compute PDE residual for 3D heat equation:
    ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    
    Args:
        model: Neural network model
        x, y, z, t: Input coordinates (requires_grad=True)
        alpha_val: Thermal diffusivity
        
    Returns:
        PDE residual tensor
    """
    # Ensure gradients are enabled
    x.requires_grad_(True)
    y.requires_grad_(True)
    z.requires_grad_(True)
    t.requires_grad_(True)
    
    # Forward pass
    u = model.forward(x, y, z, t)
    
    # First derivatives
    grad_outputs = torch.ones_like(u)
    
    du_dx = torch.autograd.grad(u, x, grad_outputs=grad_outputs, 
                                 create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, grad_outputs=grad_outputs, 
                                 create_graph=True, retain_graph=True)[0]
    du_dz = torch.autograd.grad(u, z, grad_outputs=grad_outputs, 
                                 create_graph=True, retain_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=grad_outputs, 
                                 create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=grad_outputs, 
                                   create_graph=True, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=grad_outputs, 
                                   create_graph=True, retain_graph=True)[0]
    d2u_dz2 = torch.autograd.grad(du_dz, z, grad_outputs=grad_outputs, 
                                   create_graph=True, retain_graph=True)[0]
    
    # PDE residual: ∂u/∂t - α∇²u
    laplacian = d2u_dx2 + d2u_dy2 + d2u_dz2
    residual = du_dt - alpha_val * laplacian
    
    return residual


def stratified_sampling(
    t_values: torch.Tensor, 
    n_samples: int, 
    n_bins: int = 20
) -> torch.Tensor:
    """
    Stratified temporal sampling for better coverage
    
    Args:
        t_values: All time values
        n_samples: Number of samples to select
        n_bins: Number of temporal bins
        
    Returns:
        Indices of selected samples
    """
    t_np = t_values.cpu().numpy()
    unique_times = np.unique(t_np)
    n_time_bins = min(n_bins, len(unique_times))
    samples_per_bin = n_samples // n_time_bins
    
    selected_indices = []
    
    for i in range(n_time_bins):
        if i < len(unique_times):
            t_val = unique_times[i * len(unique_times) // n_time_bins]
            mask = np.isclose(t_np, t_val, rtol=1e-5)
            bin_indices = np.where(mask)[0]
            if len(bin_indices) > 0:
                selected = np.random.choice(
                    bin_indices, 
                    size=min(samples_per_bin, len(bin_indices)), 
                    replace=False
                )
                selected_indices.extend(selected.tolist())
    
    # Fill remaining with random samples
    remaining = n_samples - len(selected_indices)
    if remaining > 0:
        all_indices = set(range(len(t_values)))
        available = list(all_indices - set(selected_indices))
        if available:
            extra = np.random.choice(available, size=min(remaining, len(available)), replace=False)
            selected_indices.extend(extra.tolist())
    
    return torch.tensor(selected_indices[:n_samples], device=t_values.device)


def compute_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions
        targets: True values
        
    Returns:
        Dictionary of metrics (MSE, MAE, Relative L2 error)
    """
    with torch.no_grad():
        mse = torch.mean((predictions - targets)**2).item()
        mae = torch.mean(torch.abs(predictions - targets)).item()
        
        # Relative L2 error
        rel_l2 = torch.norm(predictions - targets) / (torch.norm(targets) + 1e-10)
        rel_l2 = rel_l2.item()
        
        # Max error
        max_error = torch.max(torch.abs(predictions - targets)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rel_l2': rel_l2,
        'max_error': max_error
    }


def evaluate_boundary_error(
    model, 
    training_data: Dict, 
    n_samples: int = 500
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate boundary condition satisfaction
    
    Args:
        model: Trained model
        training_data: Training data dictionary
        n_samples: Number of boundary points to evaluate
        
    Returns:
        Average boundary error and per-face errors
    """
    model.eval()
    
    n_boundary = len(training_data['boundary']['x'])
    indices = torch.randperm(n_boundary)[:min(n_samples, n_boundary)]
    
    x_bd = training_data['boundary']['x'][indices]
    y_bd = training_data['boundary']['y'][indices]
    z_bd = training_data['boundary']['z'][indices]
    t_bd = training_data['boundary']['t'][indices]
    u_bd_true = training_data['boundary']['u'][indices]
    
    with torch.no_grad():
        u_bd_pred = model.forward(x_bd, y_bd, z_bd, t_bd)
        boundary_error = torch.mean((u_bd_pred - u_bd_true)**2).item()
    
    # Per-face errors (simplified)
    face_errors = {}
    eps = 1e-6
    
    # x = 0 face
    mask_x0 = (x_bd < eps).squeeze()
    if mask_x0.sum() > 0:
        face_errors['x=0'] = torch.mean(u_bd_pred[mask_x0]**2).item()
    
    # x = L face
    mask_xL = (x_bd > L - eps).squeeze()
    if mask_xL.sum() > 0:
        face_errors['x=L'] = torch.mean(u_bd_pred[mask_xL]**2).item()
    
    model.train()
    return boundary_error, face_errors


def evaluate_initial_condition_error(
    model, 
    training_data: Dict, 
    n_samples: int = 500
) -> float:
    """
    Evaluate initial condition satisfaction
    
    Args:
        model: Trained model
        training_data: Training data dictionary
        n_samples: Number of initial condition points to evaluate
        
    Returns:
        Initial condition MSE
    """
    model.eval()
    
    n_initial = len(training_data['initial']['x'])
    indices = torch.randperm(n_initial)[:min(n_samples, n_initial)]
    
    x_init = training_data['initial']['x'][indices]
    y_init = training_data['initial']['y'][indices]
    z_init = training_data['initial']['z'][indices]
    t_init = training_data['initial']['t'][indices]
    u_init_true = training_data['initial']['u'][indices]
    
    with torch.no_grad():
        u_init_pred = model.forward(x_init, y_init, z_init, t_init)
        initial_error = torch.mean((u_init_pred - u_init_true)**2).item()
    
    model.train()
    return initial_error
