#!/usr/bin/env python3
"""
Main benchmark script for PINNs vs QPINNs comparison
3D Heat Conduction Equation

Optimizers:
- PINNs: RAdam (PyTorch)
- QPINNs: SPSA (PennyLane v0.43)
- Optional: GQE-GPT circuit generation for QPINNs

Usage:
    python main.py                          # Run full benchmark
    python main.py --pinn-only              # Run only PINN
    python main.py --qpinn-only             # Run only QPINN  
    python main.py --qpinn-only --use-gqe   # Run QPINN with GQE optimization
    python main.py --epochs 1000 500        # Custom epochs (PINN, QPINN)
    python main.py --help                   # Show help
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import json
import os
import sys
from datetime import datetime

from config import (
    L, T, alpha, sigma_0, nx, ny, nz, nt,
    pinn_epochs, qnn_epochs, device,
    initial_condition, analytical_solution, print_config, set_random_seed,
    PINN_CONFIG, QPINN_CONFIG, GQE_CONFIG
)
from data_utils import (
    generate_training_data, generate_test_data,
    compute_metrics, evaluate_boundary_error, evaluate_initial_condition_error
)
from pinn_model import create_pinn_model, PINN3DHeatSolver
from qpinn_model import create_qpinn_model, QPINN3DHeatSolver


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PINNs vs QPINNs Benchmark for 3D Heat Equation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                         # Full benchmark (PINN + QPINN)
  python main.py --pinn-only             # Train and evaluate PINN only
  python main.py --qpinn-only            # Train and evaluate QPINN only
  python main.py --use-gqe               # Use GQE-GPT circuit optimization
  python main.py --epochs 2000 150       # Set PINN epochs=2000, QPINN epochs=150
  python main.py --qubits 8 --layers 6   # QPINN with 8 qubits, 6 layers
  python main.py --seed 123              # Set random seed
  python main.py --output results_exp1   # Save to custom directory
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--pinn-only', action='store_true',
        help='Run only PINN benchmark'
    )
    mode_group.add_argument(
        '--qpinn-only', action='store_true',
        help='Run only QPINN benchmark'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', nargs=2, type=int, metavar=('PINN', 'QPINN'),
        default=[pinn_epochs, qnn_epochs],
        help=f'Number of epochs for PINN and QPINN (default: {pinn_epochs} {qnn_epochs})'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # QPINN specific
    parser.add_argument(
        '--qubits', type=int, default=QPINN_CONFIG['n_qubits'],
        help=f'Number of qubits for QPINN (default: {QPINN_CONFIG["n_qubits"]})'
    )
    parser.add_argument(
        '--layers', type=int, default=QPINN_CONFIG['n_layers'],
        help=f'Number of circuit layers for QPINN (default: {QPINN_CONFIG["n_layers"]})'
    )
    
    # GQE options
    parser.add_argument(
        '--use-gqe', action='store_true', default=True,
        help='Use GQE-GPT circuit optimization for QPINN (default: enabled)'
    )
    parser.add_argument(
        '--no-gqe', action='store_true',
        help='Disable GQE-GPT circuit optimization for QPINN'
    )
    parser.add_argument(
        '--gqe-iterations', type=int, default=5,
        help='Number of GQE optimization iterations (default: 5)'
    )
    parser.add_argument(
        '--dynamic-update', action='store_true',
        help='Enable dynamic circuit updates during QPINN training'
    )
    
    # PINN specific options
    parser.add_argument(
        '--adaptive-weights', action='store_true', default=True,
        help='Use adaptive loss weighting for PINN (default: True)'
    )
    parser.add_argument(
        '--no-adaptive-weights', action='store_true',
        help='Disable adaptive loss weighting for PINN'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output directory for results (default: results_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--log-dir', type=str, default='logs',
        help='Directory for log files (default: logs)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable visualization generation'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Reduce output verbosity'
    )
    
    # Test data
    parser.add_argument(
        '--test-points', type=int, default=1000,
        help='Number of test points for evaluation (default: 1000)'
    )
    
    return parser.parse_args()


def evaluate_model(model, test_data, model_name: str = "Model", quiet: bool = False):
    """Evaluate model on test data"""
    if not quiet:
        print(f"\nEvaluating {model_name}...")
    
    x_test = test_data['x']
    y_test = test_data['y']
    z_test = test_data['z']
    t_test = test_data['t']
    u_true = test_data['u']
    
    if isinstance(model, PINN3DHeatSolver):
        model.eval()
        with torch.no_grad():
            u_pred = model.forward(x_test, y_test, z_test, t_test)
        model.train()
    else:
        # QPINN model - convert torch tensors to numpy
        x_np = x_test.cpu().numpy() if isinstance(x_test, torch.Tensor) else x_test
        y_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        z_np = z_test.cpu().numpy() if isinstance(z_test, torch.Tensor) else z_test
        t_np = t_test.cpu().numpy() if isinstance(t_test, torch.Tensor) else t_test
        
        u_pred = model.forward(x_np, y_np, z_np, t_np)
        u_pred = torch.tensor(u_pred, dtype=torch.float32, device=device)
    
    metrics = compute_metrics(u_pred, u_true)
    
    if not quiet:
        print(f"  MSE: {metrics['mse']:.6e}")
        print(f"  MAE: {metrics['mae']:.6e}")
        print(f"  Relative L2: {metrics['rel_l2']:.6e}")
        print(f"  Max Error: {metrics['max_error']:.6e}")
    
    return metrics, u_pred


def visualize_results(
    pinn_model, qpinn_model, 
    test_data, training_data,
    pinn_history, qpinn_history,
    pinn_metrics, qpinn_metrics,
    save_dir: str = "results"
):
    """Create comprehensive visualization of comparison results"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # 1. Loss curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if pinn_history is not None:
        axes[0].semilogy(pinn_history['total_loss'], label='PINN (RAdam)', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('PINN Training Loss (RAdam)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'PINN not trained', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('PINN Training Loss')
    
    if qpinn_history is not None:
        axes[1].semilogy(qpinn_history['total_loss'], label='QPINN (SPSA)', linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('QPINN Training Loss (SPSA)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'QPINN not trained', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('QPINN Training Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Solution comparison at t=0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Create evaluation grid
    x_eval = np.linspace(0, L, 50)
    y_eval = np.linspace(0, L, 50)
    X, Y = np.meshgrid(x_eval, y_eval)
    
    z_slice = L / 2
    t_slice = 0.0
    
    # True solution
    U_true = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U_true[i, j] = initial_condition(X[i, j], Y[i, j], z_slice)
    
    # Convert to tensors for PINN, numpy for QPINN
    x_flat_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    y_flat_tensor = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    z_flat_tensor = torch.full_like(x_flat_tensor, z_slice)
    t_flat_tensor = torch.full_like(x_flat_tensor, t_slice)
    
    # Numpy arrays for QPINN
    x_flat_np = X.flatten().astype(np.float64)
    y_flat_np = Y.flatten().astype(np.float64)
    z_flat_np = np.full_like(x_flat_np, z_slice)
    t_flat_np = np.full_like(x_flat_np, t_slice)
    
    # PINN prediction
    if pinn_model is not None:
        pinn_model.eval()
        with torch.no_grad():
            u_pinn = pinn_model.forward(x_flat_tensor, y_flat_tensor, z_flat_tensor, t_flat_tensor)
        U_pinn = u_pinn.cpu().numpy().reshape(X.shape)
    else:
        U_pinn = np.zeros_like(U_true)
    
    # QPINN prediction
    if qpinn_model is not None:
        u_qpinn = qpinn_model.forward(x_flat_np, y_flat_np, z_flat_np, t_flat_np)
        U_qpinn = u_qpinn.reshape(X.shape)
    else:
        U_qpinn = np.zeros_like(U_true)
    
    # Plotting
    vmin = min(U_true.min(), U_pinn.min() if pinn_model else 0, U_qpinn.min() if qpinn_model else 0)
    vmax = max(U_true.max(), U_pinn.max() if pinn_model else 1, U_qpinn.max() if qpinn_model else 1)
    
    im1 = axes[0].contourf(X, Y, U_true, levels=50, cmap='hot', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'True Solution (t={t_slice}, z={z_slice})')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    pinn_title = f'PINN (RAdam) MSE={pinn_metrics["mse"]:.2e}' if pinn_metrics else 'PINN (not trained)'
    im2 = axes[1].contourf(X, Y, U_pinn, levels=50, cmap='hot', vmin=vmin, vmax=vmax)
    axes[1].set_title(pinn_title)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    qpinn_title = f'QPINN (SPSA) MSE={qpinn_metrics["mse"]:.2e}' if qpinn_metrics else 'QPINN (not trained)'
    im3 = axes[2].contourf(X, Y, U_qpinn, levels=50, cmap='hot', vmin=vmin, vmax=vmax)
    axes[2].set_title(qpinn_title)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'solution_comparison_t0.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error distribution
    if pinn_model is not None or qpinn_model is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if pinn_model is not None:
            error_pinn = np.abs(U_pinn - U_true)
            im1 = axes[0].contourf(X, Y, error_pinn, levels=50, cmap='viridis')
            axes[0].set_title('PINN Absolute Error')
        else:
            axes[0].text(0.5, 0.5, 'PINN not trained', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('PINN Error')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        
        if qpinn_model is not None:
            error_qpinn = np.abs(U_qpinn - U_true)
            im2 = axes[1].contourf(X, Y, error_qpinn, levels=50, cmap='viridis')
            axes[1].set_title('QPINN Absolute Error')
        else:
            axes[1].text(0.5, 0.5, 'QPINN not trained', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('QPINN Error')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Metrics comparison bar chart
    if pinn_metrics is not None and qpinn_metrics is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_names = ['MSE', 'MAE', 'Rel L2', 'Max Error']
        pinn_values = [pinn_metrics['mse'], pinn_metrics['mae'], 
                       pinn_metrics['rel_l2'], pinn_metrics['max_error']]
        qpinn_values = [qpinn_metrics['mse'], qpinn_metrics['mae'],
                        qpinn_metrics['rel_l2'], qpinn_metrics['max_error']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pinn_values, width, label='PINN (RAdam)', color='steelblue')
        bars2 = ax.bar(x + width/2, qpinn_values, width, label='QPINN (SPSA)', color='orange')
        
        ax.set_ylabel('Value (log scale)')
        ax.set_title('Metrics Comparison: PINN vs QPINN')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Temperature Time Evolution Simulation
    print("  Generating temperature time evolution...")
    visualize_temperature_evolution(pinn_model, qpinn_model, save_dir)
    
    # 6. QPINN Circuit Diagram
    if qpinn_model is not None:
        print("  Generating QPINN circuit diagram...")
        visualize_qpinn_circuit(qpinn_model, save_dir)
    
    print(f"Visualizations saved to {save_dir}/")


def visualize_temperature_evolution(pinn_model, qpinn_model, save_dir: str):
    """
    Visualize temperature evolution over time for PINN, QPINN, and Analytical solution
    
    Creates a grid showing temperature distribution at different time steps
    """
    # Time steps to visualize
    time_steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_times = len(time_steps)
    
    # Create evaluation grid (2D slice at z = L/2)
    x_eval = np.linspace(0, L, 40)
    y_eval = np.linspace(0, L, 40)
    X, Y = np.meshgrid(x_eval, y_eval)
    z_slice = L / 2
    
    # Count number of models to display (Analytical always included)
    n_models = 1  # Analytical
    if pinn_model is not None:
        n_models += 1
    if qpinn_model is not None:
        n_models += 1
    
    # Prepare figure
    n_rows = n_models
    fig, axes = plt.subplots(n_rows, n_times, figsize=(4 * n_times, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Find global min/max for consistent colorbar
    all_values = []
    
    for t_idx, t_val in enumerate(time_steps):
        # Analytical solution (Fourier series)
        U_analytical = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U_analytical[i, j] = analytical_solution(X[i, j], Y[i, j], z_slice, t_val, n_terms=8)
        all_values.extend(U_analytical.flatten().tolist())
        
        # PINN prediction
        if pinn_model is not None:
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            z_flat = torch.full_like(x_flat, z_slice)
            t_flat = torch.full_like(x_flat, t_val)
            
            pinn_model.eval()
            with torch.no_grad():
                u_pinn = pinn_model.forward(x_flat, y_flat, z_flat, t_flat)
            U_pinn = u_pinn.cpu().numpy().reshape(X.shape)
            all_values.extend(U_pinn.flatten().tolist())
        
        # QPINN prediction
        if qpinn_model is not None:
            x_np = X.flatten().astype(np.float64)
            y_np = Y.flatten().astype(np.float64)
            z_np = np.full_like(x_np, z_slice)
            t_np = np.full_like(x_np, t_val)
            
            u_qpinn = qpinn_model.forward(x_np, y_np, z_np, t_np)
            U_qpinn = u_qpinn.reshape(X.shape)
            all_values.extend(U_qpinn.flatten().tolist())
    
    vmin = min(all_values)
    vmax = max(all_values)
    
    # Plot each time step
    for t_idx, t_val in enumerate(time_steps):
        row = 0
        
        # Analytical solution (always first row)
        U_analytical = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U_analytical[i, j] = analytical_solution(X[i, j], Y[i, j], z_slice, t_val, n_terms=8)
        
        ax = axes[row, t_idx]
        im = ax.contourf(X, Y, U_analytical, levels=30, cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_title(f'Analytical (t={t_val:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        row += 1
        
        # PINN prediction
        if pinn_model is not None:
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            z_flat = torch.full_like(x_flat, z_slice)
            t_flat = torch.full_like(x_flat, t_val)
            
            pinn_model.eval()
            with torch.no_grad():
                u_pinn = pinn_model.forward(x_flat, y_flat, z_flat, t_flat)
            U_pinn = u_pinn.cpu().numpy().reshape(X.shape)
            
            ax = axes[row, t_idx]
            im = ax.contourf(X, Y, U_pinn, levels=30, cmap='hot', vmin=vmin, vmax=vmax)
            ax.set_title(f'PINN (t={t_val:.2f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            row += 1
        
        # QPINN prediction
        if qpinn_model is not None:
            x_np = X.flatten().astype(np.float64)
            y_np = Y.flatten().astype(np.float64)
            z_np = np.full_like(x_np, z_slice)
            t_np = np.full_like(x_np, t_val)
            
            u_qpinn = qpinn_model.forward(x_np, y_np, z_np, t_np)
            U_qpinn = u_qpinn.reshape(X.shape)
            
            ax = axes[row, t_idx]
            im = ax.contourf(X, Y, U_qpinn, levels=30, cmap='hot', vmin=vmin, vmax=vmax)
            ax.set_title(f'QPINN (t={t_val:.2f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Temperature')
    
    plt.suptitle('Temperature Evolution (z = L/2 slice)', fontsize=14, y=0.98)
    plt.savefig(os.path.join(save_dir, 'temperature_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a line plot showing temperature decay at center point
    fig, ax = plt.subplots(figsize=(12, 7))
    
    t_fine = np.linspace(0, T, 50)
    center_x, center_y, center_z = L/2, L/2, L/2
    
    # Analytical solution (Fourier series)
    u_analytical_center = [analytical_solution(center_x, center_y, center_z, t, n_terms=10) for t in t_fine]
    ax.plot(t_fine, u_analytical_center, 'k-', linewidth=3, label='Analytical (Fourier)', zorder=10)
    
    # PINN
    if pinn_model is not None:
        u_pinn_center = []
        pinn_model.eval()
        for t in t_fine:
            x_pt = torch.tensor([[center_x]], dtype=torch.float32, device=device)
            y_pt = torch.tensor([[center_y]], dtype=torch.float32, device=device)
            z_pt = torch.tensor([[center_z]], dtype=torch.float32, device=device)
            t_pt = torch.tensor([[t]], dtype=torch.float32, device=device)
            with torch.no_grad():
                u = pinn_model.forward(x_pt, y_pt, z_pt, t_pt)
            # Handle multi-dimensional output
            u_np = u.cpu().numpy().flatten()
            u_pinn_center.append(float(u_np[0]))
        ax.plot(t_fine, u_pinn_center, 'b--', linewidth=2, label='PINN (RAdam)', marker='o', markevery=5, markersize=6)
    
    # QPINN
    if qpinn_model is not None:
        u_qpinn_center = []
        for t in t_fine:
            u = qpinn_model.forward_single(center_x, center_y, center_z, t)
            u_qpinn_center.append(u)
        ax.plot(t_fine, u_qpinn_center, 'r-.', linewidth=2, label='QPINN (SPSA)', marker='s', markevery=5, markersize=6)
    
    ax.set_xlabel('Time (t)', fontsize=14)
    ax.set_ylabel('Temperature u(L/2, L/2, L/2, t)', fontsize=14)
    ax.set_title('Temperature Decay at Domain Center\n3D Heat Conduction Equation', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])
    ax.set_ylim([0, max(u_analytical_center) * 1.1])
    
    # Add equation annotation
    eq_text = r'$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$, $\alpha = ' + f'{alpha}' + r'$'
    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temperature_center_decay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create error comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if pinn_model is not None or qpinn_model is not None:
        # Error at t=0.5
        t_mid = 0.5
        U_analytical_mid = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U_analytical_mid[i, j] = analytical_solution(X[i, j], Y[i, j], z_slice, t_mid, n_terms=8)
        
        ax_idx = 0
        if pinn_model is not None:
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
            z_flat = torch.full_like(x_flat, z_slice)
            t_flat = torch.full_like(x_flat, t_mid)
            
            pinn_model.eval()
            with torch.no_grad():
                u_pinn = pinn_model.forward(x_flat, y_flat, z_flat, t_flat)
            U_pinn_mid = u_pinn.cpu().numpy().reshape(X.shape)
            
            error_pinn = np.abs(U_pinn_mid - U_analytical_mid)
            im1 = axes[ax_idx].contourf(X, Y, error_pinn, levels=30, cmap='viridis')
            axes[ax_idx].set_title(f'PINN Absolute Error (t={t_mid})\nMax: {error_pinn.max():.4e}')
            axes[ax_idx].set_xlabel('x')
            axes[ax_idx].set_ylabel('y')
            axes[ax_idx].set_aspect('equal')
            plt.colorbar(im1, ax=axes[ax_idx])
            ax_idx = 1
        
        if qpinn_model is not None:
            x_np = X.flatten().astype(np.float64)
            y_np = Y.flatten().astype(np.float64)
            z_np = np.full_like(x_np, z_slice)
            t_np = np.full_like(x_np, t_mid)
            
            u_qpinn = qpinn_model.forward(x_np, y_np, z_np, t_np)
            U_qpinn_mid = u_qpinn.reshape(X.shape)
            
            error_qpinn = np.abs(U_qpinn_mid - U_analytical_mid)
            im2 = axes[ax_idx].contourf(X, Y, error_qpinn, levels=30, cmap='viridis')
            axes[ax_idx].set_title(f'QPINN Absolute Error (t={t_mid})\nMax: {error_qpinn.max():.4e}')
            axes[ax_idx].set_xlabel('x')
            axes[ax_idx].set_ylabel('y')
            axes[ax_idx].set_aspect('equal')
            plt.colorbar(im2, ax=axes[ax_idx])
    
    plt.suptitle('Error vs Analytical Solution (z = L/2 slice)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_vs_analytical.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_qpinn_circuit(qpinn_model, save_dir: str):
    """
    Visualize the QPINN quantum circuit
    
    Creates both a circuit diagram and a text representation
    """
    try:
        import pennylane as qml
        
        n_qubits = qpinn_model.n_qubits
        circuit_template = qpinn_model.circuit_template
        
        # Create a simple device for drawing
        dev = qml.device('default.qubit', wires=n_qubits)
        
        # Build circuit for visualization
        @qml.qnode(dev)
        def draw_circuit(params):
            # Input encoding layer
            for i in range(n_qubits):
                qml.RY(0.5, wires=i)  # Placeholder input
            
            # Apply the circuit template
            for gate_info in circuit_template.gate_sequence:
                gate_name = gate_info['gate']
                qubits = gate_info['qubits']
                
                if gate_info.get('trainable', False):
                    param_idx = gate_info['param_idx']
                    if param_idx < len(params):
                        angle = float(params[param_idx])
                    else:
                        angle = 0.1
                    
                    if gate_name == 'RX':
                        qml.RX(angle, wires=qubits[0])
                    elif gate_name == 'RY':
                        qml.RY(angle, wires=qubits[0])
                    elif gate_name == 'RZ':
                        qml.RZ(angle, wires=qubits[0])
                else:
                    if gate_name == 'CNOT':
                        qml.CNOT(wires=qubits)
                    elif gate_name == 'CZ':
                        qml.CZ(wires=qubits)
                    elif gate_name == 'H':
                        qml.Hadamard(wires=qubits[0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Get current parameters
        params = qpinn_model.circuit_params
        
        # Draw circuit using matplotlib
        fig, ax = qml.draw_mpl(draw_circuit)(params)
        plt.title(f'QPINN Quantum Circuit ({n_qubits} qubits, {len(params)} parameters)')
        plt.savefig(os.path.join(save_dir, 'qpinn_circuit_diagram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save text representation
        circuit_text = qml.draw(draw_circuit)(params)
        text_path = os.path.join(save_dir, 'qpinn_circuit_text.txt')
        with open(text_path, 'w') as f:
            f.write("QPINN Quantum Circuit\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of qubits: {n_qubits}\n")
            f.write(f"Number of circuit parameters: {len(params)}\n")
            f.write(f"Total trainable parameters: {qpinn_model.n_circuit_params + qpinn_model.n_output_params + len(qpinn_model.spatial_params)}\n")
            f.write(f"GQE-GPT optimized: {qpinn_model.use_gqe}\n\n")
            f.write("Circuit Structure:\n")
            f.write("-" * 60 + "\n")
            f.write(str(circuit_text) + "\n")
            f.write("-" * 60 + "\n\n")
            f.write("Gate Sequence:\n")
            for i, gate in enumerate(circuit_template.gate_sequence[:20]):  # Show first 20 gates
                f.write(f"  {i+1}. {gate['gate']} on qubit(s) {gate['qubits']}")
                if gate.get('trainable', False):
                    f.write(f" [trainable, param_idx={gate['param_idx']}]")
                f.write("\n")
            if len(circuit_template.gate_sequence) > 20:
                f.write(f"  ... and {len(circuit_template.gate_sequence) - 20} more gates\n")
            f.write("\nParameter values:\n")
            for i, p in enumerate(params[:20]):
                f.write(f"  param[{i}] = {p:.6f}\n")
            if len(params) > 20:
                f.write(f"  ... and {len(params) - 20} more parameters\n")
        
        print(f"    Circuit diagram saved to {save_dir}/qpinn_circuit_diagram.png")
        print(f"    Circuit text saved to {save_dir}/qpinn_circuit_text.txt")
        
    except Exception as e:
        print(f"    Warning: Could not generate circuit diagram: {e}")
        
        # Fallback: create a simple text-based representation
        text_path = os.path.join(save_dir, 'qpinn_circuit_info.txt')
        with open(text_path, 'w') as f:
            f.write("QPINN Quantum Circuit Information\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of qubits: {qpinn_model.n_qubits}\n")
            f.write(f"Number of layers: {qpinn_model.n_layers}\n")
            f.write(f"Circuit parameters: {qpinn_model.n_circuit_params}\n")
            f.write(f"Output parameters: {qpinn_model.n_output_params}\n")
            f.write(f"Spatial parameters: {len(qpinn_model.spatial_params)}\n")
            f.write(f"Total parameters: {qpinn_model.n_circuit_params + qpinn_model.n_output_params + len(qpinn_model.spatial_params)}\n")
            f.write(f"GQE-GPT optimized: {qpinn_model.use_gqe}\n\n")
            if qpinn_model.circuit_template:
                f.write(f"Template metadata: {qpinn_model.circuit_template.metadata}\n")
                f.write(f"Number of gates: {len(qpinn_model.circuit_template.gate_sequence)}\n")


def save_results(
    pinn_metrics, qpinn_metrics,
    pinn_history, qpinn_history,
    pinn_time, qpinn_time,
    args,
    save_dir: str = "results"
):
    """Save results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'problem': '3D Heat Conduction Equation',
        'domain': f'[0, {L}]³ × [0, {T}]',
        'parameters': {
            'alpha': alpha,
            'sigma_0': sigma_0,
            'discretization': {'nx': nx, 'ny': ny, 'nz': nz, 'nt': nt}
        },
        'command_line': {
            'pinn_epochs': args.epochs[0],
            'qpinn_epochs': args.epochs[1],
            'qubits': args.qubits,
            'layers': args.layers,
            'use_gqe': args.use_gqe and not getattr(args, 'no_gqe', False),
            'seed': args.seed,
            'adaptive_weights': args.adaptive_weights and not args.no_adaptive_weights,
            'log_dir': args.log_dir
        }
    }
    
    if pinn_metrics is not None:
        use_adaptive = args.adaptive_weights and not args.no_adaptive_weights
        pinn_result = {
            'optimizer': 'RAdam',
            'epochs': args.epochs[0],
            'training_time_seconds': pinn_time,
            'metrics': pinn_metrics,
            'final_loss': pinn_history['total_loss'][-1] if pinn_history else None,
            'adaptive_loss_weighting': use_adaptive
        }
        # Add final adaptive weights if available
        if pinn_history and 'loss_weights' in pinn_history and pinn_history['loss_weights']:
            pinn_result['final_loss_weights'] = {
                'initial': pinn_history['loss_weights'][-1][0],
                'peak': pinn_history['loss_weights'][-1][1],
                'boundary': pinn_history['loss_weights'][-1][2],
                'pde': pinn_history['loss_weights'][-1][3]
            }
        # Add best validation loss if available
        if pinn_history and 'validation_loss' in pinn_history:
            val_losses = [v for v in pinn_history['validation_loss'] if v is not None]
            if val_losses:
                pinn_result['best_validation_loss'] = min(val_losses)
        results['pinn'] = pinn_result
    
    if qpinn_metrics is not None:
        use_gqe_flag = args.use_gqe and not getattr(args, 'no_gqe', False)
        results['qpinn'] = {
            'optimizer': 'Manual SPSA',
            'epochs': args.epochs[1],
            'qubits': args.qubits,
            'layers': args.layers,
            'use_gqe': use_gqe_flag,
            'training_time_seconds': qpinn_time,
            'metrics': qpinn_metrics,
            'final_loss': qpinn_history['total_loss'][-1] if qpinn_history else None
        }
    
    if pinn_metrics is not None and qpinn_metrics is not None:
        results['comparison'] = {
            'mse_ratio': qpinn_metrics['mse'] / pinn_metrics['mse'] if pinn_metrics['mse'] > 0 else None,
            'mae_ratio': qpinn_metrics['mae'] / pinn_metrics['mae'] if pinn_metrics['mae'] > 0 else None,
            'time_ratio': qpinn_time / pinn_time if pinn_time and pinn_time > 0 else None,
        }
    
    with open(os.path.join(save_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_dir}/benchmark_results.json")


def run_pinn_benchmark(training_data, test_data, args, validation_data=None):
    """Run PINN benchmark"""
    print("\n" + "=" * 80)
    print("Training PINN with RAdam Optimizer")
    print("=" * 80)
    
    # Determine if adaptive weights should be used
    use_adaptive = args.adaptive_weights and not args.no_adaptive_weights
    
    pinn_model = create_pinn_model(use_adaptive_weights=use_adaptive)
    
    if not args.quiet:
        print(f"\nPINN Model created:")
        print(f"  - Total parameters: {sum(p.numel() for p in pinn_model.parameters()):,}")
        print(f"  - Adaptive loss weighting: {use_adaptive}")
    
    start_time = time.time()
    pinn_history = pinn_model.train_model(
        training_data=training_data,
        epochs=args.epochs[0],
        log_interval=100 if not args.quiet else args.epochs[0],
        validation_data=validation_data,
        log_to_file=True,
        log_dir=args.log_dir
    )
    training_time = time.time() - start_time
    
    metrics, _ = evaluate_model(pinn_model, test_data, "PINN", args.quiet)
    
    return pinn_model, pinn_history, metrics, training_time


def run_qpinn_benchmark(training_data, test_data, args, validation_data=None):
    """Run QPINN benchmark"""
    # Determine if GQE should be used
    use_gqe = args.use_gqe and not getattr(args, 'no_gqe', False)
    
    print("\n" + "=" * 80)
    print("Training QPINN with Manual SPSA Optimizer")
    if use_gqe:
        print("(GQE-GPT Circuit Optimization Enabled)")
    print("=" * 80)
    
    # Update QPINN config with command line args
    qpinn_config = QPINN_CONFIG.copy()
    qpinn_config['n_qubits'] = args.qubits
    qpinn_config['n_layers'] = args.layers
    
    # GQE config
    gqe_config = GQE_CONFIG.copy()
    gqe_config['n_iterations'] = args.gqe_iterations
    
    qpinn_model = create_qpinn_model(
        config=qpinn_config, 
        use_gqe=use_gqe,
        gqe_config=gqe_config,
        dynamic_update=args.dynamic_update
    )
    
    if not args.quiet:
        print(f"\nQPINN Model created:")
        print(f"  - Qubits: {qpinn_model.n_qubits}")
        print(f"  - Circuit parameters: {qpinn_model.n_circuit_params}")
        total_params = qpinn_model.n_circuit_params + qpinn_model.n_output_params + len(qpinn_model.spatial_params)
        print(f"  - Total parameters: {total_params}")
        print(f"  - GQE enabled: {qpinn_model.use_gqe}")
        print(f"  - Device: {qpinn_model.device_name}")
        print(f"  - Adaptive loss weighting: True")
    
    start_time = time.time()
    qpinn_history = qpinn_model.train_model(
        training_data=training_data,
        epochs=args.epochs[1],
        log_interval=20 if not args.quiet else args.epochs[1],
        log_to_file=True,
        log_dir=args.log_dir,
        validation_data=validation_data,
        adaptive_weights=True
    )
    training_time = time.time() - start_time
    
    metrics, _ = evaluate_model(qpinn_model, test_data, "QPINN", args.quiet)
    
    return qpinn_model, qpinn_history, metrics, training_time


def print_summary(pinn_metrics, qpinn_metrics, pinn_time, qpinn_time, pinn_history, qpinn_history):
    """Print benchmark summary"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    if pinn_metrics is not None and qpinn_metrics is not None:
        print(f"\n{'Metric':<25} {'PINN (RAdam)':<20} {'QPINN (SPSA)':<20} {'Ratio (Q/P)':<15}")
        print("-" * 80)
        
        for metric_name, key in [('MSE', 'mse'), ('MAE', 'mae'), ('Relative L2', 'rel_l2'), ('Max Error', 'max_error')]:
            p_val = pinn_metrics[key]
            q_val = qpinn_metrics[key]
            ratio = q_val / p_val if p_val > 0 else float('inf')
            print(f"{metric_name:<25} {p_val:<20.6e} {q_val:<20.6e} {ratio:<15.3f}")
        
        if pinn_time and qpinn_time:
            print(f"{'Training Time (s)':<25} {pinn_time:<20.1f} {qpinn_time:<20.1f} {qpinn_time/pinn_time:<15.3f}")
        
        print("\n" + "-" * 80)
        print("OPTIMIZER DETAILS")
        print("-" * 80)
        
        if pinn_history:
            print(f"\nPINN (RAdam):")
            print(f"  - Final Loss: {pinn_history['total_loss'][-1]:.6e}")
        
        if qpinn_history:
            print(f"\nQPINN (SPSA):")
            print(f"  - Final Loss: {qpinn_history['total_loss'][-1]:.6e}")
            if 'circuit_executions' in qpinn_history:
                print(f"  - Circuit Executions: {qpinn_history['circuit_executions'][-1]}")
    
    elif pinn_metrics is not None:
        print("\nPINN Results:")
        for key, val in pinn_metrics.items():
            print(f"  {key}: {val:.6e}")
        if pinn_time:
            print(f"  Training Time: {pinn_time:.1f}s")
    
    elif qpinn_metrics is not None:
        print("\nQPINN Results:")
        for key, val in qpinn_metrics.items():
            print(f"  {key}: {val:.6e}")
        if qpinn_time:
            print(f"  Training Time: {qpinn_time:.1f}s")


def main():
    """Main benchmark function"""
    args = parse_args()
    
    # Banner
    use_gqe = args.use_gqe and not getattr(args, 'no_gqe', False)
    print("\n" + "=" * 80)
    print("PINNs vs QPINNs Benchmark: 3D Heat Conduction Equation")
    print("Optimizers: RAdam (PINN) vs Manual SPSA (QPINN)")
    if use_gqe:
        print("GQE-GPT Circuit Optimization: ENABLED")
    print("=" * 80)
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Print configuration
    if not args.quiet:
        print_config()
    
    # Generate training data
    print("\nGenerating training data...")
    training_data = generate_training_data()
    
    # Generate test data
    print(f"Generating {args.test_points} test points...")
    test_data = generate_test_data(n_points=args.test_points)
    
    # Generate validation data (separate from test data for monitoring)
    print("Generating validation data (500 points)...")
    validation_data = generate_test_data(n_points=500)
    
    # Create results directory
    if args.output:
        results_dir = args.output
    else:
        results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize results
    pinn_model, pinn_history, pinn_metrics, pinn_time = None, None, None, None
    qpinn_model, qpinn_history, qpinn_metrics, qpinn_time = None, None, None, None
    
    # Run benchmarks based on mode
    if not args.qpinn_only:
        pinn_model, pinn_history, pinn_metrics, pinn_time = run_pinn_benchmark(
            training_data, test_data, args, validation_data=validation_data
        )
    
    if not args.pinn_only:
        qpinn_model, qpinn_history, qpinn_metrics, qpinn_time = run_qpinn_benchmark(
            training_data, test_data, args, validation_data=validation_data
        )
    
    # Print summary
    print_summary(pinn_metrics, qpinn_metrics, pinn_time, qpinn_time, pinn_history, qpinn_history)
    
    # Generate visualizations
    if not args.no_plot:
        visualize_results(
            pinn_model, qpinn_model,
            test_data, training_data,
            pinn_history, qpinn_history,
            pinn_metrics, qpinn_metrics,
            save_dir=results_dir
        )
    
    # Save results
    save_results(
        pinn_metrics, qpinn_metrics,
        pinn_history, qpinn_history,
        pinn_time, qpinn_time,
        args,
        save_dir=results_dir
    )
    
    # Generate detailed benchmark report
    generate_benchmark_report(
        pinn_model, qpinn_model,
        pinn_metrics, qpinn_metrics,
        pinn_history, qpinn_history,
        pinn_time, qpinn_time,
        args,
        save_dir=results_dir
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print(f"Results saved to: {results_dir}/")
    print("Files generated:")
    print(f"  - benchmark_results.json (metrics and configuration)")
    print(f"  - benchmark_report.txt (detailed analysis)")
    if not args.no_plot:
        print(f"  - loss_curves.png (training curves)")
        print(f"  - predictions_comparison.png (model outputs)")
    print(f"  - logs/ (training logs)")
    print("=" * 80)
    
    return {
        'pinn': {'model': pinn_model, 'metrics': pinn_metrics, 'history': pinn_history, 'time': pinn_time},
        'qpinn': {'model': qpinn_model, 'metrics': qpinn_metrics, 'history': qpinn_history, 'time': qpinn_time}
    }


def generate_benchmark_report(
    pinn_model, qpinn_model,
    pinn_metrics, qpinn_metrics,
    pinn_history, qpinn_history,
    pinn_time, qpinn_time,
    args,
    save_dir: str = "results"
):
    """Generate detailed benchmark report as text file"""
    os.makedirs(save_dir, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PINNs vs QPINNs BENCHMARK REPORT")
    report_lines.append("3D Heat Conduction Equation Solver")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Problem description
    report_lines.append("-" * 80)
    report_lines.append("PROBLEM DESCRIPTION")
    report_lines.append("-" * 80)
    report_lines.append(f"Equation: du/dt = alpha * (d2u/dx2 + d2u/dy2 + d2u/dz2)")
    report_lines.append(f"Domain: [0, {L}]^3 x [0, {T}]")
    report_lines.append(f"Thermal diffusivity (alpha): {alpha}")
    report_lines.append(f"Initial condition: Gaussian centered at ({L/2}, {L/2}, {L/2})")
    report_lines.append(f"Boundary conditions: Dirichlet (u = 0 on all boundaries)")
    report_lines.append("")
    
    # Configuration
    report_lines.append("-" * 80)
    report_lines.append("CONFIGURATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Random seed: {args.seed}")
    report_lines.append(f"Test points: {args.test_points}")
    report_lines.append("")
    
    # PINN results
    if pinn_metrics is not None:
        report_lines.append("-" * 80)
        report_lines.append("PINN RESULTS (RAdam Optimizer)")
        report_lines.append("-" * 80)
        report_lines.append(f"Epochs: {args.epochs[0]}")
        report_lines.append(f"Training time: {pinn_time:.2f} seconds")
        if pinn_model is not None:
            total_params = sum(p.numel() for p in pinn_model.parameters())
            report_lines.append(f"Total parameters: {total_params:,}")
        report_lines.append("")
        report_lines.append("Metrics:")
        report_lines.append(f"  MSE:          {pinn_metrics['mse']:.6e}")
        report_lines.append(f"  MAE:          {pinn_metrics['mae']:.6e}")
        report_lines.append(f"  Relative L2:  {pinn_metrics['rel_l2']:.6e}")
        report_lines.append(f"  Max Error:    {pinn_metrics['max_error']:.6e}")
        report_lines.append("")
        if pinn_history:
            report_lines.append("Training summary:")
            report_lines.append(f"  Initial loss: {pinn_history['total_loss'][0]:.6e}")
            report_lines.append(f"  Final loss:   {pinn_history['total_loss'][-1]:.6e}")
            report_lines.append(f"  Best loss:    {min(pinn_history['total_loss']):.6e}")
            if 'validation_loss' in pinn_history and any(v is not None for v in pinn_history['validation_loss']):
                val_losses = [v for v in pinn_history['validation_loss'] if v is not None]
                report_lines.append(f"  Best val loss: {min(val_losses):.6e}")
        report_lines.append("")
    
    # QPINN results
    if qpinn_metrics is not None:
        report_lines.append("-" * 80)
        report_lines.append("QPINN RESULTS (Manual SPSA Optimizer)")
        report_lines.append("-" * 80)
        report_lines.append(f"Epochs: {args.epochs[1]}")
        report_lines.append(f"Training time: {qpinn_time:.2f} seconds")
        if qpinn_model is not None:
            report_lines.append(f"Qubits: {qpinn_model.n_qubits}")
            report_lines.append(f"Circuit layers: {qpinn_model.n_layers}")
            report_lines.append(f"Circuit parameters: {qpinn_model.n_circuit_params}")
            total_params = qpinn_model.n_circuit_params + qpinn_model.n_output_params + len(qpinn_model.spatial_params)
            report_lines.append(f"Total parameters: {total_params}")
            report_lines.append(f"GQE-GPT enabled: {qpinn_model.use_gqe}")
        report_lines.append("")
        report_lines.append("Metrics:")
        report_lines.append(f"  MSE:          {qpinn_metrics['mse']:.6e}")
        report_lines.append(f"  MAE:          {qpinn_metrics['mae']:.6e}")
        report_lines.append(f"  Relative L2:  {qpinn_metrics['rel_l2']:.6e}")
        report_lines.append(f"  Max Error:    {qpinn_metrics['max_error']:.6e}")
        report_lines.append("")
        if qpinn_history:
            report_lines.append("Training summary:")
            report_lines.append(f"  Initial loss: {qpinn_history['total_loss'][0]:.6e}")
            report_lines.append(f"  Final loss:   {qpinn_history['total_loss'][-1]:.6e}")
            report_lines.append(f"  Best loss:    {min(qpinn_history['total_loss']):.6e}")
            if 'circuit_executions' in qpinn_history:
                report_lines.append(f"  Circuit executions: {qpinn_history['circuit_executions'][-1]}")
        report_lines.append("")
    
    # Comparison
    if pinn_metrics is not None and qpinn_metrics is not None:
        report_lines.append("-" * 80)
        report_lines.append("COMPARATIVE ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append("")
        report_lines.append(f"{'Metric':<20} {'PINN':<15} {'QPINN':<15} {'Ratio (Q/P)':<12} {'Winner':<10}")
        report_lines.append("-" * 72)
        
        for metric_name, key in [('MSE', 'mse'), ('MAE', 'mae'), ('Rel L2', 'rel_l2'), ('Max Error', 'max_error')]:
            p_val = pinn_metrics[key]
            q_val = qpinn_metrics[key]
            ratio = q_val / p_val if p_val > 0 else float('inf')
            winner = "QPINN" if q_val < p_val else "PINN"
            report_lines.append(f"{metric_name:<20} {p_val:<15.6e} {q_val:<15.6e} {ratio:<12.3f} {winner:<10}")
        
        if pinn_time and qpinn_time:
            time_ratio = qpinn_time / pinn_time
            time_winner = "PINN" if pinn_time < qpinn_time else "QPINN"
            report_lines.append(f"{'Training Time (s)':<20} {pinn_time:<15.1f} {qpinn_time:<15.1f} {time_ratio:<12.3f} {time_winner:<10}")
        
        report_lines.append("")
        
        # Performance improvement calculation
        mse_improvement = ((pinn_metrics['mse'] - qpinn_metrics['mse']) / pinn_metrics['mse']) * 100 if pinn_metrics['mse'] > 0 else 0
        if mse_improvement > 0:
            report_lines.append(f"QPINN achieves {mse_improvement:.2f}% better MSE than PINN")
        else:
            report_lines.append(f"PINN achieves {-mse_improvement:.2f}% better MSE than QPINN")
        
        if pinn_time and qpinn_time:
            if qpinn_time > pinn_time:
                report_lines.append(f"QPINN requires {qpinn_time/pinn_time:.2f}x more training time")
            else:
                report_lines.append(f"QPINN is {pinn_time/qpinn_time:.2f}x faster in training")
        report_lines.append("")
    
    # Conclusions
    report_lines.append("-" * 80)
    report_lines.append("CONCLUSIONS")
    report_lines.append("-" * 80)
    if pinn_metrics is not None and qpinn_metrics is not None:
        if qpinn_metrics['mse'] < pinn_metrics['mse']:
            report_lines.append("- QPINN demonstrates superior accuracy in terms of MSE")
        else:
            report_lines.append("- PINN demonstrates superior accuracy in terms of MSE")
        
        if pinn_time and qpinn_time:
            report_lines.append(f"- PINN training completed in {pinn_time:.1f}s")
            report_lines.append(f"- QPINN training completed in {qpinn_time:.1f}s")
    report_lines.append("")
    
    # References
    report_lines.append("-" * 80)
    report_lines.append("REFERENCES")
    report_lines.append("-" * 80)
    report_lines.append("1. Raissi et al. (2019) Physics-informed neural networks")
    report_lines.append("2. Xiang et al. (2022) Self-adaptive loss balanced PINNs")
    report_lines.append("3. Trahan et al. (2024) Quantum Physics-Informed Neural Networks")
    report_lines.append("4. Liu et al. (2020) RAdam optimizer")
    report_lines.append("5. Spall (1998) SPSA algorithm")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(save_dir, 'benchmark_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nDetailed benchmark report saved to: {report_path}")


if __name__ == "__main__":
    results = main()