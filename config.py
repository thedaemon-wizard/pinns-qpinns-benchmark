"""
Configuration and common parameters for PINNs vs QPINNs benchmark
3D Heat Conduction Equation Solver

Optimizers:
- PINNs: RAdam (PyTorch)
- QPINNs: SPSA (PennyLane v0.43)
"""

import numpy as np
import torch
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Environment settings
os.environ['OMP_NUM_THREADS'] = str(12)

# Set PyTorch default floating point precision
torch.set_default_dtype(torch.float32)

# Device settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================================
# Problem parameters (3D Heat Equation)
#================================================
alpha = 0.01    # Thermal diffusivity
L = 1.0         # Length of cube side
T = 1.0         # Final time
sigma_0 = 0.05  # Gaussian parameter for initial condition

#================================================
# Discretization parameters
#================================================
nx, ny, nz = 20, 20, 20  # Spatial divisions
nt = 20                   # Time divisions

#================================================
# Training parameters
#================================================
pinn_epochs = 3000        # PINN epochs with RAdam
qnn_epochs = 200          # QPINN epochs with SPSA

# Learning rates
pinn_learning_rate = 1e-3     # RAdam learning rate
qpinn_learning_rate = 0.1     # SPSA initial step size (a parameter)

# SPSA optimizer parameters (PennyLane v0.43)
SPSA_CONFIG = {
    'maxiter': qnn_epochs,
    'alpha': 0.602,         # Learning rate decay exponent
    'gamma': 0.101,         # Perturbation decay exponent
    'c': 0.1,               # Initial perturbation magnitude
    'a': None,              # Initial step size (calculated from maxiter if None)
    'A': None,              # Stability constant (calculated as 0.1 * maxiter if None)
}

# RAdam optimizer parameters
RADAM_CONFIG = {
    'lr': pinn_learning_rate,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-5,
}

#================================================
# Model architecture parameters
#================================================
# PINN architecture (PINNsFormer)
PINN_CONFIG = {
    'use_transformer': True,
    'use_fourier_features': True,
    'num_fourier_features': 64,
    'transformer_config': {
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'seq_length': 16,
    },
    'layers': [4, 256, 256, 256, 256, 1],  # Fallback MLP
    'use_hard_constraints': True,
    'boundary_epsilon': 1e-3,
}

# QPINN architecture
QPINN_CONFIG = {
    'n_qubits': 6,
    'n_layers': 4,
    'use_gpt_circuit_generation': True,
    'circuit_update_interval': 50,
    'n_spatial_features': 8,
    'n_frequencies': 4,
    'n_temporal_features': 4,
}

# GQE-GPT Configuration
GQE_CONFIG = {
    # GPT Model
    'n_embd': 256,
    'n_head': 8,
    'n_layer': 6,
    'block_size': 128,
    'dropout': 0.1,
    
    # Circuit Generation
    'n_qubits': 6,
    'n_layers': 4,
    'max_circuit_depth': 20,
    
    # Optimization
    'n_candidates': 20,
    'n_iterations': 5,
    'exploration_rate': 0.9,
    'exploration_decay': 0.85,
    
    # Energy Estimation
    'use_noise': False,
    'noise_model': 'realistic',
    'shots': 1000,
    
    # Hardware Topology
    'hardware_topology': 'linear',
    'preferred_gates': ['RY', 'RZ', 'CNOT', 'CZ'],
    
    # Multi-objective Optimization
    'n_objectives': 9,
    'objective_names': [
        'Hardware Efficiency',
        'Noise Resilience', 
        'Expressivity',
        'Mitigation Compatibility',
        'Trainability',
        'Entanglement Capability',
        'Depth Efficiency',
        'Parameter Efficiency',
        'Energy Estimation Quality'
    ]
}

#================================================
# Data classes
#================================================
@dataclass
class TrainingPoint:
    """Training data point"""
    x: float
    y: float
    z: float
    t: float
    u_true: float = None
    type: str = 'interior'


@dataclass
class QuantumCircuitTemplate:
    """Quantum circuit template for GQE optimization"""
    n_qubits: int
    n_layers: int
    gate_sequence: List[Dict[str, Any]] = field(default_factory=list)
    parameter_map: Dict[str, int] = field(default_factory=dict)
    entangling_pattern: str = 'linear'
    noise_resilience_score: float = 0.5
    hardware_efficiency: float = 0.5
    expressivity_score: float = 0.5
    estimated_energy: float = -1.0
    depth_score: float = 0.5
    diversity_score: float = 0.5
    mitigation_score: float = 0.5
    param_efficiency: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default gate sequence if empty"""
        if not self.gate_sequence:
            self.gate_sequence = self._default_gate_sequence()
        if not self.parameter_map:
            self.parameter_map = self._build_parameter_map()
    
    def _default_gate_sequence(self) -> List[Dict]:
        """Generate default hardware-efficient ansatz"""
        sequence = []
        param_idx = 0
        
        for layer in range(self.n_layers):
            # RY rotation layer
            for q in range(self.n_qubits):
                sequence.append({
                    'gate': 'RY',
                    'qubits': [q],
                    'param_idx': param_idx,
                    'trainable': True
                })
                param_idx += 1
            
            # RZ rotation layer
            for q in range(self.n_qubits):
                sequence.append({
                    'gate': 'RZ',
                    'qubits': [q],
                    'param_idx': param_idx,
                    'trainable': True
                })
                param_idx += 1
            
            # CNOT entangling layer
            if layer < self.n_layers - 1:
                for q in range(self.n_qubits - 1):
                    sequence.append({
                        'gate': 'CNOT',
                        'qubits': [q, q + 1],
                        'param_idx': None,
                        'trainable': False
                    })
        
        return sequence
    
    def _build_parameter_map(self) -> Dict[str, int]:
        """Build parameter map from gate sequence"""
        param_map = {}
        for i, gate in enumerate(self.gate_sequence):
            if gate.get('trainable', False) and gate.get('param_idx') is not None:
                param_map[f"{gate['gate']}_{i}"] = gate['param_idx']
        return param_map
    
    @property
    def n_params(self) -> int:
        """Total number of trainable parameters"""
        return len(self.parameter_map)
    
    @property
    def depth(self) -> int:
        """Circuit depth"""
        if not self.gate_sequence:
            return 0
        
        qubit_layers = {}
        max_depth = 0
        
        for gate_info in self.gate_sequence:
            qubits = gate_info['qubits']
            current_layer = 0
            
            for q in qubits:
                if q < self.n_qubits and q in qubit_layers:
                    current_layer = max(current_layer, qubit_layers[q] + 1)
            
            for q in qubits:
                if q < self.n_qubits:
                    qubit_layers[q] = current_layer
            
            max_depth = max(max_depth, current_layer + 1)
        
        return max_depth


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int
    learning_rate: float
    batch_size: int = 1024
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 500
    log_interval: int = 100


#================================================
# Physics functions
#================================================
def initial_condition(x, y, z):
    """
    Gaussian initial condition for 3D heat equation
    u(x, y, z, 0) = exp(-((x-0.5)² + (y-0.5)² + (z-0.5)²) / (2σ²))
    """
    return np.exp(-((x - L/2)**2 + (y - L/2)**2 + (z - L/2)**2) / (2 * sigma_0**2))


def boundary_condition(x, y, z, t):
    """
    Dirichlet boundary condition: u = 0 on all boundaries
    """
    return 0.0


def analytical_solution(x, y, z, t, n_terms=10):
    """
    Analytical solution for 3D heat equation with Gaussian initial condition
    Based on the fundamental solution approach from pinns_d3.py
    
    Uses time-evolving Gaussian approximation which is accurate for interior points
    """
    x0, y0, z0 = L/2, L/2, L/2
    
    if t < 1e-10:
        # At t=0, return initial condition
        return initial_condition(x, y, z)
    
    # Time-evolving sigma (fundamental solution spreading)
    sigma_t = np.sqrt(sigma_0**2 + 2*alpha*t)
    
    # Calculate peak value decay (3D amplitude decay)
    amplitude = (sigma_0/sigma_t)**3
    
    # Calculate Gaussian distribution
    r_sq = (x-x0)**2 + (y-y0)**2 + (z-z0)**2
    gauss_term = amplitude * np.exp(-r_sq / (2*sigma_t**2))
    
    # Consider boundary condition effects (simplified mirror method)
    # Correction term considering reflection at boundaries
    boundary_effect = 1.0
    
    # Decay based on distance from each boundary
    dist_from_boundaries = min(x, L-x, y, L-y, z, L-z)
    if dist_from_boundaries < 0.1 * L:  # Near boundary
        boundary_effect = dist_from_boundaries / (0.1 * L)
    
    return gauss_term * boundary_effect


def analytical_solution_simple(x, y, z, t):
    """
    Simplified analytical approximation for visualization
    Same as analytical_solution but uses smooth boundary function
    """
    if t < 1e-10:
        return initial_condition(x, y, z)
    
    x0, y0, z0 = L/2, L/2, L/2
    
    # Time-evolving sigma
    sigma_t = np.sqrt(sigma_0**2 + 2 * alpha * t)
    
    # 3D amplitude decay
    amplitude = (sigma_0 / sigma_t)**3
    
    # Gaussian profile
    r_sq = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    u = amplitude * np.exp(-r_sq / (2 * sigma_t**2))
    
    # Apply smooth boundary decay for Dirichlet BC approximation
    # Use softer boundary effect that doesn't affect interior much
    dist_from_boundaries = min(x, L-x, y, L-y, z, L-z)
    if dist_from_boundaries < 0.1 * L:
        boundary_effect = dist_from_boundaries / (0.1 * L)
    else:
        boundary_effect = 1.0
    
    return u * boundary_effect


#================================================
# Utility functions
#================================================
def to_python_float(tensor):
    """Convert tensor to Python float safely"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().item()
    elif isinstance(tensor, np.ndarray):
        return float(tensor.flatten()[0])
    return float(tensor)


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("PINNs vs QPINNs Benchmark Configuration")
    print("=" * 60)
    print(f"\nProblem: 3D Heat Conduction Equation")
    print(f"  - Domain: [0, {L}]³ × [0, {T}]")
    print(f"  - Thermal diffusivity: α = {alpha}")
    print(f"  - Initial condition: Gaussian (σ = {sigma_0})")
    
    print(f"\nDiscretization:")
    print(f"  - Spatial: {nx} × {ny} × {nz}")
    print(f"  - Temporal: {nt} steps")
    
    print(f"\nPINN Configuration:")
    print(f"  - Optimizer: RAdam")
    print(f"  - Learning rate: {RADAM_CONFIG['lr']}")
    print(f"  - Epochs: {pinn_epochs}")
    print(f"  - Architecture: PINNsFormer" if PINN_CONFIG['use_transformer'] else "  - Architecture: MLP")
    
    print(f"\nQPINN Configuration:")
    print(f"  - Optimizer: SPSA (PennyLane v0.43)")
    print(f"  - Initial step size (c): {SPSA_CONFIG['c']}")
    print(f"  - Epochs: {qnn_epochs}")
    print(f"  - Qubits: {QPINN_CONFIG['n_qubits']}")
    print(f"  - Layers: {QPINN_CONFIG['n_layers']}")
    
    print(f"\nDevice: {get_device()}")
    print("=" * 60)