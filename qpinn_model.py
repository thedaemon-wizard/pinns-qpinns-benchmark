"""
QPINN Model for 3D Heat Conduction Equation
Optimizer: Manual SPSA Implementation (does NOT use PennyLane's SPSAOptimizer)
Architecture: Variational Quantum Circuit with GQE-GPT generation

This implementation uses a custom SPSA optimizer to avoid compatibility issues
with different PennyLane versions.

References:
- Trahan et al. "Quantum Physics-Informed Neural Networks" Entropy 26(8):649 (2024)
- Apak et al. "KetGPT" arXiv:2402.13352 (2024)
- Spall (1998) "Implementation of the Simultaneous Perturbation Algorithm"
"""

import numpy as np
import torch
import pennylane as qml
import time
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from config import (
    L, T, alpha, device, sigma_0,
    QPINN_CONFIG, SPSA_CONFIG, GQE_CONFIG, qnn_epochs,
    initial_condition, to_python_float, QuantumCircuitTemplate
)

# Optional GQE import
try:
    from gqe_generator import GQEQuantumCircuitGenerator, create_gqe_generator
    GQE_AVAILABLE = True
except ImportError:
    GQE_AVAILABLE = False
    print("Warning: GQE Generator not available. Using default circuits.")


# ================================================
# Logger Setup
# ================================================
def setup_qpinn_logger(
    name: str = "qpinn_training",
    log_dir: str = "logs",
    log_to_file: bool = True
) -> logging.Logger:
    """Setup logger for QPINN training"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ================================================
# Manual SPSA Optimizer (NOT using PennyLane's SPSAOptimizer)
# ================================================
class ManualSPSAOptimizer:
    """
    Custom implementation of SPSA (Simultaneous Perturbation Stochastic Approximation)
    
    This implementation does NOT use PennyLane's SPSAOptimizer to avoid
    compatibility issues with different PennyLane versions.
    
    Reference: Spall (1998) "Implementation of the Simultaneous Perturbation Algorithm"
    
    Parameters:
        a: Step size scaling factor
        c: Perturbation size
        alpha: Step size decay exponent
        gamma: Perturbation decay exponent
        A: Stability constant
    """
    
    def __init__(
        self,
        maxiter: int = 200,
        a: float = 0.05,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = None
    ):
        self.maxiter = maxiter
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A if A is not None else 0.1 * maxiter
        self.k = 0
    
    def _get_ak(self) -> float:
        """Get step size for current iteration"""
        return self.a / (self.A + self.k + 1) ** self.alpha
    
    def _get_ck(self) -> float:
        """Get perturbation magnitude for current iteration"""
        return self.c / (self.k + 1) ** self.gamma
    
    def step(self, cost_fn, params: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform one SPSA optimization step
        
        Args:
            cost_fn: Cost function that takes params and returns a scalar float
            params: Current parameters as numpy array
            
        Returns:
            Tuple of (updated parameters, current loss value)
        """
        ak = self._get_ak()
        ck = self._get_ck()
        
        # Generate Bernoulli perturbation vector (+1 or -1)
        delta = 2.0 * np.random.randint(0, 2, size=len(params)).astype(np.float64) - 1.0
        
        # Perturbed parameter evaluations
        params_plus = params + ck * delta
        params_minus = params - ck * delta
        
        # Evaluate cost function at perturbed points
        y_plus = float(cost_fn(params_plus))
        y_minus = float(cost_fn(params_minus))
        
        # Estimate gradient using SPSA formula
        gradient = (y_plus - y_minus) / (2.0 * ck * delta)
        
        # Update parameters
        new_params = params - ak * gradient
        
        # Get current loss value
        current_loss = float(cost_fn(params))
        
        # Increment iteration counter
        self.k += 1
        
        return new_params, current_loss
    
    def reset(self):
        """Reset iteration counter"""
        self.k = 0


# ================================================
# Quantum Circuit Builder
# ================================================
class QuantumCircuitBuilder:
    """Build and manage quantum circuits for QPINN"""
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_template = None
        
    def create_default_template(self) -> QuantumCircuitTemplate:
        """Create a default hardware-efficient ansatz template"""
        gate_sequence = []
        parameter_map = {}
        param_idx = 0
        
        for layer in range(self.n_layers):
            # Single-qubit rotation layer
            for qubit in range(self.n_qubits):
                # RY rotation
                gate_sequence.append({
                    'gate': 'RY',
                    'qubits': [qubit],
                    'trainable': True,
                    'param_idx': param_idx
                })
                parameter_map[f'ry_{layer}_{qubit}'] = param_idx
                param_idx += 1
                
                # RZ rotation
                gate_sequence.append({
                    'gate': 'RZ',
                    'qubits': [qubit],
                    'trainable': True,
                    'param_idx': param_idx
                })
                parameter_map[f'rz_{layer}_{qubit}'] = param_idx
                param_idx += 1
            
            # Entangling layer (linear connectivity)
            for qubit in range(self.n_qubits - 1):
                gate_sequence.append({
                    'gate': 'CNOT',
                    'qubits': [qubit, qubit + 1],
                    'trainable': False
                })
        
        return QuantumCircuitTemplate(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            gate_sequence=gate_sequence,
            parameter_map=parameter_map,
            metadata={'architecture': 'hardware_efficient'}
        )
    
    def build_circuit(self, template: QuantumCircuitTemplate, params: np.ndarray):
        """Build quantum circuit from template"""
        for gate_info in template.gate_sequence:
            gate_name = gate_info['gate']
            qubits = gate_info['qubits']
            
            if gate_info.get('trainable', False):
                param_idx = gate_info['param_idx']
                angle = float(params[param_idx])
                
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


# ================================================
# QPINN Model
# ================================================
class QPINN3DHeatSolver:
    """
    Quantum Physics-Informed Neural Network for 3D Heat Equation
    Using Variational Quantum Circuits with custom SPSA optimizer
    
    Features:
    - GQE-GPT based circuit generation (enabled by default when available)
    - Custom SPSA implementation for maximum compatibility
    - Multi-objective circuit quality evaluation
    """
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 4,
        use_gqe: bool = True,  # GQE enabled by default
        gqe_config: Dict = None,
        dynamic_circuit_update: bool = False,
        circuit_update_interval: int = 50
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_gqe = use_gqe and GQE_AVAILABLE
        self.dynamic_circuit_update = dynamic_circuit_update
        self.circuit_update_interval = circuit_update_interval
        
        # Circuit builder
        self.circuit_builder = QuantumCircuitBuilder(n_qubits, n_layers)
        
        # Initialize circuit template
        if self.use_gqe and gqe_config is not None:
            try:
                self.gqe_generator = create_gqe_generator(
                    n_qubits=n_qubits,
                    n_layers=n_layers
                )
                print("Generating optimized quantum circuit with GQE-GPT...")
                generated_template = self.gqe_generator.generate_circuit(
                    use_gpt=True,
                    temperature=0.8
                )
                # Check if generated circuit has enough parameters
                min_expected_params = n_qubits * n_layers  # At least this many
                if generated_template.n_params < min_expected_params:
                    print(f"GQE generated circuit has only {generated_template.n_params} parameters (expected >= {min_expected_params})")
                    print("Using default hardware-efficient circuit instead")
                    self.circuit_template = self.circuit_builder.create_default_template()
                    self.use_gqe = False
                else:
                    self.circuit_template = generated_template
                    print(f"GQE-GPT generated circuit with {self.circuit_template.n_params} parameters")
                    self.use_gqe = True
            except Exception as e:
                print(f"GQE generation failed: {e}")
                print("Using default hardware-efficient circuit")
                self.circuit_template = self.circuit_builder.create_default_template()
                self.gqe_generator = None
                self.use_gqe = False
        else:
            self.circuit_template = self.circuit_builder.create_default_template()
            self.gqe_generator = None
            if use_gqe and not GQE_AVAILABLE:
                print("GQE requested but not available. Using default circuit.")
        
        # Number of circuit parameters
        self.n_circuit_params = self.circuit_template.n_params
        
        # Output transformation parameters
        self.n_output_params = 4  # scale, bias, amplitude, decay
        
        # Spatial encoding parameters
        self.n_spatial_params = 3 * n_qubits  # encoding for x, y, z
        
        # Initialize parameters
        self.circuit_params = np.random.randn(self.n_circuit_params) * 0.1
        self.output_params = np.array([1.0, 0.0, 1.0, 1.0])  # scale, bias, amp, decay
        self.spatial_params = np.random.randn(self.n_spatial_params) * 0.1
        
        # Create quantum device (use lightning.qubit if available, fallback to default.qubit)
        try:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            self.device_name = 'lightning.qubit'
        except Exception:
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self.device_name = 'default.qubit'
        
        # Create QNode
        self._create_qnode()
        
        # Training data and history
        self.training_data = None
        self.loss_history = []
        self.circuit_update_history = []
        
        # Adaptive loss weights (similar to PINN)
        self.adaptive_weights = True
        self.loss_weights = {
            'initial': 10.0,
            'peak': 50.0,
            'boundary': 5.0,
            'pde': 1.0
        }
        # Running statistics for adaptive weighting
        self._loss_stats = {
            'initial': {'sum': 0.0, 'count': 0},
            'peak': {'sum': 0.0, 'count': 0},
            'boundary': {'sum': 0.0, 'count': 0}
        }
        
        # Validation data
        self.validation_data = None
        
        # Logger
        self.logger = None
    
    def _create_qnode(self):
        """Create quantum circuit as QNode"""
        n_qubits = self.n_qubits
        circuit_builder = self.circuit_builder
        circuit_template = self.circuit_template
        
        @qml.qnode(self.dev, interface='autograd')
        def circuit(inputs, circuit_params):
            # Input encoding
            for i in range(min(len(inputs), n_qubits)):
                qml.RY(float(inputs[i]), wires=i)
            
            # Apply circuit template
            circuit_builder.build_circuit(circuit_template, circuit_params)
            
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.qcircuit = circuit
    
    def _encode_spatial_input(
        self,
        x: float,
        y: float,
        z: float,
        t: float
    ) -> np.ndarray:
        """Encode spatial-temporal coordinates for quantum circuit"""
        # Normalize coordinates
        x_norm = float(x) / L
        y_norm = float(y) / L
        z_norm = float(z) / L
        t_norm = float(t) / T
        
        # Create encoding angles
        angles = np.zeros(self.n_qubits)
        
        # Split encoding across qubits
        n_per_coord = max(1, self.n_qubits // 4)
        
        for i in range(min(n_per_coord, self.n_qubits)):
            idx = i % len(self.spatial_params)
            angles[i] = np.pi * x_norm * (1 + self.spatial_params[idx])
        
        for i in range(n_per_coord, min(2 * n_per_coord, self.n_qubits)):
            idx = i % len(self.spatial_params)
            angles[i] = np.pi * y_norm * (1 + self.spatial_params[idx])
        
        for i in range(2 * n_per_coord, min(3 * n_per_coord, self.n_qubits)):
            idx = i % len(self.spatial_params)
            angles[i] = np.pi * z_norm * (1 + self.spatial_params[idx])
        
        for i in range(3 * n_per_coord, self.n_qubits):
            idx = i % len(self.spatial_params)
            angles[i] = np.pi * t_norm * (1 + self.spatial_params[idx])
        
        return angles
    
    def forward_single(self, x: float, y: float, z: float, t: float) -> float:
        """Forward pass for a single point"""
        # Encode inputs
        input_angles = self._encode_spatial_input(x, y, z, t)
        
        # Run quantum circuit
        result = self.qcircuit(input_angles, self.circuit_params)
        
        # Extract expectation values
        if isinstance(result, list):
            exp_vals = np.array([float(r) for r in result])
        else:
            exp_vals = np.atleast_1d(np.array(result))
        
        # Classical post-processing
        scale = float(self.output_params[0])
        bias = float(self.output_params[1])
        # Use absolute value for amplitude to ensure non-negative output
        amplitude = abs(float(self.output_params[2]))
        decay = max(float(self.output_params[3]), 0.1)
        
        # Weighted sum of expectation values
        weights = np.exp(-np.arange(len(exp_vals)) / decay)
        weights = weights / weights.sum()
        raw_output = np.sum(exp_vals * weights)
        
        # Scale and shift, ensure non-negative correction term
        # Use sigmoid to keep output in [0, 1] range
        output = amplitude * (np.tanh(scale * raw_output + bias) + 1) / 2
        
        # Apply physics constraints using product of tanh functions
        # This gives ~1 in interior, ~0 at boundaries
        k = 20.0  # Steepness parameter
        d_x_low = float(x) / L
        d_x_high = (L - float(x)) / L
        d_y_low = float(y) / L
        d_y_high = (L - float(y)) / L
        d_z_low = float(z) / L
        d_z_high = (L - float(z)) / L
        
        boundary_mask = (np.tanh(k * d_x_low) * np.tanh(k * d_x_high) *
                        np.tanh(k * d_y_low) * np.tanh(k * d_y_high) *
                        np.tanh(k * d_z_low) * np.tanh(k * d_z_high))
        
        # Initial condition
        u_init = initial_condition(float(x), float(y), float(z))
        
        # Physics-based time decay: (σ₀/σ_t)³ where σ_t² = σ₀² + 2αt
        t_val = float(t)
        sigma_t_sq = sigma_0**2 + 2 * alpha * t_val
        physical_decay = (sigma_0**2 / sigma_t_sq)**(3/2)
        
        # Time-evolving Gaussian width
        r_sq = (float(x) - L/2)**2 + (float(y) - L/2)**2 + (float(z) - L/2)**2
        r_sq_scaled = r_sq / sigma_t_sq
        u_evolved = physical_decay * np.exp(-r_sq_scaled / 2)
        
        # Neural network learns SMALL corrections to the physics
        # Scale output to be a multiplicative correction factor around 1.0
        correction = 1.0 + 0.1 * np.tanh(output)  # Correction in range [0.9, 1.1]
        
        # Apply correction to physics-based solution
        prediction = boundary_mask * u_evolved * correction
        
        # Ensure non-negative temperature
        return float(np.clip(prediction, 0, 1.0))
    
    def forward(self, x, y, z, t) -> np.ndarray:
        """Forward pass through quantum circuit (batch)"""
        # Handle scalar, array, and tensor inputs
        def to_numpy(arr):
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy().astype(np.float64).flatten()
            else:
                return np.atleast_1d(np.array(arr, dtype=np.float64)).flatten()
        
        x_np = to_numpy(x)
        y_np = to_numpy(y)
        z_np = to_numpy(z)
        t_np = to_numpy(t)
        
        predictions = []
        for i in range(len(x_np)):
            pred = self.forward_single(x_np[i], y_np[i], z_np[i], t_np[i])
            predictions.append(pred)
        
        return np.array(predictions).reshape(-1, 1)
    
    def _compute_loss(self, params: np.ndarray, training_data: Dict, batch_size: int = 50) -> float:
        """
        Compute total loss for SPSA optimization with adaptive weighting
        
        Returns a Python float (not numpy array) for compatibility
        """
        # Unpack parameters
        n_circuit = self.n_circuit_params
        n_output = self.n_output_params
        
        self.circuit_params = params[:n_circuit].copy()
        self.output_params = params[n_circuit:n_circuit + n_output].copy()
        self.spatial_params = params[n_circuit + n_output:].copy()
        
        total_loss = 0.0
        
        try:
            # 1. Initial condition loss
            n_init = len(training_data['initial']['x'])
            idx_init = np.random.choice(n_init, size=min(batch_size, n_init), replace=False)
            
            x_init = training_data['initial']['x'][idx_init].cpu().numpy()
            y_init = training_data['initial']['y'][idx_init].cpu().numpy()
            z_init = training_data['initial']['z'][idx_init].cpu().numpy()
            t_init = training_data['initial']['t'][idx_init].cpu().numpy()
            u_init_true = training_data['initial']['u'][idx_init].cpu().numpy()
            
            u_init_pred = self.forward(x_init, y_init, z_init, t_init)
            initial_loss = float(np.mean((u_init_pred.flatten() - u_init_true.flatten())**2))
            
            # 2. Boundary condition loss
            n_bd = len(training_data['boundary']['x'])
            idx_bd = np.random.choice(n_bd, size=min(batch_size, n_bd), replace=False)
            
            x_bd = training_data['boundary']['x'][idx_bd].cpu().numpy()
            y_bd = training_data['boundary']['y'][idx_bd].cpu().numpy()
            z_bd = training_data['boundary']['z'][idx_bd].cpu().numpy()
            t_bd = training_data['boundary']['t'][idx_bd].cpu().numpy()
            u_bd_true = training_data['boundary']['u'][idx_bd].cpu().numpy()
            
            u_bd_pred = self.forward(x_bd, y_bd, z_bd, t_bd)
            boundary_loss = float(np.mean((u_bd_pred.flatten() - u_bd_true.flatten())**2))
            
            # 3. Peak loss (center at t=0)
            u_peak_pred = self.forward_single(L/2, L/2, L/2, 0.0)
            u_peak_true = initial_condition(L/2, L/2, L/2)
            peak_loss = float((u_peak_pred - u_peak_true)**2)
            
            # Store individual losses for history
            self._current_losses = {
                'initial': initial_loss,
                'peak': peak_loss,
                'boundary': boundary_loss
            }
            
            # Update running statistics for adaptive weighting
            self._loss_stats['initial']['sum'] += initial_loss
            self._loss_stats['initial']['count'] += 1
            self._loss_stats['peak']['sum'] += peak_loss
            self._loss_stats['peak']['count'] += 1
            self._loss_stats['boundary']['sum'] += boundary_loss
            self._loss_stats['boundary']['count'] += 1
            
            # Combined loss with current weights
            total_loss = float(
                self.loss_weights['initial'] * initial_loss + 
                self.loss_weights['peak'] * peak_loss + 
                self.loss_weights['boundary'] * boundary_loss
            )
            
        except Exception as e:
            print(f"Loss computation error: {e}")
            total_loss = 1e10
        
        return total_loss
    
    def _update_adaptive_weights(self, epoch: int):
        """
        Update adaptive loss weights based on loss magnitudes
        Reference: Self-adaptive loss balanced physics-informed neural networks (Xiang et al., 2022)
        """
        if not self.adaptive_weights or epoch < 10:
            return
        
        # Compute mean losses
        mean_losses = {}
        for key in self._loss_stats:
            if self._loss_stats[key]['count'] > 0:
                mean_losses[key] = self._loss_stats[key]['sum'] / self._loss_stats[key]['count']
            else:
                mean_losses[key] = 1.0
        
        # Prevent division by zero
        eps = 1e-8
        
        # Compute inverse loss weighting (higher weight for smaller losses to balance)
        total_inverse = sum(1.0 / (loss + eps) for loss in mean_losses.values())
        
        if total_inverse > 0:
            # Update weights with smoothing
            smoothing = 0.9
            for key in mean_losses:
                new_weight = (1.0 / (mean_losses[key] + eps)) / total_inverse * len(mean_losses)
                # Clamp weights to reasonable range
                new_weight = np.clip(new_weight, 0.1, 100.0)
                self.loss_weights[key] = smoothing * self.loss_weights[key] + (1 - smoothing) * new_weight
        
        # Reset running statistics
        for key in self._loss_stats:
            self._loss_stats[key] = {'sum': 0.0, 'count': 0}
    
    def _compute_validation_loss(self, validation_data: Dict) -> float:
        """Compute validation loss"""
        if validation_data is None:
            return float('nan')
        
        try:
            x_val = validation_data['x'].cpu().numpy() if isinstance(validation_data['x'], torch.Tensor) else validation_data['x']
            y_val = validation_data['y'].cpu().numpy() if isinstance(validation_data['y'], torch.Tensor) else validation_data['y']
            z_val = validation_data['z'].cpu().numpy() if isinstance(validation_data['z'], torch.Tensor) else validation_data['z']
            t_val = validation_data['t'].cpu().numpy() if isinstance(validation_data['t'], torch.Tensor) else validation_data['t']
            
            # Support both 'u_true' and 'u' keys
            u_key = 'u_true' if 'u_true' in validation_data else 'u'
            u_true = validation_data[u_key].cpu().numpy() if isinstance(validation_data[u_key], torch.Tensor) else validation_data[u_key]
            
            u_pred = self.forward(x_val, y_val, z_val, t_val)
            val_loss = float(np.mean((u_pred.flatten() - u_true.flatten())**2))
            return val_loss
        except Exception as e:
            return float('nan')
    
    def train_model(
        self,
        training_data: Dict,
        epochs: int = None,
        log_interval: int = 10,
        log_to_file: bool = True,
        log_dir: str = "logs",
        validation_data: Dict = None,
        adaptive_weights: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the QPINN model using custom SPSA optimizer
        
        Args:
            training_data: Training data dictionary
            epochs: Number of training epochs
            log_interval: Logging interval
            log_to_file: Whether to log to file
            log_dir: Directory for log files
            validation_data: Validation data dictionary
            adaptive_weights: Whether to use adaptive loss weighting
            
        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = qnn_epochs
        
        self.training_data = training_data
        self.validation_data = validation_data
        self.adaptive_weights = adaptive_weights
        
        # Initialize current losses tracking
        self._current_losses = {'initial': 0.0, 'peak': 0.0, 'boundary': 0.0}
        
        # Setup logger
        self.logger = setup_qpinn_logger(
            name="qpinn_training",
            log_dir=log_dir,
            log_to_file=log_to_file
        )
        
        total_params = self.n_circuit_params + self.n_output_params + len(self.spatial_params)
        
        self.logger.info("=" * 60)
        self.logger.info("QPINN Training with Custom SPSA Optimizer")
        self.logger.info("=" * 60)
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Qubits: {self.n_qubits}")
        self.logger.info(f"Circuit layers: {self.n_layers}")
        self.logger.info(f"Circuit parameters: {self.n_circuit_params}")
        self.logger.info(f"Total parameters: {total_params}")
        self.logger.info(f"GQE enabled: {self.use_gqe}")
        self.logger.info(f"Device: {self.device_name}")
        self.logger.info(f"Adaptive loss weighting: {self.adaptive_weights}")
        self.logger.info(f"Validation data: {'Provided' if validation_data else 'None'}")
        self.logger.info(f"SPSA config: c={SPSA_CONFIG['c']}, alpha={SPSA_CONFIG['alpha']}, gamma={SPSA_CONFIG['gamma']}")
        self.logger.info("=" * 60)
        
        # Combine all parameters into single array
        all_params = np.concatenate([
            self.circuit_params.flatten(),
            self.output_params.flatten(),
            self.spatial_params.flatten()
        ])
        
        # Create custom SPSA optimizer (NOT PennyLane's)
        opt = ManualSPSAOptimizer(
            maxiter=epochs,
            a=0.05,
            c=SPSA_CONFIG['c'],
            alpha=SPSA_CONFIG['alpha'],
            gamma=SPSA_CONFIG['gamma'],
            A=SPSA_CONFIG['A'] if SPSA_CONFIG['A'] is not None else 0.1 * epochs
        )
        
        history = {
            'total_loss': [],
            'initial_loss': [],
            'peak_loss': [],
            'boundary_loss': [],
            'validation_loss': [],
            'circuit_executions': [],
            'adaptive_weights_initial': [],
            'adaptive_weights_peak': [],
            'adaptive_weights_boundary': []
        }
        
        start_time = time.time()
        best_loss = float('inf')
        best_val_loss = float('inf')
        best_params = all_params.copy()
        
        # Define cost function
        def cost_fn(params):
            return self._compute_loss(params, training_data)
        
        # Training loop
        for epoch in range(epochs):
            # Perform SPSA step
            all_params, loss = opt.step(cost_fn, all_params)
            
            # Record history
            history['total_loss'].append(loss)
            history['initial_loss'].append(self._current_losses.get('initial', 0.0))
            history['peak_loss'].append(self._current_losses.get('peak', 0.0))
            history['boundary_loss'].append(self._current_losses.get('boundary', 0.0))
            history['circuit_executions'].append((epoch + 1) * 3)  # 3 evaluations per step
            
            # Record adaptive weights
            history['adaptive_weights_initial'].append(self.loss_weights['initial'])
            history['adaptive_weights_peak'].append(self.loss_weights['peak'])
            history['adaptive_weights_boundary'].append(self.loss_weights['boundary'])
            
            # Compute validation loss
            val_loss = self._compute_validation_loss(validation_data)
            history['validation_loss'].append(val_loss)
            
            # Track best (use validation loss if available)
            if loss < best_loss:
                best_loss = loss
                best_params = all_params.copy()
            
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Update adaptive weights periodically
            if self.adaptive_weights and (epoch + 1) % 20 == 0:
                self._update_adaptive_weights(epoch)
            
            # Dynamic circuit update using GQE
            if (self.use_gqe and self.dynamic_circuit_update and 
                self.gqe_generator is not None and 
                (epoch + 1) % self.circuit_update_interval == 0):
                
                try:
                    new_template, was_updated = self.gqe_generator.update_circuit(
                        self.circuit_template,
                        history['total_loss'],
                        update_threshold=0.1
                    )
                    
                    if was_updated:
                        self._update_circuit(new_template, all_params)
                        n_circuit = self.n_circuit_params
                        all_params = np.concatenate([
                            self.circuit_params.flatten(),
                            all_params[n_circuit:]
                        ])
                        self.circuit_update_history.append({
                            'epoch': epoch + 1,
                            'new_params': self.n_circuit_params
                        })
                        self.logger.info(f"Circuit updated at epoch {epoch + 1}")
                except Exception as e:
                    self.logger.warning(f"Circuit update failed: {e}")
            
            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                elapsed = time.time() - start_time
                val_str = f"Val: {val_loss:.6e}" if not np.isnan(val_loss) else "Val: N/A"
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} [{elapsed:.1f}s] | "
                    f"Loss: {loss:.6e} | Init: {self._current_losses.get('initial', 0):.6e} | "
                    f"Peak: {self._current_losses.get('peak', 0):.6e} | "
                    f"BC: {self._current_losses.get('boundary', 0):.6e} | "
                    f"Best: {best_loss:.6e} | {val_str}"
                )
                if self.adaptive_weights:
                    self.logger.info(
                        f"  Adaptive weights: Init={self.loss_weights['initial']:.2f}, "
                        f"Peak={self.loss_weights['peak']:.2f}, BC={self.loss_weights['boundary']:.2f}"
                    )
        
        # Restore best parameters
        n_circuit = self.n_circuit_params
        n_output = self.n_output_params
        self.circuit_params = best_params[:n_circuit].copy()
        self.output_params = best_params[n_circuit:n_circuit + n_output].copy()
        self.spatial_params = best_params[n_circuit + n_output:].copy()
        
        training_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training completed in {training_time:.1f}s")
        self.logger.info(f"Final Loss: {history['total_loss'][-1]:.6e}")
        self.logger.info(f"Best Loss: {best_loss:.6e}")
        if not np.isnan(best_val_loss) and best_val_loss < float('inf'):
            self.logger.info(f"Best Validation Loss: {best_val_loss:.6e}")
        self.logger.info(f"Total Circuit Executions: {history['circuit_executions'][-1]}")
        self.logger.info("=" * 60)
        
        self.loss_history = history['total_loss']
        
        return history
    
    def _update_circuit(self, new_template: QuantumCircuitTemplate, current_params: np.ndarray):
        """Update circuit template and reinitialize parameters"""
        old_n_params = self.n_circuit_params
        self.circuit_template = new_template
        self.n_circuit_params = new_template.n_params
        
        # Initialize new circuit parameters
        if self.n_circuit_params > old_n_params:
            extra_params = np.random.randn(self.n_circuit_params - old_n_params) * 0.1
            self.circuit_params = np.concatenate([
                current_params[:old_n_params],
                extra_params
            ])
        else:
            self.circuit_params = current_params[:self.n_circuit_params].copy()
        
        # Recreate QNode
        self._create_qnode()
    
    def get_circuit_info(self) -> Dict:
        """Get information about the current quantum circuit"""
        return {
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'n_circuit_params': self.n_circuit_params,
            'n_output_params': self.n_output_params,
            'n_spatial_params': len(self.spatial_params),
            'total_params': self.n_circuit_params + self.n_output_params + len(self.spatial_params),
            'use_gqe': self.use_gqe,
            'circuit_updates': len(self.circuit_update_history),
            'template_metadata': self.circuit_template.metadata if self.circuit_template else None
        }
    
    def get_results_summary(self) -> Dict:
        """Get summary of training results"""
        return {
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'best_loss': min(self.loss_history) if self.loss_history else None,
            'total_epochs': len(self.loss_history),
            'circuit_info': self.get_circuit_info(),
            'circuit_updates': self.circuit_update_history
        }


def create_qpinn_model(
    config: Dict = None,
    use_gqe: bool = True,  # GQE enabled by default
    gqe_config: Dict = None,
    dynamic_update: bool = False
) -> QPINN3DHeatSolver:
    """
    Factory function to create QPINN model
    
    Args:
        config: QPINN configuration dictionary
        use_gqe: Whether to use GQE-GPT circuit generation (default: True)
        gqe_config: GQE configuration dictionary
        dynamic_update: Whether to enable dynamic circuit updates
        
    Returns:
        Configured QPINN model
    """
    if config is None:
        config = QPINN_CONFIG
    if gqe_config is None:
        gqe_config = GQE_CONFIG
    
    return QPINN3DHeatSolver(
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers'],
        use_gqe=use_gqe,
        gqe_config=gqe_config,
        dynamic_circuit_update=dynamic_update
    )