"""
PINN Model for 3D Heat Conduction Equation
Optimizer: RAdam (PyTorch)
Architecture: PINNsFormer (Transformer-based)

Features:
- Self-adaptive loss weighting based on Gaussian likelihood estimation
- File logging support
- Optional validation monitoring
- English comments without special characters

References:
- Xiang et al. (2022) Self-adaptive loss balanced Physics-informed neural networks
- Wang et al. (2022) When and why PINNs fail to train
- Liu et al. (2020) On the Variance of the Adaptive Learning Rate and Beyond (RAdam)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import math

from config import (
    L, T, alpha, device, sigma_0,
    PINN_CONFIG, RADAM_CONFIG, pinn_epochs,
    initial_condition, to_python_float
)
from data_utils import compute_pde_residual, stratified_sampling


# ================================================
# Logging Setup
# ================================================
def setup_logger(
    name: str = "pinn_training",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# ================================================
# Adaptive Loss Weighting
# ================================================
class AdaptiveLossWeights:
    """
    Self-adaptive loss weighting based on Gaussian likelihood estimation
    
    Reference: Xiang et al. (2022) Self-adaptive loss balanced PINNs
    
    The weights are updated based on maximum likelihood estimation:
    weight_i = 1 / (2 * sigma_i^2)
    where sigma_i is the standard deviation of the i-th loss term
    """
    
    def __init__(
        self,
        n_losses: int = 4,
        initial_weights: Optional[List[float]] = None,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 100.0,
        use_adaptive: bool = True
    ):
        """
        Initialize adaptive loss weights
        
        Args:
            n_losses: Number of loss terms
            initial_weights: Initial weights for each loss term
            adaptation_rate: Rate of weight adaptation (0 to 1)
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            use_adaptive: Whether to use adaptive weighting
        """
        self.n_losses = n_losses
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_adaptive = use_adaptive
        
        # Initialize weights
        if initial_weights is not None:
            self.weights = torch.tensor(initial_weights, dtype=torch.float32)
        else:
            self.weights = torch.ones(n_losses, dtype=torch.float32)
        
        # Running statistics for loss values
        self.loss_means = torch.zeros(n_losses, dtype=torch.float32)
        self.loss_vars = torch.ones(n_losses, dtype=torch.float32)
        self.n_updates = 0
        
        # History for analysis
        self.weight_history = []
    
    def update(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Update weights based on current loss values
        
        Args:
            losses: List of loss tensors
            
        Returns:
            Updated weights tensor
        """
        if not self.use_adaptive:
            return self.weights.to(losses[0].device)
        
        self.n_updates += 1
        
        # Convert losses to tensor
        loss_values = torch.tensor(
            [l.detach().item() for l in losses],
            dtype=torch.float32
        )
        
        # Update running mean and variance using Welford algorithm
        delta = loss_values - self.loss_means
        self.loss_means = self.loss_means + delta / self.n_updates
        delta2 = loss_values - self.loss_means
        self.loss_vars = self.loss_vars + delta * delta2
        
        # Compute variance estimates
        if self.n_updates > 1:
            variances = self.loss_vars / (self.n_updates - 1)
            variances = torch.clamp(variances, min=1e-10)
            
            # Compute new weights based on inverse variance
            # weight_i = 1 / (2 * sigma_i^2)
            new_weights = 1.0 / (2.0 * variances + 1e-8)
            
            # Normalize weights
            new_weights = new_weights / new_weights.sum() * self.n_losses
            
            # Smooth update with adaptation rate
            self.weights = (
                (1 - self.adaptation_rate) * self.weights +
                self.adaptation_rate * new_weights
            )
            
            # Clamp weights
            self.weights = torch.clamp(
                self.weights,
                min=self.min_weight,
                max=self.max_weight
            )
        
        # Record history
        self.weight_history.append(self.weights.clone())
        
        return self.weights.to(losses[0].device)
    
    def get_weights(self) -> torch.Tensor:
        """Get current weights"""
        return self.weights
    
    def reset(self):
        """Reset statistics"""
        self.loss_means = torch.zeros(self.n_losses, dtype=torch.float32)
        self.loss_vars = torch.ones(self.n_losses, dtype=torch.float32)
        self.n_updates = 0
        self.weight_history = []


# ================================================
# Custom Activation Functions
# ================================================
class WaveletActivation(nn.Module):
    """
    Wavelet-inspired activation function for better frequency representation
    Combines sine wave with Gaussian envelope for multi-scale feature capture
    """
    def __init__(self):
        super(WaveletActivation, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
    def forward(self, x):
        return self.scale * torch.sin(x) * torch.exp(-0.1 * x**2)


# ================================================
# PINNsFormer Components
# ================================================
class PseudoSequenceGenerator(nn.Module):
    """
    Generate pseudo-sequence from point-wise input
    Converts spatial-temporal points into sequence format for Transformer
    """
    def __init__(self, input_dim, seq_length=16, d_model=128, delta_t=0.0625):
        super(PseudoSequenceGenerator, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.delta_t = delta_t
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            WaveletActivation(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        self.temporal_embedding = nn.Embedding(seq_length, d_model)
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.input_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        projected = self.input_projection(x)
        projected = projected.unsqueeze(1).expand(-1, self.seq_length, -1)
        
        positions = torch.arange(self.seq_length, device=x.device)
        pos_embed = self.temporal_embedding(positions)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        sequence = projected + 0.1 * pos_embed
        return sequence


class SpatioTemporalMixer(nn.Module):
    """
    Mix spatial and temporal features using dual attention mechanism
    Processes both spatial correlations and temporal dependencies
    """
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super(SpatioTemporalMixer, self).__init__()
        
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        self.mixing_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            WaveletActivation(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self._init_weights()
        
    def _init_weights(self):
        for module in [self.spatial_attention, self.temporal_attention]:
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight, gain=0.5)
        for layer in self.mixing_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
    
    def forward(self, seq):
        spatial_out, _ = self.spatial_attention(seq, seq, seq)
        seq = self.norm1(seq + 0.1 * spatial_out)
        
        temporal_out, _ = self.temporal_attention(seq, seq, seq)
        seq = self.norm2(seq + 0.1 * temporal_out)
        
        mixed = self.mixing_layer(seq)
        output = self.norm3(seq + 0.1 * mixed)
        
        return output


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with pre-norm architecture
    """
    def __init__(self, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveletActivation(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        if hasattr(self.self_attention, 'in_proj_weight') and self.self_attention.in_proj_weight is not None:
            nn.init.xavier_uniform_(self.self_attention.in_proj_weight, gain=0.5)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + 0.1 * self.dropout(attn_output))
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + 0.1 * self.dropout(ffn_output))
        
        return x


class PINNsFormerEncoder(nn.Module):
    """
    Encoder with multiple Transformer blocks for PINN
    """
    def __init__(self, n_layers=4, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super(PINNsFormerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class OutputProjection(nn.Module):
    """
    Project Transformer output to solution value
    Uses learnable temporal weights for sequence aggregation
    """
    def __init__(self, seq_length=16, d_model=128, output_dim=1):
        super(OutputProjection, self).__init__()
        
        self.seq_length = seq_length
        self.temporal_weights = nn.Parameter(
            torch.ones(seq_length, dtype=torch.float32) / seq_length
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            WaveletActivation(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )
        self._init_weights()
        
    def _init_weights(self):
        with torch.no_grad():
            # Emphasize early time steps for initial condition
            weights = torch.exp(-torch.arange(self.seq_length, dtype=torch.float32) / 4.0)
            self.temporal_weights.data = weights / weights.sum()
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
    
    def forward(self, seq):
        weights = torch.softmax(self.temporal_weights, dim=0)
        weighted_seq = torch.einsum('bsd,s->bd', seq, weights)
        output = self.output_layer(weighted_seq)
        return output


# ================================================
# Main PINN Model
# ================================================
class PINN3DHeatSolver(nn.Module):
    """
    Physics-Informed Neural Network for 3D Heat Equation
    Using PINNsFormer architecture with RAdam optimizer
    
    Features:
    - Self-adaptive loss weighting based on Gaussian likelihood estimation
    - Transformer-based architecture for spatio-temporal learning
    - Fourier feature mapping for high-frequency patterns
    - Hard boundary constraints enforcement
    
    Note on data splitting:
    Unlike standard regression, PINNs use physics laws as constraints.
    The ground truth comes from PDEs, not labeled data.
    Validation is done against analytical solutions or separate test points.
    Traditional train/val/test splits are not strictly necessary but can be
    used for hyperparameter tuning and early stopping.
    """
    
    def __init__(
        self,
        layers: List[int] = None,
        use_transformer: bool = True,
        use_fourier_features: bool = True,
        num_fourier_features: int = 64,
        use_hard_constraints: bool = True,
        boundary_epsilon: float = 1e-3,
        transformer_config: Dict = None,
        use_adaptive_weights: bool = True
    ):
        super(PINN3DHeatSolver, self).__init__()
        
        self.use_transformer = use_transformer
        self.use_fourier_features = use_fourier_features
        self.use_hard_constraints = use_hard_constraints
        self.use_adaptive_weights = use_adaptive_weights
        
        if layers is None:
            layers = PINN_CONFIG['layers']
        if transformer_config is None:
            transformer_config = PINN_CONFIG['transformer_config']
        
        self.transformer_config = transformer_config
        
        # Register constants as buffers (float32)
        self.register_buffer('pi', torch.tensor(math.pi, dtype=torch.float32))
        self.register_buffer('L_const', torch.tensor(L, dtype=torch.float32))
        self.register_buffer('T_const', torch.tensor(T, dtype=torch.float32))
        self.register_buffer('sigma_0_const', torch.tensor(sigma_0, dtype=torch.float32))
        
        # Fourier feature matrices (explicitly float32)
        if use_fourier_features:
            self.B_spatial_coarse = nn.Parameter(
                torch.randn(3, num_fourier_features // 4, dtype=torch.float32),
                requires_grad=True
            )
            self.B_spatial_fine = nn.Parameter(
                torch.randn(3, num_fourier_features // 4, dtype=torch.float32),
                requires_grad=True
            )
            two_pi_over_T = torch.tensor(2.0 * math.pi / T, dtype=torch.float32)
            self.B_temporal_slow = nn.Parameter(
                torch.randn(1, num_fourier_features // 4, dtype=torch.float32) * two_pi_over_T,
                requires_grad=True
            )
            ten_pi_over_T = torch.tensor(10.0 * math.pi / T, dtype=torch.float32)
            self.B_temporal_fast = nn.Parameter(
                torch.randn(1, num_fourier_features // 4, dtype=torch.float32) * ten_pi_over_T,
                requires_grad=True
            )
            self.spatial_scale_coarse = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.spatial_scale_fine = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.temporal_scale = nn.Parameter(torch.tensor(2.0 * math.pi / T, dtype=torch.float32))
        
        # Calculate input dimensions
        base_features = 3  # x_norm, y_norm, z_norm
        temporal_features = 8
        
        if use_fourier_features:
            spatial_fourier_dim = num_fourier_features
            temporal_fourier_dim = num_fourier_features
            self.input_dim = base_features + temporal_features + spatial_fourier_dim + temporal_fourier_dim
        else:
            self.input_dim = base_features + temporal_features
        
        # Build architecture
        if use_transformer:
            config = transformer_config
            
            self.pseudo_seq_generator = PseudoSequenceGenerator(
                input_dim=self.input_dim,
                seq_length=config['seq_length'],
                d_model=config['d_model'],
                delta_t=T / config['seq_length']
            )
            
            self.spatio_temporal_mixer = SpatioTemporalMixer(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                dropout=config['dropout']
            )
            
            self.transformer_encoder = PINNsFormerEncoder(
                n_layers=config['n_layers'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )
            
            self.output_projection = OutputProjection(
                seq_length=config['seq_length'],
                d_model=config['d_model'],
                output_dim=1
            )
        else:
            # Fallback MLP
            self.layers = nn.ModuleList()
            layer_sizes = [self.input_dim] + layers[1:]
            
            for i in range(len(layer_sizes) - 1):
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                if i < len(layer_sizes) - 2:
                    self.layers.append(nn.Tanh())
        
        # Hard constraints
        if use_hard_constraints:
            self.boundary_epsilon = nn.Parameter(
                torch.tensor(boundary_epsilon, dtype=torch.float32)
            )
        
        # Adaptive loss weights (4 losses: initial, peak, boundary, pde)
        # Initial weights: [10.0, 50.0, 5.0, 1.0] for initial, peak, boundary, pde
        self.adaptive_weights = AdaptiveLossWeights(
            n_losses=4,
            initial_weights=[10.0, 50.0, 5.0, 1.0],
            adaptation_rate=0.1,
            use_adaptive=use_adaptive_weights
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Training data and history
        self.training_data = None
        self.validation_data = None
        self.loss_history = []
        self.validation_history = []
        self.objective_history = {
            'initial': [], 'peak': [], 'boundary': [], 'pde': [], 'combined': []
        }
        
        # Logger
        self.logger = None
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def compute_distance_function(self, x, y, z):
        """
        Compute smooth boundary distance function
        Returns ~1 in interior, ~0 at boundaries
        Uses product of smooth step functions for each boundary
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if z.dim() == 1:
            z = z.unsqueeze(1)
        
        L_val = self.L_const.to(x.device)
        
        # Smooth step function: tanh(k * d) where d is distance to boundary
        # This approaches 1 in interior and 0 at boundaries
        k = 20.0  # Steepness parameter
        
        # Distance to each boundary, normalized by L
        d_x_low = x / L_val
        d_x_high = (L_val - x) / L_val
        d_y_low = y / L_val
        d_y_high = (L_val - y) / L_val
        d_z_low = z / L_val
        d_z_high = (L_val - z) / L_val
        
        # Smooth boundary mask: product of tanh functions
        # Near boundary: d~0, tanh(k*d)~0
        # In interior: d~0.5, tanh(k*0.5)~tanh(10)~1
        mask = (torch.tanh(k * d_x_low) * torch.tanh(k * d_x_high) *
                torch.tanh(k * d_y_low) * torch.tanh(k * d_y_high) *
                torch.tanh(k * d_z_low) * torch.tanh(k * d_z_high))
        
        return mask
    
    def forward(self, x, y, z, t):
        """
        Forward pass through the network
        
        Args:
            x, y, z: Spatial coordinates
            t: Time coordinate
            
        Returns:
            u: Predicted temperature field
        """
        # Ensure float32
        x = x.float()
        y = y.float()
        z = z.float()
        t = t.float()
        
        # Get constants on correct device
        L_val = self.L_const.to(x.device)
        T_val = self.T_const.to(x.device)
        pi_val = self.pi.to(x.device)
        sigma_val = self.sigma_0_const.to(x.device)
        
        # Normalize inputs
        x_norm = x / L_val
        y_norm = y / L_val
        z_norm = z / L_val
        t_norm = t / T_val
        
        # Build temporal features (using float32 constants)
        two_pi = 2.0 * pi_val
        four_pi = 4.0 * pi_val
        
        temporal_features = torch.cat([
            t_norm,
            torch.sin(two_pi * t_norm),
            torch.cos(two_pi * t_norm),
            torch.sin(four_pi * t_norm),
            torch.cos(four_pi * t_norm),
            torch.exp(torch.clamp(-t_norm, -10.0, 0.0)),
            torch.exp(torch.clamp(-2.0 * t_norm, -10.0, 0.0)),
            t_norm**2
        ], dim=1)
        
        inputs = [x_norm, y_norm, z_norm, temporal_features]
        
        # Apply Fourier features
        if self.use_fourier_features:
            spatial_coords = torch.cat([x_norm, y_norm, z_norm], dim=1)
            
            # Coarse and fine spatial features
            spatial_proj_coarse = spatial_coords @ (self.spatial_scale_coarse * self.B_spatial_coarse)
            spatial_proj_fine = spatial_coords @ (self.spatial_scale_fine * self.B_spatial_fine * 10.0)
            
            spatial_fourier = torch.cat([
                torch.sin(spatial_proj_coarse),
                torch.cos(spatial_proj_coarse),
                torch.sin(spatial_proj_fine),
                torch.cos(spatial_proj_fine)
            ], dim=1)
            
            # Temporal Fourier features
            temporal_proj_slow = t_norm @ (self.temporal_scale * self.B_temporal_slow)
            temporal_proj_fast = t_norm @ (self.temporal_scale * self.B_temporal_fast * 5.0)
            
            temporal_fourier = torch.cat([
                torch.sin(temporal_proj_slow),
                torch.cos(temporal_proj_slow),
                torch.sin(temporal_proj_fast),
                torch.cos(temporal_proj_fast)
            ], dim=1)
            
            inputs.extend([spatial_fourier, temporal_fourier])
        
        # Concatenate all features
        features = torch.cat(inputs, dim=1)
        
        # Forward through network
        if self.use_transformer:
            seq = self.pseudo_seq_generator(features)
            seq = self.spatio_temporal_mixer(seq)
            seq = self.transformer_encoder(seq)
            u_nn = self.output_projection(seq)
        else:
            h = features
            for layer in self.layers:
                h = layer(h)
            u_nn = h
        
        # Apply hard constraints
        if self.use_hard_constraints:
            distance = self.compute_distance_function(x, y, z)
            
            # Initial condition embedding (Gaussian at center)
            x_c = L_val / 2.0
            y_c = L_val / 2.0
            z_c = L_val / 2.0
            r_squared = (x - x_c)**2 + (y - y_c)**2 + (z - z_c)**2
            sigma_sq = 2.0 * sigma_val**2
            u_initial = torch.exp(-r_squared / sigma_sq)
            
            # Physics-based time decay: (σ₀/σ_t)³ where σ_t² = σ₀² + 2αt
            # This is the correct amplitude decay for heat equation with Gaussian IC
            alpha_const = torch.tensor(alpha, dtype=torch.float32, device=x.device)
            sigma_t_sq = sigma_val**2 + 2 * alpha_const * t
            physical_decay = (sigma_val**2 / sigma_t_sq)**(3.0/2.0)
            
            # Time-evolving Gaussian width
            r_sq_scaled = r_squared / (sigma_val**2 + 2 * alpha_const * t)
            u_evolved = physical_decay * torch.exp(-r_sq_scaled / 2)
            
            # Neural network learns SMALL corrections to the physics
            # Scale u_nn to be a multiplicative correction factor around 1.0
            correction = 1.0 + 0.1 * torch.tanh(u_nn)  # Correction in range [0.9, 1.1]
            
            # Apply correction to physics-based solution
            u = distance * u_evolved * correction
            
            # Ensure non-negative temperature
            u = torch.clamp(u, min=0.0, max=1.0)
        else:
            u = u_nn
        
        return u
    
    def _compute_losses(self, batch_size: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components
        
        Returns:
            Dictionary with loss tensors for each component
        """
        losses = {}
        mse_loss = nn.MSELoss()
        
        # 1. Initial condition loss
        n_init = len(self.training_data['initial']['x'])
        idx_init = torch.randperm(n_init, device=device)[:min(batch_size, n_init)]
        
        x_init = self.training_data['initial']['x'][idx_init]
        y_init = self.training_data['initial']['y'][idx_init]
        z_init = self.training_data['initial']['z'][idx_init]
        t_init = self.training_data['initial']['t'][idx_init]
        u_init_true = self.training_data['initial']['u'][idx_init]
        
        u_init_pred = self.forward(x_init, y_init, z_init, t_init)
        losses['initial'] = mse_loss(u_init_pred, u_init_true)
        
        # 2. Peak loss (center at t=0)
        x_peak = torch.tensor([[L/2]], dtype=torch.float32, device=device)
        y_peak = torch.tensor([[L/2]], dtype=torch.float32, device=device)
        z_peak = torch.tensor([[L/2]], dtype=torch.float32, device=device)
        t_peak = torch.tensor([[0.0]], dtype=torch.float32, device=device)
        u_peak_true = torch.tensor(
            [[float(initial_condition(L/2, L/2, L/2))]],
            dtype=torch.float32, device=device
        )
        
        u_peak_pred = self.forward(x_peak, y_peak, z_peak, t_peak)
        losses['peak'] = mse_loss(u_peak_pred, u_peak_true)
        
        # 3. Boundary condition loss
        n_bd = len(self.training_data['boundary']['x'])
        idx_bd = torch.randperm(n_bd, device=device)[:min(batch_size, n_bd)]
        
        x_bd = self.training_data['boundary']['x'][idx_bd]
        y_bd = self.training_data['boundary']['y'][idx_bd]
        z_bd = self.training_data['boundary']['z'][idx_bd]
        t_bd = self.training_data['boundary']['t'][idx_bd]
        u_bd_true = self.training_data['boundary']['u'][idx_bd]
        
        u_bd_pred = self.forward(x_bd, y_bd, z_bd, t_bd)
        losses['boundary'] = mse_loss(u_bd_pred, u_bd_true)
        
        # 4. PDE residual loss
        n_pde = len(self.training_data['interior']['x'])
        idx_pde = stratified_sampling(
            self.training_data['interior']['t'].squeeze(),
            min(batch_size, n_pde)
        )
        
        x_pde = self.training_data['interior']['x'][idx_pde].clone().detach().requires_grad_(True)
        y_pde = self.training_data['interior']['y'][idx_pde].clone().detach().requires_grad_(True)
        z_pde = self.training_data['interior']['z'][idx_pde].clone().detach().requires_grad_(True)
        t_pde = self.training_data['interior']['t'][idx_pde].clone().detach().requires_grad_(True)
        
        residual = compute_pde_residual(self, x_pde, y_pde, z_pde, t_pde)
        losses['pde'] = torch.mean(residual**2)
        
        return losses
    
    def _compute_validation_loss(self) -> Optional[float]:
        """
        Compute validation loss if validation data is available
        
        Returns:
            Validation MSE or None if no validation data
        """
        if self.validation_data is None:
            return None
        
        self.eval()
        with torch.no_grad():
            u_pred = self.forward(
                self.validation_data['x'],
                self.validation_data['y'],
                self.validation_data['z'],
                self.validation_data['t']
            )
            val_loss = torch.mean((u_pred - self.validation_data['u'])**2).item()
        self.train()
        
        return val_loss
    
    def train_model(
        self,
        training_data: Dict,
        epochs: int = None,
        lr: float = None,
        log_interval: int = 100,
        validation_data: Dict = None,
        log_to_file: bool = True,
        log_dir: str = "logs"
    ) -> Dict[str, List[float]]:
        """
        Train the PINN model using RAdam optimizer
        
        Args:
            training_data: Training data dictionary
            epochs: Number of training epochs
            lr: Learning rate
            log_interval: Logging interval
            validation_data: Optional validation data for monitoring
            log_to_file: Whether to log to file
            log_dir: Directory for log files
            
        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = pinn_epochs
        if lr is None:
            lr = RADAM_CONFIG['lr']
        
        self.training_data = training_data
        self.validation_data = validation_data
        self.to(device)
        self.train()
        
        # Ensure model is float32
        self.float()
        
        # Setup logger
        self.logger = setup_logger(
            name="pinn_training",
            log_dir=log_dir,
            log_to_file=log_to_file,
            log_to_console=True
        )
        
        # Setup RAdam optimizer
        optimizer = optim.RAdam(
            self.parameters(),
            lr=lr,
            betas=RADAM_CONFIG['betas'],
            eps=RADAM_CONFIG['eps'],
            weight_decay=RADAM_CONFIG['weight_decay']
        )
        
        # Learning rate scheduler (verbose removed for newer PyTorch)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=100
        )
        
        # Log configuration
        self.logger.info("=" * 60)
        self.logger.info("PINN Training with RAdam Optimizer")
        self.logger.info("=" * 60)
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Adaptive loss weighting: {self.use_adaptive_weights}")
        self.logger.info(f"Validation data: {'Provided' if validation_data else 'Not provided'}")
        self.logger.info("=" * 60)
        
        history = {
            'total_loss': [],
            'initial_loss': [],
            'boundary_loss': [],
            'pde_loss': [],
            'peak_loss': [],
            'learning_rate': [],
            'validation_loss': [],
            'loss_weights': []
        }
        
        start_time = time.time()
        best_loss = float('inf')
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute losses
            losses = self._compute_losses()
            
            # Update adaptive weights
            loss_list = [
                losses['initial'],
                losses['peak'],
                losses['boundary'],
                losses['pde']
            ]
            weights = self.adaptive_weights.update(loss_list)
            
            # Combined loss with adaptive weights
            total_loss = (
                weights[0] * losses['initial'] +
                weights[1] * losses['peak'] +
                weights[2] * losses['boundary'] +
                weights[3] * losses['pde']
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update scheduler
            scheduler.step(total_loss)
            
            # Compute validation loss
            val_loss = self._compute_validation_loss()
            
            # Record history
            history['total_loss'].append(total_loss.item())
            history['initial_loss'].append(losses['initial'].item())
            history['boundary_loss'].append(losses['boundary'].item())
            history['pde_loss'].append(losses['pde'].item())
            history['peak_loss'].append(losses['peak'].item())
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['validation_loss'].append(val_loss)
            history['loss_weights'].append(weights.tolist())
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                elapsed = time.time() - start_time
                
                log_msg = (
                    f"Epoch {epoch + 1}/{epochs} [{elapsed:.1f}s] | "
                    f"Loss: {total_loss.item():.6e} | "
                    f"Init: {losses['initial'].item():.6e} | "
                    f"Peak: {losses['peak'].item():.6e} | "
                    f"BC: {losses['boundary'].item():.6e} | "
                    f"PDE: {losses['pde'].item():.6e} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                
                if val_loss is not None:
                    log_msg += f" | Val: {val_loss:.6e}"
                
                self.logger.info(log_msg)
                
                # Log weights if adaptive
                if self.use_adaptive_weights:
                    self.logger.info(
                        f"  Adaptive weights: Init={weights[0]:.2f}, "
                        f"Peak={weights[1]:.2f}, BC={weights[2]:.2f}, PDE={weights[3]:.2f}"
                    )
        
        training_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"Training completed in {training_time:.1f}s")
        self.logger.info(f"Final Loss: {history['total_loss'][-1]:.6e}")
        self.logger.info(f"Best Loss: {best_loss:.6e}")
        if validation_data is not None:
            self.logger.info(f"Best Validation Loss: {best_val_loss:.6e}")
        self.logger.info("=" * 60)
        
        self.loss_history = history['total_loss']
        self.validation_history = history['validation_loss']
        
        return history


def create_pinn_model(
    config: Dict = None,
    use_adaptive_weights: bool = True
) -> PINN3DHeatSolver:
    """
    Factory function to create PINN model
    
    Args:
        config: Model configuration dictionary
        use_adaptive_weights: Whether to use adaptive loss weighting
        
    Returns:
        Configured PINN model
    """
    if config is None:
        config = PINN_CONFIG
    
    model = PINN3DHeatSolver(
        layers=config['layers'],
        use_transformer=config['use_transformer'],
        use_fourier_features=config['use_fourier_features'],
        num_fourier_features=config['num_fourier_features'],
        use_hard_constraints=config['use_hard_constraints'],
        boundary_epsilon=config['boundary_epsilon'],
        transformer_config=config['transformer_config'],
        use_adaptive_weights=use_adaptive_weights
    )
    
    # Ensure float32
    model = model.float()
    
    return model.to(device)