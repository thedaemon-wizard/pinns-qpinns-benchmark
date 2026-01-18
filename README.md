# PINNs vs QPINNs Benchmark

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.43+-green.svg)](https://pennylane.ai/)

A comprehensive benchmark comparing **Physics-Informed Neural Networks (PINNs)** with **Quantum Physics-Informed Neural Networks (QPINNs)** for solving the 3D heat conduction equation.

## Features

- **PINN**: Transformer-based architecture (PINNsFormer) with RAdam optimizer
- **QPINN**: Variational quantum circuits with SPSA optimizer (PennyLane v0.43)
- **GQE-GPT**: Optional GPT-based quantum circuit generation and optimization
- **Multi-objective Optimization**: 9-objective circuit quality evaluation
- **Command-line Interface**: Flexible benchmark execution with various options

## Problem Description

3D Heat Conduction Equation:
```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
```

- **Domain**: [0, L]³ × [0, T] where L=1.0, T=1.0
- **Initial condition**: Gaussian centered at (L/2, L/2, L/2)
- **Boundary conditions**: Dirichlet (u = 0 on all boundaries)
- **Thermal diffusivity**: α = 0.01

## Installation

### From Source

```bash
git clone https://github.com/yourusername/pinns-qpinns-benchmark.git
cd pinns-qpinns-benchmark
pip install -r requirements.txt
```

### As Package

```bash
pip install -e .
```

### With GPU Support (Optional)

```bash
pip install pennylane-lightning-gpu
```

## Quick Start

### Run Full Benchmark

```bash
python main.py
```

### Run PINN Only

```bash
python main.py --pinn-only
```

### Run QPINN Only

```bash
python main.py --qpinn-only
```

### Run QPINN with GQE Circuit Optimization

```bash
python main.py --qpinn-only --use-gqe
```

## Command Line Options

```
usage: main.py [-h] [--pinn-only | --qpinn-only] [--epochs PINN QPINN]
               [--seed SEED] [--qubits QUBITS] [--layers LAYERS] [--use-gqe]
               [--gqe-iterations GQE_ITERATIONS] [--dynamic-update]
               [--output OUTPUT] [--no-plot] [--quiet] [--test-points TEST_POINTS]

Options:
  --pinn-only           Run only PINN benchmark
  --qpinn-only          Run only QPINN benchmark
  --epochs PINN QPINN   Number of epochs (default: 3000 200)
  --seed SEED           Random seed (default: 42)
  --qubits QUBITS       Number of qubits for QPINN (default: 6)
  --layers LAYERS       Number of circuit layers (default: 4)
  --use-gqe             Enable GQE-GPT circuit optimization
  --gqe-iterations N    GQE optimization iterations (default: 5)
  --dynamic-update      Enable dynamic circuit updates during training
  --output, -o DIR      Output directory for results
  --no-plot             Disable visualization generation
  --quiet, -q           Reduce output verbosity
  --test-points N       Number of test points (default: 1000)
```

## Examples

```bash
# Full benchmark with custom epochs
python main.py --epochs 2000 150

# QPINN with 8 qubits and 6 layers
python main.py --qpinn-only --qubits 8 --layers 6

# QPINN with GQE optimization (10 iterations)
python main.py --qpinn-only --use-gqe --gqe-iterations 10

# Quick test run
python main.py --epochs 100 20 --test-points 100 --quiet

# Save to specific directory
python main.py --output my_experiment
```

## Project Structure

```
pinns-qpinns-benchmark/
├── main.py              # CLI entry point
├── config.py            # Configuration and parameters
├── data_utils.py        # Data generation utilities
├── pinn_model.py        # PINN with RAdam optimizer
├── qpinn_model.py       # QPINN with SPSA optimizer
├── gpt_model.py         # GPT model for circuit generation
├── energy_estimator.py  # Unsupervised energy estimation
├── gqe_generator.py     # GQE circuit generator
├── requirements.txt     # Dependencies
├── setup.py             # Package setup
├── pyproject.toml       # Build configuration
├── LICENSE              # MIT License
└── README.md            # This file
```

## GQE-GPT Circuit Generation

The project includes an advanced quantum circuit generation system:

### Components

- **QuantumCircuitGPT**: GPT-2 based model for generating circuit token sequences
- **CircuitTokenizer**: Converts between gate sequences and token IDs
- **UnsupervisedQuantumEnergyEstimator**: Estimates circuit quality without labels
- **CircuitQualityEvaluator**: Multi-objective evaluation (9 objectives)
- **GQEQuantumCircuitGenerator**: Main generator combining all components

### Multi-objective Optimization Objectives

1. **Hardware Efficiency** - Gate compatibility with real hardware
2. **Noise Resilience** - Robustness to quantum noise
3. **Expressivity** - Circuit's ability to represent complex functions
4. **Mitigation Compatibility** - Error mitigation friendliness
5. **Trainability** - Avoiding barren plateaus
6. **Entanglement Capability** - Quantum correlations
7. **Depth Efficiency** - Gate parallelization
8. **Parameter Efficiency** - Optimal parameter count
9. **Energy Estimation Quality** - Prediction accuracy

### Usage

```python
from qpinn_model import create_qpinn_with_optimized_circuit

# Create QPINN with GQE-optimized circuit
model = create_qpinn_with_optimized_circuit(
    training_data,
    gqe_iterations=5
)

# Train the model
history = model.train_model(training_data, epochs=200)
```

## Optimizers

### RAdam (PINN)

Rectified Adam optimizer for stable neural network training.

```python
optimizer = torch.optim.RAdam(
    params,
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
```

**Reference**: Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)

### SPSA (QPINN)

Simultaneous Perturbation Stochastic Approximation from PennyLane v0.43.

```python
opt = qml.SPSAOptimizer(
    maxiter=200,
    alpha=0.602,    # Learning rate decay
    gamma=0.101,    # Perturbation decay
    c=0.1,          # Initial perturbation
    A=20            # Stability constant
)

for _ in range(epochs):
    params, loss = opt.step_and_cost(cost_fn, params)
```

**Key Features**:
- Only 2 cost function evaluations per iteration
- Gradient-free optimization
- Robust to quantum noise

**Reference**: Spall, "An Overview of the Simultaneous Perturbation Method for Efficient Optimization"

## Output

The benchmark generates:

- `loss_comparison.png` - Training loss curves
- `solution_comparison_t0.png` - Solution visualization at t=0
- `error_comparison.png` - Error distribution maps
- `metrics_comparison.png` - Metrics bar chart
- `benchmark_results.json` - Numerical results

## Configuration

Key parameters in `config.py`:

```python
# Problem
alpha = 0.01    # Thermal diffusivity
L = 1.0         # Domain size
T = 1.0         # Final time

# Training
pinn_epochs = 3000
qnn_epochs = 200

# QPINN
QPINN_CONFIG = {
    'n_qubits': 6,
    'n_layers': 4,
    'use_gpt_circuit_generation': True,
}

# GQE
GQE_CONFIG = {
    'n_candidates': 20,
    'n_iterations': 5,
    'exploration_rate': 0.9,
}
```

## References

1. Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (2019)
2. Trahan et al., "Quantum Physics-Informed Neural Networks" Entropy 26(8):649 (2024)
3. Apak et al., "KetGPT - Dataset Augmentation of Quantum Circuits using Transformers" arXiv:2402.13352 (2024)
4. Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)
5. PennyLane SPSA: https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
