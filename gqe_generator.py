"""
GQE Quantum Circuit Generator with GPT Integration

References:
- Apak et al. "KetGPT - Dataset Augmentation of Quantum Circuits 
  using Transformers" arXiv:2402.13352 (2024)
- Meyer et al. "Fisher Information in Noisy Intermediate-Scale Quantum Applications"
  Quantum 5, 539 (2021)
"""

import torch
import torch.nn.functional as F
import numpy as np
import pennylane as qml
from typing import List, Dict, Optional, Tuple, Any
import time
import os
import json

from config import (
    QuantumCircuitTemplate, 
    device, 
    L, T,
    GQE_CONFIG
)
from gpt_model import (
    QuantumCircuitGPT,
    CircuitTokenizer,
    QuantumCircuitDataset,
    create_gpt_model,
    train_gpt_on_circuits
)
from energy_estimator import (
    UnsupervisedQuantumEnergyEstimator,
    CircuitQualityEvaluator
)


class GQEQuantumCircuitGenerator:
    """
    GPT-based GQE quantum circuit generator
    
    Combines GPT language model with unsupervised energy estimation
    for quantum circuit optimization.
    """
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 4,
        noise_budget: float = 0.01,
        hardware_topology: str = 'linear',
        use_pretrained_gpt: bool = False,
        use_energy_prediction: bool = True,
        max_circuit_depth: int = 20
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_budget = noise_budget
        self.hardware_topology = hardware_topology
        self.use_pretrained_gpt = use_pretrained_gpt
        self.use_energy_prediction = use_energy_prediction
        self.max_circuit_depth = max_circuit_depth
        
        # Preferred gates for hardware efficiency
        self.preferred_gates = ['RY', 'RZ', 'CNOT', 'CZ']
        
        # Initialize GPT model and tokenizer
        self.gpt_model, self.tokenizer = create_gpt_model(n_qubits)
        self.gpt_optimizer = torch.optim.AdamW(self.gpt_model.parameters(), lr=5e-4)
        
        # Initialize energy estimator
        if use_energy_prediction:
            self.energy_estimator = UnsupervisedQuantumEnergyEstimator(
                n_qubits=n_qubits,
                n_layers=n_layers,
                use_noise=(noise_budget > 0),
                shots=1000
            )
        else:
            self.energy_estimator = None
        
        # Circuit quality evaluator
        self.quality_evaluator = CircuitQualityEvaluator(n_qubits, self.energy_estimator)
        
        # History tracking
        self.circuit_history = []
        self.energy_history = []
        self.generation_history = []
        
        # Search parameters
        self.exploration_rate = 0.9
        self.exploration_decay = 0.85
        self.diversity_bonus = 0.2
        
        # Training data cache
        self.cached_training_data = None
        self.cached_prepared_inputs = None
        
        # Initialize with KetGPT data if available
        self._initialize_ketgpt_dataset()
        
        print(f"GQE Generator initialized: {n_qubits} qubits, {n_layers} layers")
    
    def _initialize_ketgpt_dataset(self):
        """Load and preprocess KetGPT dataset for pretraining"""
        try:
            # Try to load PennyLane KetGPT dataset
            [ketgpt_dataset] = qml.data.load("ketgpt")
            
            circuits_data = []
            for circuit in ketgpt_dataset.circuits:
                circuits_data.append(circuit)
            
            print(f"KetGPT dataset loaded: {len(circuits_data)} circuits")
            del ketgpt_dataset
            
            # Prepare pretraining data
            pretrain_data = []
            for i, circuit in enumerate(circuits_data):
                gate_sequence = self._pennylane_to_gate_sequence(circuit)
                if gate_sequence and len(gate_sequence) <= self.max_circuit_depth:
                    pretrain_data.append({
                        'gate_sequence': gate_sequence,
                        'energy': -1.0 - 0.01 * i,
                        'score': 0.8 + 0.001 * i
                    })
            
            # Pretrain GPT model
            if self.gpt_model is not None and pretrain_data:
                print("Pretraining GPT model with KetGPT data...")
                self._train_gpt_on_circuits(pretrain_data, epochs=min(100, len(pretrain_data) * 10))
                
        except Exception as e:
            print(f"KetGPT dataset not available: {e}")
            print("Using random initialization")
    
    def _pennylane_to_gate_sequence(self, pennylane_ops) -> List[Dict]:
        """Convert PennyLane operations to gate sequence format"""
        gate_sequence = []
        param_counter = 0
        
        gate_map = {
            'PAULIX': 'X', 'PAULIY': 'Y', 'PAULIZ': 'Z',
            'HADAMARD': 'H', 'CNOT': 'CNOT', 'CZ': 'CZ',
            'RX': 'RX', 'RY': 'RY', 'RZ': 'RZ'
        }
        
        for op in pennylane_ops:
            gate_name = op.name.upper()
            if gate_name not in gate_map:
                continue
            
            wires = list(op.wires)
            if any(w >= self.n_qubits for w in wires):
                continue
            
            trainable = hasattr(op, 'parameters') and len(op.parameters) > 0
            
            gate_info = {
                'gate': gate_map[gate_name],
                'qubits': wires,
                'trainable': trainable,
                'param_idx': param_counter if trainable else None
            }
            
            if trainable:
                param_counter += 1
            
            gate_sequence.append(gate_info)
        
        return gate_sequence
    
    def set_training_data(self, training_data: Dict):
        """Set training data from external source for energy estimation"""
        self.cached_training_data = training_data
        self.cached_prepared_inputs = self._prepare_training_inputs()
    
    def _prepare_training_inputs(self) -> List[Dict]:
        """Prepare training inputs for energy estimation"""
        if self.cached_training_data is None:
            return None
        
        prepared = []
        all_points = []
        
        for data_type in ['initial_points', 'boundary_points', 'interior_points']:
            if data_type in self.cached_training_data:
                all_points.extend(self.cached_training_data[data_type])
        
        for point in all_points:
            coords = np.array([point.x / L, point.y / L, point.z / L, point.t / T])
            true_val = getattr(point, 'u', 0.0)
            
            prepared.append({
                'coordinates': coords,
                'true_value': true_val
            })
        
        return prepared
    
    def generate_circuit(
        self,
        use_gpt: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> QuantumCircuitTemplate:
        """
        Generate a quantum circuit template
        
        Args:
            use_gpt: Whether to use GPT for generation
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated circuit template
        """
        if use_gpt and self.gpt_model is not None:
            template = self._generate_with_gpt(temperature, top_k, top_p)
        else:
            template = self._generate_hardware_efficient()
        
        # Evaluate and store
        if self.energy_estimator is not None:
            energy = self._estimate_circuit_energy(template)
            self.energy_history.append(energy)
        
        self.circuit_history.append({
            'template': template,
            'timestamp': time.time()
        })
        
        return template
    
    def _generate_with_gpt(
        self,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> QuantumCircuitTemplate:
        """Generate circuit using GPT model"""
        self.gpt_model.eval()
        
        # Start with START token
        start_tokens = torch.tensor(
            [[self.tokenizer.start_token_id]],
            dtype=torch.long,
            device=device
        )
        
        # Generate sequence
        with torch.no_grad():
            generated = self.gpt_model.generate(
                start_tokens,
                max_new_tokens=self.max_circuit_depth * 2,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                end_token_id=self.tokenizer.end_token_id
            )
        
        # Decode to gate sequence
        token_ids = generated[0].tolist()
        gate_sequence, parameter_map = self.tokenizer.decode(token_ids)
        
        # Validate and fix
        gate_sequence = self._validate_gate_sequence(gate_sequence)
        
        if not gate_sequence:
            return self._generate_hardware_efficient()
        
        return QuantumCircuitTemplate(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            gate_sequence=gate_sequence,
            parameter_map=parameter_map
        )
    
    def _generate_hardware_efficient(self) -> QuantumCircuitTemplate:
        """Generate hardware-efficient ansatz circuit"""
        gate_sequence = []
        parameter_map = {}
        param_counter = 0
        
        n_layers = min(self.n_layers, self.max_circuit_depth // (self.n_qubits + 1))
        
        for layer in range(n_layers):
            # RY rotation layer
            for q in range(self.n_qubits):
                gate_sequence.append({
                    'gate': 'RY',
                    'qubits': [q],
                    'param_idx': param_counter,
                    'trainable': True
                })
                parameter_map[f'ry_l{layer}_q{q}'] = param_counter
                param_counter += 1
            
            # RZ rotation layer
            for q in range(self.n_qubits):
                gate_sequence.append({
                    'gate': 'RZ',
                    'qubits': [q],
                    'param_idx': param_counter,
                    'trainable': True
                })
                parameter_map[f'rz_l{layer}_q{q}'] = param_counter
                param_counter += 1
            
            # CNOT entangling layer (linear topology)
            if layer < n_layers - 1:
                for q in range(self.n_qubits - 1):
                    gate_sequence.append({
                        'gate': 'CNOT',
                        'qubits': [q, q + 1],
                        'param_idx': None,
                        'trainable': False
                    })
                
                # Ring connection
                if self.n_qubits > 2:
                    gate_sequence.append({
                        'gate': 'CNOT',
                        'qubits': [self.n_qubits - 1, 0],
                        'param_idx': None,
                        'trainable': False
                    })
        
        return QuantumCircuitTemplate(
            n_qubits=self.n_qubits,
            n_layers=n_layers,
            gate_sequence=gate_sequence,
            parameter_map=parameter_map
        )
    
    def _validate_gate_sequence(self, gate_sequence: List[Dict]) -> List[Dict]:
        """Validate and fix gate sequence"""
        valid_sequence = []
        
        for gate_info in gate_sequence:
            # Check qubit indices
            qubits = gate_info.get('qubits', [])
            if any(q < 0 or q >= self.n_qubits for q in qubits):
                continue
            
            # Check for duplicate qubits in two-qubit gates
            if len(qubits) >= 2 and qubits[0] == qubits[1]:
                continue
            
            # Validate gate type
            gate_type = gate_info.get('gate', '')
            valid_single = ['RX', 'RY', 'RZ', 'H', 'S', 'T', 'X', 'Y', 'Z']
            valid_two = ['CNOT', 'CZ', 'SWAP']
            
            if len(qubits) == 1 and gate_type not in valid_single:
                continue
            if len(qubits) == 2 and gate_type not in valid_two:
                continue
            
            valid_sequence.append(gate_info)
        
        return valid_sequence
    
    def _estimate_circuit_energy(self, template: QuantumCircuitTemplate) -> float:
        """Estimate circuit energy using unsupervised method"""
        if self.energy_estimator is None:
            return -1.0 + 0.01 * len(template.parameter_map)
        
        try:
            # Use cached training data if available
            if self.cached_prepared_inputs:
                sample_idx = np.random.randint(len(self.cached_prepared_inputs))
                point_data = self.cached_prepared_inputs[sample_idx]
                input_data = self._create_quantum_state(point_data)
            else:
                input_dim = 2**self.n_qubits
                input_data = np.random.randn(input_dim)
                input_data = input_data / (np.linalg.norm(input_data) + 1e-10)
            
            energy = self.energy_estimator.estimate_energy_unsupervised(template, input_data)
            return float(energy)
            
        except Exception as e:
            return -1.0 + 0.01 * len(template.parameter_map)
    
    def _create_quantum_state(self, point_data: Dict) -> np.ndarray:
        """Create quantum state from training point data"""
        coords = point_data['coordinates']
        true_val = point_data['true_value']
        
        state_dim = 2**self.n_qubits
        amplitudes = np.zeros(state_dim, dtype=complex)
        
        qubit_regions = np.linspace(0, 1, self.n_qubits + 1)
        
        for basis_idx in range(state_dim):
            basis_binary = format(basis_idx, f'0{self.n_qubits}b')
            amplitude = 1.0
            phase = 0.0
            
            for qubit_idx, bit in enumerate(basis_binary):
                region_center = (qubit_regions[qubit_idx] + qubit_regions[qubit_idx + 1]) / 2
                
                if bit == '1':
                    for dim_idx, coord in enumerate(coords[:3]):
                        distance = abs(coord - region_center)
                        amplitude *= np.exp(-distance**2 / 0.02)
                        phase += coord * np.pi * (qubit_idx + 1) / self.n_qubits
                else:
                    for dim_idx, coord in enumerate(coords[:3]):
                        distance = abs(coord - region_center)
                        amplitude *= (1 - 0.5 * np.exp(-distance**2 / 0.08))
            
            time_factor = coords[3]
            amplitude *= np.exp(-time_factor * basis_idx / (2 * state_dim))
            phase += 2 * np.pi * true_val * basis_idx / state_dim
            amplitudes[basis_idx] = amplitude * np.exp(1j * phase)
        
        # Normalize
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-10:
            amplitudes = amplitudes / norm
        else:
            amplitudes = np.ones(state_dim) / np.sqrt(state_dim)
        
        return np.abs(amplitudes).astype(np.float64)
    
    def optimize_circuit(
        self,
        n_candidates: int = 20,
        n_iterations: int = 5,
        selection_method: str = 'pareto'
    ) -> QuantumCircuitTemplate:
        """
        Optimize circuit through iterative generation and selection
        
        Args:
            n_candidates: Number of candidate circuits per iteration
            n_iterations: Number of optimization iterations
            selection_method: 'pareto' or 'weighted'
            
        Returns:
            Best circuit template
        """
        best_template = None
        best_score = -float('inf')
        
        all_candidates = []
        
        for iteration in range(n_iterations):
            candidates = []
            
            # Generate candidates
            for _ in range(n_candidates):
                # Decay exploration rate
                use_gpt = np.random.random() < self.exploration_rate
                temperature = 0.5 + 0.5 * self.exploration_rate
                
                template = self.generate_circuit(
                    use_gpt=use_gpt,
                    temperature=temperature
                )
                
                # Evaluate quality
                quality_scores = self.quality_evaluator.evaluate(template)
                
                # Estimate energy
                energy = self._estimate_circuit_energy(template)
                
                candidates.append({
                    'template': template,
                    'quality': quality_scores,
                    'energy': energy,
                    'score': np.mean(list(quality_scores.values()))
                })
            
            all_candidates.extend(candidates)
            
            # Select best candidates
            if selection_method == 'pareto':
                selected = self._pareto_selection(candidates)
            else:
                selected = self._weighted_selection(candidates)
            
            # Update best
            for candidate in selected:
                if candidate['score'] > best_score:
                    best_score = candidate['score']
                    best_template = candidate['template']
            
            # Train GPT on good candidates
            training_data = []
            for candidate in selected[:5]:
                training_data.append({
                    'gate_sequence': candidate['template'].gate_sequence,
                    'energy': candidate['energy'],
                    'score': candidate['score']
                })
            
            if training_data:
                self._train_gpt_on_circuits(training_data, epochs=10)
            
            # Decay exploration
            self.exploration_rate *= self.exploration_decay
            
            print(f"Iteration {iteration + 1}/{n_iterations}: "
                  f"Best score = {best_score:.4f}, "
                  f"Exploration rate = {self.exploration_rate:.3f}")
        
        # Record generation history
        self.generation_history.append({
            'n_candidates': len(all_candidates),
            'best_score': best_score,
            'timestamp': time.time()
        })
        
        return best_template if best_template else self._generate_hardware_efficient()
    
    def _pareto_selection(
        self,
        candidates: List[Dict],
        n_select: int = 10
    ) -> List[Dict]:
        """Select candidates using Pareto dominance"""
        n = len(candidates)
        if n == 0:
            return []
        
        # Extract objective values
        objectives = []
        for c in candidates:
            obj_values = list(c['quality'].values())
            objectives.append(obj_values)
        
        objectives = np.array(objectives)
        
        # Find Pareto front
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # j dominates i if all objectives are >= and at least one is >
                    if (np.all(objectives[j] >= objectives[i]) and 
                        np.any(objectives[j] > objectives[i])):
                        is_dominated[i] = True
                        break
        
        # Select non-dominated solutions
        pareto_indices = np.where(~is_dominated)[0]
        selected = [candidates[i] for i in pareto_indices]
        
        # If not enough, add by score
        if len(selected) < n_select:
            remaining = [c for i, c in enumerate(candidates) if is_dominated[i]]
            remaining.sort(key=lambda x: x['score'], reverse=True)
            selected.extend(remaining[:n_select - len(selected)])
        
        return selected[:n_select]
    
    def _weighted_selection(
        self,
        candidates: List[Dict],
        n_select: int = 10
    ) -> List[Dict]:
        """Select candidates using weighted scoring"""
        # Sort by score
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates[:n_select]
    
    def _train_gpt_on_circuits(self, training_data: List[Dict], epochs: int = 10):
        """Train GPT model on circuit data"""
        if self.gpt_model is None or not training_data:
            return
        
        # Prepare sequences
        sequences = []
        energies = []
        
        for data in training_data:
            tokens = self.tokenizer.encode(data['gate_sequence'])
            sequences.append(tokens)
            energies.append(data.get('energy', -1.0))
        
        # Normalize energies
        energies = np.array(energies)
        if energies.std() > 1e-6:
            energies = (energies - energies.mean()) / energies.std()
        energies = energies.tolist()
        
        dataset = QuantumCircuitDataset(sequences, energies)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=min(16, len(dataset)), shuffle=True
        )
        
        self.gpt_model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for seq_batch, energy_batch, _ in dataloader:
                seq_batch = seq_batch.to(device)
                energy_batch = energy_batch.to(device)
                
                self.gpt_optimizer.zero_grad()
                logits, loss, _ = self.gpt_model(
                    seq_batch,
                    targets=seq_batch,
                    energies=energy_batch
                )
                
                if loss is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.gpt_model.parameters(), 1.0)
                    self.gpt_optimizer.step()
                    total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(1, len(dataloader))
                print(f"  GPT Training Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    
    def update_circuit(
        self,
        current_template: QuantumCircuitTemplate,
        loss_history: List[float],
        update_threshold: float = 0.1
    ) -> Tuple[QuantumCircuitTemplate, bool]:
        """
        Dynamically update circuit based on training progress
        
        Args:
            current_template: Current circuit template
            loss_history: Recent loss values
            update_threshold: Threshold for triggering update
            
        Returns:
            (new_template, was_updated)
        """
        if len(loss_history) < 10:
            return current_template, False
        
        # Check if loss is stagnating
        recent_loss = np.mean(loss_history[-5:])
        older_loss = np.mean(loss_history[-10:-5])
        
        improvement = (older_loss - recent_loss) / (abs(older_loss) + 1e-8)
        
        if improvement < update_threshold:
            # Generate new candidate
            new_template = self.generate_circuit(use_gpt=True, temperature=0.7)
            
            # Evaluate
            new_quality = self.quality_evaluator.evaluate(new_template)
            current_quality = self.quality_evaluator.evaluate(current_template)
            
            new_score = np.mean(list(new_quality.values()))
            current_score = np.mean(list(current_quality.values()))
            
            if new_score > current_score:
                print(f"Circuit updated: score {current_score:.4f} -> {new_score:.4f}")
                return new_template, True
        
        return current_template, False
    
    def save_state(self, save_path: str = 'gqe_state'):
        """Save generator state"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save GPT model
        if self.gpt_model is not None:
            model_path = os.path.join(save_path, 'gpt_model.pth')
            torch.save({
                'model_state_dict': self.gpt_model.state_dict(),
                'optimizer_state_dict': self.gpt_optimizer.state_dict(),
                'vocab_size': self.tokenizer.vocab_size
            }, model_path)
        
        # Save history
        history_path = os.path.join(save_path, 'history.json')
        history = {
            'energy_history': self.energy_history,
            'generation_history': self.generation_history,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'exploration_rate': self.exploration_rate
        }
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"GQE state saved to {save_path}")
    
    def load_state(self, load_path: str = 'gqe_state'):
        """Load generator state"""
        # Load GPT model
        model_path = os.path.join(load_path, 'gpt_model.pth')
        if os.path.exists(model_path) and self.gpt_model is not None:
            checkpoint = torch.load(model_path, map_location=device)
            self.gpt_model.load_state_dict(checkpoint['model_state_dict'])
            self.gpt_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("GPT model loaded")
        
        # Load history
        history_path = os.path.join(load_path, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            self.energy_history = history.get('energy_history', [])
            self.generation_history = history.get('generation_history', [])
            self.exploration_rate = history.get('exploration_rate', 0.9)
            print("History loaded")


def create_gqe_generator(
    n_qubits: int = 6,
    n_layers: int = 4,
    use_noise: bool = False,
    pretrained_path: Optional[str] = None
) -> GQEQuantumCircuitGenerator:
    """
    Factory function to create GQE generator
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of circuit layers
        use_noise: Whether to model noise
        pretrained_path: Path to pretrained GPT weights
        
    Returns:
        Configured GQE generator
    """
    generator = GQEQuantumCircuitGenerator(
        n_qubits=n_qubits,
        n_layers=n_layers,
        noise_budget=0.01 if use_noise else 0.0,
        use_energy_prediction=True
    )
    
    if pretrained_path:
        generator.load_state(pretrained_path)
    
    return generator
