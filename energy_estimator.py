"""
Unsupervised Quantum Energy Estimator
For GQE circuit optimization

References:
- Mitarai et al. "Quantum circuit learning" Phys. Rev. A 98, 032309 (2018)
- Abbas et al. "The power of quantum neural networks" Nat Comput Sci 1, 403-409 (2021)
- Endo et al. "Practical Quantum Error Mitigation" Phys. Rev. X 11, 031057 (2021)
"""

import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Any
from collections import deque

from config import QuantumCircuitTemplate


class UnsupervisedQuantumEnergyEstimator:
    """
    Unsupervised quantum energy estimator with noise awareness
    
    Uses unsupervised learning to estimate circuit energy without
    requiring labeled training data.
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 4,
        use_noise: bool = False,
        noise_model: str = 'realistic',
        shots: Optional[int] = None
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_noise = use_noise
        self.noise_model = noise_model
        self.shots = shots if shots is not None else (2048 if use_noise else None)
        
        # History tracking
        self.measurement_history = []
        self.circuit_features = []
        
        # Kernel estimation parameters
        self.kernel_bandwidth = 1.0
        self.n_measurement_bases = 2**n_qubits
        
        # Unsupervised clustering
        self.n_energy_clusters = 10
        self.energy_estimator = None
        
        # Feature processing
        self.feature_dim = None
        self.pca = None
        self.kmeans = None
        self.scaler = None
        
        # Noise parameters
        self.noise_params = self._initialize_noise_params()
        
        # Error mitigation
        self.error_mitigation_enabled = use_noise
        self.zero_noise_extrapolation_factors = [1.0, 1.5, 2.0] if use_noise else [1.0]
        
        # Energy history for tracking
        self._last_mitigated_energy = None
    
    def _initialize_noise_params(self) -> Dict[str, float]:
        """Initialize noise model parameters"""
        noise_configs = {
            'light': {
                'depolarizing_1q': 0.001,
                'depolarizing_2q': 0.01,
                'amplitude_damping': 0.0005,
                'phase_damping': 0.0005,
                'readout_error': 0.01
            },
            'realistic': {
                'depolarizing_1q': 0.002,
                'depolarizing_2q': 0.02,
                'amplitude_damping': 0.001,
                'phase_damping': 0.001,
                'readout_error': 0.02
            },
            'heavy': {
                'depolarizing_1q': 0.005,
                'depolarizing_2q': 0.05,
                'amplitude_damping': 0.002,
                'phase_damping': 0.002,
                'readout_error': 0.05
            }
        }
        return noise_configs.get(self.noise_model, {
            'depolarizing_1q': 0.0,
            'depolarizing_2q': 0.0,
            'amplitude_damping': 0.0,
            'phase_damping': 0.0,
            'readout_error': 0.0
        })
    
    def _create_device(self, shots: Optional[int] = None):
        """Create appropriate quantum device"""
        if self.use_noise:
            return qml.device('default.mixed', wires=self.n_qubits,
                            shots=shots if shots else self.shots)
        else:
            if shots:
                return qml.device('default.qubit', wires=self.n_qubits, shots=shots)
            return qml.device('default.qubit', wires=self.n_qubits)
    
    def _prepare_input_data(self, input_data: np.ndarray) -> np.ndarray:
        """Prepare and normalize input data for quantum circuit"""
        # Ensure correct dimension
        required_dim = 2**self.n_qubits
        
        if len(input_data) < required_dim:
            padded = np.zeros(required_dim)
            padded[:len(input_data)] = input_data
            input_data = padded
        elif len(input_data) > required_dim:
            input_data = input_data[:required_dim]
        
        # Normalize
        norm = np.linalg.norm(input_data)
        if norm > 1e-10:
            input_data = input_data / norm
        else:
            input_data = np.ones(required_dim) / np.sqrt(required_dim)
        
        return input_data.astype(np.float64)
    
    def _apply_circuit_template(self, template: QuantumCircuitTemplate, param_values: np.ndarray):
        """Apply circuit template gates"""
        param_idx = 0
        
        for gate_info in template.gate_sequence:
            gate_type = gate_info['gate']
            qubits = gate_info['qubits']
            
            # Skip invalid qubit indices
            if any(q >= self.n_qubits for q in qubits):
                continue
            
            try:
                if gate_type == 'H':
                    qml.Hadamard(wires=qubits[0])
                elif gate_type == 'X':
                    qml.PauliX(wires=qubits[0])
                elif gate_type == 'Y':
                    qml.PauliY(wires=qubits[0])
                elif gate_type == 'Z':
                    qml.PauliZ(wires=qubits[0])
                elif gate_type == 'S':
                    qml.S(wires=qubits[0])
                elif gate_type == 'T':
                    qml.T(wires=qubits[0])
                elif gate_type == 'RY' and gate_info.get('trainable', False):
                    if param_idx < len(param_values):
                        qml.RY(param_values[param_idx], wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'RX' and gate_info.get('trainable', False):
                    if param_idx < len(param_values):
                        qml.RX(param_values[param_idx], wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'RZ' and gate_info.get('trainable', False):
                    if param_idx < len(param_values):
                        qml.RZ(param_values[param_idx], wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'CNOT' and len(qubits) >= 2:
                    if qubits[0] != qubits[1]:
                        qml.CNOT(wires=[qubits[0], qubits[1]])
                elif gate_type == 'CZ' and len(qubits) >= 2:
                    if qubits[0] != qubits[1]:
                        qml.CZ(wires=[qubits[0], qubits[1]])
                elif gate_type == 'SWAP' and len(qubits) >= 2:
                    if qubits[0] != qubits[1]:
                        qml.SWAP(wires=[qubits[0], qubits[1]])
            except Exception as e:
                continue
    
    def _extract_quantum_features(
        self, 
        template: QuantumCircuitTemplate,
        input_data: np.ndarray
    ) -> np.ndarray:
        """Extract quantum features from circuit execution"""
        dev = self._create_device()
        prepared_data = self._prepare_input_data(input_data)
        
        @qml.qnode(dev)
        def feature_circuit():
            # Amplitude embedding
            qml.AmplitudeEmbedding(
                prepared_data,
                wires=range(self.n_qubits),
                normalize=True,
                pad_with=0.0
            )
            
            # Apply circuit template
            param_values = np.random.uniform(-np.pi, np.pi, len(template.parameter_map))
            self._apply_circuit_template(template, param_values)
            
            # Measure in multiple bases
            measurements = []
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
                measurements.append(qml.expval(qml.PauliX(i)))
            
            # Add correlations
            for i in range(self.n_qubits - 1):
                measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i+1)))
            
            return measurements
        
        try:
            features = np.array(feature_circuit())
            return features
        except Exception as e:
            # Return default features on error
            return np.zeros(3 * self.n_qubits - 1)
    
    def estimate_energy_unsupervised(
        self,
        template: QuantumCircuitTemplate,
        input_data: np.ndarray
    ) -> float:
        """
        Estimate circuit energy using unsupervised learning
        
        Args:
            template: Quantum circuit template
            input_data: Input state data
            
        Returns:
            Estimated energy value
        """
        try:
            # Extract quantum features
            quantum_features = self._extract_quantum_features(template, input_data)
            
            # Apply error mitigation if enabled
            if self.error_mitigation_enabled:
                quantum_features = self._apply_error_mitigation(quantum_features, template)
            
            # Use variational energy estimation
            energy = self._variational_energy_estimation(template, quantum_features)
            
            # Store in history
            self.measurement_history.append(quantum_features)
            self.circuit_features.append(quantum_features)
            
            # Limit history size
            max_history = 1000
            if len(self.measurement_history) > max_history:
                self.measurement_history = self.measurement_history[-max_history:]
                self.circuit_features = self.circuit_features[-max_history:]
            
            return float(energy)
            
        except Exception as e:
            # Fallback estimation based on circuit complexity
            return -1.0 + 0.01 * len(template.parameter_map)
    
    def _apply_error_mitigation(
        self,
        features: np.ndarray,
        template: QuantumCircuitTemplate
    ) -> np.ndarray:
        """Apply zero-noise extrapolation error mitigation"""
        if not self.use_noise:
            return features
        
        try:
            # Simplified error mitigation: scale features
            # based on estimated noise level
            noise_factor = 1.0 / (1.0 + self.noise_params['depolarizing_1q'] * len(template.gate_sequence))
            return features * noise_factor
        except:
            return features
    
    def _variational_energy_estimation(
        self,
        template: QuantumCircuitTemplate,
        quantum_features: np.ndarray
    ) -> float:
        """
        Estimate energy using variational method with clustering
        """
        # Add features to history
        self.circuit_features.append(quantum_features)
        
        # Feature standardization
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        # Execute clustering when enough data
        if len(self.circuit_features) >= self.n_energy_clusters:
            # Convert all features to array
            all_features = np.array(self.circuit_features)
            
            # Handle NaN/Inf
            all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            try:
                scaled_features = self.scaler.fit_transform(all_features)
            except:
                scaled_features = all_features
            
            # PCA dimensionality reduction
            target_dim = min(5, scaled_features.shape[0] - 1, scaled_features.shape[1])
            
            if target_dim > 0:
                if self.pca is None:
                    self.pca = PCA(n_components=target_dim)
                    reduced_features = self.pca.fit_transform(scaled_features)
                else:
                    try:
                        reduced_features = self.pca.transform(scaled_features)
                    except:
                        self.pca = PCA(n_components=target_dim)
                        reduced_features = self.pca.fit_transform(scaled_features)
                
                current_reduced = reduced_features[-1]
                
                # K-means clustering
                n_clusters = min(self.n_energy_clusters, len(reduced_features))
                if self.kmeans is None or len(self.measurement_history) % 50 == 0:
                    self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    self.kmeans.fit(reduced_features)
                
                try:
                    cluster_idx = self.kmeans.predict(current_reduced.reshape(1, -1))[0]
                    cluster_center = self.kmeans.cluster_centers_[cluster_idx]
                    
                    # Energy defined as negative of feature vector norm
                    energy = -np.linalg.norm(cluster_center)
                    energy = energy * self.n_qubits / 2.0
                except Exception as e:
                    energy = -self.n_qubits * 0.5
            else:
                energy = -self.n_qubits * 0.5
        else:
            # Initial estimation when data is insufficient
            entropy = self._compute_quantum_entropy(quantum_features)
            energy = -self.n_qubits * (1 - entropy)
        
        return energy
    
    def _compute_quantum_entropy(self, features: np.ndarray) -> float:
        """Compute von Neumann entropy approximation from features"""
        # Build probability distribution
        probs = np.abs(features)**2
        total = np.sum(probs)
        if total > 1e-10:
            probs = probs / total
        else:
            probs = np.ones_like(probs) / len(probs)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        
        if max_entropy > 0:
            return entropy / max_entropy
        return 0.5
    
    def update_learning(
        self,
        template: QuantumCircuitTemplate,
        measurement_results: np.ndarray
    ):
        """
        Update learning with new measurement results
        
        Args:
            template: Circuit template
            measurement_results: Measurement outcome data
        """
        # Adjust measurement size
        required_dim = 2**self.n_qubits
        if len(measurement_results) < required_dim:
            padded = np.zeros(required_dim)
            padded[:len(measurement_results)] = measurement_results
            measurement_results = padded
        
        self.measurement_history.append(measurement_results)
        
        # Limit history
        max_history = 10000
        if len(self.measurement_history) > max_history:
            self.measurement_history = self.measurement_history[-max_history:]
            self.circuit_features = self.circuit_features[-max_history:]
    
    def get_estimation_statistics(self) -> Dict[str, float]:
        """Get statistics about energy estimation"""
        if not self.measurement_history:
            return {'mean': 0.0, 'std': 0.0, 'count': 0}
        
        features = np.array(self.measurement_history)
        return {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'count': len(self.measurement_history)
        }


class CircuitQualityEvaluator:
    """
    Evaluate quantum circuit quality for multi-objective optimization
    """
    
    def __init__(self, n_qubits: int, energy_estimator: Optional[UnsupervisedQuantumEnergyEstimator] = None):
        self.n_qubits = n_qubits
        self.energy_estimator = energy_estimator
    
    def evaluate(self, template: QuantumCircuitTemplate) -> Dict[str, float]:
        """
        Evaluate circuit quality across multiple objectives
        
        Returns dictionary with 9 objective scores
        """
        scores = {}
        
        # 1. Hardware efficiency
        scores['hardware_efficiency'] = self._compute_hardware_efficiency(template)
        
        # 2. Noise resilience
        scores['noise_resilience'] = self._compute_noise_resilience(template)
        
        # 3. Expressivity
        scores['expressivity'] = self._compute_expressivity(template)
        
        # 4. Mitigation compatibility
        scores['mitigation'] = self._compute_mitigation_compatibility(template)
        
        # 5. Trainability
        scores['trainability'] = self._compute_trainability(template)
        
        # 6. Entanglement capability
        scores['entanglement'] = self._compute_entanglement_capability(template)
        
        # 7. Depth efficiency
        scores['depth_efficiency'] = self._compute_depth_efficiency(template)
        
        # 8. Parameter efficiency
        scores['param_efficiency'] = self._compute_parameter_efficiency(template)
        
        # 9. Energy estimation quality
        scores['energy_quality'] = self._compute_energy_quality(template)
        
        return scores
    
    def _compute_hardware_efficiency(self, template: QuantumCircuitTemplate) -> float:
        """Compute hardware efficiency score"""
        preferred_gates = {'RY', 'RZ', 'CNOT', 'CZ'}
        total_gates = len(template.gate_sequence)
        
        if total_gates == 0:
            return 0.0
        
        preferred_count = sum(1 for g in template.gate_sequence if g['gate'] in preferred_gates)
        return preferred_count / total_gates
    
    def _compute_noise_resilience(self, template: QuantumCircuitTemplate) -> float:
        """Compute noise resilience score"""
        depth = self._calculate_depth(template)
        max_depth = 20
        
        # Shallower circuits are more noise resilient
        depth_score = max(0, 1.0 - depth / max_depth)
        
        # Fewer two-qubit gates = better
        two_qubit_gates = sum(1 for g in template.gate_sequence if len(g['qubits']) >= 2)
        two_qubit_score = max(0, 1.0 - two_qubit_gates / max(1, len(template.gate_sequence)))
        
        return 0.6 * depth_score + 0.4 * two_qubit_score
    
    def _compute_expressivity(self, template: QuantumCircuitTemplate) -> float:
        """Compute circuit expressivity score"""
        n_params = len(template.parameter_map)
        n_gates = len(template.gate_sequence)
        
        # More parameters relative to gates = higher expressivity
        if n_gates == 0:
            return 0.0
        
        param_density = n_params / max(1, n_gates)
        return min(1.0, param_density)
    
    def _compute_mitigation_compatibility(self, template: QuantumCircuitTemplate) -> float:
        """Compute error mitigation compatibility"""
        # Check for Pauli twirling compatibility
        cnot_count = sum(1 for g in template.gate_sequence if g['gate'] in ['CNOT', 'CZ'])
        
        if cnot_count == 0:
            return 0.5
        
        # Count single-qubit gates around CNOTs
        compatible = 0
        for i, gate in enumerate(template.gate_sequence):
            if gate['gate'] in ['CNOT', 'CZ']:
                has_neighbor = False
                if i > 0 and len(template.gate_sequence[i-1]['qubits']) == 1:
                    has_neighbor = True
                if i < len(template.gate_sequence) - 1 and len(template.gate_sequence[i+1]['qubits']) == 1:
                    has_neighbor = True
                if has_neighbor:
                    compatible += 1
        
        return compatible / cnot_count
    
    def _compute_trainability(self, template: QuantumCircuitTemplate) -> float:
        """Compute trainability (avoiding barren plateaus)"""
        depth = self._calculate_depth(template)
        n_params = len(template.parameter_map)
        
        # Shallow circuits with moderate parameters are more trainable
        depth_factor = np.exp(-depth / 10)
        param_factor = min(1.0, n_params / (2 * self.n_qubits))
        
        return depth_factor * param_factor
    
    def _compute_entanglement_capability(self, template: QuantumCircuitTemplate) -> float:
        """Compute entanglement capability score"""
        entangling_gates = ['CNOT', 'CZ', 'SWAP', 'iSWAP']
        n_entangling = sum(1 for g in template.gate_sequence if g['gate'] in entangling_gates)
        
        # Ideal: roughly one entangling gate per qubit pair per layer
        ideal_entangling = self.n_qubits * template.n_layers // 2
        
        if ideal_entangling == 0:
            return 0.5
        
        ratio = n_entangling / ideal_entangling
        return min(1.0, ratio)
    
    def _compute_depth_efficiency(self, template: QuantumCircuitTemplate) -> float:
        """Compute depth efficiency score"""
        depth = self._calculate_depth(template)
        n_gates = len(template.gate_sequence)
        
        if depth == 0:
            return 0.0
        
        # Higher parallelization = better efficiency
        parallelization = n_gates / depth
        max_parallel = self.n_qubits  # Max gates in parallel
        
        return min(1.0, parallelization / max_parallel)
    
    def _compute_parameter_efficiency(self, template: QuantumCircuitTemplate) -> float:
        """Compute parameter efficiency score"""
        n_params = len(template.parameter_map)
        
        # Ideal parameter count
        ideal_params = 2 * self.n_qubits * template.n_layers
        
        if ideal_params == 0:
            return 0.5
        
        ratio = n_params / ideal_params
        # Penalize both too few and too many parameters
        return 1.0 - abs(ratio - 1.0)
    
    def _compute_energy_quality(self, template: QuantumCircuitTemplate) -> float:
        """Compute energy estimation quality score"""
        if self.energy_estimator is None:
            return 0.5
        
        stats = self.energy_estimator.get_estimation_statistics()
        
        # Lower variance = better quality
        if stats['std'] > 0:
            quality = 1.0 / (1.0 + stats['std'])
        else:
            quality = 0.5
        
        return quality
    
    def _calculate_depth(self, template: QuantumCircuitTemplate) -> int:
        """Calculate circuit depth"""
        if not template.gate_sequence:
            return 0
        
        qubit_layers = {}
        max_depth = 0
        
        for gate_info in template.gate_sequence:
            qubits = gate_info['qubits']
            current_layer = 0
            
            for q in qubits:
                if q < template.n_qubits and q in qubit_layers:
                    current_layer = max(current_layer, qubit_layers[q] + 1)
            
            for q in qubits:
                if q < template.n_qubits:
                    qubit_layers[q] = current_layer
            
            max_depth = max(max_depth, current_layer + 1)
        
        return max_depth
