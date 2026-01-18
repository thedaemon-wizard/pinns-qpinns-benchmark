"""
GPT Model for Quantum Circuit Generation
Based on KetGPT methodology

References:
- Apak et al. "KetGPT - Dataset Augmentation of Quantum Circuits 
  using Transformers" arXiv:2402.13352 (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Config
import numpy as np
from typing import List, Dict, Optional, Tuple
import os

from config import device


class QuantumCircuitGPT(nn.Module):
    """
    GPT model for quantum circuit generation
    
    Generates quantum circuit token sequences and predicts
    expected energy values for circuit optimization.
    """
    
    def __init__(
        self, 
        vocab_size: int,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        block_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # GPT-2 configuration
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_ctx=block_size,
            n_positions=block_size,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False
        )
        
        # GPT-2 transformer backbone
        self.transformer = GPT2Model(self.config)
        
        # Language modeling head (next token prediction)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Energy prediction head (predicts expected circuit energy)
        self.energy_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(n_embd // 2, 1)
        )
        
        # Circuit quality prediction head
        self.quality_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd // 2, 9)  # 9 objective scores
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with Xavier uniform"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
        qualities: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass
        
        Args:
            idx: Input token indices [batch, seq_len]
            targets: Target token indices for loss computation
            energies: Target energy values
            qualities: Target quality scores (9 objectives)
            
        Returns:
            logits: Next token logits
            loss: Combined loss (if targets provided)
            energy_pred: Predicted energy values
        """
        # Transformer processing
        transformer_outputs = self.transformer(idx)
        hidden_states = transformer_outputs.last_hidden_state
        
        # Language modeling output
        logits = self.lm_head(hidden_states)
        
        # Energy prediction (from last token hidden state)
        energy_pred = self.energy_head(hidden_states[:, -1, :])
        
        # Quality prediction
        quality_pred = self.quality_head(hidden_states[:, -1, :])
        
        loss = None
        if targets is not None:
            # Cross entropy loss (next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0  # Ignore padding token
            )
            
            loss = loss_ce
            
            # Energy prediction loss
            if energies is not None:
                loss_energy = F.mse_loss(energy_pred.squeeze(-1), energies)
                loss = loss + 0.1 * loss_energy
            
            # Quality prediction loss
            if qualities is not None:
                loss_quality = F.mse_loss(quality_pred, qualities)
                loss = loss + 0.05 * loss_quality
        
        return logits, loss, energy_pred
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        end_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate quantum circuit token sequences
        
        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            end_token_id: Token ID to stop generation
            
        Returns:
            Generated token sequence
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to block size if needed
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            
            # Get predictions
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find positions exceeding top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop if end token generated
            if end_token_id is not None and idx_next.item() == end_token_id:
                break
        
        return idx
    
    def predict_circuit_quality(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Predict quality scores for a circuit
        
        Args:
            idx: Token indices representing the circuit
            
        Returns:
            Quality scores for 9 objectives
        """
        self.eval()
        with torch.no_grad():
            transformer_outputs = self.transformer(idx)
            hidden_states = transformer_outputs.last_hidden_state
            quality_pred = self.quality_head(hidden_states[:, -1, :])
        return quality_pred


class QuantumCircuitDataset(Dataset):
    """
    Dataset for training the quantum circuit GPT model
    """
    
    def __init__(
        self,
        sequences: List[List[int]],
        energies: List[float],
        qualities: Optional[List[List[float]]] = None,
        block_size: int = 128
    ):
        self.sequences = sequences
        self.energies = energies
        self.qualities = qualities
        self.block_size = block_size
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        energy = self.energies[idx]
        
        # Padding or truncation
        if len(seq) > self.block_size:
            seq = seq[:self.block_size]
        else:
            seq = seq + [0] * (self.block_size - len(seq))  # Pad with 0
        
        # Quality scores
        if self.qualities is not None:
            quality = self.qualities[idx]
        else:
            quality = [0.5] * 9  # Default quality
        
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(energy, dtype=torch.float32),
            torch.tensor(quality, dtype=torch.float32)
        )


class CircuitTokenizer:
    """
    Tokenizer for quantum circuits
    Converts circuit gate sequences to token IDs and vice versa
    """
    
    def __init__(self, n_qubits: int = 6):
        self.n_qubits = n_qubits
        self.gate_tokens = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build the gate vocabulary"""
        # Special tokens
        self.gate_tokens = ['[PAD]', '[START]', '[END]', '[SEP]']
        
        # Single-qubit gates
        for gate in ['RX', 'RY', 'RZ', 'H', 'S', 'T', 'X', 'Y', 'Z']:
            for q in range(self.n_qubits):
                self.gate_tokens.append(f'{gate}_{q}')
        
        # Two-qubit gates
        for gate in ['CNOT', 'CZ', 'SWAP']:
            for q1 in range(self.n_qubits):
                for q2 in range(self.n_qubits):
                    if q1 != q2:
                        self.gate_tokens.append(f'{gate}_{q1}_{q2}')
        
        # Parameter value tokens (discretized angles)
        for i in range(16):
            self.gate_tokens.append(f'PARAM_{i}')
        
        # Build mappings
        self.token_to_id = {token: i for i, token in enumerate(self.gate_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.gate_tokens)}
        self.vocab_size = len(self.gate_tokens)
    
    def encode(self, gate_sequence: List[Dict]) -> List[int]:
        """
        Convert gate sequence to token IDs
        
        Args:
            gate_sequence: List of gate dictionaries
            
        Returns:
            List of token IDs
        """
        tokens = [self.token_to_id['[START]']]
        
        for gate_info in gate_sequence:
            try:
                gate_type = gate_info['gate']
                qubits = gate_info['qubits']
                
                # Generate gate token
                if len(qubits) == 1:
                    token_str = f'{gate_type}_{qubits[0]}'
                elif len(qubits) == 2:
                    token_str = f'{gate_type}_{qubits[0]}_{qubits[1]}'
                else:
                    continue
                
                if token_str in self.token_to_id:
                    tokens.append(self.token_to_id[token_str])
                
                # Add parameter token if trainable
                if gate_info.get('trainable', False):
                    param_idx = gate_info.get('param_idx', 0)
                    if isinstance(param_idx, int):
                        param_token = f'PARAM_{param_idx % 16}'
                        if param_token in self.token_to_id:
                            tokens.append(self.token_to_id[param_token])
                            
            except Exception as e:
                continue
        
        tokens.append(self.token_to_id['[END]'])
        return tokens
    
    def decode(self, token_ids: List[int]) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Convert token IDs back to gate sequence
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            gate_sequence: List of gate dictionaries
            parameter_map: Mapping of parameter names to indices
        """
        gate_sequence = []
        parameter_map = {}
        param_counter = 0
        
        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]
            
            # Skip special tokens
            if token_id in [
                self.token_to_id['[PAD]'],
                self.token_to_id['[START]'],
                self.token_to_id['[END]'],
                self.token_to_id.get('[SEP]', -1)
            ]:
                i += 1
                continue
            
            token_str = self.id_to_token.get(token_id, '')
            
            # Parse gate tokens
            if '_' in token_str and not token_str.startswith('PARAM'):
                parts = token_str.split('_')
                gate_type = parts[0]
                
                if gate_type in ['RX', 'RY', 'RZ', 'H', 'S', 'T', 'X', 'Y', 'Z']:
                    # Single-qubit gate
                    qubit = int(parts[1])
                    trainable = gate_type in ['RX', 'RY', 'RZ']
                    
                    gate_info = {
                        'gate': gate_type,
                        'qubits': [qubit],
                        'trainable': trainable,
                        'param_idx': param_counter if trainable else None
                    }
                    
                    if trainable:
                        parameter_map[f'{gate_type}_gate_{len(gate_sequence)}'] = param_counter
                        param_counter += 1
                    
                    gate_sequence.append(gate_info)
                    
                elif gate_type in ['CNOT', 'CZ', 'SWAP'] and len(parts) == 3:
                    # Two-qubit gate
                    q1, q2 = int(parts[1]), int(parts[2])
                    
                    gate_info = {
                        'gate': gate_type,
                        'qubits': [q1, q2],
                        'trainable': False,
                        'param_idx': None
                    }
                    
                    gate_sequence.append(gate_info)
            
            i += 1
        
        return gate_sequence, parameter_map
    
    @property
    def start_token_id(self) -> int:
        return self.token_to_id['[START]']
    
    @property
    def end_token_id(self) -> int:
        return self.token_to_id['[END]']
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id['[PAD]']


def create_gpt_model(n_qubits: int = 6, pretrained_path: Optional[str] = None) -> Tuple[QuantumCircuitGPT, CircuitTokenizer]:
    """
    Factory function to create GPT model and tokenizer
    
    Args:
        n_qubits: Number of qubits in quantum circuits
        pretrained_path: Path to pretrained model weights
        
    Returns:
        model: QuantumCircuitGPT model
        tokenizer: CircuitTokenizer
    """
    tokenizer = CircuitTokenizer(n_qubits)
    
    model = QuantumCircuitGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=128,
        dropout=0.1
    ).to(device)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained GPT model from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    print(f"GPT Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer


def train_gpt_on_circuits(
    model: QuantumCircuitGPT,
    tokenizer: CircuitTokenizer,
    circuit_data: List[Dict],
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 5e-4
):
    """
    Train GPT model on circuit data
    
    Args:
        model: QuantumCircuitGPT model
        tokenizer: CircuitTokenizer
        circuit_data: List of circuit dictionaries with 'gate_sequence', 'energy', 'score'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    # Prepare data
    sequences = []
    energies = []
    
    for circuit in circuit_data:
        tokens = tokenizer.encode(circuit['gate_sequence'])
        sequences.append(tokens)
        energies.append(circuit.get('energy', -1.0))
    
    dataset = QuantumCircuitDataset(sequences, energies)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (seq, energy, quality) in enumerate(dataloader):
            seq = seq.to(device)
            energy = energy.to(device)
            
            optimizer.zero_grad()
            logits, loss, _ = model(seq, targets=seq, energies=energy)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
