# transformer_encoder.py
# Processes sequence of past states
# Learns temporal patterns in workload
# Key innovation of TASCAR!

import torch
import torch.nn as nn
import numpy as np
from config import (
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_FF_DIM,
    DROPOUT_RATE,
    NUM_QUEUES
)


class PositionalEncoding(nn.Module):
    """
    Adds position information to sequence
    Transformer needs to know ORDER
    of states in sequence!

    Like numbering sentences:
    State 1 came first
    State 2 came second
    State 10 is most recent
    """
    def __init__(self, dim,
                 max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(
            0, max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0, dim, 2
            ).float() *
            (-np.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(
            position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[
            :, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """
    Main Transformer Encoder

    Takes last 10 S-Cache states
    Learns temporal patterns:
    - Burst behavior
    - Periodic patterns
    - Long range dependencies

    Returns enriched state for SAC agent!

    Input:  (batch, seq_len, state_dim)
    Output: (batch, TRANSFORMER_DIM)
    """
    def __init__(self, state_dim):
        super().__init__()

        self.state_dim = state_dim

        # Project input to transformer dim
        self.input_projection = nn.Linear(
            state_dim,
            TRANSFORMER_DIM)

        # Positional encoding
        self.pos_encoding = (
            PositionalEncoding(
                TRANSFORMER_DIM))

        # Transformer encoder layers
        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=TRANSFORMER_DIM,
                nhead=TRANSFORMER_HEADS,
                dim_feedforward=TRANSFORMER_FF_DIM,
                dropout=DROPOUT_RATE,
                batch_first=True))

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=TRANSFORMER_LAYERS))

        # Cross-queue attention
        # Queues communicate with each other!
        self.cross_queue_attention = (
            nn.MultiheadAttention(
                embed_dim=TRANSFORMER_DIM,
                num_heads=TRANSFORMER_HEADS,
                dropout=DROPOUT_RATE,
                batch_first=True))

        # Final output projection
        self.output_projection = (
            nn.Sequential(
                nn.Linear(
                    TRANSFORMER_DIM,
                    TRANSFORMER_DIM),
                nn.ReLU(),
                nn.Linear(
                    TRANSFORMER_DIM,
                    TRANSFORMER_DIM)))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(
            TRANSFORMER_DIM)

        # Dropout
        self.dropout = nn.Dropout(
            DROPOUT_RATE)

    def forward(self, state_sequence):
        """
        Process sequence of states

        state_sequence:
        (batch, seq_len, state_dim)
        OR
        (seq_len, state_dim)
        """
        # Add batch dim if needed
        if state_sequence.dim() == 2:
            state_sequence = (
                state_sequence.unsqueeze(0))

        # Project to transformer dim
        x = self.input_projection(
            state_sequence)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer layers
        # Learns temporal patterns!
        x = self.transformer(x)

        # Cross-queue attention
        # Last timestep as query
        query = x[:, -1:, :]

        attended, _ = (
            self.cross_queue_attention(
                query, x, x))

        # Combine last hidden
        # with attended features
        last_hidden = x[:, -1, :]
        combined = (
            last_hidden +
            attended.squeeze(1))
        combined = self.layer_norm(
            combined)

        # Final projection
        output = (
            self.output_projection(
                combined))

        return output.squeeze(0)

    def get_output_dim(self):
        return TRANSFORMER_DIM


class StateHistoryBuffer:
    """
    Stores last N states
    Feeds sequence to Transformer

    Short term memory for agent!
    Remembers last 10 observations
    Transformer finds patterns in them!
    """
    def __init__(self,
                 sequence_length,
                 state_dim):
        self.sequence_length = (
            sequence_length)
        self.state_dim = state_dim
        self.buffer    = []

    def add(self, state):
        """Add new state to buffer"""
        self.buffer.append(
            np.array(state,
                     dtype=np.float32))

        # Keep only last N states
        if len(self.buffer) > (
                self.sequence_length):
            self.buffer.pop(0)

    def get_sequence(self):
        """
        Get padded sequence
        If not enough history:
        pad beginning with zeros
        """
        if len(self.buffer) == 0:
            return np.zeros((
                self.sequence_length,
                self.state_dim),
                dtype=np.float32)

        # Pad if history too short
        if len(self.buffer) < (
                self.sequence_length):
            padding_needed = (
                self.sequence_length -
                len(self.buffer))
            padding = [
                np.zeros(
                    self.state_dim,
                    dtype=np.float32)
                for _ in range(
                    padding_needed)]
            sequence = (
                padding + self.buffer)
        else:
            sequence = self.buffer[
                -self.sequence_length:]

        return np.array(
            sequence,
            dtype=np.float32)

    def reset(self):
        """Clear at episode start"""
        self.buffer = []

    def is_ready(self):
        """Has at least one state"""
        return len(self.buffer) >= 1