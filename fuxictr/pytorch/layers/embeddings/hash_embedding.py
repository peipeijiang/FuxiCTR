"""
Hash Embedding layer for FuxiCTR.

This module provides hash-based embedding as an alternative to label encoder approach.
Key features:
- No preprocessing required (no LabelEncoder needed)
- Handles high-cardinality features efficiently
- Supports online learning with new categories
- Tracks trained hash positions to handle OOV properly in inference
"""
import torch
import torch.nn as nn
import numpy as np


class HashEmbedding(nn.Module):
    """
    Hash-based embedding layer that maps feature values directly to embedding indices.

    This layer uses a hash function to map feature values to embedding indices,
    eliminating the need for preprocessing with LabelEncoder. It maintains a
    trained_mask to track which hash positions have been seen during training,
    allowing proper handling of OOV (Out of Vocabulary) values in inference.

    Args:
        num_buckets: Number of hash buckets (embedding table size)
        embedding_dim: Dimension of embedding vectors
        hash_function: Hash function to use ('uniform' or 'murmurhash3')
        padding_idx: If specified, the entries at padding_idx are not updated
                     and default to zeros during training

    Example:
        >>> hash_emb = HashEmbedding(num_buckets=100000, embedding_dim=64)
        >>> # Directly use feature values (no encoding needed)
        >>> values = torch.tensor([1, 2, 3, 1, 999999])  # 999999 is OOV
        >>> embeddings = hash_emb(values)
    """

    def __init__(self, num_buckets, embedding_dim, hash_function='uniform', padding_idx=None):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.hash_function = hash_function
        self.padding_idx = padding_idx

        # Main embedding table
        self.embedding = nn.Embedding(
            num_buckets,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Track which hash positions have been seen during training
        self.register_buffer('trained_mask', torch.zeros(num_buckets, dtype=torch.bool))

        # Unknown embedding (zeros initialization - implicit, not configurable)
        # This is used for OOV values in inference mode
        self.register_buffer('unknown_embedding', torch.zeros(embedding_dim))

    def _hash(self, feature_values):
        """
        Hash feature values to bucket indices.

        Args:
            feature_values: Tensor of feature values (any dtype)

        Returns:
            Tensor of hash indices in range [0, num_buckets)
        """
        if self.hash_function == 'uniform':
            # Simple uniform hash using modulo
            # Convert to positive values for modulo
            values = feature_values.long()
            indices = torch.remainder(values, self.num_buckets)
        elif self.hash_function == 'murmurhash3':
            # Use murmurhash3 for better distribution
            indices = self._murmurhash3(feature_values)
        else:
            raise ValueError(f"Unknown hash function: {self.hash_function}")

        return indices

    def _murmurhash3(self, feature_values):
        """
        MurmurHash3 implementation for better hash distribution.

        Uses 32-bit MurmurHash3 finalizer for better distribution.
        """
        values = feature_values.long()

        # Constants for MurmurHash3
        c1 = 0xcc9e2d51
        c2 = 0x1b873593
        r1 = 15
        r2 = 13
        m = 5
        n = 0xe6546b64

        # Apply MurmurHash3 mixing
        h = values
        h = h * c1
        # Use bitwise_right_shift instead of shift_logical_right
        h = torch.bitwise_xor(torch.bitwise_right_shift(h, r1), h)
        h = h * c2
        h = torch.bitwise_xor(torch.bitwise_right_shift(h, r2), h)
        h = h * m + n

        # Finalizer
        h = torch.bitwise_xor(torch.bitwise_right_shift(h, 16), h)
        h = h * 0x85ebca6b
        h = torch.bitwise_xor(torch.bitwise_right_shift(h, 13), h)
        h = h * 0xc2b2ae35
        h = torch.bitwise_xor(torch.bitwise_right_shift(h, 16), h)

        # Convert to positive and apply modulo
        h = torch.remainder(h, self.num_buckets)
        return torch.remainder(h, self.num_buckets).long()

    def forward(self, feature_values):
        """
        Forward pass for hash embedding.

        Args:
            feature_values: Tensor of feature values (shape: [batch_size]
                           or [batch_size, seq_len] for sequences)

        Returns:
            Embedded tensor (shape: [batch_size, embedding_dim]
                           or [batch_size, seq_len, embedding_dim] for sequences)
        """
        original_shape = feature_values.shape
        flat_values = feature_values.reshape(-1)

        # Handle padding index
        if self.padding_idx is not None:
            padding_mask = (flat_values == self.padding_idx)
            # Replace padding with 0 temporarily for hashing
            flat_values = torch.where(padding_mask, torch.zeros_like(flat_values), flat_values)

        # Compute hash indices
        indices = self._hash(flat_values)

        # Restore padding
        if self.padding_idx is not None:
            indices = torch.where(padding_mask, torch.tensor(self.padding_idx), indices)

        # Get embeddings
        embeddings = self.embedding(indices)

        # Handle OOV: replace untrained positions with zeros in inference mode
        if not self.training:
            is_oov = ~self.trained_mask[indices]
            embeddings[is_oov] = self.unknown_embedding
        else:
            # Update trained mask during training
            self.trained_mask.scatter_(0, indices, 1)

        # Reshape to original shape
        if len(original_shape) == 1:
            return embeddings
        else:
            seq_len = original_shape[1]
            return embeddings.reshape(-1, seq_len, self.embedding_dim)

    def extra_repr(self):
        """Extra representation for print."""
        return f'num_buckets={self.num_buckets}, embedding_dim={self.embedding_dim}, hash_function={self.hash_function}'


class SequenceHashEmbedding(nn.Module):
    """
    Hash-based embedding for sequence features.

    Similar to HashEmbedding but specifically designed for sequence features.
    Handles variable-length sequences with proper masking.

    Args:
        num_buckets: Number of hash buckets (embedding table size)
        embedding_dim: Dimension of embedding vectors
        hash_function: Hash function to use ('uniform' or 'murmurhash3')
        padding_idx: Padding index (default: 0)

    Example:
        >>> seq_hash_emb = SequenceHashEmbedding(num_buckets=100000, embedding_dim=64)
        >>> sequences = torch.tensor([[1, 2, 3, 0], [4, 0, 0, 0]])  # 0 is padding
        >>> embeddings = seq_hash_emb(sequences)
    """

    def __init__(self, num_buckets, embedding_dim, hash_function='uniform', padding_idx=0):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.hash_embedding = HashEmbedding(
            num_buckets=num_buckets,
            embedding_dim=embedding_dim,
            hash_function=hash_function,
            padding_idx=padding_idx
        )

    def forward(self, sequences):
        """
        Forward pass for sequence hash embedding.

        Args:
            sequences: Tensor of shape [batch_size, seq_len]

        Returns:
            Embedded tensor of shape [batch_size, seq_len, embedding_dim]
        """
        return self.hash_embedding(sequences)

    def extra_repr(self):
        """Extra representation for print."""
        return f'num_buckets={self.num_buckets}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}'
