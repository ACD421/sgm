#!/usr/bin/env python3
"""
GateMesh: Boolean circuit network with evolvable wiring and truth tables.

Each gate has:
- 2 input wires (indices into previous layer)
- 1 truth table (4-bit, encodes any 2-input Boolean function)
- 1 locked flag (frozen when converged)

Memory: 5 bytes per gate (2 + 2 + 1 packed)
"""
import numpy as np


class GateMesh:
    """
    A mesh of Boolean gates with evolvable wiring.

    Architecture:
        depth layers of width gates each.
        Each gate reads 2 inputs from previous layer and applies a 4-bit truth table.

    Truth table encoding:
        Bits [3:0] encode outputs for inputs (A,B) = (0,0), (0,1), (1,0), (1,1)
        Example: NAND = 0b0111 = 7, AND = 0b1000 = 8, XOR = 0b0110 = 6

    Plasticity:
        1% of gates remain permanently unlocked for hyper-plasticity.
        These "plastic gates" allow rapid adaptation even after convergence.

        Empirically optimal: 1% gives best efficiency (improvement per gate).
        Higher fractions show diminishing returns after 3%.
    """

    PLASTIC_FRACTION = 0.01  # 1% always plastic (empirically optimal)

    def __init__(self, depth: int = 12, width: int = 1024):
        self.depth = depth
        self.width = width

        # Wire indices: which neurons in prev layer feed each gate
        self.wire_a = np.random.randint(0, width, (depth, width), dtype=np.uint16)
        self.wire_b = np.random.randint(0, width, (depth, width), dtype=np.uint16)

        # Truth tables: random 4-bit functions (not all NAND - more diversity)
        self.truth_tables = np.random.randint(0, 16, (depth, width), dtype=np.uint8)

        # Locked gates cannot be mutated
        self.locked = np.zeros((depth, width), dtype=np.uint8)

        # Permanently plastic gate indices (never lock these)
        n_plastic = max(1, int(width * self.PLASTIC_FRACTION))
        self.plastic_indices = np.random.choice(width, n_plastic, replace=False)

        # Activation layers for forward pass
        self.layers = [np.zeros(width, dtype=np.uint8) for _ in range(depth + 1)]

    def forward_vectorized(self, inputs: np.ndarray, residual: bool = True) -> np.ndarray:
        """
        Fast forward pass using numpy fancy indexing.

        Args:
            inputs: Binary input vector (width bits)
            residual: If True, XOR output with input (prevents attractor collapse)

        Returns:
            Binary output vector (width bits)
        """
        self.layers[0] = inputs.astype(np.uint8)

        for d in range(self.depth):
            # Gather inputs from previous layer
            a_vals = self.layers[d][self.wire_a[d]]
            b_vals = self.layers[d][self.wire_b[d]]

            # Compute truth table index: (a << 1) | b
            tt_idx = (a_vals.astype(np.uint16) << 1) | b_vals.astype(np.uint16)

            # Apply truth table: (tt >> idx) & 1
            self.layers[d + 1] = (
                (self.truth_tables[d].astype(np.uint16) >> tt_idx) & 1
            ).astype(np.uint8)

        output = self.layers[self.depth]

        # Residual connection: prevents deep networks from collapsing to attractors
        if residual:
            output = output ^ inputs.astype(np.uint8)

        return output

    def mutate_unlocked(self, rate: float = 0.02) -> int:
        """
        Mutate unlocked gates.

        Returns number of mutations applied.
        """
        mutations = 0

        for d in range(self.depth):
            for w in range(self.width):
                if self.locked[d, w]:
                    continue

                if np.random.random() < rate:
                    choice = np.random.randint(3)
                    if choice == 0:
                        self.wire_a[d, w] = np.random.randint(self.width)
                    elif choice == 1:
                        self.wire_b[d, w] = np.random.randint(self.width)
                    else:
                        self.truth_tables[d, w] = np.random.randint(16)
                    mutations += 1

        return mutations

    def lock_layer(self, layer: int):
        """Lock gates in a layer, preserving plastic gates for hyper-plasticity."""
        self.locked[layer, :] = 1
        # Keep plastic gates unlocked
        self.locked[layer, self.plastic_indices] = 0

    def n_locked(self) -> int:
        """Count locked gates."""
        return int(self.locked.sum())

    def n_plastic(self) -> int:
        """Count permanently plastic gates (per layer)."""
        return len(self.plastic_indices)

    def total_plastic(self) -> int:
        """Count total plastic gates across all layers."""
        return self.depth * len(self.plastic_indices)

    def save(self, path: str):
        """Save gate mesh to file."""
        np.savez_compressed(
            path,
            wire_a=self.wire_a,
            wire_b=self.wire_b,
            truth_tables=self.truth_tables,
            locked=self.locked,
            plastic_indices=self.plastic_indices,
            depth=self.depth,
            width=self.width,
        )

    @classmethod
    def load(cls, path: str) -> 'GateMesh':
        """Load gate mesh from file."""
        data = np.load(path)
        mesh = cls(int(data['depth']), int(data['width']))
        mesh.wire_a = data['wire_a']
        mesh.wire_b = data['wire_b']
        mesh.truth_tables = data['truth_tables']
        mesh.locked = data['locked']
        if 'plastic_indices' in data:
            mesh.plastic_indices = data['plastic_indices']
        return mesh

    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        return (
            self.wire_a.nbytes +
            self.wire_b.nbytes +
            self.truth_tables.nbytes +
            self.locked.nbytes
        )
