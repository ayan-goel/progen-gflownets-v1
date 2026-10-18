"""Fixed-length substitution constraints and finite, prefix-sharing task trees."""

import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def hamming(first: str, second: str) -> int:
    """Count substitutions; never silently compare different lengths."""
    if len(first) != len(second):
        raise ValueError("Hamming distance requires equal lengths")
    return sum(left != right for left, right in zip(first, second))


@dataclass(frozen=True)
class SequenceContext:
    """A known reference and maximum number of substitutions, including zero."""

    reference: str
    mutation_budget: int
    alphabet: str = AMINO_ACIDS

    def __post_init__(self):
        if not self.reference or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("Need a nonempty reference and distinct alphabet symbols")
        if not set(self.reference) <= set(self.alphabet):
            raise ValueError("Reference contains residues outside the alphabet")
        if not 0 <= self.mutation_budget <= len(self.reference):
            raise ValueError("Mutation budget must be between zero and sequence length")

    def legal_actions(self, prefix: str) -> list[str]:
        """Return residues that leave a legal completion; forced residues stay explicit."""
        if len(prefix) > len(self.reference) or not set(prefix) <= set(self.alphabet):
            raise ValueError("Invalid prefix")
        used = hamming(prefix, self.reference[: len(prefix)])
        if used > self.mutation_budget:
            raise ValueError("Prefix exceeds mutation budget")
        if len(prefix) == len(self.reference):
            return []
        if used == self.mutation_budget:
            return [self.reference[len(prefix)]]
        return list(self.alphabet)

    def enumerate_sequences(self, maximum: int = 20000) -> list[str]:
        """Enumerate only small tasks; fail before constructing an oversized neighborhood."""
        count = sum(
            math.comb(len(self.reference), changes)
            * (len(self.alphabet) - 1) ** changes
            for changes in range(self.mutation_budget + 1)
        )
        if count > maximum:
            raise ValueError(
                f"Support has {count} sequences; configured limit is {maximum}"
            )
        sequences = []
        for changes in range(self.mutation_budget + 1):
            for positions in combinations(range(len(self.reference)), changes):
                choices = [
                    [
                        residue
                        for residue in self.alphabet
                        if residue != self.reference[position]
                    ]
                    for position in positions
                ]
                for replacements in product(*choices):
                    sequence = list(self.reference)
                    for position, residue in zip(positions, replacements):
                        sequence[position] = residue
                    sequences.append("".join(sequence))
        return sorted(sequences)


class PrefixTree:
    """Compile a declared finite support without consulting any stability labels.

    Internal nodes are prefixes. Edges index residues, and leaf identifiers follow
    sorted sequence order. Every full sequence has exactly one trajectory.
    """

    def __init__(self, context: SequenceContext, sequences: list[str]):
        self.context = context
        self.sequences = sorted(set(sequences))
        if not self.sequences or len(self.sequences) != len(sequences):
            raise ValueError(
                "Support must be nonempty and contain no duplicate sequences"
            )
        for sequence in self.sequences:
            if not set(sequence) <= set(context.alphabet):
                raise ValueError("Noncanonical candidate")
            if hamming(sequence, context.reference) > context.mutation_budget:
                raise ValueError("Candidate outside mutation budget")
        prefixes = {
            sequence[:position]
            for sequence in self.sequences
            for position in range(len(context.reference))
        }
        self.prefixes = sorted(prefixes, key=lambda prefix: (len(prefix), prefix))
        self.node_index = {prefix: index for index, prefix in enumerate(self.prefixes)}
        shape = (len(self.prefixes), len(context.alphabet))
        self.children = np.full(shape, -1, dtype=np.int64)
        self.leaves = np.full(shape, -1, dtype=np.int64)
        self.paths = np.empty((len(sequences), len(context.reference)), dtype=np.int64)
        self.actions = np.empty_like(self.paths)
        for leaf, sequence in enumerate(self.sequences):
            for position, residue in enumerate(sequence):
                node = self.node_index[sequence[:position]]
                action = context.alphabet.index(residue)
                self.paths[leaf, position] = node
                self.actions[leaf, position] = action
                if position + 1 == len(sequence):
                    self.leaves[node, action] = leaf
                else:
                    self.children[node, action] = self.node_index[
                        sequence[: position + 1]
                    ]
        self.legal = (self.children >= 0) | (self.leaves >= 0)

    def context_features(self) -> np.ndarray:
        """Position, remaining budget, and one-hot reference residue at each prefix."""
        features = np.zeros(
            (len(self.prefixes), 2 + len(self.context.alphabet)), dtype=np.float32
        )
        for index, prefix in enumerate(self.prefixes):
            position = len(prefix)
            used = hamming(prefix, self.context.reference[:position])
            features[index, 0] = position / len(self.context.reference)
            features[index, 1] = (self.context.mutation_budget - used) / max(
                1, self.context.mutation_budget
            )
            features[
                index, 2 + self.context.alphabet.index(self.context.reference[position])
            ] = 1
        return features
