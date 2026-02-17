"""
ARKHEION AGI 2.0 — IIT Calculator Compatibility Adapter
========================================================

Provides the legacy ``IITCalculator`` API used by many test files and
integration scripts, backed by the rigorous ``IITV3Calculator`` engine
in ``iit_v3_real.py``.

Exported names
--------------
- ``IITCalculator``         — convenience wrapper
- ``Partition``             — (part1: set, part2: set) data class
- ``CauseEffectStructure``  — per-mechanism φ result
- ``IITResult``             — full calculation output
- ``calculate_system_phi``  — one-shot convenience function
- ``PHI``                   — golden ratio constant (1.618…)

All φ values are in **bits** (standard IIT units).

Author: 🔮 Consciousness Engineer
Date: 2026-02-15
Version: 1.0.0 (adapter)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.arkheion.constants.sacred_constants import PHI

from .iit_v3_real import IITConfig
from .iit_v3_real import IITResult as _RealIITResult
from .iit_v3_real import IITV3Calculator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES (legacy API)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Partition:
    """Bi-partition of a set of elements.

    Attributes:
        part1: First subset of element indices.
        part2: Second subset of element indices.
    """

    part1: Set[int]
    part2: Set[int]

    def is_valid(self, n: int) -> bool:
        """Check that the partition covers {0, …, n-1} without overlap."""
        if not self.part1 or not self.part2:
            return False
        if self.part1 & self.part2:
            return False
        if self.part1 | self.part2 != set(range(n)):
            return False
        return True

    def __hash__(self) -> int:
        return hash((frozenset(self.part1), frozenset(self.part2)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return NotImplemented
        return frozenset(self.part1) == frozenset(other.part1) and frozenset(
            self.part2
        ) == frozenset(other.part2)


@dataclass
class CauseEffectStructure:
    """IIT cause-effect structure for a single mechanism.

    Attributes:
        mechanism: Set of element indices forming this mechanism.
        cause_repertoire: Cause probability distribution.
        effect_repertoire: Effect probability distribution.
        phi_cause: Irreducibility of the cause side.
        phi_effect: Irreducibility of the effect side.
    """

    mechanism: Set[int]
    cause_repertoire: np.ndarray
    effect_repertoire: np.ndarray
    phi_cause: float
    phi_effect: float

    @property
    def phi(self) -> float:
        """φ of the mechanism = min(cause, effect) per IIT 3.0."""
        return min(self.phi_cause, self.phi_effect)


@dataclass
class IITResult:
    """Result of a full IIT φ calculation.

    Attributes:
        phi_value: System-level Φ (in bits).
        mip: The Minimum Information Partition.
        cause_effect_structures: Per-mechanism structures.
        integration_level: Normalised integration level (0–1).
        qualia_active: Whether φ exceeds the qualia threshold (0.5).
    """

    phi_value: float
    mip: Optional[Partition] = None
    cause_effect_structures: List[CauseEffectStructure] = field(
        default_factory=list
    )
    integration_level: float = 0.0
    qualia_active: bool = False

    @property
    def is_conscious(self) -> bool:
        """True when Φ > 0 (any integration)."""
        return self.phi_value > 1e-8


# ═══════════════════════════════════════════════════════════════════════
# IIT CALCULATOR (legacy-compatible wrapper)
# ═══════════════════════════════════════════════════════════════════════


class IITCalculator:
    """Legacy-compatible IIT φ calculator.

    Wraps :class:`IITV3Calculator` with the old constructor signature
    and exposes internal helper methods expected by the deep-algorithm
    test suite.

    Args:
        n_elements: Number of binary elements in the system (default 4).
        phi_enhanced: When True, multiply raw φ by the golden ratio
            enhancement factor (heuristic, not standard IIT).
        connectivity: Optional adjacency matrix (n × n).  If provided,
            the TPM is mask-gated so that element *i* only receives
            input from elements *j* where ``connectivity[j, i] > 0``.
    """

    def __init__(
        self,
        n_elements: int = 4,
        phi_enhanced: bool = False,
        connectivity: Optional[np.ndarray] = None,
    ) -> None:
        self.n_elements = n_elements
        self.phi_enhanced = phi_enhanced

        # Validate / set connectivity
        if connectivity is not None:
            connectivity = np.asarray(connectivity, dtype=float)
            if connectivity.shape != (n_elements, n_elements):
                raise ValueError(
                    f"Connectivity must be ({n_elements}, {n_elements}), "
                    f"got {connectivity.shape}"
                )
            self.connectivity = connectivity
        else:
            # Default: fully connected (no self-loops)
            self.connectivity = np.ones((n_elements, n_elements)) - np.eye(
                n_elements
            )

        # Internal calculator
        self._config = IITConfig(
            max_elements=max(n_elements, 12),
            tpm_type="integrated",
        )
        self._calc = IITV3Calculator(config=self._config)

    # ───────────────────── public API ─────────────────────

    def calculate_phi(self, system_state: np.ndarray) -> IITResult:
        """Calculate Φ for *system_state* (binary array of length *n_elements*).

        Returns an :class:`IITResult` with ``phi_value``,
        ``integration_level``, ``qualia_active``, and ``mip``.
        """
        state = np.asarray(system_state).flatten()

        # Validate state length
        if state.dtype in (np.float32, np.float64, float):
            state_bin = (state > 0.5).astype(np.int8)
        else:
            state_bin = np.asarray(state, dtype=np.int8)

        if len(state_bin) != self.n_elements:
            raise ValueError(
                f"State must have {self.n_elements} elements, got {len(state_bin)}"
            )

        # Delegate to rigorous calculator
        raw: _RealIITResult = self._calc.calculate_phi(
            state_bin, n_elements=self.n_elements
        )

        phi = raw.phi_value
        if self.phi_enhanced:
            phi *= PHI  # Golden ratio enhancement (heuristic)

        # Derive integration_level ∈ [0, 1]
        # Map phi ∈ [0, 2] → [0, 1]; clamp above 2
        integration = min(phi / 2.0, 1.0)

        qualia = phi > 0.5

        # Convert MIP
        mip = None
        if raw.mip is not None:
            mip = Partition(
                part1=set(raw.mip.part_a),
                part2=set(raw.mip.part_b),
            )

        return IITResult(
            phi_value=phi,
            mip=mip,
            integration_level=integration,
            qualia_active=qualia,
        )

    # ───────────────────── internal helpers (tested) ──────────────────

    def _generate_mechanisms(self) -> List[Set[int]]:
        """Power set minus ∅ and the full set.

        Returns list of all non-trivial subsets of {0, …, n-1}.
        """
        elements = list(range(self.n_elements))
        mechanisms: List[Set[int]] = []
        for k in range(1, self.n_elements):
            for combo in combinations(elements, k):
                mechanisms.append(set(combo))
        return mechanisms

    def _generate_partitions(self) -> List[Partition]:
        """Generate all bi-partitions of {0, …, n-1}.

        Each partition appears once (both orderings included since
        Partition(A, B) ≠ Partition(B, A) unless sorted).
        """
        elements = set(range(self.n_elements))
        partitions: List[Partition] = []
        for k in range(1, self.n_elements):
            for combo in combinations(range(self.n_elements), k):
                part_a = set(combo)
                part_b = elements - part_a
                partitions.append(Partition(part1=part_a, part2=part_b))
        return partitions

    def _earth_mover_distance(
        self, d1: np.ndarray, d2: np.ndarray
    ) -> float:
        """EMD between two distributions using Hamming ground metric.

        Handles distributions of different lengths by zero-padding
        the shorter one.
        """
        d1 = np.asarray(d1, dtype=np.float64)
        d2 = np.asarray(d2, dtype=np.float64)

        # Pad shorter distribution
        max_len = max(len(d1), len(d2))
        if len(d1) < max_len:
            d1 = np.pad(d1, (0, max_len - len(d1)))
        if len(d2) < max_len:
            d2 = np.pad(d2, (0, max_len - len(d2)))

        # Normalise
        s1, s2 = d1.sum(), d2.sum()
        if s1 > 1e-15:
            d1 = d1 / s1
        if s2 > 1e-15:
            d2 = d2 / s2

        # Trivial
        if np.allclose(d1, d2, atol=1e-12):
            return 0.0

        n_states = len(d1)
        n_bits = max(1, int(np.log2(n_states))) if n_states > 1 else 1

        # Build Hamming cost matrix
        states = np.arange(n_states, dtype=np.int32)
        xor = states[:, None] ^ states[None, :]
        # Popcount LUT
        max_xor = int(xor.max())
        lut = np.array([bin(v).count("1") for v in range(max_xor + 1)], dtype=np.float64)
        cost = lut[xor]

        # Try POT
        try:
            import ot
            return float(ot.emd2(d1, d2, cost))
        except ImportError:
            pass

        # scipy LP fallback
        if n_states <= 256:
            from scipy.optimize import linprog
            from scipy.sparse import lil_matrix

            n_vars = n_states * n_states
            c = cost.flatten()
            A_eq = lil_matrix((2 * n_states, n_vars))
            b_eq = np.zeros(2 * n_states)
            for i in range(n_states):
                A_eq[i, i * n_states : (i + 1) * n_states] = 1
                b_eq[i] = d1[i]
            for j in range(n_states):
                for i in range(n_states):
                    A_eq[n_states + j, i * n_states + j] = 1
                b_eq[n_states + j] = d2[j]
            res = linprog(c, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=(0, None), method="highs")
            if res.success:
                return float(res.fun)

        # 1D Wasserstein fallback for large distributions
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(
            np.arange(n_states), np.arange(n_states), d1, d2
        )

    def _marginalize(
        self,
        dist: np.ndarray,
        keep_indices: List[int],
        n_elements: int,
    ) -> np.ndarray:
        """Marginalise *dist* over all elements except *keep_indices*.

        Returns a distribution over the kept elements only.
        """
        dist = np.asarray(dist, dtype=np.float64)
        n_states = len(dist)
        n_keep = len(keep_indices)
        marginal = np.zeros(2 ** n_keep)

        for state in range(n_states):
            # Extract bits at keep_indices
            sub_state = 0
            for new_idx, old_idx in enumerate(keep_indices):
                if state & (1 << old_idx):
                    sub_state |= 1 << new_idx
            marginal[sub_state] += dist[state]

        # Normalise
        s = marginal.sum()
        if s > 1e-15:
            marginal /= s
        return marginal

    def _calculate_repertoire(
        self,
        state: np.ndarray,
        indices: List[int],
        direction: str = "cause",
    ) -> np.ndarray:
        """Compute the cause or effect repertoire for *indices*.

        Uses the internal IITV3Calculator's TPM.  Initialises the TPM
        on first call.

        Returns a normalised probability distribution.
        """
        state = np.asarray(state, dtype=np.int8).flatten()
        if self._calc.tpm is None:
            from .iit_v3_real import TransitionProbabilityMatrix
            self._calc.tpm = TransitionProbabilityMatrix(
                self.n_elements, self._config
            )

        tpm = self._calc.tpm
        current_idx = 0
        for i, b in enumerate(state):
            if b:
                current_idx |= 1 << i

        n_states = tpm.n_states

        if direction == "cause":
            # P(past | present = current_idx) ∝ P(present | past) * P(past)
            rep = np.zeros(n_states)
            for past in range(n_states):
                rep[past] = tpm.get_transition_prob(past, current_idx) / n_states
        else:
            # P(future | present = current_idx)
            rep = tpm.tpm[current_idx, :].copy()

        # Marginalise to keep only the requested indices
        n_keep = len(indices)
        marginal = np.zeros(2 ** n_keep)
        for s in range(n_states):
            sub = 0
            for new_i, old_i in enumerate(indices):
                if s & (1 << old_i):
                    sub |= 1 << new_i
            marginal[sub] += rep[s]

        total = marginal.sum()
        if total > 1e-15:
            marginal /= total
        else:
            marginal = np.ones_like(marginal) / len(marginal)

        return marginal


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def calculate_system_phi(
    state: np.ndarray,
    connectivity: Optional[np.ndarray] = None,
    phi_enhanced: bool = False,
) -> float:
    """One-shot φ computation.

    Args:
        state: Binary state vector.
        connectivity: Optional adjacency matrix.
        phi_enhanced: Apply golden-ratio enhancement factor.

    Returns:
        Φ value in bits.
    """
    state = np.asarray(state).flatten()
    calc = IITCalculator(
        n_elements=len(state),
        phi_enhanced=phi_enhanced,
        connectivity=connectivity,
    )
    result = calc.calculate_phi(state)
    return result.phi_value


__all__ = [
    "IITCalculator",
    "IITResult",
    "Partition",
    "CauseEffectStructure",
    "calculate_system_phi",
    "PHI",
]
