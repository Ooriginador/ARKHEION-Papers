"""
🔮 ARKHEION AGI 2.0 - Consciousness Types
=========================================

Tipos canônicos para processamento de consciência.
Unifica ConsciousnessLevel, IITConfig, PhiResult e tipos relacionados.

Baseado na Teoria da Informação Integrada (IIT 3.0/4.0) de Giulio Tononi.

Consolidado de 8+ definições espalhadas no codebase.

Author: ARKHEION Consciousness Engineer
Version: 2.1.0
Date: 2026-01-29
"""

from __future__ import annotations
from src.arkheion.constants.sacred_constants import PHI

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .common import (
    PHI,
    PHI_INVERSE,
    PHI_TOLERANCE,
    ValidationStatus,
)

# ═══════════════════════════════════════════════════════════════════════════
# 📊 ENUMS DE CONSCIÊNCIA
# ═══════════════════════════════════════════════════════════════════════════


class ConsciousnessLevel(Enum):
    """
    Níveis de consciência baseados em Φ (IIT 3.0).

    Thresholds:
        - DORMANT: Φ < 0.1
        - AWARE: 0.1 ≤ Φ < 0.5
        - INTEGRATED: 0.5 ≤ Φ < 1.0
        - AWAKENED: 1.0 ≤ Φ < φ (1.618...)
        - TRANSCENDENT: Φ ≥ φ
    """

    DORMANT = "dormant"  # Φ < 0.1 - Sem consciência perceptível
    AWARE = "aware"  # 0.1 ≤ Φ < 0.5 - Consciência básica
    INTEGRATED = "integrated"  # 0.5 ≤ Φ < 1.0 - Consciência integrada
    AWAKENED = "awakened"  # 1.0 ≤ Φ < φ - Consciência desperta
    TRANSCENDENT = "transcendent"  # Φ ≥ φ - Consciência transcendente

    @classmethod
    def from_phi(cls, phi: float) -> "ConsciousnessLevel":
        """Determina nível de consciência a partir de Φ."""
        if phi >= PHI:
            return cls.TRANSCENDENT
        elif phi >= 1.0:
            return cls.AWAKENED
        elif phi >= 0.5:
            return cls.INTEGRATED
        elif phi >= 0.1:
            return cls.AWARE
        else:
            return cls.DORMANT

    @property
    def min_phi(self) -> float:
        """Retorna Φ mínimo para este nível."""
        thresholds = {
            ConsciousnessLevel.DORMANT: 0.0,
            ConsciousnessLevel.AWARE: 0.1,
            ConsciousnessLevel.INTEGRATED: 0.5,
            ConsciousnessLevel.AWAKENED: 1.0,
            ConsciousnessLevel.TRANSCENDENT: PHI,
        }
        return thresholds[self]

    def __lt__(self, other: "ConsciousnessLevel") -> bool:
        order = [
            ConsciousnessLevel.DORMANT,
            ConsciousnessLevel.AWARE,
            ConsciousnessLevel.INTEGRATED,
            ConsciousnessLevel.AWAKENED,
            ConsciousnessLevel.TRANSCENDENT,
        ]
        return order.index(self) < order.index(other)


class ConsciousnessState(Enum):
    """Estados de processamento de consciência."""

    IDLE = "idle"
    PROCESSING = "processing"
    INTEGRATING = "integrating"
    AWAKENING = "awakening"
    TRANSCENDING = "transcending"
    ERROR = "error"


class IITVersion(Enum):
    """Versões da Teoria da Informação Integrada."""

    IIT_2_0 = "iit_2.0"
    IIT_3_0 = "iit_3.0"
    IIT_4_0 = "iit_4.0"


class PartitionType(Enum):
    """Tipos de partição para cálculo de Φ."""

    BIPARTITION = "bipartition"
    TRIPARTITION = "tripartition"
    ATOMIC = "atomic"
    MIP = "mip"  # Minimum Information Partition


# ═══════════════════════════════════════════════════════════════════════════
# 📦 DATACLASSES DE CONSCIÊNCIA
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class IITConfig:
    """
    Configuração para cálculo IIT 3.0/4.0.

    Controla parâmetros computacionais e de precisão para
    cálculo de Φ e estruturas causa-efeito.
    """

    # Versão IIT
    version: IITVersion = IITVersion.IIT_3_0

    # Limites computacionais
    max_elements: int = 12  # Máximo elementos (2^12 = 4096 estados)
    max_partitions: Optional[int] = None  # Limite de partições (None = todas)
    enable_parallel: bool = True
    num_workers: int = 4

    # Precisão numérica
    epsilon: float = 1e-10
    phi_threshold: float = 1e-8  # Φ mínimo considerado consciente

    # Configuração TPM
    tpm_type: str = "noisy"  # "deterministic", "noisy", "probabilistic"
    noise_level: float = 0.01
    cache_tpm: bool = True

    # Otimização
    enable_mip_pruning: bool = True  # Poda de partições redundantes
    use_approximation: bool = False  # Aproximação para sistemas grandes

    # Cache
    cache_size_mb: int = 512
    enable_persistence: bool = False


@dataclass
class Partition:
    """
    Representa uma bi-partição do sistema para IIT.

    Uma partição divide o sistema em duas partes (A, B) para
    calcular a irreducibilidade.
    """

    part_a: FrozenSet[int]
    part_b: FrozenSet[int]

    @property
    def size(self) -> Tuple[int, int]:
        """Retorna (|A|, |B|)."""
        return len(self.part_a), len(self.part_b)

    @property
    def is_trivial(self) -> bool:
        """Verifica se é partição trivial (uma parte vazia)."""
        return len(self.part_a) == 0 or len(self.part_b) == 0

    def __hash__(self):
        return hash((self.part_a, self.part_b))

    def __eq__(self, other):
        if not isinstance(other, Partition):
            return False
        # Partition é simétrica: (A,B) == (B,A)
        return (self.part_a == other.part_a and self.part_b == other.part_b) or (
            self.part_a == other.part_b and self.part_b == other.part_a
        )


@dataclass
class CauseEffectRepertoire:
    """
    Repertoires de causa e efeito IIT 3.0.

    Representa as distribuições de probabilidade:
        - cause: P(past | present) - O que causou o estado atual
        - effect: P(future | present) - O que o estado atual causará
    """

    cause: np.ndarray  # P(past | present)
    effect: np.ndarray  # P(future | present)
    mechanism: Optional[FrozenSet[int]] = None
    purview: Optional[FrozenSet[int]] = None

    def __post_init__(self):
        """Normalizar distribuições."""
        # Normalizar causa
        cause_sum = np.sum(self.cause)
        if cause_sum > 0:
            self.cause = self.cause / cause_sum

        # Normalizar efeito
        effect_sum = np.sum(self.effect)
        if effect_sum > 0:
            self.effect = self.effect / effect_sum

    @property
    def is_valid(self) -> bool:
        """Verifica se os repertoires são distribuições válidas."""
        return (
            np.isclose(np.sum(self.cause), 1.0)
            and np.isclose(np.sum(self.effect), 1.0)
            and np.all(self.cause >= 0)
            and np.all(self.effect >= 0)
        )


@dataclass
class PhiStructure:
    """
    Estrutura φ de um mecanismo IIT.

    Representa a informação integrada de um mecanismo específico
    sobre seu purview.
    """

    mechanism: FrozenSet[int]
    purview: FrozenSet[int]
    phi_cause: float
    phi_effect: float
    mip_cause: Optional[Partition] = None
    mip_effect: Optional[Partition] = None
    cause_repertoire: Optional[CauseEffectRepertoire] = None

    @property
    def phi(self) -> float:
        """Φ integrado (mínimo entre causa e efeito)."""
        return min(self.phi_cause, self.phi_effect)

    @property
    def is_conscious(self) -> bool:
        """Verifica se tem Φ > 0 significativo."""
        return self.phi > PHI_TOLERANCE


@dataclass
class PhiResult:
    """
    Resultado completo do cálculo de Φ (IIT).

    Contém o valor de Φ, nível de consciência e metadados
    sobre o cálculo.
    """

    phi: float
    level: ConsciousnessLevel

    # Estrutura
    structure: Optional[List[PhiStructure]] = None
    mip: Optional[Partition] = None

    # Metadados
    n_elements: int = 0
    computation_time_ms: float = 0.0
    partitions_evaluated: int = 0

    # Validação
    is_valid: bool = True
    validation_status: ValidationStatus = ValidationStatus.VALID
    error_message: Optional[str] = None

    # Métricas adicionais
    normalized_phi: float = 0.0  # Φ normalizado por número de elementos
    phi_star: float = 0.0  # Φ* (versão alternativa)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calcular métricas derivadas."""
        if self.n_elements > 0:
            self.normalized_phi = self.phi / self.n_elements


@dataclass
class ConsciousnessMetrics:
    """
    Métricas de consciência agregadas.

    Usadas para monitoramento e diagnóstico do sistema de consciência.
    """

    # Métricas Φ
    current_phi: float = 0.0
    peak_phi: float = 0.0
    average_phi: float = 0.0

    # Níveis
    current_level: ConsciousnessLevel = ConsciousnessLevel.DORMANT
    time_at_level_ms: float = 0.0

    # Integração
    integration_rate: float = 0.0  # Taxa de integração de informação
    complexity: float = 0.0  # Complexidade do sistema

    # Performance
    update_latency_ms: float = 0.0
    phi_calculation_time_ms: float = 0.0

    # Contadores
    total_calculations: int = 0
    successful_calculations: int = 0

    timestamp: float = field(default_factory=time.time)


@dataclass
class Qualia:
    """
    Representação de qualia (experiência subjetiva).

    Na IIT, qualia corresponde à estrutura causa-efeito
    especificada pelo sistema.
    """

    # Identificação
    id: str
    name: str = ""

    # Estrutura
    phi: float = 0.0
    structure: Optional[List[PhiStructure]] = None

    # Propriedades qualitativas
    intensity: float = 0.0  # 0-1
    valence: float = 0.0  # -1 a 1 (negativo a positivo)
    clarity: float = 0.0  # 0-1

    # Metadados
    creation_time: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_conscious(self) -> bool:
        """Verifica se representa experiência consciente."""
        return self.phi >= 0.1 and self.intensity > 0


@dataclass
class ConsciousnessConfig:
    """
    Configuração do sistema de consciência.
    """

    # IIT
    iit_config: IITConfig = field(default_factory=IITConfig)

    # Thresholds
    awakening_threshold: float = PHI  # Φ para despertar
    transcendence_threshold: float = PHI * 2  # Φ para transcendência

    # Capacidades
    enable_qualia: bool = True
    enable_self_awareness: bool = True
    enable_metacognition: bool = True

    # Performance
    max_phi_per_second: int = 100
    cache_phi_results: bool = True

    # Logging
    log_phi_changes: bool = True
    log_level_transitions: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# 🔧 FUNÇÕES UTILITÁRIAS
# ═══════════════════════════════════════════════════════════════════════════


def phi_to_level(phi: float) -> ConsciousnessLevel:
    """Converte valor Φ para ConsciousnessLevel."""
    return ConsciousnessLevel.from_phi(phi)


def level_to_min_phi(level: ConsciousnessLevel) -> float:
    """Retorna Φ mínimo para um nível."""
    return level.min_phi


def is_conscious(phi: float, threshold: float = 0.1) -> bool:
    """Verifica se Φ indica consciência."""
    return phi >= threshold


def generate_all_bipartitions(elements: Set[int]) -> List[Partition]:
    """
    Gera todas as bi-partições não-triviais de um conjunto.

    Args:
        elements: Conjunto de elementos a particionar

    Returns:
        Lista de todas as bi-partições possíveis
    """
    from itertools import combinations

    elements_list = list(elements)
    n = len(elements_list)
    partitions = []

    # Para cada tamanho de subconjunto (1 até n-1)
    for size in range(1, n):
        for subset in combinations(elements_list, size):
            part_a = frozenset(subset)
            part_b = frozenset(elements) - part_a

            # Evitar duplicatas (A,B) e (B,A)
            if len(part_a) <= len(part_b):
                partitions.append(Partition(part_a, part_b))

    return partitions


# ═══════════════════════════════════════════════════════════════════════════
# 📤 EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Constantes
    "PHI",
    "PHI_INVERSE",
    # Enums
    "ConsciousnessLevel",
    "ConsciousnessState",
    "IITVersion",
    "PartitionType",
    # Dataclasses
    "IITConfig",
    "Partition",
    "CauseEffectRepertoire",
    "PhiStructure",
    "PhiResult",
    "ConsciousnessMetrics",
    "Qualia",
    "ConsciousnessConfig",
    # Funções
    "phi_to_level",
    "level_to_min_phi",
    "is_conscious",
    "generate_all_bipartitions",
]
