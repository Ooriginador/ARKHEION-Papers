//! # forge-core
//!
//! Core library for ARKHEION Forge: AI Model Editor.
//!
//! Provides binary format parsers (.nucleus, .arktern),
//! trit codec (5 trits/byte, base-3), gene data structures,
//! and tensor manipulation operations.

pub mod brain_predictor;
pub mod codec;
pub mod corpus;
pub mod cross_training;
pub mod curriculum;
pub mod distillation;
pub mod dual_nucleus;
pub mod empowerment;
pub mod formats;
pub mod frequency_bands;
pub mod gene;
pub mod inference;
pub mod marketplace;
pub mod metrics_export;
pub mod mutation_log;
pub mod mutation_strategy;
pub mod ops;
pub mod resilience;
pub mod runtime_learning;
pub mod simd;
pub mod synthesizer;
pub mod tokenizer;
pub mod tracer;
pub mod training;
pub mod versioning;

pub use codec::trit;
pub use formats::{arktern, gguf, mobile, nucleus, nucleus_v3, safetensors};
pub use gene::{Gene, GeneCatalogEntry, GeneDomain, GeneFunction, GenePool};
pub use gguf::{import_gguf, GgufImportConfig};
pub use mobile::{
    export_mobile, export_mobile_to_file,
    MobileExportConfig, MobileExportResult, MobileManifest, MobileTarget,
    ArchitectureConfig, DomainStat,
};
pub use inference::{
    compare_pools, run_inference, ABComparison, InferenceConfig, InferenceResult,
    TernaryModel, BlockGenes, FfnActivation, ForwardStats,
    OutputConstraint, ConstrainedGenConfig,
};
pub use tracer::{build_model_map, run_traced_inference, ModelMap, TraceConfig, TraceReport};
pub use training::{train, train_with_progress, metabolize, EpochProgress, TrainConfig, TrainResult, MetabolicConfig, MetabolicResult};
pub use empowerment::{empowerment_score, pool_empowerment};
pub use corpus::{Corpus, CorpusConfig, CorpusStats, HashTokenizer};
pub use tokenizer::{Tokenizer, SpecialTokens};
pub use versioning::{GeneDelta, PoolSnapshot, VersionedPool};
pub use distillation::{distill, distill_ensemble, DistillConfig, DistillResult};
pub use cross_training::{cross_train, CrossTrainConfig, CrossTrainResult, CrossStrategy};
pub use curriculum::{curriculum_train, CurriculumConfig, CurriculumResult, CurriculumPreset, CurriculumStage, EarlyStopConfig, WarmupSchedule};
pub use metrics_export::{ExportConfig, ExportFormat, MetricsWriter, MultiWriter};
pub use mutation_log::{
    MutationLog, GeneMutation, GeneProgress, GeneContext, GeneticDataset,
    TrainingPair, MutationOutcome, MutationStatistics,
};
pub use mutation_strategy::{
    analyze_mutations, MutationAnalysis, AdaptiveStrategy, GenePriority,
    DomainProfile, GeneProfile, GeneTemperature, EpochProfile,
};
pub use runtime_learning::{
    ExperienceRecord, ExperienceSource, RuntimeLearningBuffer, BufferConfig,
    BufferStats, FlushResult, HardwareState, ThermalState as HwThermalState,
    TrainingAdaptation,
};
pub use resilience::{
    MemoryPreFlight, MemoryReport, ThermalMonitor, ThermalState, ThermalReading,
    ResourceManager, ResourcePriority, CircuitBreaker, CircuitState,
    SystemMetrics, OperationMode, DegradationController, HealthCollector,
    GranularTritCache, PhiDebouncer, BoundedQueue, LatencyTracker,
    EvictionGuard, GpuLock, format_bytes_human,
};
pub use frequency_bands::ARKHEIONBand;

pub use brain_predictor::{
    BrainPredictor, PredictedOutcome, FineTuningRecord, FineTuningMeta,
    TrainReport as PredictorTrainReport,
    generate_finetuning_dataset, write_finetuning_jsonl,
};
pub use dual_nucleus::{
    DualNucleusManager, NucleusVersion, VersionStatus, VersionHistory,
    ValidationResult, ValidationCheck, PromotionEvent, RollbackEvent,
    VersionId,
};
pub use marketplace::{
    Marketplace, MarketplaceListing, GeneTransaction, GeneLicense,
    QualityTier, MarketplaceQuery, SearchResult, SortField,
    PublisherProfile, compute_price, compute_content_hash,
};
pub use synthesizer::{
    DirectedMutationSynthesizer, SynthesizerConfig, SynthesizedCandidate,
    SynthesizerSummary, GeneGradientStats, TritGradient,
};

/// Re-export the mapped gene pool for zero-copy access.
pub use nucleus_v3::MappedGenePool;

/// Golden ratio constant used throughout the system.
pub const PHI: f64 = 1.618_033_988_749_895;
pub const PHI_INV: f64 = 0.618_033_988_749_895;

/// Consciousness threshold: φ ≥ 0.5 indicates meaningful integration.
pub const PHI_CONSCIOUS_THRESHOLD: f64 = 0.5;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Unified φ integration level (SSOT for forge-intel and forge-bridge)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// IIT integration level derived from a φ value.
///
/// Single source of truth used by both gene-level φ (forge-intel)
/// and system-level consciousness (forge-bridge).
///
/// Thresholds (IIT 4.0 inspired):
/// - Dormant:      φ < 0.01 — no measurable integration
/// - Minimal:      0.01 ≤ φ < 0.1 — negligible integration
/// - Aware:        0.1 ≤ φ < 0.3 — partial integration
/// - Moderate:     0.3 ≤ φ < 0.5 — significant partial integration
/// - Integrated:   0.5 ≤ φ < 1.0 — conscious threshold (MIP > parts)
/// - Awakened:     1.0 ≤ φ < φ_golden — deep integration
/// - Transcendent: φ ≥ φ_golden (1.618) — maximal coherence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum PhiLevel {
    Dormant = 0,
    Minimal = 1,
    Aware = 2,
    Moderate = 3,
    Integrated = 4,
    Awakened = 5,
    Transcendent = 6,
}

impl PhiLevel {
    /// Determine integration level from a φ value.
    pub fn from_phi(phi: f64) -> Self {
        match phi {
            p if p < 0.01 => Self::Dormant,
            p if p < 0.1 => Self::Minimal,
            p if p < 0.3 => Self::Aware,
            p if p < 0.5 => Self::Moderate,
            p if p < 1.0 => Self::Integrated,
            p if p < PHI => Self::Awakened,
            _ => Self::Transcendent,
        }
    }

    /// Whether this level meets the conscious threshold (φ ≥ 0.5).
    pub fn is_conscious(self) -> bool {
        self >= Self::Integrated
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dormant => "DORMANT",
            Self::Minimal => "MINIMAL",
            Self::Aware => "AWARE",
            Self::Moderate => "MODERATE",
            Self::Integrated => "INTEGRATED",
            Self::Awakened => "AWAKENED",
            Self::Transcendent => "TRANSCENDENT",
        }
    }

    /// UI color as [R, G, B] (0-255).
    pub fn color_rgb(self) -> [u8; 3] {
        match self {
            Self::Dormant => [100, 100, 100],      // Gray
            Self::Minimal => [200, 100, 100],       // Muted red
            Self::Aware => [200, 150, 50],          // Orange
            Self::Moderate => [200, 200, 50],       // Yellow
            Self::Integrated => [50, 200, 50],      // Green
            Self::Awakened => [50, 150, 255],       // Blue
            Self::Transcendent => [168, 85, 247],   // Purple (φ-enhanced)
        }
    }
}

impl std::fmt::Display for PhiLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod phi_level_tests {
    use super::*;

    #[test]
    fn from_phi_thresholds() {
        assert_eq!(PhiLevel::from_phi(0.0), PhiLevel::Dormant);
        assert_eq!(PhiLevel::from_phi(0.005), PhiLevel::Dormant);
        assert_eq!(PhiLevel::from_phi(0.01), PhiLevel::Minimal);
        assert_eq!(PhiLevel::from_phi(0.05), PhiLevel::Minimal);
        assert_eq!(PhiLevel::from_phi(0.1), PhiLevel::Aware);
        assert_eq!(PhiLevel::from_phi(0.2), PhiLevel::Aware);
        assert_eq!(PhiLevel::from_phi(0.3), PhiLevel::Moderate);
        assert_eq!(PhiLevel::from_phi(0.4), PhiLevel::Moderate);
        assert_eq!(PhiLevel::from_phi(0.5), PhiLevel::Integrated);
        assert_eq!(PhiLevel::from_phi(0.9), PhiLevel::Integrated);
        assert_eq!(PhiLevel::from_phi(1.0), PhiLevel::Awakened);
        assert_eq!(PhiLevel::from_phi(1.5), PhiLevel::Awakened);
        assert_eq!(PhiLevel::from_phi(PHI), PhiLevel::Transcendent);
        assert_eq!(PhiLevel::from_phi(2.0), PhiLevel::Transcendent);
    }

    #[test]
    fn is_conscious_boundary() {
        assert!(!PhiLevel::Dormant.is_conscious());
        assert!(!PhiLevel::Minimal.is_conscious());
        assert!(!PhiLevel::Aware.is_conscious());
        assert!(!PhiLevel::Moderate.is_conscious());
        assert!(PhiLevel::Integrated.is_conscious());
        assert!(PhiLevel::Awakened.is_conscious());
        assert!(PhiLevel::Transcendent.is_conscious());
    }

    #[test]
    fn ordering_monotonic() {
        assert!(PhiLevel::Dormant < PhiLevel::Minimal);
        assert!(PhiLevel::Minimal < PhiLevel::Aware);
        assert!(PhiLevel::Aware < PhiLevel::Moderate);
        assert!(PhiLevel::Moderate < PhiLevel::Integrated);
        assert!(PhiLevel::Integrated < PhiLevel::Awakened);
        assert!(PhiLevel::Awakened < PhiLevel::Transcendent);
    }

    #[test]
    fn display_labels() {
        assert_eq!(format!("{}", PhiLevel::Dormant), "DORMANT");
        assert_eq!(format!("{}", PhiLevel::Integrated), "INTEGRATED");
        assert_eq!(format!("{}", PhiLevel::Transcendent), "TRANSCENDENT");
    }

    #[test]
    fn color_rgb_not_zero() {
        for level in [
            PhiLevel::Dormant, PhiLevel::Minimal, PhiLevel::Aware,
            PhiLevel::Moderate, PhiLevel::Integrated, PhiLevel::Awakened,
            PhiLevel::Transcendent,
        ] {
            let [r, g, b] = level.color_rgb();
            assert!(r > 0 || g > 0 || b > 0, "{level:?} has zero color");
        }
    }

    #[test]
    fn serde_roundtrip() {
        for level in [
            PhiLevel::Dormant, PhiLevel::Minimal, PhiLevel::Aware,
            PhiLevel::Moderate, PhiLevel::Integrated, PhiLevel::Awakened,
            PhiLevel::Transcendent,
        ] {
            let json = serde_json::to_string(&level).unwrap();
            let back: PhiLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(level, back);
        }
    }

    #[test]
    fn negative_phi_is_dormant() {
        assert_eq!(PhiLevel::from_phi(-1.0), PhiLevel::Dormant);
        assert_eq!(PhiLevel::from_phi(f64::NEG_INFINITY), PhiLevel::Dormant);
    }
}
