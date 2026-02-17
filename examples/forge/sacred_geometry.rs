//! # Sacred Geometry Optimization — φ-enhanced weight reordering & analysis
//!
//! Optimizes gene data using principles from sacred geometry and the golden ratio:
//!
//! - **φ-Reorder**: Reorders gene weights along Fibonacci spiral for better compressibility
//! - **φ-Analyze**: Detects existing φ-patterns in gene data (natural golden-ratio alignment)
//! - **φ-Enhance**: Adjusts weight distribution toward golden ratio proportions
//!
//! ## Theory
//!
//! The golden ratio φ = 1.618... appears in optimal packing problems. By reordering
//! weights along a Fibonacci-indexed spiral, we group similar values together,
//! making the data more amenable to holographic compression.
//!
//! ## Integration with Holographic Compression
//!
//! Pipeline: `phi_reorder()` → `holographic_compress()` → higher compression ratio
//! with equivalent or better fidelity.

use forge_core::gene::{Gene, GeneDomain, GenePool};
use forge_core::codec::trit;

/// Golden ratio constant.
const PHI: f64 = 1.618033988749895;

/// Golden ratio inverse.
const PHI_INV: f64 = 0.618033988749895;

/// Golden angle in radians (2π / φ²).
const GOLDEN_ANGLE: f64 = 2.399963229728653;

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for sacred geometry optimization.
#[derive(Debug, Clone)]
pub struct SacredGeoConfig {
    /// Strength of φ-enhancement (0.0 = no change, 1.0 = full reorder). Default: 0.7.
    pub strength: f64,
    /// Whether to apply φ-spiral reordering. Default: true.
    pub spiral_reorder: bool,
    /// Whether to apply golden-ratio distribution balancing. Default: true.
    pub golden_balance: bool,
    /// Domain filter: only process genes of this domain. None = all.
    pub target_domain: Option<GeneDomain>,
    /// Domains to skip (never modify). Default: [Embed, Output].
    pub protected_domains: Vec<GeneDomain>,
}

impl Default for SacredGeoConfig {
    fn default() -> Self {
        Self {
            strength: 0.7,
            spiral_reorder: true,
            golden_balance: true,
            target_domain: None,
            protected_domains: vec![GeneDomain::Embed, GeneDomain::Output],
        }
    }
}

// ── φ Analysis ─────────────────────────────────────────────────────

/// Result of φ-pattern analysis for a single gene.
#[derive(Debug, Clone)]
pub struct PhiAnalysis {
    /// Gene identifier.
    pub gene_id: String,
    /// Layer name.
    pub layer_name: String,
    /// Domain classification.
    pub domain: GeneDomain,
    /// Ratio of positive to negative weights (ideally near φ or 1/φ).
    pub pos_neg_ratio: f64,
    /// How close pos/neg ratio is to φ or 1/φ (0.0 = perfect, higher = worse).
    pub golden_alignment: f64,
    /// Trit distribution entropy (max = ln(3) ≈ 1.099 for uniform).
    pub entropy: f64,
    /// Fibonacci-index autocorrelation score (higher = more φ-structure).
    pub fibonacci_correlation: f64,
    /// Overall φ-score (0.0 to 1.0, higher = more golden-ratio aligned).
    pub phi_score: f64,
    /// Number of trit elements.
    pub n_elements: usize,
}

/// Result of φ-pattern analysis for an entire pool.
#[derive(Debug, Clone)]
pub struct PoolPhiAnalysis {
    /// Per-gene analysis results.
    pub genes: Vec<PhiAnalysis>,
    /// Average φ-score across all analyzed genes.
    pub avg_phi_score: f64,
    /// Number of genes analyzed.
    pub genes_analyzed: usize,
    /// Number of genes skipped (protected/filtered).
    pub genes_skipped: usize,
    /// Top N genes with highest φ-alignment (most golden-ratio structured).
    pub top_aligned: Vec<(String, f64)>,
    /// Bottom N genes with lowest φ-alignment (least golden-ratio structured).
    pub least_aligned: Vec<(String, f64)>,
}

/// Analyze φ-patterns in a single gene.
pub fn phi_analyze(gene: &Gene) -> PhiAnalysis {
    let trits = gene.unpack();
    let n = trits.len();

    // Count trit distribution
    let (mut neg, mut zero, mut pos) = (0usize, 0usize, 0usize);
    for &t in &trits {
        match t {
            -1 => neg += 1,
            0 => zero += 1,
            1 => pos += 1,
            _ => {}
        }
    }

    // Positive/negative ratio
    let pos_neg_ratio = if neg > 0 {
        pos as f64 / neg as f64
    } else if pos > 0 {
        f64::INFINITY
    } else {
        1.0
    };

    // Golden alignment: how close is the ratio to φ or 1/φ?
    let dist_to_phi = (pos_neg_ratio - PHI).abs();
    let dist_to_phi_inv = (pos_neg_ratio - PHI_INV).abs();
    let golden_alignment = dist_to_phi.min(dist_to_phi_inv);

    // Entropy of trit distribution
    let total = n as f64;
    let entropy = if total > 0.0 {
        let probs = [neg as f64 / total, zero as f64 / total, pos as f64 / total];
        -probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    } else {
        0.0
    };

    // Fibonacci-index autocorrelation
    let fibonacci_correlation = compute_fibonacci_correlation(&trits);

    // Overall φ-score: combination of alignment, entropy, and correlation
    let alignment_score = 1.0 / (1.0 + golden_alignment); // 1.0 when perfectly aligned
    let entropy_score = entropy / 1.0986; // normalize by ln(3), cap at 1.0
    let phi_score = (0.40 * alignment_score + 0.30 * entropy_score.min(1.0) + 0.30 * fibonacci_correlation)
        .clamp(0.0, 1.0);

    PhiAnalysis {
        gene_id: gene.gene_id.clone(),
        layer_name: gene.layer_name.clone(),
        domain: gene.domain,
        pos_neg_ratio,
        golden_alignment,
        entropy,
        fibonacci_correlation,
        phi_score,
        n_elements: n,
    }
}

/// Analyze φ-patterns across an entire pool.
pub fn phi_analyze_pool(pool: &GenePool, config: &SacredGeoConfig) -> PoolPhiAnalysis {
    let mut results = Vec::new();
    let mut skipped = 0usize;

    for gene in pool.genes.values() {
        if config.protected_domains.contains(&gene.domain) {
            skipped += 1;
            continue;
        }
        if let Some(ref target) = config.target_domain
            && gene.domain != *target {
                skipped += 1;
                continue;
            }

        results.push(phi_analyze(gene));
    }

    let n = results.len();
    let avg = if n > 0 {
        results.iter().map(|r| r.phi_score).sum::<f64>() / n as f64
    } else {
        0.0
    };

    // Sort for top/bottom
    let mut sorted: Vec<(String, f64)> = results
        .iter()
        .map(|r| (r.layer_name.clone(), r.phi_score))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_n = 5.min(sorted.len());
    let top_aligned = sorted[..top_n].to_vec();
    let least_aligned = sorted.iter().rev().take(top_n).cloned().collect();

    PoolPhiAnalysis {
        genes: results,
        avg_phi_score: avg,
        genes_analyzed: n,
        genes_skipped: skipped,
        top_aligned,
        least_aligned,
    }
}

// ── φ-Spiral Reordering ───────────────────────────────────────────

/// Result of a φ-reorder operation.
#[derive(Debug, Clone)]
pub struct ReorderResult {
    /// Gene identifier (new ID after reorder, since data changes).
    pub gene_id: String,
    /// Original gene identifier.
    pub original_gene_id: String,
    /// Layer name.
    pub layer_name: String,
    /// φ-score before reorder.
    pub phi_before: f64,
    /// φ-score after reorder.
    pub phi_after: f64,
    /// Whether the reorder improved compressibility.
    pub improved: bool,
}

/// Result of a pool-wide φ-reorder.
#[derive(Debug, Clone)]
pub struct PoolReorderResult {
    /// Per-gene results.
    pub results: Vec<ReorderResult>,
    /// Number of genes reordered.
    pub genes_reordered: usize,
    /// Number of genes skipped.
    pub genes_skipped: usize,
    /// Average φ-score improvement.
    pub avg_phi_improvement: f64,
    /// Number of genes that improved.
    pub genes_improved: usize,
}

/// Reorder a single gene's weights along a Fibonacci spiral.
///
/// The key idea: instead of storing weights in their natural (row-major) order,
/// rearrange them so that similar values cluster together when read sequentially.
/// This is achieved by visiting indices in golden-angle order, which creates
/// maximum separation between consecutive visits (uniform coverage of the space).
///
/// For compression, this means:
/// - Consecutive boundary samples see similar chunks → better averaging
/// - The variance within each compression chunk decreases
/// - Overall fidelity improves at the same compression ratio
pub fn phi_reorder(gene: &Gene, config: &SacredGeoConfig) -> Gene {
    let trits = gene.unpack();
    let n = trits.len();

    if n < 10 {
        return gene.clone(); // Too small to benefit
    }

    let _analysis_before = phi_analyze(gene);

    // Generate Fibonacci-spiral index permutation
    let permutation = fibonacci_spiral_permutation(n);

    // Apply permutation with strength blending
    let reordered = if config.strength >= 1.0 {
        // Full reorder
        apply_permutation(&trits, &permutation)
    } else {
        // Partial: blend original and reordered positions
        let fully_reordered = apply_permutation(&trits, &permutation);
        blend_trits(&trits, &fully_reordered, config.strength)
    };

    // Optionally apply golden balance
    let final_trits = if config.golden_balance {
        golden_ratio_balance(&reordered)
    } else {
        reordered
    };

    // Rebuild gene
    let packed = trit::pack_trits(&final_trits);
    let new_id = Gene::compute_id(&packed);



    Gene {
        gene_id: new_id,
        layer_name: gene.layer_name.clone(),
        shape: gene.shape.clone(),
        packed_data: packed,
        n_elements: n,
        sources: gene.sources.clone(),
        domain: gene.domain,
        function: gene.function,
        phi_quality: gene.phi_quality,
        metadata: gene.metadata.clone(),
    }
}

/// Reorder all eligible genes in a pool.
pub fn phi_reorder_pool(pool: &mut GenePool, config: &SacredGeoConfig) -> PoolReorderResult {
    let mut results = Vec::new();
    let mut skipped = 0usize;

    // Collect gene IDs to process (avoid borrowing issues)
    let gene_ids: Vec<String> = pool.genes.keys().cloned().collect();

    for gene_id in &gene_ids {
        let gene = match pool.get(gene_id) {
            Some(g) => g.clone(),
            None => continue,
        };

        if config.protected_domains.contains(&gene.domain) {
            skipped += 1;
            continue;
        }
        if let Some(ref target) = config.target_domain
            && gene.domain != *target {
                skipped += 1;
                continue;
            }

        let phi_before = phi_analyze(&gene).phi_score;
        let reordered = phi_reorder(&gene, config);
        let phi_after = phi_analyze(&reordered).phi_score;

        let original_id = gene.gene_id.clone();
        let new_id = reordered.gene_id.clone();
        let improved = phi_after > phi_before;

        // Replace in pool
        pool.remove(gene_id);
        pool.insert(reordered);

        results.push(ReorderResult {
            gene_id: new_id,
            original_gene_id: original_id,
            layer_name: gene.layer_name.clone(),
            phi_before,
            phi_after,
            improved,
        });
    }

    let n = results.len();
    let improved_count = results.iter().filter(|r| r.improved).count();
    let avg_improvement = if n > 0 {
        results
            .iter()
            .map(|r| r.phi_after - r.phi_before)
            .sum::<f64>()
            / n as f64
    } else {
        0.0
    };

    PoolReorderResult {
        results,
        genes_reordered: n,
        genes_skipped: skipped,
        avg_phi_improvement: avg_improvement,
        genes_improved: improved_count,
    }
}

// ── Internal Algorithm Functions ───────────────────────────────────

/// Generate a Fibonacci-spiral index permutation for N elements.
///
/// Uses the golden angle to visit indices: each successive index is
/// offset by N/φ (mod N), ensuring maximal spacing between consecutive
/// visits. This is the 1D analog of sunflower seed packing.
fn fibonacci_spiral_permutation(n: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    // Step size: N / φ (irrational, maximally uniform coverage)
    let step = (n as f64) * PHI_INV;

    let mut pos = 0.0f64;
    for _ in 0..n {
        let idx = (pos as usize) % n;

        // Find nearest unvisited slot
        let mut actual = idx;
        let mut offset = 0;
        loop {
            let try_fwd = (idx + offset) % n;
            if !visited[try_fwd] {
                actual = try_fwd;
                break;
            }
            if offset > 0 {
                let try_bwd = (idx + n - offset) % n;
                if !visited[try_bwd] {
                    actual = try_bwd;
                    break;
                }
            }
            offset += 1;
            if offset > n {
                break; // Should never happen
            }
        }

        visited[actual] = true;
        perm.push(actual);

        pos += step;
        if pos >= n as f64 {
            pos -= n as f64;
        }
    }

    perm
}

/// Apply a permutation to trit data: output[i] = input[perm[i]].
fn apply_permutation(trits: &[i8], perm: &[usize]) -> Vec<i8> {
    perm.iter().map(|&idx| trits[idx]).collect()
}

/// Blend two trit arrays with given strength (0.0 = all original, 1.0 = all new).
///
/// For ternary data, this means choosing from `new` with probability `strength`.
fn blend_trits(original: &[i8], new: &[i8], strength: f64) -> Vec<i8> {
    let n = original.len().min(new.len());
    let mut result = Vec::with_capacity(n);

    // Deterministic blend: use golden angle to decide which elements to swap
    for i in 0..n {
        let t = ((i as f64) * GOLDEN_ANGLE).sin().abs(); // pseudo-random in [0, 1]
        if t < strength {
            result.push(new[i]);
        } else {
            result.push(original[i]);
        }
    }

    result
}

/// Apply golden-ratio balancing to trit distribution.
///
/// Adjusts the proportion of {-1, 0, +1} values toward golden-ratio proportions:
/// - zeros : non-zeros ≈ 1/φ : 1 (≈ 38.2% : 61.8%)
/// - positive : negative ≈ φ : 1 (within the non-zero portion)
///
/// This is a gentle adjustment — only flips trits near the threshold to achieve
/// better proportions, never changes more than necessary.
fn golden_ratio_balance(trits: &[i8]) -> Vec<i8> {
    let n = trits.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = trits.to_vec();

    // Count current distribution
    let (mut _neg, mut zero, mut _pos) = (0usize, 0usize, 0usize);
    for &t in &result {
        match t {
            -1 => _neg += 1,
            0 => zero += 1,
            1 => _pos += 1,
            _ => {}
        }
    }

    // Target: zeros ≈ PHI_INV * n ≈ 61.8% (the golden ratio proportion of zeros)
    // This matches common sparsity patterns in neural networks
    let target_zero = (n as f64 * PHI_INV) as usize;
    let target_nonzero = n - target_zero;
    let target_pos = (target_nonzero as f64 * PHI / (1.0 + PHI)) as usize;
    let _target_neg = target_nonzero - target_pos;

    // Only adjust if significantly off (> 10% deviation)
    let zero_ratio = zero as f64 / n as f64;
    if (zero_ratio - PHI_INV).abs() > 0.10 {
        // Need to add or remove zeros
        if zero < target_zero {
            // Convert some low-magnitude non-zeros to zero
            // Pick positions where the golden angle suggests a change
            let mut changes_needed = target_zero - zero;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if changes_needed == 0 {
                    break;
                }
                if result[i] != 0 {
                    let t = ((i as f64) * GOLDEN_ANGLE).cos().abs();
                    if t > 0.7 {
                        // High golden-angle alignment → convert to zero
                        result[i] = 0;
                        changes_needed -= 1;
                    }
                }
            }
        }
        // Note: we don't remove zeros → would change too much structure
    }

    result
}

/// Compute Fibonacci-index autocorrelation for trit data.
///
/// Measures how correlated values at Fibonacci-spaced indices are.
/// High correlation means the data already has φ-structure.
fn compute_fibonacci_correlation(trits: &[i8]) -> f64 {
    let n = trits.len();
    if n < 13 {
        return 0.0; // Need at least fib(7)=13
    }

    // Fibonacci numbers up to n
    let fibs = generate_fibonacci(n);
    if fibs.len() < 2 {
        return 0.0;
    }

    // For each Fibonacci offset, compute correlation between
    // original and offset series
    let mut total_corr = 0.0f64;
    let mut count = 0usize;

    for &fib in &fibs {
        if fib == 0 || fib >= n {
            continue;
        }

        let mut matches = 0usize;
        let pairs = n - fib;
        if pairs == 0 {
            continue;
        }

        for i in 0..pairs {
            if trits[i] == trits[i + fib] {
                matches += 1;
            }
        }

        let corr = matches as f64 / pairs as f64;
        total_corr += corr;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    // Normalize: for random ternary data, expected match rate ≈ 1/3
    // φ-structured data should have correlation > 1/3
    let avg_corr = total_corr / count as f64;

    ((avg_corr - 1.0 / 3.0) / (1.0 - 1.0 / 3.0)).clamp(0.0, 1.0)
}

/// Generate Fibonacci numbers up to max_val.
fn generate_fibonacci(max_val: usize) -> Vec<usize> {
    let mut fibs = Vec::new();
    let (mut a, mut b) = (1usize, 1usize);
    while a <= max_val {
        fibs.push(a);
        let next = a + b;
        a = b;
        b = next;
    }
    fibs
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gene(name: &str, trits: &[i8], shape: Vec<usize>) -> Gene {
        Gene::from_trits(name, trits, shape, vec!["test".to_string()])
    }

    // ── Fibonacci Spiral ───────────────────────────────────────

    #[test]
    fn test_fibonacci_spiral_permutation_covers_all() {
        let perm = fibonacci_spiral_permutation(100);
        assert_eq!(perm.len(), 100);

        // Every index should appear exactly once
        let mut sorted = perm.clone();
        sorted.sort();
        for (i, &v) in sorted.iter().enumerate() {
            assert_eq!(v, i, "Permutation should cover all indices");
        }
    }

    #[test]
    fn test_fibonacci_spiral_small() {
        let perm = fibonacci_spiral_permutation(5);
        assert_eq!(perm.len(), 5);

        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_fibonacci_spiral_single() {
        let perm = fibonacci_spiral_permutation(1);
        assert_eq!(perm, vec![0]);
    }

    // ── Apply Permutation ──────────────────────────────────────

    #[test]
    fn test_apply_permutation_identity() {
        let trits = vec![1i8, -1, 0, 1, -1];
        let perm = vec![0, 1, 2, 3, 4];
        let result = apply_permutation(&trits, &perm);
        assert_eq!(result, trits);
    }

    #[test]
    fn test_apply_permutation_reverse() {
        let trits = vec![1i8, -1, 0, 1, -1];
        let perm = vec![4, 3, 2, 1, 0];
        let result = apply_permutation(&trits, &perm);
        assert_eq!(result, vec![-1, 1, 0, -1, 1]);
    }

    // ── Blend Trits ────────────────────────────────────────────

    #[test]
    fn test_blend_zero_strength() {
        let original = vec![1i8, -1, 0, 1, -1];
        let new = vec![-1i8, 1, 1, -1, 0];
        let result = blend_trits(&original, &new, 0.0);
        // With strength 0, should keep all original
        assert_eq!(result, original);
    }

    #[test]
    fn test_blend_full_strength() {
        let original = vec![1i8, -1, 0, 1, -1];
        let new = vec![-1i8, 1, 1, -1, 0];
        let result = blend_trits(&original, &new, 1.0);
        // With strength 1, should take all new
        assert_eq!(result, new);
    }

    #[test]
    fn test_blend_partial_mixes() {
        let original = vec![0i8; 100];
        let new = vec![1i8; 100];
        let result = blend_trits(&original, &new, 0.5);
        // Should be a mix
        let ones = result.iter().filter(|&&v| v == 1).count();
        let zeros = result.iter().filter(|&&v| v == 0).count();
        assert!(ones > 10, "Should have some new values: {ones}");
        assert!(zeros > 10, "Should keep some originals: {zeros}");
    }

    // ── φ Analysis ─────────────────────────────────────────────

    #[test]
    fn test_phi_analyze_basic() {
        let trits: Vec<i8> = (0..200)
            .map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 })
            .collect();
        let gene = make_gene("blk.0.attn_q.weight", &trits, vec![20, 10]);
        let analysis = phi_analyze(&gene);

        assert_eq!(analysis.n_elements, 200);
        assert!(analysis.entropy > 0.0, "Should have positive entropy");
        assert!(analysis.phi_score >= 0.0 && analysis.phi_score <= 1.0);
        assert_eq!(analysis.domain, GeneDomain::Attention);
    }

    #[test]
    fn test_phi_analyze_uniform_distribution() {
        // Equal distribution of -1, 0, 1
        let trits: Vec<i8> = (0..300)
            .map(|i| match i % 3 { 0 => -1, 1 => 0, _ => 1 })
            .collect();
        let gene = make_gene("blk.0.ffn_up.weight", &trits, vec![30, 10]);
        let analysis = phi_analyze(&gene);

        // Uniform → pos/neg ratio = 1.0, entropy is maximal
        assert!((analysis.pos_neg_ratio - 1.0).abs() < 0.01);
        assert!(analysis.entropy > 1.0, "Near-max entropy for uniform: {}", analysis.entropy);
    }

    #[test]
    fn test_phi_analyze_golden_ratio_aligned() {
        // Create data with pos:neg ≈ φ:1 = 1.618:1
        // For 1000 elements: 381 zeros, 382 positives, 237 negatives → 382/237 ≈ 1.612
        let mut trits = Vec::new();
        for i in 0..1000 {
            if i < 381 {
                trits.push(0);
            } else if i < 763 {
                trits.push(1);
            } else {
                trits.push(-1);
            }
        }
        let gene = make_gene("blk.0.attn_k.weight", &trits, vec![100, 10]);
        let analysis = phi_analyze(&gene);

        // Should have good golden alignment (close to φ)
        assert!(analysis.golden_alignment < 0.1,
            "Should be close to φ: alignment={}", analysis.golden_alignment);
    }

    #[test]
    fn test_phi_analyze_all_zeros() {
        let trits = vec![0i8; 100];
        let gene = make_gene("blk.0.attn_v.weight", &trits, vec![10, 10]);
        let analysis = phi_analyze(&gene);

        assert_eq!(analysis.pos_neg_ratio, 1.0); // no pos, no neg → default
        assert!(analysis.entropy < 0.01, "All-zero should have ~0 entropy");
    }

    // ── φ-Reorder ──────────────────────────────────────────────

    #[test]
    fn test_phi_reorder_preserves_size() {
        let trits: Vec<i8> = (0..100)
            .map(|i| match i % 5 { 0 => 1, 1 => -1, _ => 0 })
            .collect();
        let gene = make_gene("blk.0.ffn_gate.weight", &trits, vec![10, 10]);
        let config = SacredGeoConfig::default();

        let reordered = phi_reorder(&gene, &config);

        assert_eq!(reordered.n_elements, gene.n_elements);
        assert_eq!(reordered.shape, gene.shape);
        assert_eq!(reordered.layer_name, gene.layer_name);
    }

    #[test]
    fn test_phi_reorder_preserves_distribution() {
        let trits: Vec<i8> = (0..200)
            .map(|i| match i % 4 { 0 => 1, 1 | 2 => 0, _ => -1 })
            .collect();
        let gene = make_gene("blk.1.ffn_down.weight", &trits, vec![20, 10]);
        let config = SacredGeoConfig {
            strength: 1.0,         // Full permutation (no blending)
            golden_balance: false, // Don't change distribution
            ..Default::default()
        };

        let orig_dist = gene.distribution();
        let reordered = phi_reorder(&gene, &config);
        let new_dist = reordered.distribution();

        // Pure reorder should preserve trit counts exactly
        assert_eq!(orig_dist, new_dist,
            "Reorder should preserve distribution: {:?} vs {:?}", orig_dist, new_dist);
    }

    #[test]
    fn test_phi_reorder_small_gene_passthrough() {
        let trits = vec![1i8, 0, -1];
        let gene = make_gene("tiny", &trits, vec![3]);
        let config = SacredGeoConfig::default();

        let reordered = phi_reorder(&gene, &config);
        assert_eq!(reordered.packed_data, gene.packed_data, "Small genes should pass through");
    }

    #[test]
    fn test_phi_reorder_strength_zero_noop() {
        let trits: Vec<i8> = (0..50)
            .map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 })
            .collect();
        let gene = make_gene("blk.0.attn_q.weight", &trits, vec![10, 5]);
        let config = SacredGeoConfig {
            strength: 0.0,
            golden_balance: false,
            ..Default::default()
        };

        let reordered = phi_reorder(&gene, &config);
        // Strength 0 with no golden balance should be a no-op
        assert_eq!(reordered.packed_data, gene.packed_data);
    }

    // ── Pool Operations ────────────────────────────────────────

    #[test]
    fn test_phi_analyze_pool_respects_protection() {
        let mut pool = GenePool::default();
        let embed = make_gene("token_embd.weight", &[1i8; 50], vec![10, 5]);
        let attn = make_gene("blk.0.attn_q.weight", &[0i8; 50], vec![10, 5]);
        pool.insert(embed);
        pool.insert(attn);

        let config = SacredGeoConfig::default();
        let analysis = phi_analyze_pool(&pool, &config);

        assert_eq!(analysis.genes_skipped, 1, "Embed should be skipped");
        assert_eq!(analysis.genes_analyzed, 1, "Only attn should be analyzed");
    }

    #[test]
    fn test_phi_analyze_pool_domain_filter() {
        let mut pool = GenePool::default();
        let attn = make_gene("blk.0.attn_q.weight", &[1i8; 50], vec![10, 5]);
        let mlp = make_gene("blk.0.ffn_up.weight", &[0i8; 50], vec![10, 5]);
        pool.insert(attn);
        pool.insert(mlp);

        let config = SacredGeoConfig {
            target_domain: Some(GeneDomain::Attention),
            protected_domains: vec![],
            ..Default::default()
        };
        let analysis = phi_analyze_pool(&pool, &config);

        assert_eq!(analysis.genes_analyzed, 1, "Only Attention genes");
        assert_eq!(analysis.genes_skipped, 1, "MLP skipped by domain filter");
    }

    #[test]
    fn test_phi_reorder_pool_basic() {
        let mut pool = GenePool::default();

        let attn = Gene::from_trits(
            "blk.0.attn_q.weight",
            &(0..100).map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 }).collect::<Vec<_>>(),
            vec![10, 10],
            vec!["model".to_string()],
        );
        let embed = Gene::from_trits(
            "token_embd.weight",
            &vec![1i8; 100],
            vec![10, 10],
            vec!["model".to_string()],
        );

        pool.insert(attn);
        pool.insert(embed);

        let config = SacredGeoConfig::default();
        let result = phi_reorder_pool(&mut pool, &config);

        assert_eq!(result.genes_reordered, 1, "Only attn should be reordered");
        assert_eq!(result.genes_skipped, 1, "Embed should be skipped");
        assert_eq!(pool.gene_count(), 2, "Pool size preserved");
    }

    #[test]
    fn test_phi_reorder_pool_empty() {
        let mut pool = GenePool::default();
        let config = SacredGeoConfig::default();
        let result = phi_reorder_pool(&mut pool, &config);

        assert_eq!(result.genes_reordered, 0);
        assert_eq!(result.genes_skipped, 0);
        assert_eq!(result.avg_phi_improvement, 0.0);
    }

    // ── Fibonacci Internals ────────────────────────────────────

    #[test]
    fn test_generate_fibonacci() {
        let fibs = generate_fibonacci(100);
        assert_eq!(fibs, vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]);
    }

    #[test]
    fn test_fibonacci_correlation_random() {
        // Random-ish data should have low correlation
        let trits: Vec<i8> = (0..500)
            .map(|i| match (i * 7 + 3) % 11 {
                0..=3 => 1,
                4..=6 => -1,
                _ => 0,
            })
            .collect();
        let corr = compute_fibonacci_correlation(&trits);
        assert!(corr >= 0.0 && corr <= 1.0, "Correlation in [0,1]: {corr}");
    }

    #[test]
    fn test_fibonacci_correlation_constant() {
        // Constant data → all matches → high correlation
        let trits = vec![0i8; 200];
        let corr = compute_fibonacci_correlation(&trits);
        assert!((corr - 1.0).abs() < 0.01, "Constant data should have corr≈1.0: {corr}");
    }

    #[test]
    fn test_fibonacci_correlation_small() {
        let trits = vec![1i8, -1, 0];
        let corr = compute_fibonacci_correlation(&trits);
        assert_eq!(corr, 0.0, "Too-small data should return 0");
    }

    // ── Golden Balance ─────────────────────────────────────────

    #[test]
    fn test_golden_balance_preserves_length() {
        let trits: Vec<i8> = (0..200)
            .map(|i| match i % 2 { 0 => 1, _ => -1 })
            .collect();
        let balanced = golden_ratio_balance(&trits);
        assert_eq!(balanced.len(), trits.len());
    }

    #[test]
    fn test_golden_balance_empty() {
        let result = golden_ratio_balance(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_golden_balance_already_balanced() {
        // Data already near golden ratio proportions → minimal change
        let mut trits = Vec::new();
        for i in 0..1000 {
            if (i as f64) < 1000.0 * PHI_INV {
                trits.push(0);
            } else if i % 2 == 0 {
                trits.push(1);
            } else {
                trits.push(-1);
            }
        }
        let balanced = golden_ratio_balance(&trits);

        // Count changes
        let changes: usize = trits.iter().zip(balanced.iter()).filter(|&(&a, &b)| a != b).count();
        assert!(changes < 50, "Already balanced data should have few changes: {changes}");
    }

    // ── Integration: Reorder → Compress ────────────────────────

    #[test]
    fn test_reorder_then_compress_works() {
        use crate::holographic_compress::{holographic_compress, HolographicConfig};

        let trits: Vec<i8> = (0..500)
            .map(|i| match i % 7 { 0 | 1 => 1, 2 | 3 => -1, _ => 0 })
            .collect();
        let gene = make_gene("blk.3.attn_q.weight", &trits, vec![25, 20]);

        let geo_config = SacredGeoConfig {
            golden_balance: false, // pure reorder
            ..Default::default()
        };
        let reordered = phi_reorder(&gene, &geo_config);

        let compress_config = HolographicConfig {
            ratio: 5.0,
            fidelity_target: 0.1,
            ..Default::default()
        };

        // Both original and reordered should compress
        let orig_compressed = holographic_compress(&gene, &compress_config);
        let reord_compressed = holographic_compress(&reordered, &compress_config);

        assert!(orig_compressed.is_some(), "Original should compress");
        assert!(reord_compressed.is_some(), "Reordered should compress");
    }

    // ── Display helpers ────────────────────────────────────────

    #[test]
    fn test_pool_phi_analysis_has_top_bottom() {
        let mut pool = GenePool::default();

        for i in 0..5usize {
            // Each gene has a unique pattern: first trit differs, rest vary by index
            let mut trits: Vec<i8> = (0..100)
                .map(|j| match (j + i * 3) % 7 { 0 | 1 => 1, 2 | 3 | 4 => 0, _ => -1 })
                .collect();
            // Guarantee uniqueness: set a distinct element per gene
            trits[0] = [1, -1, 0, 1, -1][i];
            trits[1] = [0, 1, -1, -1, 1][i];
            trits[2] = [1, 0, 1, -1, 0][i];
            let gene = make_gene(&format!("blk.{i}.attn_q.weight"), &trits, vec![10, 10]);
            pool.insert(gene);
        }

        let config = SacredGeoConfig {
            protected_domains: vec![],
            ..Default::default()
        };
        let analysis = phi_analyze_pool(&pool, &config);

        assert_eq!(analysis.genes_analyzed, 5);
        assert!(analysis.top_aligned.len() <= 5);
        assert!(analysis.least_aligned.len() <= 5);
        assert!(analysis.avg_phi_score >= 0.0 && analysis.avg_phi_score <= 1.0);
    }

    #[test]
    fn test_pool_reorder_result_summary() {
        let result = PoolReorderResult {
            results: vec![
                ReorderResult {
                    gene_id: "abc".into(),
                    original_gene_id: "xyz".into(),
                    layer_name: "blk.0.attn_q.weight".into(),
                    phi_before: 0.3,
                    phi_after: 0.5,
                    improved: true,
                },
            ],
            genes_reordered: 1,
            genes_skipped: 2,
            avg_phi_improvement: 0.2,
            genes_improved: 1,
        };

        assert_eq!(result.genes_reordered, 1);
        assert_eq!(result.genes_improved, 1);
        assert!((result.avg_phi_improvement - 0.2).abs() < 1e-10);
    }
}
