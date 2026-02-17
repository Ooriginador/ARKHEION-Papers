/**
 * ARKHEION Advanced φ-Optimization System Implementation
 * =====================================================
 * 
 * Implementação completa do sistema de otimização baseado na proporção áurea.
 * Inclui algoritmos genéticos, geometria sagrada, e sincronização temporal.
 * 
 * Author: ARKHEION Team
 * Date: 2025-09-10
 * Version: 1.0.0
 */

#include "arkheion_phi_optimization.hpp"
#include "arkheion_consciousness_interface.hpp"
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>
#include <unordered_set>
#include <numeric>
#include <iostream>

namespace arkheion {

// Static random generator for optimization
static std::random_device rd;
static std::mt19937 gen(rd());

// ============================================================================
// Golden Spiral Generator Implementation
// ============================================================================

std::vector<glm::vec3> GoldenSpiralGenerator::generateSpiral(
    int point_count,
    const SpiralParams& params
) {
    std::vector<glm::vec3> points;
    points.reserve(point_count);
    
    for (int i = 0; i < point_count; ++i) {
        float t = static_cast<float>(i) / params.point_density;
        
        // Golden spiral in polar coordinates
        float radius = params.radius_factor * std::exp(t * FIBONACCI_SPIRAL_RATE);
        float angle = t * GOLDEN_ANGLE + params.rotation_offset;
        float height = params.height_factor * t * params.pitch;
        
        // Convert to Cartesian coordinates
        float x = radius * std::cos(angle);
        float y = radius * std::sin(angle);
        float z = height;
        
        points.emplace_back(x, y, z);
    }
    
    return points;
}

float GoldenSpiralGenerator::calculateSpiralParameter(const glm::vec3& point, const glm::vec3& center) {
    glm::vec3 relative = point - center;
    float radius = glm::length(glm::vec2(relative.x, relative.y));
    
    if (radius < 1e-6f) return 0.0f;
    
    // Calculate parameter t from radius using inverse of spiral equation
    float t = std::log(radius) / FIBONACCI_SPIRAL_RATE;
    return std::max(0.0f, t);
}

float GoldenSpiralGenerator::getSpiralRadius(float parameter) {
    return std::exp(parameter * FIBONACCI_SPIRAL_RATE);
}

float GoldenSpiralGenerator::getSpiralAngle(float parameter) {
    return parameter * GOLDEN_ANGLE;
}

// ============================================================================
// Sacred Geometry Generator Implementation
// ============================================================================

std::vector<glm::vec3> SacredGeometryGenerator::generateFlowerOfLife(
    int layers,
    float radius,
    const glm::vec3& center
) {
    std::vector<glm::vec3> points;
    
    // Central circle
    points.push_back(center);
    
    // Generate concentric layers
    for (int layer = 1; layer <= layers; ++layer) {
        int circles_in_layer = 6 * layer;
        float layer_radius = radius * layer;
        
        for (int i = 0; i < circles_in_layer; ++i) {
            float angle = (2.0f * M_PI * i) / circles_in_layer;
            float x = center.x + layer_radius * std::cos(angle);
            float y = center.y + layer_radius * std::sin(angle);
            float z = center.z;
            
            points.emplace_back(x, y, z);
        }
    }
    
    return points;
}

std::vector<glm::vec3> SacredGeometryGenerator::generateVesicaPiscis(
    float radius,
    int point_density,
    const glm::vec3& center
) {
    std::vector<glm::vec3> points;
    
    // Two intersecting circles
    glm::vec3 circle1_center = center + glm::vec3(-radius * 0.5f, 0.0f, 0.0f);
    glm::vec3 circle2_center = center + glm::vec3(radius * 0.5f, 0.0f, 0.0f);
    
    // Generate points on both circles
    for (int i = 0; i < point_density; ++i) {
        float angle = (2.0f * M_PI * i) / point_density;
        
        // Circle 1
        float x1 = circle1_center.x + radius * std::cos(angle);
        float y1 = circle1_center.y + radius * std::sin(angle);
        points.emplace_back(x1, y1, center.z);
        
        // Circle 2
        float x2 = circle2_center.x + radius * std::cos(angle);
        float y2 = circle2_center.y + radius * std::sin(angle);
        points.emplace_back(x2, y2, center.z);
    }
    
    return points;
}

std::vector<glm::vec3> SacredGeometryGenerator::generatePentagram(
    float radius,
    const glm::vec3& center
) {
    std::vector<glm::vec3> points;
    
    // Five points of the pentagram
    for (int i = 0; i < 5; ++i) {
        float angle = (2.0f * M_PI * i) / 5.0f - M_PI / 2.0f; // Start from top
        float x = center.x + radius * std::cos(angle);
        float y = center.y + radius * std::sin(angle);
        points.emplace_back(x, y, center.z);
    }
    
    // Inner pentagon points (φ ratio)
    float inner_radius = radius * INV_PHI * INV_PHI;
    for (int i = 0; i < 5; ++i) {
        float angle = (2.0f * M_PI * i) / 5.0f - M_PI / 2.0f + M_PI / 5.0f; // Rotated
        float x = center.x + inner_radius * std::cos(angle);
        float y = center.y + inner_radius * std::sin(angle);
        points.emplace_back(x, y, center.z);
    }
    
    return points;
}

std::vector<glm::vec3> SacredGeometryGenerator::generatePlatonicSolid(
    SacredPattern type,
    float scale,
    const glm::vec3& center
) {
    std::vector<glm::vec3> points;
    
    switch (type) {
        case SacredPattern::PLATONIC_SOLIDS: {
            // Tetrahedron vertices
            float a = scale;
            points.push_back(center + glm::vec3(a, a, a));
            points.push_back(center + glm::vec3(a, -a, -a));
            points.push_back(center + glm::vec3(-a, a, -a));
            points.push_back(center + glm::vec3(-a, -a, a));
            break;
        }
        default:
            // Default to tetrahedron
            points = generatePlatonicSolid(SacredPattern::PLATONIC_SOLIDS, scale, center);
            break;
    }
    
    return points;
}

// ============================================================================
// φ-Genetic Optimizer Implementation
// ============================================================================

PhiGeneticOptimizer::PhiGeneticOptimizer(const PhiOptimizationParams& params)
    : params(params), generation(0) {
    
    // Initialize population based on quality level
    int population_size = 20; // Base size
    switch (params.quality_level) {
        case PhiQualityLevel::FAST: population_size = 10; break;
        case PhiQualityLevel::BALANCED: population_size = 20; break;
        case PhiQualityLevel::HIGH: population_size = 50; break;
        case PhiQualityLevel::ULTRA: population_size = 100; break;
    }
    
    population.resize(population_size);
}

void PhiGeneticOptimizer::initialize(const std::vector<glm::vec3>& initial_points) {
    for (auto& chromosome : population) {
        chromosome.positions = initial_points;
        
        // Add random variations
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
        for (auto& pos : chromosome.positions) {
            pos.x += dis(gen) * params.pattern_scale;
            pos.y += dis(gen) * params.pattern_scale;
            pos.z += dis(gen) * params.pattern_scale * 0.5f; // Less variation in Z
        }
        
        chromosome.fitness = calculateFitness(chromosome.positions);
    }
    
    // Sort by fitness
    std::sort(population.begin(), population.end(),
        [](const Chromosome& a, const Chromosome& b) {
            return a.fitness > b.fitness;
        });
}

bool PhiGeneticOptimizer::evolve() {
    if (generation >= params.max_iterations) {
        return true; // Max generations reached
    }
    
    // Selection
    selection();
    
    // Crossover
    crossover();
    
    // Mutation
    mutation();
    
    // Evaluate fitness
    for (auto& chromosome : population) {
        chromosome.fitness = calculateFitness(chromosome.positions);
    }
    
    // Sort by fitness
    std::sort(population.begin(), population.end(),
        [](const Chromosome& a, const Chromosome& b) {
            return a.fitness > b.fitness;
        });
    
    generation++;
    
    // Check convergence
    if (generation > 10) {
        float best_fitness = population[0].fitness;
        float avg_fitness = 0.0f;
        for (const auto& chromosome : population) {
            avg_fitness += chromosome.fitness;
        }
        avg_fitness /= population.size();
        
        float convergence = (best_fitness - avg_fitness) / best_fitness;
        return convergence < params.convergence_threshold;
    }
    
    return false;
}

std::vector<glm::vec3> PhiGeneticOptimizer::getBestSolution() const {
    if (population.empty()) return {};
    return population[0].positions;
}

PhiMetrics PhiGeneticOptimizer::getBestMetrics() const {
    if (population.empty()) return PhiMetrics{};
    return population[0].metrics;
}

float PhiGeneticOptimizer::calculateFitness(const std::vector<glm::vec3>& positions) {
    float phi_compliance = evaluatePhiCompliance(positions);
    
    // Add other fitness components based on mode
    float consciousness_factor = 1.0f;
    float aesthetic_factor = 1.0f;
    float performance_factor = 1.0f;
    
    // Weighted fitness
    float fitness = phi_compliance * params.phi_weight +
                   consciousness_factor * params.consciousness_weight +
                   aesthetic_factor * params.aesthetic_weight +
                   performance_factor * params.performance_weight;
    
    return fitness / (params.phi_weight + params.consciousness_weight + 
                     params.aesthetic_weight + params.performance_weight);
}

void PhiGeneticOptimizer::selection() {
    // Tournament selection
    int tournament_size = std::max(2, static_cast<int>(population.size() / 10));
    std::vector<Chromosome> new_population;
    new_population.reserve(population.size());
    
    std::uniform_int_distribution<> dis(0, population.size() - 1);
    
    for (size_t i = 0; i < population.size(); ++i) {
        // Tournament
        Chromosome best = population[dis(gen)];
        for (int j = 1; j < tournament_size; ++j) {
            Chromosome candidate = population[dis(gen)];
            if (candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        new_population.push_back(best);
    }
    
    population = std::move(new_population);
}

void PhiGeneticOptimizer::crossover() {
    std::uniform_real_distribution<float> prob_dis(0.0f, 1.0f);
    std::uniform_int_distribution<> index_dis(0, population.size() - 1);
    
    for (size_t i = 0; i < population.size(); i += 2) {
        if (prob_dis(gen) < 0.8f) { // Crossover probability
            size_t parent1_idx = i;
            size_t parent2_idx = std::min(i + 1, population.size() - 1);
            
            auto& parent1 = population[parent1_idx];
            auto& parent2 = population[parent2_idx];
            
            // Single-point crossover
            if (!parent1.positions.empty() && !parent2.positions.empty()) {
                size_t crossover_point = index_dis(gen) % parent1.positions.size();
                
                for (size_t j = crossover_point; j < parent1.positions.size(); ++j) {
                    std::swap(parent1.positions[j], parent2.positions[j]);
                }
            }
        }
    }
}

void PhiGeneticOptimizer::mutation() {
    std::uniform_real_distribution<float> prob_dis(0.0f, 1.0f);
    std::uniform_real_distribution<float> mutation_dis(-0.1f, 0.1f);
    
    for (auto& chromosome : population) {
        for (auto& position : chromosome.positions) {
            if (prob_dis(gen) < 0.1f) { // Mutation probability
                position.x += mutation_dis(gen) * params.pattern_scale;
                position.y += mutation_dis(gen) * params.pattern_scale;
                position.z += mutation_dis(gen) * params.pattern_scale * 0.5f;
            }
        }
    }
}

float PhiGeneticOptimizer::evaluatePhiCompliance(const std::vector<glm::vec3>& positions) {
    if (positions.size() < 2) return 0.0f;
    
    float total_compliance = 0.0f;
    int comparisons = 0;
    
    // Calculate center of mass
    glm::vec3 center(0.0f);
    for (const auto& pos : positions) {
        center += pos;
    }
    center /= static_cast<float>(positions.size());
    
    // Check φ ratios between points
    for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            float distance = glm::distance(positions[i], positions[j]);
            float distance_to_center_i = glm::distance(positions[i], center);
            float distance_to_center_j = glm::distance(positions[j], center);
            
            if (distance > 1e-6f && distance_to_center_i > 1e-6f) {
                float ratio = distance / distance_to_center_i;
                
                // Check if ratio is close to φ or 1/φ
                float phi_error = std::min(
                    std::abs(ratio - PHI),
                    std::abs(ratio - INV_PHI)
                );
                
                float compliance = std::exp(-phi_error * 10.0f); // Exponential decay
                total_compliance += compliance;
                comparisons++;
            }
        }
    }
    
    return comparisons > 0 ? total_compliance / comparisons : 0.0f;
}

// ============================================================================
// Main φ-Optimization System Implementation
// ============================================================================

ARKHEIONPhiOptimizationSystem::ARKHEIONPhiOptimizationSystem() {
    // Initialize default parameters
    params = PhiOptimizationParams{};
}

ARKHEIONPhiOptimizationSystem::~ARKHEIONPhiOptimizationSystem() {
    shutdown();
}

bool ARKHEIONPhiOptimizationSystem::initialize(const PhiOptimizationParams& init_params) {
    params = init_params;
    
    // Initialize genetic optimizer
    genetic_optimizer = std::make_unique<PhiGeneticOptimizer>(params);
    
    // Initialize consciousness interface if needed
    if (params.mode == PhiOptimizationMode::CONSCIOUSNESS_AWARE ||
        params.mode == PhiOptimizationMode::FULL_INTEGRATION) {
        
        consciousness_interface = std::make_unique<ARKHEIONConsciousnessInterface>();
        if (!consciousness_interface->initialize()) {
            std::cerr << "Warning: Consciousness interface failed to initialize" << std::endl;
            consciousness_interface.reset();
        }
    }
    
    std::cout << "✅ ARKHEION φ-Optimization System initialized" << std::endl;
    return true;
}

void ARKHEIONPhiOptimizationSystem::shutdown() {
    stopRealtimeOptimization();
    
    genetic_optimizer.reset();
    consciousness_interface.reset();
    
    // Clear cache
    std::lock_guard<std::mutex> lock(cache_mutex);
    metrics_cache.clear();
}

PhiMetrics ARKHEIONPhiOptimizationSystem::optimizePoints(
    std::vector<PhiOptimizedPoint>& points,
    float time_factor
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PhiMetrics metrics{};
    
    // Check cache first
    uint64_t points_hash = calculatePointsHash(points);
    if (getCachedMetrics(points_hash, metrics)) {
        return metrics;
    }
    
    // Extract positions for optimization
    std::vector<glm::vec3> positions;
    positions.reserve(points.size());
    for (const auto& point : points) {
        positions.push_back(point.position);
    }
    
    // Optimize based on mode
    switch (params.mode) {
        case PhiOptimizationMode::SPATIAL_ONLY:
            metrics = optimizeSpatialDistribution(points);
            break;
            
        case PhiOptimizationMode::TEMPORAL_ONLY:
            metrics = optimizeTemporalSynchronization(points, time_factor);
            break;
            
        case PhiOptimizationMode::CONSCIOUSNESS_AWARE:
            if (consciousness_interface) {
                // Get consciousness metrics first
                // Then optimize with consciousness
                metrics = optimizeSpatialDistribution(points);
                // Add consciousness optimization here
            } else {
                metrics = optimizeSpatialDistribution(points);
            }
            break;
            
        case PhiOptimizationMode::FULL_INTEGRATION:
            metrics = optimizeSpatialDistribution(points);
            PhiMetrics temporal_metrics = optimizeTemporalSynchronization(points, time_factor);
            
            // Combine metrics
            metrics.temporal_phi_sync = temporal_metrics.temporal_phi_sync;
            metrics.pattern_stability = temporal_metrics.pattern_stability;
            break;
    }
    
    // Calculate processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics.optimization_time_ms = duration.count() / 1000.0f;
    
    // Update performance tracking
    total_optimizations++;
    float new_avg_time = average_optimization_time.load() * 0.9f + metrics.optimization_time_ms * 0.1f;
    average_optimization_time.store(new_avg_time);
    
    float new_avg_compliance = average_phi_compliance.load() * 0.9f + metrics.golden_ratio_compliance * 0.1f;
    average_phi_compliance.store(new_avg_compliance);
    
    // Cache results
    cacheMetrics(points_hash, metrics);
    
    return metrics;
}

PhiMetrics ARKHEIONPhiOptimizationSystem::optimizeSpatialDistribution(std::vector<PhiOptimizedPoint>& points) {
    PhiMetrics metrics{};
    
    if (points.empty()) return metrics;
    
    // Extract positions
    std::vector<glm::vec3> positions;
    for (const auto& point : points) {
        positions.push_back(point.position);
    }
    
    // Initialize genetic optimizer
    genetic_optimizer->initialize(positions);
    
    // Evolve until convergence or max iterations
    int iterations = 0;
    while (iterations < params.max_iterations && !genetic_optimizer->evolve()) {
        iterations++;
    }
    
    // Get optimized positions
    std::vector<glm::vec3> optimized_positions = genetic_optimizer->getBestSolution();
    
    // Update points with optimized positions
    for (size_t i = 0; i < points.size() && i < optimized_positions.size(); ++i) {
        points[i].optimized_position = optimized_positions[i];
        
        // Calculate individual φ compliance
        points[i].phi_compliance = calculateGoldenRatioCompliance({optimized_positions[i]});
        
        // Calculate spiral parameter
        glm::vec3 center = glm::vec3(0.0f); // Assume center for now
        points[i].spiral_parameter = GoldenSpiralGenerator::calculateSpiralParameter(
            optimized_positions[i], center
        );
    }
    
    // Calculate overall metrics
    metrics.golden_ratio_compliance = calculateGoldenRatioCompliance(optimized_positions);
    metrics.spatial_harmony = calculateSpatialHarmony(optimized_positions);
    metrics.fibonacci_alignment = calculateFibonacciAlignment(optimized_positions);
    metrics.sacred_geometry_score = calculateSacredGeometryScore(optimized_positions);
    metrics.aesthetic_quality = calculateAestheticQuality(points);
    
    metrics.iterations_used = iterations;
    metrics.converged = (iterations < params.max_iterations);
    
    return metrics;
}

PhiMetrics ARKHEIONPhiOptimizationSystem::optimizeTemporalSynchronization(
    std::vector<PhiOptimizedPoint>& points,
    float time_factor
) {
    PhiMetrics metrics{};
    
    // Calculate φ-based temporal synchronization
    float base_frequency = 1.0f;
    float phi_frequency = base_frequency * PHI;
    
    for (auto& point : points) {
        // Calculate phase offset based on position
        float position_hash = point.position.x + point.position.y * PHI + point.position.z * PHI_SQUARED;
        point.phase_offset = std::fmod(position_hash, 2.0f * M_PI);
        
        // Calculate beat frequency based on φ harmonics
        point.beat_frequency = phi_frequency * std::fmod(point.spiral_parameter, PHI);
        
        // Update intensity based on temporal φ synchronization
        float temporal_factor = std::sin(time_factor * point.beat_frequency + point.phase_offset);
        temporal_factor = (temporal_factor + 1.0f) * 0.5f; // Normalize to [0,1]
        
        point.intensity *= (0.7f + 0.3f * temporal_factor); // Modulate intensity
    }
    
    // Calculate temporal metrics
    metrics.temporal_phi_sync = calculateTemporalSync(points, time_factor);
    metrics.pattern_stability = 0.9f; // Simplified calculation
    
    return metrics;
}

float ARKHEIONPhiOptimizationSystem::calculateGoldenRatioCompliance(const std::vector<glm::vec3>& positions) {
    if (positions.size() < 2) return 0.0f;
    
    float total_compliance = 0.0f;
    int comparisons = 0;
    
    for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            float distance = glm::distance(positions[i], positions[j]);
            
            // Compare with φ-based distances
            for (size_t k = 0; k < positions.size(); ++k) {
                if (k != i && k != j) {
                    float other_distance = glm::distance(positions[i], positions[k]);
                    
                    if (other_distance > 1e-6f) {
                        float ratio = distance / other_distance;
                        
                        // Check φ compliance
                        float phi_error = std::min(
                            std::abs(ratio - PHI),
                            std::abs(ratio - INV_PHI)
                        );
                        
                        float compliance = std::exp(-phi_error * 5.0f);
                        total_compliance += compliance;
                        comparisons++;
                    }
                }
            }
        }
    }
    
    return comparisons > 0 ? total_compliance / comparisons : 0.0f;
}

float ARKHEIONPhiOptimizationSystem::calculateSpatialHarmony(const std::vector<glm::vec3>& positions) {
    if (positions.empty()) return 0.0f;
    
    // Calculate center of mass
    glm::vec3 center(0.0f);
    for (const auto& pos : positions) {
        center += pos;
    }
    center /= static_cast<float>(positions.size());
    
    // Calculate spatial distribution harmony
    float harmony = 0.0f;
    for (const auto& pos : positions) {
        float distance_to_center = glm::distance(pos, center);
        float spiral_distance = phi_utils::distance_to_golden_spiral(pos, center);
        
        float harmony_factor = 1.0f - (spiral_distance / (distance_to_center + 1e-6f));
        harmony += std::max(0.0f, harmony_factor);
    }
    
    return harmony / positions.size();
}

float ARKHEIONPhiOptimizationSystem::calculateTemporalSync(
    const std::vector<PhiOptimizedPoint>& points,
    float time_factor
) {
    if (points.empty()) return 0.0f;
    
    float sync_measure = 0.0f;
    
    for (const auto& point : points) {
        float expected_phase = std::fmod(time_factor * point.beat_frequency, 2.0f * M_PI);
        float actual_phase = point.phase_offset;
        
        float phase_difference = std::abs(expected_phase - actual_phase);
        phase_difference = std::min(phase_difference, 2.0f * M_PI - phase_difference);
        
        float sync_factor = 1.0f - (phase_difference / M_PI);
        sync_measure += std::max(0.0f, sync_factor);
    }
    
    return sync_measure / points.size();
}

// Additional helper method implementations
float ARKHEIONPhiOptimizationSystem::calculateFibonacciAlignment(const std::vector<glm::vec3>& positions) {
    // Simplified Fibonacci alignment calculation
    return 0.8f; // Placeholder
}

float ARKHEIONPhiOptimizationSystem::calculateSacredGeometryScore(const std::vector<glm::vec3>& positions) {
    // Simplified sacred geometry score
    return 0.7f; // Placeholder
}

float ARKHEIONPhiOptimizationSystem::calculateAestheticQuality(const std::vector<PhiOptimizedPoint>& points) {
    // Aesthetic quality based on φ compliance and spatial harmony
    float total_quality = 0.0f;
    for (const auto& point : points) {
        total_quality += point.phi_compliance;
    }
    return points.empty() ? 0.0f : total_quality / points.size();
}

uint64_t ARKHEIONPhiOptimizationSystem::calculatePointsHash(const std::vector<PhiOptimizedPoint>& points) {
    // Simple hash calculation
    uint64_t hash = 0;
    for (const auto& point : points) {
        hash ^= std::hash<float>{}(point.position.x) << 1;
        hash ^= std::hash<float>{}(point.position.y) << 2;
        hash ^= std::hash<float>{}(point.position.z) << 3;
    }
    return hash;
}

bool ARKHEIONPhiOptimizationSystem::getCachedMetrics(uint64_t hash, PhiMetrics& metrics) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = metrics_cache.find(hash);
    if (it != metrics_cache.end()) {
        metrics = it->second;
        return true;
    }
    return false;
}

void ARKHEIONPhiOptimizationSystem::cacheMetrics(uint64_t hash, const PhiMetrics& metrics) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Limit cache size
    if (metrics_cache.size() > 1000) {
        metrics_cache.clear();
    }
    
    metrics_cache[hash] = metrics;
}

ARKHEIONPhiOptimizationSystem::PerformanceMetrics ARKHEIONPhiOptimizationSystem::getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    return PerformanceMetrics{
        .total_optimizations = total_optimizations.load(),
        .average_optimization_time = average_optimization_time.load(),
        .average_phi_compliance = average_phi_compliance.load(),
        .cache_hit_rate = 0.0f, // TODO: Calculate actual hit rate
        .cache_size = metrics_cache.size()
    };
}

// ============================================================================
// φ Utility Functions Implementation
// ============================================================================

namespace phi_utils {

glm::vec3 golden_spiral_point(float t, float radius_scale) {
    float radius = radius_scale * std::exp(t * FIBONACCI_SPIRAL_RATE);
    float angle = t * GOLDEN_ANGLE;
    
    return glm::vec3(
        radius * std::cos(angle),
        radius * std::sin(angle),
        t * radius_scale * 0.1f // Slight vertical component
    );
}

float distance_to_golden_spiral(const glm::vec3& point, const glm::vec3& center) {
    glm::vec3 relative = point - center;
    float radius = glm::length(glm::vec2(relative.x, relative.y));
    
    if (radius < 1e-6f) return 0.0f;
    
    // Find closest point on spiral
    float t = std::log(radius) / FIBONACCI_SPIRAL_RATE;
    glm::vec3 spiral_point = golden_spiral_point(t);
    spiral_point += center;
    
    return glm::distance(point, spiral_point);
}

float phi_temporal_wave(float time, float frequency) {
    return std::sin(time * frequency * PHI) * 0.5f + 0.5f;
}

float normalize_phi_metric(float value, float reference) {
    return std::tanh(value / reference); // Normalize to [0,1] range
}

} // namespace phi_utils

} // namespace arkheion
