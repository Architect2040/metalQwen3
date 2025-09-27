#pragma once

#include "MetalQwen3.h"
#include "Qwen3Original.h"
#include "Benchmark.h"
#include <vector>
#include <string>
#include <memory>
#include <chrono>

struct BenchmarkResult {
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    size_t memory_usage_mb;
    double throughput_tokens_per_sec;
    bool correctness_passed;
    std::vector<double> individual_times;
};

struct InferenceMetrics {
    BenchmarkResult vulkan_results;
    BenchmarkResult cpu_results;
    double speedup_ratio;
    double memory_efficiency_ratio;
    std::string test_name;
};

class Qwen3Benchmark {
public:
    Qwen3Benchmark(MetalContext& context);
    ~Qwen3Benchmark();

    // Setup and configuration
    bool loadModels(const std::string& checkpoint_path, int ctx_length = 0);
    void setWarmupIterations(int iterations) { warmup_iterations = iterations; }
    void setBenchmarkIterations(int iterations) { benchmark_iterations = iterations; }

    // Individual component benchmarks
    InferenceMetrics benchmarkMatrixMultiplication(const std::vector<int>& sizes);
    InferenceMetrics benchmarkAttentionMechanism(int seq_len, int n_heads, int head_dim);
    InferenceMetrics benchmarkFullInference(const std::vector<int>& test_tokens);
    InferenceMetrics benchmarkMemoryUsage();

    // Comprehensive benchmark suite
    std::vector<InferenceMetrics> runFullBenchmarkSuite();

    // Performance analysis
    void generateReport(const std::vector<InferenceMetrics>& results, const std::string& output_file = "");
    void compareAccuracy(const std::vector<int>& test_tokens, double tolerance = 1e-5);

private:
    MetalContext& vulkanContext;
    std::unique_ptr<MetalQwen3> vulkan_model;
    std::unique_ptr<Qwen3Original> cpu_model;

    int warmup_iterations = 10;
    int benchmark_iterations = 100;
    bool models_loaded = false;

    // Utility functions
    BenchmarkResult benchmarkFunction(std::function<void()> func, const std::string& description);
    double calculateStandardDeviation(const std::vector<double>& times);
    size_t getMemoryUsage() const;
    void warmup(std::function<void()> func, int iterations);

    // Test data generation
    std::vector<int> generateRandomTokenSequence(int length);
    std::vector<std::vector<float>> generateRandomMatrices(int size);

    // Correctness validation
    bool validateResults(const std::vector<float>& vulkan_output,
                        const std::vector<float>& cpu_output,
                        double tolerance);

    // Reporting helpers
    void printBenchmarkHeader();
    void printInferenceMetrics(const InferenceMetrics& metrics);
    void saveMetricsToFile(const std::vector<InferenceMetrics>& results, const std::string& filename);
};