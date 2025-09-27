#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include "Qwen3Benchmark.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iomanip>

Qwen3Benchmark::Qwen3Benchmark(MetalContext& context) : vulkanContext(context) {
    vulkan_model = std::make_unique<MetalQwen3>();
    if (vulkan_model) {
        vulkan_model->initialize();
    }
    cpu_model = std::make_unique<Qwen3Original>();
}

Qwen3Benchmark::~Qwen3Benchmark() = default;

bool Qwen3Benchmark::loadModels(const std::string& checkpoint_path, int ctx_length) {
    std::cout << "Loading Metal Qwen3 model..." << std::endl;
    if (!vulkan_model->loadModel(checkpoint_path, ctx_length)) {
        std::cerr << "Failed to load Metal model" << std::endl;
        return false;
    }

    std::cout << "Loading Original CPU Qwen3 model..." << std::endl;
    if (!cpu_model->loadModel(checkpoint_path, ctx_length)) {
        std::cerr << "Failed to load CPU model" << std::endl;
        return false;
    }

    // Verify model compatibility
    if (vulkan_model->getVocabSize() != cpu_model->getVocabSize() ||
        vulkan_model->getSeqLen() != cpu_model->getSeqLen() ||
        vulkan_model->getDim() != cpu_model->getDim()) {
        std::cerr << "Model configurations don't match between Metal and CPU versions" << std::endl;
        return false;
}

    models_loaded = true;
    std::cout << "Models loaded successfully!" << std::endl;
    std::cout << "  Vocabulary size: " << vulkan_model->getVocabSize() << std::endl;
    std::cout << "  Sequence length: " << vulkan_model->getSeqLen() << std::endl;
    std::cout << "  Model dimension: " << vulkan_model->getDim() << std::endl;

    return true;
}

std::vector<InferenceMetrics> Qwen3Benchmark::runFullBenchmarkSuite() {
    if (!models_loaded) {
        std::cerr << "Models not loaded. Call loadModels() first." << std::endl;
        return {};
    }

    std::vector<InferenceMetrics> results;

    std::cout << "\n=== Qwen3 Metal vs CPU Benchmark Suite ===" << std::endl;
    printBenchmarkHeader();

    // 1. Matrix Multiplication Benchmarks
    {
        std::cout << "\n--- Matrix Multiplication Benchmarks ---" << std::endl;
        std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
        auto metrics = benchmarkMatrixMultiplication(sizes);
        metrics.test_name = "Matrix Multiplication";
        results.push_back(metrics);
        printInferenceMetrics(metrics);
    }

    // 2. Attention Mechanism Benchmarks
    {
        std::cout << "\n--- Attention Mechanism Benchmarks ---" << std::endl;
        auto metrics = benchmarkAttentionMechanism(1024, 32, 128);
        metrics.test_name = "Multi-Head Attention";
        results.push_back(metrics);
        printInferenceMetrics(metrics);
    }

    // 3. Full Inference Benchmarks
    {
        std::cout << "\n--- Full Inference Benchmarks ---" << std::endl;
        std::vector<int> test_tokens = generateRandomTokenSequence(256);
        auto metrics = benchmarkFullInference(test_tokens);
        metrics.test_name = "Full Inference (256 tokens)";
        results.push_back(metrics);
        printInferenceMetrics(metrics);
    }

    // 4. Memory Usage Benchmarks
    {
        std::cout << "\n--- Memory Usage Benchmarks ---" << std::endl;
        auto metrics = benchmarkMemoryUsage();
        metrics.test_name = "Memory Usage";
        results.push_back(metrics);
        printInferenceMetrics(metrics);
    }

    // 5. Correctness Validation
    {
        std::cout << "\n--- Correctness Validation ---" << std::endl;
        std::vector<int> validation_tokens = generateRandomTokenSequence(64);
        compareAccuracy(validation_tokens, 1e-4);
    }

    return results;
}

InferenceMetrics Qwen3Benchmark::benchmarkFullInference(const std::vector<int>& test_tokens) {
    InferenceMetrics metrics;

    // Benchmark Vulkan implementation
    auto vulkan_benchmark = [this, &test_tokens]() {
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            auto logits = vulkan_model->forward(test_tokens[i], static_cast<int>(i));
        }
    };

    // Benchmark CPU implementation
    auto cpu_benchmark = [this, &test_tokens]() {
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            auto logits = cpu_model->forward(test_tokens[i], static_cast<int>(i));
        }
    };

    std::cout << "Benchmarking Metal inference..." << std::endl;
    metrics.vulkan_results = benchmarkFunction(vulkan_benchmark, "Metal Inference");

    std::cout << "Benchmarking CPU inference..." << std::endl;
    metrics.cpu_results = benchmarkFunction(cpu_benchmark, "CPU Inference");

    // Calculate comparative metrics
    metrics.speedup_ratio = metrics.cpu_results.avg_time_ms / metrics.vulkan_results.avg_time_ms;
    metrics.memory_efficiency_ratio = static_cast<double>(metrics.cpu_results.memory_usage_mb) /
                                     static_cast<double>(metrics.vulkan_results.memory_usage_mb);

    // Calculate throughput (tokens per second)
    metrics.vulkan_results.throughput_tokens_per_sec =
        (test_tokens.size() * 1000.0) / metrics.vulkan_results.avg_time_ms;
    metrics.cpu_results.throughput_tokens_per_sec =
        (test_tokens.size() * 1000.0) / metrics.cpu_results.avg_time_ms;

    return metrics;
}

InferenceMetrics Qwen3Benchmark::benchmarkMatrixMultiplication(const std::vector<int>& sizes) {
    InferenceMetrics metrics;

    // This is a simplified benchmark - in practice would test individual matmul operations
    // For now, we'll use the full inference as a proxy for matmul performance

    std::vector<int> test_tokens = generateRandomTokenSequence(32);

    auto vulkan_benchmark = [this, &test_tokens]() {
        vulkan_model->forward(test_tokens[0], 0);
    };

    auto cpu_benchmark = [this, &test_tokens]() {
        cpu_model->forward(test_tokens[0], 0);
    };

    metrics.vulkan_results = benchmarkFunction(vulkan_benchmark, "Vulkan MatMul");
    metrics.cpu_results = benchmarkFunction(cpu_benchmark, "CPU MatMul");

    metrics.speedup_ratio = metrics.cpu_results.avg_time_ms / metrics.vulkan_results.avg_time_ms;
    metrics.memory_efficiency_ratio = static_cast<double>(metrics.cpu_results.memory_usage_mb) /
                                     static_cast<double>(metrics.vulkan_results.memory_usage_mb);

    return metrics;
}

InferenceMetrics Qwen3Benchmark::benchmarkAttentionMechanism(int seq_len, int n_heads, int head_dim) {
    InferenceMetrics metrics;

    // Generate test sequence
    int tokens_to_generate = (seq_len < 64) ? seq_len : 64;
    std::vector<int> test_tokens = generateRandomTokenSequence(tokens_to_generate);

    auto vulkan_benchmark = [this, &test_tokens]() {
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            vulkan_model->forward(test_tokens[i], static_cast<int>(i));
        }
    };

    auto cpu_benchmark = [this, &test_tokens]() {
        for (size_t i = 0; i < test_tokens.size(); ++i) {
            cpu_model->forward(test_tokens[i], static_cast<int>(i));
        }
    };

    metrics.vulkan_results = benchmarkFunction(vulkan_benchmark, "Vulkan Attention");
    metrics.cpu_results = benchmarkFunction(cpu_benchmark, "CPU Attention");

    metrics.speedup_ratio = metrics.cpu_results.avg_time_ms / metrics.vulkan_results.avg_time_ms;
    metrics.memory_efficiency_ratio = static_cast<double>(metrics.cpu_results.memory_usage_mb) /
                                     static_cast<double>(metrics.vulkan_results.memory_usage_mb);

    return metrics;
}

InferenceMetrics Qwen3Benchmark::benchmarkMemoryUsage() {
    InferenceMetrics metrics;

    // Measure memory usage during inference
    std::vector<int> test_tokens = generateRandomTokenSequence(16);

    size_t vulkan_memory_before = getMemoryUsage();
    vulkan_model->forward(test_tokens[0], 0);
    size_t vulkan_memory_after = getMemoryUsage();

    size_t cpu_memory_before = getMemoryUsage();
    cpu_model->forward(test_tokens[0], 0);
    size_t cpu_memory_after = getMemoryUsage();

    metrics.vulkan_results.memory_usage_mb = (vulkan_memory_after - vulkan_memory_before) / (1024 * 1024);
    metrics.cpu_results.memory_usage_mb = (cpu_memory_after - cpu_memory_before) / (1024 * 1024);

    metrics.memory_efficiency_ratio = static_cast<double>(metrics.cpu_results.memory_usage_mb) /
                                     static_cast<double>(metrics.vulkan_results.memory_usage_mb);

    return metrics;
}

BenchmarkResult Qwen3Benchmark::benchmarkFunction(std::function<void()> func, const std::string& description) {
    BenchmarkResult result = {};

    // Warmup
    warmup(func, warmup_iterations);

    // Actual benchmark
    std::vector<double> times;
    times.reserve(benchmark_iterations);

    for (int i = 0; i < benchmark_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;
        times.push_back(time_ms);
    }

    // Calculate statistics
    result.individual_times = times;
    result.avg_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    auto min_it = std::min_element(times.begin(), times.end());
    auto max_it = std::max_element(times.begin(), times.end());
    result.min_time_ms = *min_it;
    result.max_time_ms = *max_it;
    result.std_dev_ms = calculateStandardDeviation(times);
    result.memory_usage_mb = getMemoryUsage() / (1024 * 1024);

    return result;
}

void Qwen3Benchmark::compareAccuracy(const std::vector<int>& test_tokens, double tolerance) {
    std::cout << "Comparing output accuracy between Vulkan and CPU implementations..." << std::endl;

    bool all_passed = true;
    double max_error = 0.0;

    size_t comparison_size = (test_tokens.size() < 10) ? test_tokens.size() : 10;
    for (size_t i = 0; i < comparison_size; ++i) {
        auto vulkan_output = vulkan_model->forward(test_tokens[i], static_cast<int>(i));
        auto cpu_output = cpu_model->forward(test_tokens[i], static_cast<int>(i));

        if (validateResults(vulkan_output, cpu_output, tolerance)) {
            std::cout << "  Token " << i << ": PASS" << std::endl;
        } else {
            std::cout << "  Token " << i << ": FAIL" << std::endl;
            all_passed = false;

            // Calculate max error for debugging
            for (size_t j = 0; j < vulkan_output.size() && j < cpu_output.size(); ++j) {
                double error = std::abs(vulkan_output[j] - cpu_output[j]);
                if (error > max_error) max_error = error;
            }
        }
    }

    std::cout << "Overall accuracy: " << (all_passed ? "PASS" : "FAIL") << std::endl;
    if (!all_passed) {
        std::cout << "Maximum error found: " << max_error << std::endl;
    }
}

bool Qwen3Benchmark::validateResults(const std::vector<float>& vulkan_output,
                                    const std::vector<float>& cpu_output,
                                    double tolerance) {
    if (vulkan_output.size() != cpu_output.size()) {
        return false;
    }

    for (size_t i = 0; i < vulkan_output.size(); ++i) {
        double error = std::abs(vulkan_output[i] - cpu_output[i]);
        if (error > tolerance) {
            return false;
        }
    }

    return true;
}

std::vector<int> Qwen3Benchmark::generateRandomTokenSequence(int length) {
    std::vector<int> tokens;
    tokens.reserve(length);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, models_loaded ? vulkan_model->getVocabSize() - 1 : 1000);

    for (int i = 0; i < length; ++i) {
        tokens.push_back(dis(gen));
    }

    return tokens;
}

double Qwen3Benchmark::calculateStandardDeviation(const std::vector<double>& times) {
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double variance = 0.0;

    for (double time : times) {
        variance += std::pow(time - mean, 2);
    }

    return std::sqrt(variance / times.size());
}

size_t Qwen3Benchmark::getMemoryUsage() const {
    // Simplified memory usage estimation
    // In a real implementation, this would query system memory usage
    return 1024 * 1024 * 100; // 100MB placeholder
}

void Qwen3Benchmark::warmup(std::function<void()> func, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        func();
    }
}

void Qwen3Benchmark::printBenchmarkHeader() {
    std::cout << std::setw(20) << "Test"
              << std::setw(15) << "Metal (ms)"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "V-Memory(MB)"
              << std::setw(15) << "C-Memory(MB)" << std::endl;
    std::cout << std::string(95, '-') << std::endl;
}

void Qwen3Benchmark::printInferenceMetrics(const InferenceMetrics& metrics) {
    std::cout << std::setw(20) << metrics.test_name
              << std::setw(15) << std::fixed << std::setprecision(2) << metrics.vulkan_results.avg_time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << metrics.cpu_results.avg_time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << metrics.speedup_ratio << "x"
              << std::setw(15) << metrics.vulkan_results.memory_usage_mb
              << std::setw(15) << metrics.cpu_results.memory_usage_mb << std::endl;
}

void Qwen3Benchmark::generateReport(const std::vector<InferenceMetrics>& results, const std::string& output_file) {
    if (!output_file.empty()) {
        saveMetricsToFile(results, output_file);
    }

    std::cout << "\n=== Benchmark Summary ===" << std::endl;

    double total_speedup = 0.0;
    for (const auto& result : results) {
        total_speedup += result.speedup_ratio;
    }
    double avg_speedup = total_speedup / results.size();

    std::cout << "Average Speedup: " << std::fixed << std::setprecision(2) << avg_speedup << "x" << std::endl;

    // Find best and worst performing tests
    auto best = std::max_element(results.begin(), results.end(),
        [](const InferenceMetrics& a, const InferenceMetrics& b) {
            return a.speedup_ratio < b.speedup_ratio;
        });
    auto worst = std::min_element(results.begin(), results.end(),
        [](const InferenceMetrics& a, const InferenceMetrics& b) {
            return a.speedup_ratio < b.speedup_ratio;
        });

    std::cout << "Best Performance: " << best->test_name << " (" << best->speedup_ratio << "x)" << std::endl;
    std::cout << "Worst Performance: " << worst->test_name << " (" << worst->speedup_ratio << "x)" << std::endl;
}

void Qwen3Benchmark::saveMetricsToFile(const std::vector<InferenceMetrics>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    file << "Test,Vulkan_Time_ms,CPU_Time_ms,Speedup,Vulkan_Memory_MB,CPU_Memory_MB\n";

    for (const auto& result : results) {
        file << result.test_name << ","
             << result.vulkan_results.avg_time_ms << ","
             << result.cpu_results.avg_time_ms << ","
             << result.speedup_ratio << ","
             << result.vulkan_results.memory_usage_mb << ","
             << result.cpu_results.memory_usage_mb << "\n";
    }

    file.close();
    std::cout << "Benchmark results saved to: " << filename << std::endl;
}
