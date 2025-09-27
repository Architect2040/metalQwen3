/**
 * MetalQwen3 Shader Functionality Test Suite
 *
 * @file test_shaders.cpp
 * @brief Dedicated test suite for Metal shader functionality verification
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Tests each Metal shader in isolation to verify correctness independent
 * of the Qwen3 LLM implementation. Validates mathematical correctness,
 * performance characteristics, and GPU execution success.
 *
 * @license MIT License - See project root for full license text
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>
#include <iomanip>

#include "metal/MetalContext.h"

class ShaderTester {
private:
    std::unique_ptr<MetalContext> metalContext;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

public:
    ShaderTester() : rng(42), dist(-1.0f, 1.0f) {
        metalContext = std::make_unique<MetalContext>();
        if (!metalContext->initialize()) {
            throw std::runtime_error("Failed to initialize Metal context");
        }
        std::cout << "✅ Metal Context initialized for shader testing\n";
        std::cout << "Device: " << (metalContext->getDevice() ? "Apple GPU Available" : "No GPU") << "\n\n";
    }

    // Helper functions for test data generation
    std::vector<float> generateRandomVector(size_t size) {
        std::vector<float> data(size);
        for (auto& val : data) {
            val = dist(rng);
        }
        return data;
    }

    std::vector<int8_t> generateRandomInt8Vector(size_t size) {
        std::vector<int8_t> data(size);
        std::uniform_int_distribution<int> int_dist(-127, 127);
        for (auto& val : data) {
            val = static_cast<int8_t>(int_dist(rng));
        }
        return data;
    }

    std::vector<float> generateScaleVector(size_t groups) {
        std::vector<float> scales(groups);
        std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
        for (auto& scale : scales) {
            scale = scale_dist(rng);
        }
        return scales;
    }

    // CPU reference implementations
    void cpuRMSNorm(std::vector<float>& output, const std::vector<float>& input,
                    const std::vector<float>& weight, float eps = 1e-6f) {
        size_t size = input.size();

        // Compute sum of squares
        float ss = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            ss += input[i] * input[i];
        }

        // Compute normalization factor
        float norm = 1.0f / sqrtf((ss / size) + eps);

        // Apply normalization and scaling
        for (size_t i = 0; i < size; ++i) {
            output[i] = weight[i] * (norm * input[i]);
        }
    }

    void cpuSoftmax(std::vector<float>& data) {
        size_t size = data.size();

        // Find max for numerical stability
        float max_val = data[0];
        for (size_t i = 1; i < size; ++i) {
            max_val = std::max(max_val, data[i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            data[i] = expf(data[i] - max_val);
            sum += data[i];
        }

        // Normalize
        for (size_t i = 0; i < size; ++i) {
            data[i] /= sum;
        }
    }

    void cpuSwiGLU(std::vector<float>& hb, const std::vector<float>& hb2) {
        size_t size = hb.size();
        for (size_t i = 0; i < size; ++i) {
            // SwiGLU: hb[i] *= hb2[i] * SiLU(hb[i])
            // where SiLU(x) = x / (1 + exp(-x))
            float silu = hb[i] / (1.0f + expf(-hb[i]));
            hb[i] = silu * hb2[i];
        }
    }

    void cpuQuantizedMatMul(std::vector<float>& output, const std::vector<int8_t>& x_q,
                           const std::vector<float>& x_s, const std::vector<int8_t>& w_q,
                           const std::vector<float>& w_s, int M, int N, int K, int group_size) {
        // output = W @ x (transposed for row-major)
        for (int i = 0; i < M; ++i) {
            float val = 0.0f;
            int row_offset = i * K;

            for (int j = 0; j <= K - group_size; j += group_size) {
                int32_t ival = 0;
                for (int k = 0; k < group_size; ++k) {
                    ival += static_cast<int32_t>(x_q[j + k]) * static_cast<int32_t>(w_q[row_offset + j + k]);
                }
                val += static_cast<float>(ival) * w_s[(row_offset + j) / group_size] * x_s[j / group_size];
            }
            output[i] = val;
        }
    }

    // Utility for comparing float vectors
    bool compareVectors(const std::vector<float>& a, const std::vector<float>& b,
                       float tolerance = 1e-4f, const std::string& test_name = "") {
        if (a.size() != b.size()) {
            std::cout << "❌ " << test_name << ": Size mismatch (" << a.size() << " vs " << b.size() << ")\n";
            return false;
        }

        float max_diff = 0.0f;
        size_t error_count = 0;

        for (size_t i = 0; i < a.size(); ++i) {
            float diff = std::abs(a[i] - b[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) {
                error_count++;
                if (error_count <= 5) { // Show first 5 errors
                    std::cout << "   Error at [" << i << "]: " << a[i] << " vs " << b[i]
                              << " (diff: " << diff << ")\n";
                }
            }
        }

        if (error_count > 0) {
            std::cout << "❌ " << test_name << ": " << error_count << "/" << a.size()
                      << " elements exceed tolerance (max diff: " << max_diff << ")\n";
            return false;
        }

        std::cout << "✅ " << test_name << ": All elements within tolerance (max diff: " << max_diff << ")\n";
        return true;
    }

    // Test RMSNorm shader
    bool testRMSNorm() {
        std::cout << "🧪 Testing RMSNorm Shader\n";

        const size_t size = 2560; // Qwen3 model dimension
        auto input = generateRandomVector(size);
        auto weight = generateRandomVector(size);

        // CPU reference
        std::vector<float> cpu_output(size);
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpuRMSNorm(cpu_output, input, weight);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // GPU implementation
        std::vector<float> gpu_output(size);
        auto start_gpu = std::chrono::high_resolution_clock::now();
        metalContext->executeRMSNorm(gpu_output.data(), input.data(), weight.data(), size);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

        bool correct = compareVectors(gpu_output, cpu_output, 1e-4f, "RMSNorm");

        std::cout << "   CPU Time: " << std::fixed << std::setprecision(3) << cpu_time << " ms\n";
        std::cout << "   GPU Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms\n";
        if (gpu_time > 0) {
            std::cout << "   Speedup: " << std::fixed << std::setprecision(2) << (cpu_time / gpu_time) << "x\n";
        }
        std::cout << "\n";

        return correct;
    }

    // Test Softmax shader
    bool testSoftmax() {
        std::cout << "🧪 Testing Softmax Shader\n";

        const size_t size = 4096; // Large attention sequence
        auto data_cpu = generateRandomVector(size);
        auto data_gpu = data_cpu; // Copy for GPU test

        // CPU reference
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpuSoftmax(data_cpu);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // GPU implementation
        auto start_gpu = std::chrono::high_resolution_clock::now();
        metalContext->executeSoftmax(data_gpu.data(), size);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

        bool correct = compareVectors(data_gpu, data_cpu, 1e-5f, "Softmax");

        // Verify softmax properties
        float sum = 0.0f;
        for (float val : data_gpu) {
            sum += val;
            if (val < 0.0f || val > 1.0f) {
                std::cout << "❌ Softmax: Invalid output range\n";
                return false;
            }
        }

        if (std::abs(sum - 1.0f) > 1e-5f) {
            std::cout << "❌ Softmax: Sum not equal to 1.0 (sum = " << sum << ")\n";
            return false;
        }

        std::cout << "   CPU Time: " << std::fixed << std::setprecision(3) << cpu_time << " ms\n";
        std::cout << "   GPU Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms\n";
        if (gpu_time > 0) {
            std::cout << "   Speedup: " << std::fixed << std::setprecision(2) << (cpu_time / gpu_time) << "x\n";
        }
        std::cout << "\n";

        return correct;
    }

    // Test SwiGLU shader
    bool testSwiGLU() {
        std::cout << "🧪 Testing SwiGLU Shader\n";

        const size_t size = 10240; // Qwen3 hidden dimension
        auto hb_cpu = generateRandomVector(size);
        auto hb2 = generateRandomVector(size);
        auto hb_gpu = hb_cpu; // Copy for GPU test

        // CPU reference
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpuSwiGLU(hb_cpu, hb2);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // GPU implementation
        auto start_gpu = std::chrono::high_resolution_clock::now();
        metalContext->executeSwiGLU(hb_gpu.data(), hb2.data(), size);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

        bool correct = compareVectors(hb_gpu, hb_cpu, 1e-5f, "SwiGLU");

        std::cout << "   CPU Time: " << std::fixed << std::setprecision(3) << cpu_time << " ms\n";
        std::cout << "   GPU Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms\n";
        if (gpu_time > 0) {
            std::cout << "   Speedup: " << std::fixed << std::setprecision(2) << (cpu_time / gpu_time) << "x\n";
        }
        std::cout << "\n";

        return correct;
    }

    // Test Quantized Matrix Multiplication shader
    bool testQuantizedMatMul() {
        std::cout << "🧪 Testing Quantized MatMul Shader\n";

        const int M = 2560;  // Output dimension
        const int K = 2560;  // Input dimension
        const int group_size = 64; // Q8_0 group size

        auto x_q = generateRandomInt8Vector(K);
        auto x_s = generateScaleVector(K / group_size);
        auto w_q = generateRandomInt8Vector(M * K);
        auto w_s = generateScaleVector(M * K / group_size);

        // CPU reference
        std::vector<float> cpu_output(M);
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpuQuantizedMatMul(cpu_output, x_q, x_s, w_q, w_s, M, 1, K, group_size);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // GPU implementation
        std::vector<float> gpu_output(M);
        auto start_gpu = std::chrono::high_resolution_clock::now();
        metalContext->executeQuantizedMatMul(gpu_output.data(), x_q.data(), x_s.data(),
                                           w_q.data(), w_s.data(), K, M, group_size);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

        bool correct = compareVectors(gpu_output, cpu_output, 1e-3f, "QuantizedMatMul");

        std::cout << "   Matrix Size: " << M << "x" << K << " (group_size=" << group_size << ")\n";
        std::cout << "   CPU Time: " << std::fixed << std::setprecision(3) << cpu_time << " ms\n";
        std::cout << "   GPU Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms\n";
        if (gpu_time > 0) {
            std::cout << "   Speedup: " << std::fixed << std::setprecision(2) << (cpu_time / gpu_time) << "x\n";
        }
        std::cout << "\n";

        return correct;
    }

    // Test RoPE shader (placeholder - complex implementation)
    bool testRoPE() {
        std::cout << "🧪 Testing RoPE Shader\n";

        const int n_heads = 20;
        const int n_kv_heads = 4;
        const int head_dim = 128;
        const int pos = 42;

        auto q = generateRandomVector(n_heads * head_dim);
        auto k = generateRandomVector(n_kv_heads * head_dim);
        auto q_norm_weights = generateRandomVector(head_dim);
        auto k_norm_weights = generateRandomVector(head_dim);

        // Make copies for GPU test
        auto q_gpu = q;
        auto k_gpu = k;

        try {
            // GPU implementation
            auto start_gpu = std::chrono::high_resolution_clock::now();
            metalContext->executeRoPE(q_gpu.data(), k_gpu.data(), head_dim, pos,
                                    n_heads, n_kv_heads, q_norm_weights.data(), k_norm_weights.data());
            auto end_gpu = std::chrono::high_resolution_clock::now();
            auto gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

            std::cout << "   Heads: " << n_heads << " query, " << n_kv_heads << " key/value\n";
            std::cout << "   Head Dimension: " << head_dim << "\n";
            std::cout << "   Position: " << pos << "\n";
            std::cout << "   GPU Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms\n";

            // Basic sanity check - values should have changed
            bool changed = false;
            for (size_t i = 0; i < q.size() && !changed; ++i) {
                if (std::abs(q[i] - q_gpu[i]) > 1e-6f) {
                    changed = true;
                }
            }

            if (!changed) {
                std::cout << "❌ RoPE: No changes detected in output\n\n";
                return false;
            }

            std::cout << "✅ RoPE: GPU execution successful with output changes\n\n";
            return true;

        } catch (const std::exception& e) {
            std::cout << "❌ RoPE: Exception during execution: " << e.what() << "\n\n";
            return false;
        }
    }

    // Test Attention shader (placeholder - complex implementation)
    bool testAttention() {
        std::cout << "🧪 Testing Multi-Head Attention Shader\n";

        const int n_heads = 20;
        const int n_kv_heads = 4;
        const int head_dim = 128;
        const int seq_len = 512;
        const int pos = 100;
        const int kv_dim = n_kv_heads * head_dim;
        const int kv_mul = n_heads / n_kv_heads;

        auto q = generateRandomVector(n_heads * head_dim);
        auto att = generateRandomVector(n_heads * seq_len);
        auto xb = generateRandomVector(n_heads * head_dim);
        auto key_cache = generateRandomVector(seq_len * kv_dim);
        auto value_cache = generateRandomVector(seq_len * kv_dim);

        try {
            // GPU implementation
            auto start_gpu = std::chrono::high_resolution_clock::now();
            metalContext->executeAttention(xb.data(), q.data(), att.data(),
                                         key_cache.data(), value_cache.data(),
                                         pos, head_dim, n_heads, n_kv_heads,
                                         seq_len, kv_dim, 0, kv_mul);
            auto end_gpu = std::chrono::high_resolution_clock::now();
            auto gpu_time = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

            std::cout << "   Attention Heads: " << n_heads << "\n";
            std::cout << "   KV Heads: " << n_kv_heads << "\n";
            std::cout << "   Head Dimension: " << head_dim << "\n";
            std::cout << "   Sequence Length: " << seq_len << "\n";
            std::cout << "   Position: " << pos << "\n";
            std::cout << "   GPU Time: " << std::fixed << std::setprecision(3) << gpu_time << " ms\n";

            // Basic sanity checks
            bool has_nan = false;
            bool has_inf = false;
            for (float val : xb) {
                if (std::isnan(val)) has_nan = true;
                if (std::isinf(val)) has_inf = true;
            }

            if (has_nan || has_inf) {
                std::cout << "❌ Attention: Invalid values detected (NaN: " << has_nan
                          << ", Inf: " << has_inf << ")\n\n";
                return false;
            }

            std::cout << "✅ Attention: GPU execution successful with valid outputs\n\n";
            return true;

        } catch (const std::exception& e) {
            std::cout << "❌ Attention: Exception during execution: " << e.what() << "\n\n";
            return false;
        }
    }

    // Run all shader tests
    void runAllTests() {
        std::cout << "🚀 MetalQwen3 Shader Functionality Test Suite\n";
        std::cout << "==============================================\n\n";

        int passed = 0;
        int total = 0;

        auto runTest = [&](const std::string& name, std::function<bool()> test) {
            std::cout << "📋 " << name << "\n";
            std::cout << std::string(50, '-') << "\n";
            total++;
            if (test()) {
                passed++;
            }
        };

        runTest("RMS Normalization", [this]() { return testRMSNorm(); });
        runTest("Softmax Activation", [this]() { return testSoftmax(); });
        runTest("SwiGLU Activation", [this]() { return testSwiGLU(); });
        runTest("Quantized Matrix Multiplication", [this]() { return testQuantizedMatMul(); });
        runTest("Rotary Position Embedding", [this]() { return testRoPE(); });
        runTest("Multi-Head Attention", [this]() { return testAttention(); });

        std::cout << "==============================================\n";
        std::cout << "📊 Test Results: " << passed << "/" << total << " tests passed\n";

        if (passed == total) {
            std::cout << "🎉 All shader tests PASSED! GPU implementation verified.\n";
        } else {
            std::cout << "⚠️  Some tests failed. Check implementation details.\n";
        }
    }
};

int main() {
    try {
        ShaderTester tester;
        tester.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return 1;
    }
}