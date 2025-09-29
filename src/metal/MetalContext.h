/**
 * MetalQwen3 - High-Performance Transformer Inference on Apple Silicon
 *
 * @file MetalContext.h
 * @brief Metal GPU context and compute shader management header
 * @author Shlomo Kashnai
 * @date 2024
 *
 * Metal context using metal-cpp C++ bindings for clean GPU integration.
 * Implements command batching and buffer pooling optimizations for
 * efficient transformer inference on Apple Silicon.
 *
 * @license MIT License - See project root for full license text
 */

#pragma once

// Forward declarations to avoid duplicate symbols
namespace MTL {
    class Device;
    class CommandQueue;
    class Buffer;
    class ComputePipelineState;
    class CommandBuffer;
    class ComputeCommandEncoder;
    class Library;
}

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

class MetalContext {
public:
    MetalContext();
    ~MetalContext();

    bool initialize();
    void cleanup();

    bool isInitialized() const { return initialized; }

    // Metal device and queue access
    MTL::Device* getDevice() const { return device; }
    MTL::CommandQueue* getCommandQueue() const { return commandQueue; }

    // Buffer management
    MTL::Buffer* createBuffer(size_t size, const void* data = nullptr);
    void releaseBuffer(MTL::Buffer* buffer);

    // Compute pipeline management
    MTL::ComputePipelineState* createComputePipeline(const std::string& shaderName, const std::string& functionName);
    void releaseComputePipeline(MTL::ComputePipelineState* pipeline);

    // Batched command execution - key optimization
    void beginBatch();
    void endBatch();
    bool isBatching() const { return batchCommandBuffer != nullptr; }

    // Command execution
    MTL::CommandBuffer* createCommandBuffer();
    void commitCommandBuffer(MTL::CommandBuffer* commandBuffer);
    void waitForCompletion(MTL::CommandBuffer* commandBuffer);

    // High-level batched forward pass for entire layer
    void executeTransformerLayer(
        float* x, float* xb, float* hb, float* hb2,
        float* q, float* k, float* v, float* att,
        const float* rms_att_weight, const float* rms_ffn_weight,
        const void* wq, const void* wk, const void* wv, const void* wo,
        const void* w1, const void* w2, const void* w3,
        int dim, int hidden_dim, int n_heads, int n_kv_heads, int head_dim, int pos,
        float* key_cache, float* value_cache, int seq_len, int kv_dim, uint64_t loff
    );

    // Optimized shader execution methods
    void executeRMSNorm(float* output, const float* input, const float* weight, int size);
    void executeSoftmax(float* x, int size);
    void executeQuantizedMatMul(float* output, const int8_t* x_q, const float* x_s,
                               const int8_t* w_q, const float* w_s, int n, int d, int group_size);
    void executeSwiGLU(float* hb, const float* hb2, int hidden_dim);
    void executeRoPE(float* q, float* k, int head_dim, int pos, int n_heads, int n_kv_heads,
                    const float* q_norm_weights, const float* k_norm_weights);
    void executeAttention(float* xb, const float* q, float* att, float* key_cache, float* value_cache,
                         int pos, int head_dim, int n_heads, int n_kv_heads, int seq_len, int kv_dim, uint64_t loff, int kv_mul);

private:
    bool initialized;
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;

    // Batching optimization
    MTL::CommandBuffer* batchCommandBuffer;
    MTL::ComputeCommandEncoder* batchEncoder;

    // Cache compiled pipelines
    std::unordered_map<std::string, MTL::ComputePipelineState*> pipelineCache;

    // GPU buffer pool for reduced allocation overhead
    std::vector<MTL::Buffer*> bufferPool;
    std::unordered_map<size_t, std::vector<MTL::Buffer*>> sizedBufferPools;

    // Helper methods
    MTL::Library* loadLibrary(const std::string& libraryName);
    std::string findLibraryPath(const std::string& libraryName);
    void logError(const std::string& message);

    // Optimized buffer management
    MTL::Buffer* getPooledBuffer(size_t size);
    void returnBufferToPool(MTL::Buffer* buffer, size_t size);

    // Internal shader execution (for batching)
    void internalExecuteRMSNorm(MTL::ComputeCommandEncoder* encoder, MTL::Buffer* output, MTL::Buffer* input, MTL::Buffer* weight, int size);
    void internalExecuteQuantizedMatMul(MTL::ComputeCommandEncoder* encoder, MTL::Buffer* output, MTL::Buffer* x_buffer, MTL::Buffer* w_buffer, MTL::Buffer* x_scales, MTL::Buffer* w_scales, int n, int d, int group_size);

    // All operations use Metal GPU shaders exclusively
};