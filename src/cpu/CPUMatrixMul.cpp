#include "CPUMatrixMul.h"
#include <random>
#include <cmath>
#include <algorithm>

std::vector<float> CPUMatrixMul::multiply(const std::vector<float>& A, const std::vector<float>& B, size_t N) {
    std::vector<float> C(N * N, 0.0f);

    // Standard matrix multiplication: C = A * B
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    return C;
}

std::vector<float> CPUMatrixMul::multiplyOptimized(const std::vector<float>& A, const std::vector<float>& B, size_t N) {
    std::vector<float> C(N * N, 0.0f);

    // Cache-optimized matrix multiplication with loop tiling
    const size_t blockSize = 64; // Tune for cache size

    for (size_t ii = 0; ii < N; ii += blockSize) {
        for (size_t jj = 0; jj < N; jj += blockSize) {
            for (size_t kk = 0; kk < N; kk += blockSize) {
                // Compute block
                size_t iEnd = std::min(ii + blockSize, N);
                size_t jEnd = std::min(jj + blockSize, N);
                size_t kEnd = std::min(kk + blockSize, N);

                for (size_t i = ii; i < iEnd; ++i) {
                    for (size_t j = jj; j < jEnd; ++j) {
                        float sum = C[i * N + j];
                        for (size_t k = kk; k < kEnd; ++k) {
                            sum += A[i * N + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }

    return C;
}

std::vector<float> CPUMatrixMul::generateRandomMatrix(size_t N) {
    std::vector<float> matrix(N * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < N * N; ++i) {
        matrix[i] = dis(gen);
    }

    return matrix;
}

bool CPUMatrixMul::verifyResults(const std::vector<float>& result1, const std::vector<float>& result2,
                                size_t N, float tolerance) {
    if (result1.size() != result2.size() || result1.size() != N * N) {
        return false;
    }

    for (size_t i = 0; i < N * N; ++i) {
        float diff = std::abs(result1[i] - result2[i]);
        if (diff > tolerance) {
            return false;
        }
    }

    return true;
}