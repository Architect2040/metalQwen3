#pragma once

#include <vector>

class CPUMatrixMul {
public:
    static std::vector<float> multiply(const std::vector<float>& A, const std::vector<float>& B, size_t N);
    static std::vector<float> multiplyOptimized(const std::vector<float>& A, const std::vector<float>& B, size_t N);
    static std::vector<float> generateRandomMatrix(size_t N);
    static bool verifyResults(const std::vector<float>& result1, const std::vector<float>& result2,
                             size_t N, float tolerance = 1e-4f);
};