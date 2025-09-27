#pragma once

#include <chrono>
#include <vector>
#include <complex>
#include <string>
#include <functional>

class Benchmark {
public:
    struct Result {
        std::string name;
        double avgTimeMs;
        double minTimeMs;
        double maxTimeMs;
        double totalTimeMs;
        int runs;
        bool success;
        std::string error;
        size_t dataSize;
    };

    static Result timeFunction(const std::string& name, std::function<void()> func, int numRuns = 1);
    static Result timeFunction(const std::string& name, std::function<void()> func, size_t dataSize, int numRuns = 1);
    static void printResults(const std::vector<Result>& results);
    static void printDetailedResults(const std::vector<Result>& results);
    static std::vector<std::complex<float>> generateTestData(size_t size);
    static std::vector<float> generateMatrixTestData(size_t size);
    static bool verifyResults(const std::vector<std::complex<float>>& result1,
                             const std::vector<std::complex<float>>& result2,
                             float tolerance = 1e-5f);
    static bool verifyMatrixResults(const std::vector<float>& result1,
                                   const std::vector<float>& result2,
                                   float tolerance = 1e-5f);
    static double calculateSpeedup(const Result& baseline, const Result& comparison, bool useBest = false);
    static std::string formatTime(double timeMs);
};