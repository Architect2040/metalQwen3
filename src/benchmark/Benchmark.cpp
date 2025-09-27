#include "Benchmark.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <functional>
#include <algorithm>
#include <limits>
#include <sstream>

Benchmark::Result Benchmark::timeFunction(const std::string& name, std::function<void()> func, int numRuns) {
    return timeFunction(name, func, 0, numRuns);
}

Benchmark::Result Benchmark::timeFunction(const std::string& name, std::function<void()> func, size_t dataSize, int numRuns) {
    Result result;
    result.name = name;
    result.success = true;
    result.runs = numRuns;
    result.dataSize = dataSize;
    result.minTimeMs = std::numeric_limits<double>::max();
    result.maxTimeMs = 0.0;
    result.totalTimeMs = 0.0;

    try {
        for (int run = 0; run < numRuns; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            double timeMs = duration.count() / 1000000.0;

            result.totalTimeMs += timeMs;
            result.minTimeMs = std::min(result.minTimeMs, timeMs);
            result.maxTimeMs = std::max(result.maxTimeMs, timeMs);
        }

        result.avgTimeMs = result.totalTimeMs / numRuns;
    }
    catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        result.avgTimeMs = result.minTimeMs = result.maxTimeMs = result.totalTimeMs = 0.0;
    }

    return result;
}

void Benchmark::printResults(const std::vector<Result>& results) {
    std::cout << "\n=== Benchmark Results ===\n";
    std::cout << std::left << std::setw(25) << "Method"
              << std::setw(15) << "Avg Time (ms)"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(25) << result.name;
        if (result.success) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(3) << result.avgTimeMs
                      << std::setw(10) << "SUCCESS";
        } else {
            std::cout << std::setw(15) << "N/A"
                      << std::setw(10) << "FAILED";
        }
        std::cout << "\n";
        if (!result.success && !result.error.empty()) {
            std::cout << "  Error: " << result.error << "\n";
        }
    }

    if (results.size() >= 2 && results[0].success && results[1].success) {
        double speedup = calculateSpeedup(results[1], results[0]);
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
}

void Benchmark::printDetailedResults(const std::vector<Result>& results) {
    std::cout << "\n=== Detailed Benchmark Results ===\n";
    std::cout << std::left
              << std::setw(20) << "Method"
              << std::setw(12) << "Runs"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(15) << "Total (ms)"
              << std::setw(10) << "Status" << "\n";
    std::cout << std::string(93, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.name;
        if (result.success) {
            std::cout << std::setw(12) << result.runs
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.avgTimeMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.minTimeMs
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.maxTimeMs
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.totalTimeMs
                      << std::setw(10) << "SUCCESS";
        } else {
            std::cout << std::setw(12) << "N/A"
                      << std::setw(12) << "N/A"
                      << std::setw(12) << "N/A"
                      << std::setw(12) << "N/A"
                      << std::setw(15) << "N/A"
                      << std::setw(10) << "FAILED";
        }
        std::cout << "\n";
        if (!result.success && !result.error.empty()) {
            std::cout << "  Error: " << result.error << "\n";
        }
    }

    if (results.size() >= 2 && results[0].success && results[1].success) {
        double speedup = calculateSpeedup(results[1], results[0]);
        std::cout << "\nSpeedup (avg): " << std::fixed << std::setprecision(2) << speedup << "x\n";
        double bestSpeedup = calculateSpeedup(results[1], results[0], true);
        std::cout << "Speedup (best): " << std::fixed << std::setprecision(2) << bestSpeedup << "x\n";
    }
}

std::vector<std::complex<float>> Benchmark::generateTestData(size_t size) {
    std::vector<std::complex<float>> data(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = std::complex<float>(dis(gen), dis(gen));
    }

    return data;
}

std::vector<float> Benchmark::generateMatrixTestData(size_t size) {
    std::vector<float> data(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }

    return data;
}

bool Benchmark::verifyResults(const std::vector<std::complex<float>>& result1,
                             const std::vector<std::complex<float>>& result2,
                             float tolerance) {
    if (result1.size() != result2.size()) {
        return false;
    }

    for (size_t i = 0; i < result1.size(); ++i) {
        float diff_real = std::abs(result1[i].real() - result2[i].real());
        float diff_imag = std::abs(result1[i].imag() - result2[i].imag());

        if (diff_real > tolerance || diff_imag > tolerance) {
            return false;
        }
    }

    return true;
}

bool Benchmark::verifyMatrixResults(const std::vector<float>& result1,
                                   const std::vector<float>& result2,
                                   float tolerance) {
    if (result1.size() != result2.size()) {
        return false;
    }

    for (size_t i = 0; i < result1.size(); ++i) {
        float diff = std::abs(result1[i] - result2[i]);
        if (diff > tolerance) {
            return false;
        }
    }

    return true;
}

double Benchmark::calculateSpeedup(const Result& baseline, const Result& comparison, bool useBest) {
    if (!baseline.success || !comparison.success) {
        return 0.0;
    }

    double baselineTime = useBest ? baseline.minTimeMs : baseline.avgTimeMs;
    double comparisonTime = useBest ? comparison.minTimeMs : comparison.avgTimeMs;

    return baselineTime / comparisonTime;
}

std::string Benchmark::formatTime(double timeMs) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);

    if (timeMs < 1.0) {
        ss << (timeMs * 1000.0) << " μs";
    } else if (timeMs < 1000.0) {
        ss << timeMs << " ms";
    } else {
        ss << (timeMs / 1000.0) << " s";
    }

    return ss.str();
}