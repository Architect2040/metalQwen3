#include <iostream>
#include <string>
#include <vector>
#include "MetalContext.h"
#include "Qwen3Benchmark.h"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <checkpoint_file> [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --ctx-length <N>     Set context length (default: model default)\n";
    std::cout << "  --warmup <N>         Set warmup iterations (default: 10)\n";
    std::cout << "  --iterations <N>     Set benchmark iterations (default: 100)\n";
    std::cout << "  --output <file>      Save results to CSV file\n";
    std::cout << "  --help               Show this help message\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " qwen3-1.5B-instruct.bin --iterations 50 --output results.csv\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string checkpoint_file = argv[1];
    int ctx_length = 0;
    int warmup_iterations = 10;
    int benchmark_iterations = 100;
    std::string output_file;

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--ctx-length" && i + 1 < argc) {
            ctx_length = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            benchmark_iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== Qwen3 Metal vs CPU Benchmark ===" << std::endl;
    std::cout << "Checkpoint file: " << checkpoint_file << std::endl;
    std::cout << "Context length: " << (ctx_length > 0 ? std::to_string(ctx_length) : "default") << std::endl;
    std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
    std::cout << "Benchmark iterations: " << benchmark_iterations << std::endl;
    if (!output_file.empty()) {
        std::cout << "Output file: " << output_file << std::endl;
    }
    std::cout << std::endl;

    try {
        // Initialize Metal context
        std::cout << "Initializing Metal context..." << std::endl;
        MetalContext metalContext;
        if (!metalContext.initialize()) {
            std::cerr << "Failed to initialize Metal context" << std::endl;
            return 1;
        }

        // Create benchmark suite
        Qwen3Benchmark benchmark(metalContext);
        benchmark.setWarmupIterations(warmup_iterations);
        benchmark.setBenchmarkIterations(benchmark_iterations);

        // Load models
        std::cout << "Loading models..." << std::endl;
        if (!benchmark.loadModels(checkpoint_file, ctx_length)) {
            std::cerr << "Failed to load models" << std::endl;
            return 1;
        }

        // Run benchmark suite
        auto results = benchmark.runFullBenchmarkSuite();

        // Generate report
        std::cout << "\n=== Final Report ===" << std::endl;
        benchmark.generateReport(results, output_file);

        std::cout << "\nBenchmark completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}
