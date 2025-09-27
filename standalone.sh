#!/bin/bash

# Metal Qwen3 - Standalone Build Script for macOS
# This script performs a complete build-compile-test cycle

set -e  # Exit on any error

echo "🚀 Starting Metal Qwen3 build process..."

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
echo "📱 macOS Version: $MACOS_VERSION"

# Check for Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode not found. Please install Xcode and Command Line Tools"
    exit 1
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install CMake 3.20+"
    exit 1
fi

# Check for Metal compiler
if ! command -v xcrun &> /dev/null || ! xcrun -find metal &> /dev/null; then
    echo "❌ Metal compiler not found. Ensure Xcode is properly installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Clean previous build artifacts
echo "🧹 Cleaning previous build artifacts..."
rm -rf build/
rm -rf bin/
rm -rf Release/

# Set up Xcode environment
echo "🔧 Setting up Xcode environment..."
export PATH="/Applications/Xcode.app/Contents/Developer/usr/bin:$PATH"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

# Create build directory
mkdir -p build

# Configure with CMake
echo "⚙️  Configuring project with CMake..."
cd build
cmake -DCMAKE_BUILD_TYPE=Release -G "Xcode" ..

# Build the project
echo "🔨 Building Metal Qwen3..."
cmake --build . --config Release

# Check if executables were built
cd ..
if [ ! -f "build/scripts/Release/TestQwen3Benchmark" ]; then
    echo "❌ TestQwen3Benchmark executable not found"
    exit 1
fi

if [ ! -f "build/scripts/Release/Qwen3ApiServer" ]; then
    echo "❌ Qwen3ApiServer executable not found"
    exit 1
fi

echo "✅ Build completed successfully!"

# Executables are built in build/scripts/Release/ directory
echo "📂 Executables built in build/scripts/Release/ directory"
echo "   📄 build/scripts/Release/TestQwen3Benchmark"
echo "   📄 build/scripts/Release/Qwen3ApiServer"

# Metal libraries are in build/scripts/Release/ directory too
echo "   📄 Metal libraries: build/scripts/Release/*.metallib"

echo ""
echo "🎉 Metal Qwen3 Build Complete!"
echo ""
echo "📋 Usage Instructions (UPDATED):"
echo ""
echo "1. Download a model (if needed):"
echo "   python3 scripts/download_qwen3_4b.py --output-dir models"
echo ""
echo "2. Run benchmark tool:"
echo "   ./build/scripts/Release/TestQwen3Benchmark models/qwen3-4B.bin"
echo ""
echo "3. Start optimized API server:"
echo "   ./build/scripts/Release/Qwen3ApiServer models/qwen3-4B.bin --port 8080"
echo ""
echo "4. Test actual responses:"
echo "   python3 scripts/actual_response_test.py"
echo ""
echo "5. Get performance report:"
echo "   python3 scripts/final_performance_report.py"
echo ""
echo "6. Run comprehensive benchmarks:"
echo "   python3 scripts/comprehensive_benchmark.py"
echo ""

# Check Metal support
echo "🔍 Metal GPU Information:"
system_profiler SPDisplaysDataType | grep -A 5 "Metal"
echo ""

echo "Build completed at: $(date)"
echo "Ready for Metal-accelerated Qwen3 inference! 🚀"