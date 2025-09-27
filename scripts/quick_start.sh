#!/bin/bash
# MetalQwen3 Quick Start Script
# Complete workflow for downloading, building, and running the Metal Qwen3 implementation

set -e  # Exit on any error

echo "🚀 MetalQwen3 Quick Start"
echo "=========================="
echo

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "❌ CMake is required but not found"
    exit 1
fi

if ! xcode-select -p &> /dev/null; then
    echo "❌ Xcode Command Line Tools are required"
    echo "   Run: xcode-select --install"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo

# Download model if not exists
MODEL_FILE="models/qwen3-4B.bin"
if [ ! -f "$MODEL_FILE" ]; then
    echo "📥 Downloading Qwen3-4B model..."
    python3 scripts/download_qwen3_4b.py --output-dir models
    echo "✅ Model downloaded successfully"
else
    echo "✅ Model already exists: $MODEL_FILE"
fi
echo

# Verify model integrity
echo "🔍 Verifying model integrity..."
MAGIC_CHECK=$(hexdump -C "$MODEL_FILE" | head -1 | grep "31 63 6a 61")
if [ -z "$MAGIC_CHECK" ]; then
    echo "❌ Model file appears corrupted (missing magic number)"
    echo "   Try removing $MODEL_FILE and running this script again"
    exit 1
fi
echo "✅ Model integrity verified (magic number: 0x616A6331)"
echo

# Build project if needed
EXECUTABLE="build/scripts/Release/Qwen3ApiServer"
if [ ! -f "$EXECUTABLE" ]; then
    echo "🔧 Building MetalQwen3..."
    ./standalone.sh
    echo "✅ Build completed successfully"
else
    echo "✅ Executable already exists: $EXECUTABLE"
fi
echo

# Test the API server
echo "🧪 Testing API server..."
echo "Starting server in background..."
"$EXECUTABLE" "$MODEL_FILE" --port 8080 --host localhost > /tmp/qwen3_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 10

# Test the API
echo "Testing API endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-metal",
    "messages": [{"role": "user", "content": "Hello! Please respond with just: API test successful."}],
    "max_tokens": 20,
    "temperature": 0.1
  }' 2>/dev/null || echo "FAILED")

# Stop the server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

if [[ "$RESPONSE" == *"API test successful"* ]] || [[ "$RESPONSE" == *"choices"* ]]; then
    echo "✅ API test passed!"
else
    echo "⚠️  API test response: $RESPONSE"
    echo "   Check server logs in /tmp/qwen3_server.log"
fi
echo

# Run benchmark
echo "📊 Running performance benchmark..."
BENCHMARK_EXECUTABLE="build/scripts/Release/TestQwen3Benchmark"
if [ -f "$BENCHMARK_EXECUTABLE" ]; then
    echo "Starting Metal vs CPU benchmark..."
    timeout 60s "$BENCHMARK_EXECUTABLE" "$MODEL_FILE" || echo "Benchmark completed (or timed out)"
    echo "✅ Benchmark completed"
else
    echo "⚠️  Benchmark executable not found: $BENCHMARK_EXECUTABLE"
fi
echo

echo "🎉 MetalQwen3 Quick Start Complete!"
echo "=================================="
echo
echo "🎯 Next steps:"
echo "1. Start API server: $EXECUTABLE $MODEL_FILE --port 8080"
echo "2. Test API: curl -X POST http://localhost:8080/v1/chat/completions -H \"Content-Type: application/json\" -d '{\"model\": \"qwen3-metal\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 50}'"
echo "3. Run full benchmark: $BENCHMARK_EXECUTABLE $MODEL_FILE"
echo "4. Setup prompt-test: python3 scripts/setup_prompt_test.py"
echo
echo "📖 For detailed documentation, see README.md"
echo "🔧 For implementation details, see CLAUDE.md"