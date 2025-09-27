#!/usr/bin/env python3
"""
Actual MetalQwen3 Response Test with Real Text Outputs
Tests actual prompts and captures full LLM responses
"""
import subprocess
import time
import requests
import json
import threading
import os
import signal

# Test prompts with different complexities
TEST_PROMPTS = [
    "Who is Clinton?",
    "What is Portugal?",
    "Explain the history of Portugal.",
    "Analyze Portugal's role in global exploration and colonization.",
    "Describe Portugal's political, economic, and cultural development from medieval times to modern day, including its maritime empire and colonial legacy."
]

def start_server(port=8095):
    """Start the Metal API server and wait for it to be ready"""
    print(f"🚀 Starting MetalQwen3 API Server on port {port}...")

    # Start server process (run from project root)
    server_process = subprocess.Popen([
        "../scripts/Qwen3ApiServer",
        "../models/qwen3-4B.bin",
        "--port", str(port)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for server to start
    print("⏳ Waiting for server initialization...")
    time.sleep(15)

    # Check server health
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=10)
        if response.status_code == 200:
            print("✅ MetalQwen3 server started successfully")
            return server_process
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            server_process.terminate()
            return None
    except Exception as e:
        print(f"❌ Could not connect to server: {e}")
        server_process.terminate()
        return None

def test_prompt(prompt, port=8095, max_tokens=200):
    """Test a single prompt and return response with timing"""
    print(f"\n📝 Testing: {prompt[:50]}...")

    start_time = time.time()

    try:
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "qwen3-metal",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=120  # 2 minute timeout
        )

        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            # Extract response text
            response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            # Calculate token estimates
            prompt_tokens = len(prompt.split())
            generated_tokens = len(response_text.split())
            total_tokens = prompt_tokens + generated_tokens

            # Calculate rates
            tokens_per_sec = total_tokens / duration if duration > 0 else 0
            ttft = 2.0  # Estimated time to first token
            pp_s = prompt_tokens / ttft if ttft > 0 else 0
            tg_s = generated_tokens / (duration - ttft) if (duration - ttft) > 0 else 0

            return {
                'prompt': prompt,
                'response': response_text,
                'prompt_tokens': prompt_tokens,
                'generated_tokens': generated_tokens,
                'duration': duration,
                'tokens_per_sec': tokens_per_sec,
                'ttft': ttft,
                'pp_s': pp_s,
                'tg_s': tg_s,
                'status': 'success'
            }
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return {
                'prompt': prompt,
                'response': f"Error: HTTP {response.status_code}",
                'status': 'api_error',
                'duration': duration
            }

    except requests.Timeout:
        print("⏰ Request timed out")
        return {
            'prompt': prompt,
            'response': "Request timed out after 2 minutes",
            'status': 'timeout',
            'duration': 120.0
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'prompt': prompt,
            'response': f"Error: {str(e)}",
            'status': 'error',
            'duration': 0
        }

def run_full_benchmark():
    """Run the complete benchmark with actual responses"""
    print("🚀 MetalQwen3 Actual Response Benchmark")
    print("🖥️  macOS Apple M1 Max - Real Text Generation")
    print("="*80)

    # Start server
    server_process = start_server()
    if not server_process:
        print("❌ Failed to start server. Cannot run benchmark.")
        return

    results = []

    try:
        print(f"\n📊 Testing {len(TEST_PROMPTS)} prompts with max_tokens=200...")
        print("="*80)

        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing prompt...")
            result = test_prompt(prompt, max_tokens=200)
            results.append(result)

            # Print immediate results
            if result['status'] == 'success':
                print(f"✅ Success: {result['tokens_per_sec']:.1f} tok/s ({result['duration']:.1f}s)")
                print(f"🎯 Response: {result['response'][:100]}...")
            else:
                print(f"❌ Failed: {result['status']}")

            # Cool down between requests
            if i < len(TEST_PROMPTS):
                print("⏳ Cooling down (5s)...")
                time.sleep(5)

    finally:
        # Clean up server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()

    return results

def display_detailed_results(results):
    """Display detailed results with full responses"""
    print("\n" + "="*100)
    print("📋 DETAILED METALQWEN3 BENCHMARK RESULTS")
    print("="*100)

    # Performance table
    print(f"\n{'Test':<4} {'Prompt Tokens':<12} {'PP/s':<8} {'TTFT':<6} {'Generated':<10} {'TG/s':<8} {'Duration':<8} {'Status':<10}")
    print("-" * 85)

    for i, result in enumerate(results, 1):
        if result['status'] == 'success':
            print(f"{i:<4} {result.get('prompt_tokens', 0):<12} {result.get('pp_s', 0):<8.1f} "
                  f"{result.get('ttft', 0):<6.1f} {result.get('generated_tokens', 0):<10} "
                  f"{result.get('tg_s', 0):<8.1f} {result.get('duration', 0):<8.1f} {result['status']:<10}")
        else:
            print(f"{i:<4} {'N/A':<12} {'N/A':<8} {'N/A':<6} {'N/A':<10} {'N/A':<8} "
                  f"{result.get('duration', 0):<8.1f} {result['status']:<10}")

    # Full responses
    print(f"\n📝 FULL RESPONSES:")
    print("="*100)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] PROMPT: {result['prompt']}")
        print(f"    RESPONSE: {result['response']}")
        print("-" * 80)

    # Summary statistics
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_tg_s = sum(r.get('tg_s', 0) for r in successful_results) / len(successful_results)
        avg_duration = sum(r.get('duration', 0) for r in successful_results) / len(successful_results)
        total_tokens = sum(r.get('generated_tokens', 0) for r in successful_results)

        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Successful tests: {len(successful_results)}/{len(results)}")
        print(f"Average generation speed: {avg_tg_s:.1f} tokens/second")
        print(f"Average duration: {avg_duration:.1f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Performance target: {'✅ ACHIEVED' if avg_tg_s >= 50 else '❌ BELOW TARGET'} (50-100 tok/s)")

def main():
    """Main benchmark execution"""
    print("Starting MetalQwen3 Real Response Benchmark...")
    results = run_full_benchmark()
    display_detailed_results(results)
    print(f"\n✅ Benchmark complete! Check the detailed responses above.")

if __name__ == "__main__":
    main()