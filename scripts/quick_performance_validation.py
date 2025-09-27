#!/usr/bin/env python3
"""
Quick MetalQwen3 Performance Validation
Tests current system with a few prompts to ensure everything works before comprehensive testing
"""

import subprocess
import time
import requests
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Quick validation prompts
VALIDATION_PROMPTS = [
    "Who is Clinton?",
    "What is Portugal?",
    "Explain the history of Portugal in detail."
]

def test_cpu_reference():
    """Quick test of CPU reference for baseline"""
    print("🔧 Testing CPU Reference (qwen3c_original)")
    results = []

    for i, prompt in enumerate(VALIDATION_PROMPTS, 1):
        print(f"[{i}] Testing: {prompt[:30]}...")

        start_time = time.time()
        try:
            result = subprocess.run([
                "../qwen3c_original/runq",
                "../models/qwen3-4B.bin",
                "-m", "generate",
                "-i", prompt
            ], capture_output=True, text=True, timeout=30)

            duration = time.time() - start_time

            if result.returncode == 0 and result.stdout:
                words = len(result.stdout.split())
                tps = words / duration if duration > 0 else 0
                print(f"    ✅ {tps:.1f} tok/s ({duration:.1f}s)")
                results.append({
                    'prompt': prompt[:30],
                    'tokens': words,
                    'duration': duration,
                    'tps': tps,
                    'status': 'success'
                })
            else:
                print(f"    ❌ Failed")
                results.append({'status': 'failed'})

        except subprocess.TimeoutExpired:
            print(f"    ⏰ Timeout")
            results.append({'status': 'timeout'})

    return results

def test_metal_api():
    """Quick test of Metal API server"""
    print("\n🔧 Testing Metal API Server")

    # Start server
    print("🚀 Starting Metal server...")
    server = subprocess.Popen([
        "../scripts/Qwen3ApiServer",
        "../models/qwen3-4B.bin",
        "--port", "8098"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(15)  # Wait for startup

    results = []

    try:
        # Test health
        health = requests.get("http://localhost:8098/health", timeout=5)
        if health.status_code != 200:
            print("❌ Server health check failed")
            return []

        print("✅ Server healthy, testing prompts...")

        for i, prompt in enumerate(VALIDATION_PROMPTS, 1):
            print(f"[{i}] Testing: {prompt[:30]}...")

            start_time = time.time()
            try:
                response = requests.post(
                    "http://localhost:8098/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "qwen3-metal",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                        "temperature": 0.7
                    },
                    timeout=60
                )

                duration = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    tokens = len(response_text.split())
                    tps = tokens / duration if duration > 0 else 0

                    print(f"    ✅ {tps:.1f} tok/s ({duration:.1f}s)")
                    results.append({
                        'prompt': prompt[:30],
                        'tokens': tokens,
                        'duration': duration,
                        'tps': tps,
                        'response': response_text[:100] + "...",
                        'status': 'success'
                    })
                else:
                    print(f"    ❌ API Error: {response.status_code}")
                    results.append({'status': 'api_error'})

            except Exception as e:
                print(f"    ❌ Error: {e}")
                results.append({'status': 'error'})

    finally:
        print("🛑 Stopping server...")
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    return results

def create_validation_plot(cpu_results, metal_results):
    """Create simple validation plot"""
    print("\n📊 Creating validation plot...")

    # Filter successful results
    cpu_success = [r for r in cpu_results if r.get('status') == 'success']
    metal_success = [r for r in metal_results if r.get('status') == 'success']

    if not cpu_success or not metal_success:
        print("❌ Insufficient data for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MetalQwen3 vs CPU Performance Validation', fontsize=14, fontweight='bold')

    # Performance comparison
    prompts = [r['prompt'] for r in cpu_success]
    cpu_tps = [r['tps'] for r in cpu_success]
    metal_tps = [r['tps'] for r in metal_success] if len(metal_success) == len(cpu_success) else [0] * len(cpu_success)

    x = range(len(prompts))
    width = 0.35

    ax1.bar([i - width/2 for i in x], cpu_tps, width, label='CPU (qwen3c)', alpha=0.7, color='orange')
    ax1.bar([i + width/2 for i in x], metal_tps, width, label='Metal GPU', alpha=0.7, color='green')

    ax1.set_xlabel('Test Prompts')
    ax1.set_ylabel('Tokens/Second')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[:10] + "..." for p in prompts], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Response length comparison
    cpu_tokens = [r['tokens'] for r in cpu_success]
    metal_tokens = [r['tokens'] for r in metal_success] if len(metal_success) == len(cpu_success) else [0] * len(cpu_success)

    ax2.bar([i - width/2 for i in x], cpu_tokens, width, label='CPU Generated', alpha=0.7, color='orange')
    ax2.bar([i + width/2 for i in x], metal_tokens, width, label='Metal Generated', alpha=0.7, color='green')

    ax2.set_xlabel('Test Prompts')
    ax2.set_ylabel('Generated Tokens')
    ax2.set_title('Response Length Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p[:10] + "..." for p in prompts], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save validation plot
    plot_path = Path("../validation_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Validation plot saved to: {plot_path}")

    plt.show()

def main():
    """Main validation execution"""
    print("🔍 MetalQwen3 Performance Validation")
    print("📊 Quick test to verify system works before comprehensive testing")
    print()

    # Test CPU reference
    cpu_results = test_cpu_reference()

    # Test Metal API
    metal_results = test_metal_api()

    # Create validation plot
    create_validation_plot(cpu_results, metal_results)

    # Summary
    cpu_success = len([r for r in cpu_results if r.get('status') == 'success'])
    metal_success = len([r for r in metal_results if r.get('status') == 'success'])

    print(f"\n✅ Validation Results:")
    print(f"   CPU Tests: {cpu_success}/{len(VALIDATION_PROMPTS)} successful")
    print(f"   Metal Tests: {metal_success}/{len(VALIDATION_PROMPTS)} successful")

    if cpu_success > 0 and metal_success > 0:
        print(f"✅ System validation PASSED - ready for comprehensive testing")
        print(f"🚀 Run: python3 scripts/comprehensive_performance_test.py")
    else:
        print(f"❌ System validation FAILED - check setup before comprehensive testing")

if __name__ == "__main__":
    main()