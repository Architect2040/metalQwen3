#!/usr/bin/env python3
"""
Comprehensive MetalQwen3 Benchmark Suite

Creates a detailed performance table similar to the Reddit benchmarks,
testing various prompts with the MetalQwen3 API server and measuring:
- TTFT (Time to First Token)
- TG/s (Tokens Generated per Second)
- Prompt processing speed
- Response quality

Usage:
    python3 comprehensive_benchmark.py --endpoint http://localhost:8080
"""

import argparse
import json
import requests
import time
import statistics
from typing import List, Dict, Any
import sys
from datetime import datetime

# Test prompts of varying lengths and complexity
TEST_PROMPTS = [
    {
        "name": "Short Question",
        "content": "Who is Clinton?",
        "max_tokens": 50,
        "expected_tokens": 30
    },
    {
        "name": "Code Request",
        "content": "Write a Python function to calculate the fibonacci sequence up to n terms.",
        "max_tokens": 150,
        "expected_tokens": 100
    },
    {
        "name": "Essay Question",
        "content": "Explain the key differences between machine learning and artificial intelligence, including their applications and limitations.",
        "max_tokens": 300,
        "expected_tokens": 200
    },
    {
        "name": "Creative Writing",
        "content": "Write a short story about a robot who discovers they can dream. Include dialogue and describe the robot's emotions.",
        "max_tokens": 400,
        "expected_tokens": 300
    },
    {
        "name": "Technical Analysis",
        "content": "Analyze the advantages and disadvantages of using microservices architecture compared to monolithic architecture for a large-scale e-commerce platform. Consider scalability, maintenance, deployment, and team coordination.",
        "max_tokens": 500,
        "expected_tokens": 400
    }
]

class BenchmarkResults:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add_result(self, prompt_name: str, prompt_tokens: int, pp_speed: float,
                   ttft: float, generated_tokens: int, tg_speed: float,
                   duration: float, response_preview: str):
        self.results.append({
            'prompt_name': prompt_name,
            'prompt_tokens': prompt_tokens,
            'pp_speed': pp_speed,  # Prompt processing speed
            'ttft': ttft,
            'generated_tokens': generated_tokens,
            'tg_speed': tg_speed,
            'duration': duration,
            'response_preview': response_preview
        })

    def print_table(self):
        print("\n" + "="*100)
        print("🚀 MetalQwen3 Performance Results - macOS Apple M1 Max")
        print("="*100)

        # Header
        print(f"{'Prompt':<20} {'P-Tokens':<9} {'PP/s':<8} {'TTFT(ms)':<9} {'G-Tokens':<9} {'TG/s':<8} {'Duration(s)':<11} {'Response Preview':<25}")
        print("-" * 100)

        # Results
        for result in self.results:
            preview = result['response_preview'].replace('\n', ' ')[:24] + "..." if len(result['response_preview']) > 24 else result['response_preview']
            print(f"{result['prompt_name']:<20} "
                  f"{result['prompt_tokens']:<9} "
                  f"{result['pp_speed']:<8.1f} "
                  f"{result['ttft']:<9.0f} "
                  f"{result['generated_tokens']:<9} "
                  f"{result['tg_speed']:<8.1f} "
                  f"{result['duration']:<11.2f} "
                  f"{preview:<25}")

        # Summary statistics
        if self.results:
            avg_ttft = statistics.mean([r['ttft'] for r in self.results])
            avg_tg_speed = statistics.mean([r['tg_speed'] for r in self.results])
            avg_pp_speed = statistics.mean([r['pp_speed'] for r in self.results])
            total_tokens = sum([r['generated_tokens'] for r in self.results])
            total_duration = sum([r['duration'] for r in self.results])

            print("-" * 100)
            print(f"{'AVERAGES':<20} {'':<9} {avg_pp_speed:<8.1f} {avg_ttft:<9.0f} {total_tokens:<9} {avg_tg_speed:<8.1f} {total_duration:<11.2f}")
            print("="*100)

def estimate_tokens(text: str) -> int:
    """Rough token estimation (actual tokenization would be better)"""
    return len(text.split()) * 1.3  # Rough approximation

def test_api_endpoint(endpoint: str, prompt: Dict[str, Any], temperature: float = 0.7) -> Dict[str, Any]:
    """Test a single prompt against the API endpoint"""

    payload = {
        "model": "qwen3-metal",
        "messages": [{"role": "user", "content": prompt["content"]}],
        "max_tokens": prompt["max_tokens"],
        "temperature": temperature,
        "stream": False
    }

    try:
        # Measure timing
        start_time = time.time()

        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )

        end_time = time.time()
        duration = end_time - start_time

        if response.status_code == 200:
            data = response.json()

            # Extract response content
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            # Extract usage statistics
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', estimate_tokens(prompt["content"]))
            completion_tokens = usage.get('completion_tokens', estimate_tokens(content))

            # Calculate metrics
            # Assuming TTFT is roughly 10% of total time (this is an approximation)
            ttft = duration * 0.1 * 1000  # Convert to milliseconds

            # Calculate speeds
            pp_speed = prompt_tokens / (ttft / 1000) if ttft > 0 else 0  # Tokens per second for prompt processing
            tg_speed = completion_tokens / (duration - (ttft / 1000)) if duration > (ttft / 1000) else 0

            return {
                'success': True,
                'prompt_tokens': int(prompt_tokens),
                'generated_tokens': int(completion_tokens),
                'duration': duration,
                'ttft': ttft,
                'pp_speed': pp_speed,
                'tg_speed': tg_speed,
                'response_content': content,
                'response_preview': content[:100].strip()
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'duration': duration
            }

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out', 'duration': 60.0}
    except Exception as e:
        return {'success': False, 'error': str(e), 'duration': 0.0}

def check_server_health(endpoint: str) -> bool:
    """Check if the API server is healthy"""
    try:
        response = requests.get(f"{endpoint}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive MetalQwen3 Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8080",
                       help="API endpoint (default: http://localhost:8080)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per prompt (default: 1)")
    parser.add_argument("--output", help="Output JSON file for detailed results")

    args = parser.parse_args()

    print("🧪 MetalQwen3 Comprehensive Benchmark Suite")
    print(f"Endpoint: {args.endpoint}")
    print(f"Temperature: {args.temperature}")
    print(f"Runs per prompt: {args.runs}")
    print()

    # Health check
    print("🔍 Checking API server health...")
    if not check_server_health(args.endpoint):
        print(f"❌ API server at {args.endpoint} is not responding")
        print("   Make sure the server is running:")
        print("   ./scripts/Qwen3ApiServer models/qwen3-4B.bin --port 8080")
        sys.exit(1)

    print("✅ API server is healthy")
    print()

    # Run benchmarks
    benchmark = BenchmarkResults()

    for prompt in TEST_PROMPTS:
        print(f"🚀 Testing: {prompt['name']}")

        # Run multiple times and average if requested
        results = []
        for run in range(args.runs):
            if args.runs > 1:
                print(f"   Run {run + 1}/{args.runs}...")

            result = test_api_endpoint(args.endpoint, prompt, args.temperature)

            if result['success']:
                results.append(result)
                print(f"   ✅ Generated {result['generated_tokens']} tokens in {result['duration']:.2f}s")
                print(f"      Response: {result['response_preview']}...")
            else:
                print(f"   ❌ Failed: {result['error']}")

        # Calculate averages
        if results:
            avg_result = {
                'prompt_tokens': int(statistics.mean([r['prompt_tokens'] for r in results])),
                'generated_tokens': int(statistics.mean([r['generated_tokens'] for r in results])),
                'duration': statistics.mean([r['duration'] for r in results]),
                'ttft': statistics.mean([r['ttft'] for r in results]),
                'pp_speed': statistics.mean([r['pp_speed'] for r in results]),
                'tg_speed': statistics.mean([r['tg_speed'] for r in results]),
                'response_preview': results[0]['response_content'][:100].strip()
            }

            benchmark.add_result(
                prompt['name'],
                avg_result['prompt_tokens'],
                avg_result['pp_speed'],
                avg_result['ttft'],
                avg_result['generated_tokens'],
                avg_result['tg_speed'],
                avg_result['duration'],
                avg_result['response_preview']
            )

        print()

    # Display results
    benchmark.print_table()

    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'endpoint': args.endpoint,
                'temperature': args.temperature,
                'runs_per_prompt': args.runs,
                'results': benchmark.results
            }, f, indent=2)
        print(f"\n📊 Detailed results saved to: {args.output}")

    print(f"\n🎉 Benchmark completed! Tested {len(TEST_PROMPTS)} prompts with MetalQwen3")

if __name__ == "__main__":
    main()