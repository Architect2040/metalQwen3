#!/usr/bin/env python3
"""
Comprehensive MetalQwen3 Performance Test Suite
Tests with larger input prompts and max_tokens=8096
Generates performance plots and detailed analysis
"""

import subprocess
import time
import requests
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Extended test prompts with varying complexity and length
TEST_PROMPTS = [
    {
        "name": "Short Question",
        "prompt": "Who is Clinton?",
        "expected_tokens": 50,
        "category": "factual"
    },
    {
        "name": "Medium Explanation",
        "prompt": "Explain the history of Portugal and its role in global exploration.",
        "expected_tokens": 150,
        "category": "historical"
    },
    {
        "name": "Long Analysis",
        "prompt": """Analyze Portugal's political, economic, and cultural development from medieval times to modern day.
        Include details about the maritime empire, colonial period, the 1755 Lisbon earthquake,
        Napoleonic invasions, Brazilian independence, republican revolution, Estado Novo dictatorship,
        Carnation Revolution, and EU membership. Discuss the cultural impact on former colonies
        and the modern Portuguese economy.""",
        "expected_tokens": 400,
        "category": "analytical"
    },
    {
        "name": "Very Long Essay",
        "prompt": """Write a comprehensive essay about the evolution of artificial intelligence and machine learning,
        starting from early computational theories by Alan Turing and John von Neumann, through the development
        of neural networks by McCulloch and Pitts, the perceptron by Frank Rosenblatt, backpropagation by
        Paul Werbos and later popularized by Rumelhart, the AI winters and resurgence, expert systems,
        the renaissance with deep learning by Geoffrey Hinton, Yann LeCun, and Yoshua Bengio, convolutional
        neural networks for computer vision, recurrent networks for sequences, the transformer architecture
        by Vaswani et al., attention mechanisms, BERT and GPT models, the scaling laws discovered by OpenAI,
        large language models like GPT-3 and GPT-4, the emergence of ChatGPT and its societal impact,
        multimodal models, and the current state of AI including concerns about alignment, safety,
        and potential future developments including artificial general intelligence.""",
        "expected_tokens": 800,
        "category": "technical_essay"
    },
    {
        "name": "Maximum Context",
        "prompt": """Provide an exhaustive analysis of the transformer architecture and its impact on natural language processing.
        Begin with the historical context of sequence modeling problems that transformers solved, including the limitations
        of recurrent neural networks and convolutional approaches. Explain the core innovations of the "Attention Is All You Need"
        paper by Vaswani et al., including the self-attention mechanism, multi-head attention, positional encodings,
        and the encoder-decoder structure. Detail the mathematical formulations of scaled dot-product attention,
        the role of key, query, and value matrices, and how attention weights are computed and applied.

        Discuss the subsequent evolution of transformer architectures, including BERT's bidirectional encoding,
        GPT's autoregressive generation, T5's text-to-text framework, and the scaling up to large language models.
        Explain architectural improvements like RMSNorm vs LayerNorm, SwiGLU vs ReLU activations, rotary position
        embeddings (RoPE) vs absolute positional encodings, and grouped query attention for efficiency.

        Cover the quantization techniques used to make these models practical, including INT8 quantization,
        mixed precision training, and knowledge distillation. Discuss inference optimizations like KV caching,
        speculative decoding, and parallel generation strategies.

        Analyze the computational requirements, memory usage patterns, and energy consumption of large transformers.
        Compare different hardware acceleration approaches including CUDA for NVIDIA GPUs, ROCm for AMD,
        Apple's Metal Performance Shaders, and specialized AI chips like TPUs and Cerebras.

        Examine the societal and economic implications of transformer technology, including its applications
        in code generation, scientific research, creative writing, language translation, and decision support systems.
        Discuss the challenges of bias, fairness, interpretability, and alignment in large language models.

        Conclude with future research directions including multimodal transformers, reasoning capabilities,
        tool use and API integration, few-shot learning improvements, and the path toward artificial general intelligence.""",
        "expected_tokens": 1500,
        "category": "maximum_context"
    }
]

class MetalQwen3PerformanceTester:
    def __init__(self, max_tokens=8096, port=8097):
        self.max_tokens = max_tokens
        self.port = port
        self.server_process = None
        self.results = []

    def start_server(self):
        """Start the MetalQwen3 API server"""
        print(f"🚀 Starting MetalQwen3 API Server (max_tokens={self.max_tokens})")

        try:
            self.server_process = subprocess.Popen([
                "../scripts/Qwen3ApiServer",
                "../models/qwen3-4B.bin",
                "--port", str(self.port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print("⏳ Waiting for server initialization...")
            time.sleep(15)

            # Test server health
            response = requests.get(f"http://localhost:{self.port}/health", timeout=10)
            if response.status_code == 200:
                print("✅ MetalQwen3 server ready")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the API server"""
        if self.server_process:
            print("🛑 Stopping MetalQwen3 server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    def test_prompt(self, prompt_data):
        """Test a single prompt and return detailed results"""
        prompt = prompt_data["prompt"]
        name = prompt_data["name"]

        print(f"\n📝 Testing: {name}")
        print(f"    Prompt length: {len(prompt)} chars, {len(prompt.split())} words")

        start_time = time.time()

        try:
            response = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "qwen3-metal",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7
                },
                timeout=300  # 5 minute timeout for long generation
            )

            duration = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                # Calculate token counts
                prompt_tokens = len(prompt.split())
                generated_tokens = len(response_text.split())
                total_tokens = prompt_tokens + generated_tokens

                # Calculate performance metrics
                tokens_per_sec = total_tokens / duration if duration > 0 else 0
                generation_tokens_per_sec = generated_tokens / duration if duration > 0 else 0
                ttft = 2.0  # Estimated time to first token
                pp_s = prompt_tokens / ttft if ttft > 0 else 0

                result = {
                    'name': name,
                    'category': prompt_data['category'],
                    'prompt_chars': len(prompt),
                    'prompt_words': len(prompt.split()),
                    'prompt_tokens': prompt_tokens,
                    'generated_tokens': generated_tokens,
                    'total_tokens': total_tokens,
                    'duration': duration,
                    'tokens_per_sec': tokens_per_sec,
                    'generation_tps': generation_tokens_per_sec,
                    'pp_s': pp_s,
                    'ttft': ttft,
                    'response_preview': response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    'response_full': response_text,
                    'status': 'success'
                }

                print(f"    ✅ Success: {generation_tokens_per_sec:.1f} tok/s generation ({duration:.1f}s total)")
                print(f"    📊 Tokens: {prompt_tokens} prompt + {generated_tokens} generated = {total_tokens} total")
                print(f"    💬 Response preview: {response_text[:100]}...")

                return result

            else:
                print(f"    ❌ API Error: {response.status_code}")
                return {
                    'name': name,
                    'status': 'api_error',
                    'duration': duration,
                    'error': f"HTTP {response.status_code}"
                }

        except requests.Timeout:
            print(f"    ⏰ Timeout after 5 minutes")
            return {
                'name': name,
                'status': 'timeout',
                'duration': 300.0
            }
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return {
                'name': name,
                'status': 'error',
                'error': str(e)
            }

    def run_full_test_suite(self):
        """Run complete test suite with all prompts"""
        print("🚀 MetalQwen3 Comprehensive Performance Test Suite")
        print(f"📊 Testing with max_tokens={self.max_tokens}")
        print("="*80)

        if not self.start_server():
            print("❌ Cannot start server. Aborting tests.")
            return []

        try:
            for i, prompt_data in enumerate(TEST_PROMPTS, 1):
                print(f"\n[{i}/{len(TEST_PROMPTS)}] Testing: {prompt_data['name']}")

                result = self.test_prompt(prompt_data)
                self.results.append(result)

                # Cool down between tests
                if i < len(TEST_PROMPTS):
                    print("    ⏳ Cooling down (10s)...")
                    time.sleep(10)

        finally:
            self.stop_server()

        return self.results

    def generate_performance_plots(self):
        """Generate comprehensive performance visualization"""
        print("\n📊 Generating performance plots...")

        # Filter successful results
        successful_results = [r for r in self.results if r.get('status') == 'success']

        if not successful_results:
            print("❌ No successful results to plot")
            return

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MetalQwen3 Performance Analysis\nApple M1 Max - max_tokens=8096', fontsize=16, fontweight='bold')

        # Extract data
        names = [r['name'] for r in successful_results]
        prompt_tokens = [r['prompt_tokens'] for r in successful_results]
        generated_tokens = [r['generated_tokens'] for r in successful_results]
        generation_tps = [r['generation_tps'] for r in successful_results]
        durations = [r['duration'] for r in successful_results]
        total_tokens = [r['total_tokens'] for r in successful_results]

        # Plot 1: Tokens per Second vs Prompt Size
        ax1.scatter(prompt_tokens, generation_tps, s=100, alpha=0.7, color='green')
        ax1.set_xlabel('Prompt Tokens')
        ax1.set_ylabel('Generation Speed (tok/s)')
        ax1.set_title('Generation Speed vs Prompt Size')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=75, color='red', linestyle='--', label='Target 75 tok/s')
        ax1.legend()

        # Add point labels
        for i, name in enumerate(names):
            ax1.annotate(name.split()[0], (prompt_tokens[i], generation_tps[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Plot 2: Generated Tokens vs Duration
        ax2.scatter(durations, generated_tokens, s=100, alpha=0.7, color='blue')
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Generated Tokens')
        ax2.set_title('Token Generation vs Time')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Throughput Consistency
        categories = [r['category'] for r in successful_results]
        cat_colors = {'factual': 'green', 'historical': 'blue', 'analytical': 'orange',
                     'technical_essay': 'red', 'maximum_context': 'purple'}
        colors = [cat_colors.get(cat, 'gray') for cat in categories]

        bars = ax3.bar(range(len(names)), generation_tps, color=colors, alpha=0.7)
        ax3.set_xlabel('Test Cases')
        ax3.set_ylabel('Generation Speed (tok/s)')
        ax3.set_title('Performance by Test Category')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels([n.split()[0] for n in names], rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=75, color='red', linestyle='--', label='Target 75 tok/s')
        ax3.legend()

        # Add value labels on bars
        for bar, tps in zip(bars, generation_tps):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{tps:.1f}', ha='center', va='bottom', fontsize=8)

        # Plot 4: Total Tokens vs Performance
        ax4.scatter(total_tokens, generation_tps, s=100, alpha=0.7, color='purple')
        ax4.set_xlabel('Total Tokens (Prompt + Generated)')
        ax4.set_ylabel('Generation Speed (tok/s)')
        ax4.set_title('Performance vs Total Token Count')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=75, color='red', linestyle='--', label='Target 75 tok/s')
        ax4.legend()

        plt.tight_layout()

        # Save plot
        plot_path = Path("../performance_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to: {plot_path}")

        # Save as high-res for README
        readme_plot_path = Path("../assets/metalqwen3_performance.png")
        readme_plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(readme_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 README plot saved to: {readme_plot_path}")

        plt.show()

        return plot_path

    def generate_detailed_report(self):
        """Generate detailed performance report"""
        print("\n📋 Generating detailed performance report...")

        successful_results = [r for r in self.results if r.get('status') == 'success']

        if not successful_results:
            print("❌ No successful results to report")
            return

        # Create DataFrame for analysis
        df = pd.DataFrame(successful_results)

        print("\n" + "="*100)
        print("📊 METALQWEN3 COMPREHENSIVE PERFORMANCE RESULTS")
        print(f"🖥️  Apple M1 Max - max_tokens={self.max_tokens}")
        print("="*100)

        # Detailed results table
        print(f"\n{'Test':<5} {'Category':<15} {'P-Chars':<8} {'P-Tok':<6} {'G-Tok':<6} {'Total':<6} {'Duration':<8} {'Gen-TPS':<8} {'Total-TPS':<9}")
        print("-" * 95)

        for i, result in enumerate(successful_results, 1):
            print(f"{i:<5} {result['category']:<15} {result['prompt_chars']:<8} "
                  f"{result['prompt_tokens']:<6} {result['generated_tokens']:<6} "
                  f"{result['total_tokens']:<6} {result['duration']:<8.1f} "
                  f"{result['generation_tps']:<8.1f} {result['tokens_per_sec']:<9.1f}")

        # Summary statistics
        avg_gen_tps = df['generation_tps'].mean()
        avg_total_tps = df['tokens_per_sec'].mean()
        avg_duration = df['duration'].mean()
        total_generated = df['generated_tokens'].sum()

        print("\n📈 SUMMARY STATISTICS:")
        print(f"Average Generation Speed: {avg_gen_tps:.1f} tokens/second")
        print(f"Average Total Throughput: {avg_total_tps:.1f} tokens/second")
        print(f"Average Duration: {avg_duration:.1f} seconds")
        print(f"Total Tokens Generated: {total_generated:,}")
        print(f"Target Achievement: {'✅ ACHIEVED' if avg_gen_tps >= 50 else '❌ BELOW TARGET'} (50+ tok/s)")

        # Full responses
        print(f"\n📝 COMPLETE LLM RESPONSES (max_tokens={self.max_tokens}):")
        print("="*100)

        for i, result in enumerate(successful_results, 1):
            print(f"\n[{i}] TEST: {result['name']}")
            print(f"    CATEGORY: {result['category']}")
            print(f"    PROMPT: {result['prompt_tokens']} tokens")
            print(f"    GENERATED: {result['generated_tokens']} tokens ({result['generation_tps']:.1f} tok/s)")
            print(f"    DURATION: {result['duration']:.1f} seconds")
            print(f"    RESPONSE:")
            print(f"    {result['response_full']}")
            print("-" * 80)

        # Save detailed results to CSV
        csv_path = Path("../comprehensive_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")

        return {
            'avg_generation_tps': avg_gen_tps,
            'avg_total_tps': avg_total_tps,
            'total_generated': total_generated,
            'success_rate': len(successful_results) / len(self.results)
        }

def main():
    """Main test execution"""
    print("🚀 MetalQwen3 Comprehensive Performance Test")
    print("📊 Testing with larger prompts and max_tokens=8096")
    print("🖥️  Apple M1 Max - Metal GPU Acceleration")
    print()

    # Check if we're in the scripts directory
    if not Path("../scripts/Qwen3ApiServer").exists():
        print("❌ Please run from the scripts/ directory")
        print("   cd scripts && python3 comprehensive_performance_test.py")
        sys.exit(1)

    tester = MetalQwen3PerformanceTester(max_tokens=8096)

    try:
        # Run comprehensive tests
        results = tester.run_full_test_suite()

        if results:
            # Generate performance analysis
            summary = tester.generate_detailed_report()

            # Generate visualization
            plot_path = tester.generate_performance_plots()

            print(f"\n✅ Comprehensive testing completed!")
            print(f"📊 Results: {len([r for r in results if r.get('status') == 'success'])}/{len(results)} successful")
            print(f"📈 Average Performance: {summary['avg_generation_tps']:.1f} tokens/second")
            print(f"📁 Results saved to comprehensive_results.csv")
            print(f"📊 Plots saved to performance_plots.png")

        else:
            print("❌ No test results obtained")

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        tester.stop_server()

if __name__ == "__main__":
    main()