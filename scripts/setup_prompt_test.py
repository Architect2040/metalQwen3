#!/usr/bin/env python3
"""
Setup script for prompt-test benchmark with Qwen3 Vulkan API server.
"""

import os
import sys
import subprocess
import requests
import time
import json

def check_server_health(base_url="http://localhost:8080"):
    """Check if the Qwen3 API server is running and healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✓ Server is healthy:")
            print(f"  GPU: {health_data.get('gpu', 'Unknown')}")
            print(f"  Model: {health_data.get('model', 'Unknown')}")
            print(f"  Vocab size: {health_data.get('vocab_size', 'Unknown')}")
            return True
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Server not reachable: {e}")
        return False

def test_chat_completion(base_url="http://localhost:8080"):
    """Test the chat completions endpoint."""
    print("\nTesting chat completions endpoint...")

    try:
        payload = {
            "model": "qwen3-vulkan",
            "messages": [
                {"role": "user", "content": "Hello! Can you tell me a short joke?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data['usage']

            print("✓ Chat completion successful:")
            print(f"  Response: {content[:100]}{'...' if len(content) > 100 else ''}")
            print(f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion")
            return True
        else:
            print(f"✗ Chat completion failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print(f"✗ Chat completion test failed: {e}")
        return False

def setup_prompt_test():
    """Clone and setup the prompt-test repository."""
    print("Setting up prompt-test benchmark...")

    repo_dir = "prompt-test"

    if os.path.exists(repo_dir):
        print(f"✓ {repo_dir} already exists")
    else:
        print("Cloning prompt-test repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/chigkim/prompt-test.git"
            ], check=True)
            print("✓ Cloned prompt-test repository")
        except subprocess.CalledProcessError:
            print("✗ Failed to clone prompt-test repository")
            return False

    # Install required packages
    print("Installing OpenAI package...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "openai"
        ], check=True)
        print("✓ OpenAI package installed")
    except subprocess.CalledProcessError:
        print("✗ Failed to install OpenAI package")
        return False

    return True

def create_test_config(base_url="http://localhost:8080"):
    """Create a test configuration for our Qwen3 server."""

    config_content = f"""#!/usr/bin/env python3
import os
from openai import OpenAI

# Configure for local Qwen3 Vulkan server
client = OpenAI(
    base_url="{base_url}/v1",
    api_key="dummy-key"  # Local server doesn't need real API key
)

# Test parameters
MODEL = "qwen3-vulkan"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# Benchmark configuration
PROMPT_FILE = "prompt.txt"
OUTPUT_FILE = "qwen3_vulkan_results.json"

def run_benchmark():
    \"\"\"Run the benchmark with our Qwen3 Vulkan server.\"\"\"

    # Read prompt
    if not os.path.exists(PROMPT_FILE):
        print(f"Error: {{PROMPT_FILE}} not found!")
        print("Please create a prompt.txt file or copy from prompt-test repository.")
        return

    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    print(f"Running benchmark with prompt length: {{len(prompt)}} characters")
    print(f"Model: {{MODEL}}")
    print(f"Max tokens: {{MAX_TOKENS}}")
    print()

    import time
    results = []

    # Run multiple iterations for statistical significance
    iterations = 3
    for i in range(iterations):
        print(f"Iteration {{i+1}}/{{iterations}}...")

        start_time = time.time()
        first_token_time = None
        generated_tokens = 0

        try:
            # Make streaming request to measure TTFT
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{{"role": "user", "content": prompt}}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.time()

                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    generated_tokens += 1

            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else 0
            generation_time = total_time - ttft

            tokens_per_sec = generated_tokens / generation_time if generation_time > 0 else 0

            result = {{
                "iteration": i + 1,
                "total_time": total_time,
                "ttft": ttft,
                "generation_time": generation_time,
                "generated_tokens": generated_tokens,
                "tokens_per_second": tokens_per_sec,
                "response_length": len(full_response)
            }}

            results.append(result)

            print(f"  TTFT: {{ttft:.3f}}s")
            print(f"  Tokens/sec: {{tokens_per_sec:.2f}}")
            print(f"  Total time: {{total_time:.3f}}s")
            print()

        except Exception as e:
            print(f"  Error: {{e}}")
            continue

    # Calculate averages
    if results:
        avg_ttft = sum(r["ttft"] for r in results) / len(results)
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_total = sum(r["total_time"] for r in results) / len(results)

        summary = {{
            "model": MODEL,
            "iterations": len(results),
            "average_ttft": avg_ttft,
            "average_tokens_per_second": avg_tps,
            "average_total_time": avg_total,
            "individual_results": results
        }}

        # Save results
        import json
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(summary, f, indent=2)

        print("=== BENCHMARK RESULTS ===")
        print(f"Model: {{MODEL}}")
        print(f"Iterations: {{len(results)}}")
        print(f"Average TTFT: {{avg_ttft:.3f}}s")
        print(f"Average Tokens/sec: {{avg_tps:.2f}}")
        print(f"Average Total time: {{avg_total:.3f}}s")
        print(f"Results saved to: {{OUTPUT_FILE}}")
    else:
        print("No successful iterations!")

if __name__ == "__main__":
    run_benchmark()
"""

    with open("qwen3_benchmark.py", "w") as f:
        f.write(config_content)

    print("✓ Created qwen3_benchmark.py")
    return True

def create_sample_prompt():
    """Create a sample prompt file for testing."""

    # Create a prompt similar to what prompt-test uses
    sample_prompt = """You are a helpful AI assistant. Please analyze the following scenario and provide a comprehensive response:

Scenario: A software engineering team is deciding between using a microservices architecture versus a monolithic architecture for their new e-commerce platform. The team consists of 10 developers, and they expect to handle moderate traffic initially but want to scale significantly over the next 2 years.

Please provide:
1. A detailed analysis of both architectural approaches
2. Pros and cons of each approach for this specific scenario
3. Your recommendation with clear reasoning
4. Implementation considerations and potential challenges
5. How the choice might affect team productivity and code maintainability

Consider factors such as development complexity, deployment, monitoring, performance, and scalability in your analysis."""

    with open("prompt.txt", "w") as f:
        f.write(sample_prompt)

    print("✓ Created sample prompt.txt file")
    return True

def main():
    print("=== Qwen3 Vulkan API Server - Benchmark Setup ===")

    # Check if server is running
    if not check_server_health():
        print("\n❌ Qwen3 API server is not running!")
        print("\nTo start the server:")
        print("  bin64\\Qwen3ApiServer.exe model.bin --host localhost --port 8080")
        print("\nThen run this script again to set up benchmarking.")
        return 1

    # Test API functionality
    if not test_chat_completion():
        print("\n❌ API server is not working correctly!")
        return 1

    # Setup prompt-test
    print("\n" + "="*50)
    if not setup_prompt_test():
        print("❌ Failed to set up prompt-test")
        return 1

    # Create test configuration
    if not create_test_config():
        print("❌ Failed to create test configuration")
        return 1

    # Create sample prompt
    if not create_sample_prompt():
        print("❌ Failed to create sample prompt")
        return 1

    print("\n" + "="*50)
    print("🎉 Setup complete!")
    print("\nTo run benchmarks:")
    print("1. Make sure your Qwen3 API server is running:")
    print("   bin64\\Qwen3ApiServer.exe model.bin")
    print("\n2. Run the benchmark:")
    print("   python qwen3_benchmark.py")
    print("\n3. For comparison with other models, use the prompt-test suite:")
    print("   cd prompt-test")
    print("   # Follow their README to test other models")
    print("\n4. Compare results between different implementations!")

    return 0

if __name__ == "__main__":
    sys.exit(main())