#!/usr/bin/env python3
"""
Check if our API is actually doing real inference or just returning errors
"""
import requests
import time

def test_api_accuracy():
    print("🔍 CHECKING API ACCURACY")
    print("Testing if API returns real responses or just errors...")

    prompts = ["2+2", "Hi", "What is the capital of France?"]

    for prompt in prompts:
        print(f"\nTesting: '{prompt}'")

        url = "http://localhost:8080/v1/chat/completions"
        payload = {
            "model": "qwen3-metal",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.7
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print(f"Response: '{content}'")

                # Check if it's a real response or error
                if "error" in content.lower() or "timeout" in content.lower():
                    print("❌ API returning error, not real inference")
                else:
                    print("✅ API returning real response")
            else:
                print(f"❌ HTTP Error: {response.status_code}")

        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_api_accuracy()