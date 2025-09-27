#!/usr/bin/env python3
"""
Download Qwen3-4B model from HuggingFace and convert to qwen3.c format.

This script will:
1. Download the Qwen3-4B model from https://huggingface.co/Qwen/Qwen3-4B
2. Convert it to qwen3.c's optimized binary format with Q8_0 quantization
3. Create tokenizer and prompt template files
4. Verify the conversion by checking the magic number

Usage:
    python3 download_qwen3_4b.py [--output-dir models] [--force-redownload]
"""

import argparse
import os
import sys
import struct
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['transformers', 'torch', 'huggingface_hub']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def download_and_convert_model(output_dir="models", force_redownload=False):
    """Download Qwen3-4B and convert to qwen3.c format."""

    if not check_requirements():
        return False

    # Import after checking requirements
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        import torch
    except ImportError as e:
        print(f"Error importing required packages: {e}")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model_name = "Qwen/Qwen3-4B"
    output_file = output_path / "qwen3-4B.bin"

    # Skip if already exists and not forcing redownload
    if output_file.exists() and not force_redownload:
        file_size = output_file.stat().st_size
        if file_size > 1024 * 1024:  # At least 1MB (real model should be GB)
            print(f"Model already exists at {output_file} ({file_size:,} bytes)")
            print("Use --force-redownload to download again")
            return True

    print(f"Downloading {model_name}...")

    try:
        # Download model to cache
        print("Step 1: Downloading model files...")
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=None,  # Use default cache
            ignore_patterns=["*.safetensors.index.json"]  # Skip index files
        )
        print(f"Model downloaded to: {local_dir}")

        # Load model and tokenizer
        print("Step 2: Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            torch_dtype=torch.float32,  # Use float32 for conversion
            device_map="cpu"  # Keep on CPU for conversion
        )
        tokenizer = AutoTokenizer.from_pretrained(local_dir)

        print(f"Model config:")
        print(f"  - Hidden size: {model.config.hidden_size}")
        print(f"  - Num layers: {model.config.num_hidden_layers}")
        print(f"  - Num heads: {model.config.num_attention_heads}")
        print(f"  - Vocab size: {model.config.vocab_size}")

        # Import export functions
        sys.path.append(str(Path(__file__).parent))
        from export import load_hf_model, model_export, build_tokenizer, build_prompts

        print("Step 3: Converting to qwen3.c format...")

        # Use the existing export functions but load from local directory
        qwen3_model = load_hf_model(local_dir)
        if qwen3_model is None:
            print("Failed to load model for conversion")
            return False

        # Export model
        print("Step 4: Exporting quantized model...")
        model_export(qwen3_model, str(output_file))

        # Build tokenizer and templates
        print("Step 5: Creating tokenizer files...")
        build_tokenizer(qwen3_model, str(output_file))
        build_prompts(qwen3_model, str(output_file))

        # Verify the conversion
        print("Step 6: Verifying conversion...")
        if verify_model_file(output_file):
            print(f"✅ Model successfully converted to {output_file}")
            print(f"   File size: {output_file.stat().st_size:,} bytes")

            # List all created files
            created_files = []
            for suffix in ['', '.tokenizer', '.template', '.template.with-thinking',
                          '.template.with-system', '.template.with-system-and-thinking']:
                file_path = Path(str(output_file) + suffix)
                if file_path.exists():
                    created_files.append(file_path)

            print("\nCreated files:")
            for file_path in created_files:
                size = file_path.stat().st_size
                print(f"  - {file_path} ({size:,} bytes)")

            print(f"\nUsage:")
            print(f"  ./runq {output_file} -i \"Hello, how are you?\"")

            return True
        else:
            print("❌ Model conversion failed verification")
            return False

    except Exception as e:
        print(f"Error during download/conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_file(file_path):
    """Verify that the model file has correct magic number and structure."""
    try:
        with open(file_path, "rb") as f:
            # Read magic number
            magic_bytes = f.read(4)
            if len(magic_bytes) != 4:
                return False

            magic = struct.unpack("I", magic_bytes)[0]
            expected_magic = 0x616A6331  # "ajc1" in ASCII

            if magic != expected_magic:
                print(f"❌ Invalid magic number: 0x{magic:08X} (expected 0x{expected_magic:08X})")
                return False

            # Read version
            version_bytes = f.read(4)
            if len(version_bytes) != 4:
                return False

            version = struct.unpack("i", version_bytes)[0]
            if version != 1:
                print(f"❌ Invalid version: {version} (expected 1)")
                return False

            print(f"✅ Magic number: 0x{magic:08X}")
            print(f"✅ Version: {version}")

            return True

    except Exception as e:
        print(f"❌ Error verifying file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-4B model and convert to qwen3.c format"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory for model files (default: models)"
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force redownload even if model already exists"
    )

    args = parser.parse_args()

    print("🚀 Qwen3-4B Model Downloader")
    print("=" * 50)
    print(f"Model: https://huggingface.co/Qwen/Qwen3-4B")
    print(f"Output directory: {args.output_dir}")
    print(f"Force redownload: {args.force_redownload}")
    print()

    success = download_and_convert_model(
        output_dir=args.output_dir,
        force_redownload=args.force_redownload
    )

    if success:
        print("\n🎉 Download and conversion completed successfully!")
        return 0
    else:
        print("\n❌ Download and conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())