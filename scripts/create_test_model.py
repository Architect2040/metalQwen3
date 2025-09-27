#!/usr/bin/env python3
"""
Create a minimal test qwen3.c model file with proper header structure.
This demonstrates the correct binary format expected by qwen3.c.
"""

import struct

def create_test_model(filename):
    """Create a minimal qwen3.c model file with proper header."""

    with open(filename, "wb") as f:
        # Write the magic number (0x616A6331 = "ajc1" in ASCII)
        f.write(struct.pack("I", 0x616A6331))

        # Write version (1)
        f.write(struct.pack("i", 1))

        # Write model parameters (simplified values for a tiny test model)
        # These would normally come from the actual model
        header = struct.pack(
            "iiiiiiiiii",
            128,    # dim (hidden size)
            512,    # hidden_dim (MLP size)
            2,      # n_layers
            4,      # n_heads
            4,      # n_kv_heads
            1000,   # vocab_size
            512,    # max_seq_len
            32,     # head_dim
            1,      # shared_classifier
            64      # group_size
        )
        f.write(header)

        # Pad to 256 bytes for header
        current_pos = f.tell()
        pad_size = 256 - current_pos
        f.write(b"\0" * pad_size)

        # Write some dummy weights (this would be actual model weights)
        # For a real model, this would be gigabytes of quantized weights
        # Just write a small amount to make it a valid file
        dummy_weights = b"\x00" * 1024  # 1KB of dummy weights
        f.write(dummy_weights)

    print(f"Created test model: {filename}")
    print(f"File size: {open(filename, 'rb').seek(0, 2) or open(filename, 'rb').tell()} bytes")

if __name__ == "__main__":
    create_test_model("./models/test-qwen3.bin")

    # Verify the magic number
    with open("./models/test-qwen3.bin", "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        version = struct.unpack("i", f.read(4))[0]
        print(f"Magic number: 0x{magic:08X} (should be 0x616A6331)")
        print(f"Version: {version} (should be 1)")
        print(f"Magic matches: {'✅' if magic == 0x616A6331 else '❌'}")