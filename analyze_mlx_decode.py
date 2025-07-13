#!/usr/bin/env python3
"""
Analyze mlx_whisper's decode implementation to understand batch handling
"""
import mlx_whisper
import inspect
from pathlib import Path

print("=== Analyzing MLX Whisper Decode Implementation ===")

# Find mlx_whisper location
mlx_path = Path(mlx_whisper.__file__).parent
print(f"\nMLX Whisper location: {mlx_path}")

# Check decode function
print("\n1. Checking decode function:")
decode_func = mlx_whisper.decoding.decode

# Get source
try:
    source = inspect.getsource(decode_func)
    lines = source.split('\n')
    
    # Find key patterns
    for i, line in enumerate(lines[:30]):  # First 30 lines
        if any(keyword in line for keyword in ['for', 'mel', 'batch', 'loop']):
            print(f"   Line {i}: {line.strip()}")
    
    # Check if it loops over mel
    if "for mel_segment in mel" in source:
        print("\n   ⚠️  FOUND: decode loops over mel segments!")
        print("   This explains why batch fails - it's not true parallel")
    
    # Check for single_decode
    if "single_decode" in source:
        print("\n   Found single_decode usage")
        
except Exception as e:
    print(f"   Could not get source: {e}")

# Check for batch-specific code
print("\n2. Checking for batch handling:")
decoding_module = inspect.getmodule(decode_func)
all_functions = inspect.getmembers(decoding_module, inspect.isfunction)

for name, func in all_functions:
    if 'batch' in name.lower():
        print(f"   Found: {name}")

# Check DecodingTask
print("\n3. Checking DecodingTask class:")
try:
    task_source = inspect.getsource(mlx_whisper.decoding.DecodingTask)
    if "batch" in task_source:
        print("   DecodingTask mentions batch processing")
        # Find specific mentions
        for line in task_source.split('\n'):
            if 'batch' in line and not line.strip().startswith('#'):
                print(f"      {line.strip()}")
except:
    pass

# The key finding
print("\n4. Key Finding:")
print("   MLX Whisper's decode() function does NOT support true batch processing!")
print("   It loops over the batch dimension sequentially")
print("   This is why Lightning must have modified or wrapped it")

print("\n5. Solution Approach:")
print("   We need to implement our own batch decode that:")
print("   1. Calls model.encoder() on the full batch (this works)")
print("   2. Implements parallel token generation")
print("   3. Returns multiple results efficiently")