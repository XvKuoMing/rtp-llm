#!/usr/bin/env python3
"""
Test script to verify that RTLLM_DEBUG environment variable works correctly.
This script tests the behavior with and without the environment variable set.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_env_debug():
    """Test that RTLLM_DEBUG environment variable is respected."""
    
    # Get the project root
    project_root = Path(__file__).parent
    
    print("=== Testing RTLLM_DEBUG Environment Variable ===\n")
    
    # Test 1: Without RTLLM_DEBUG set
    print("Test 1: Running without RTLLM_DEBUG environment variable")
    env_no_debug = os.environ.copy()
    env_no_debug.pop('RTLLM_DEBUG', None)  # Remove if it exists
    
    try:
        result = subprocess.run([
            sys.executable, '-c', 
            'from src.entrypoint.settings import AppSettings; s = AppSettings(); print(f"Debug: {s.debug}")'
        ], env=env_no_debug, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(f"✓ Success: {result.stdout.strip()}")
        else:
            print(f"✗ Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()
    
    # Test 2: With RTLLM_DEBUG=true
    print("Test 2: Running with RTLLM_DEBUG=true")
    env_with_debug = os.environ.copy()
    env_with_debug['RTLLM_DEBUG'] = 'true'
    
    try:
        result = subprocess.run([
            sys.executable, '-c', 
            'from src.entrypoint.settings import AppSettings; s = AppSettings(); print(f"Debug: {s.debug}")'
        ], env=env_with_debug, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(f"✓ Success: {result.stdout.strip()}")
        else:
            print(f"✗ Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    print()
    
    # Test 3: With RTLLM_DEBUG=false
    print("Test 3: Running with RTLLM_DEBUG=false")
    env_false_debug = os.environ.copy()
    env_false_debug['RTLLM_DEBUG'] = 'false'
    
    try:
        result = subprocess.run([
            sys.executable, '-c', 
            'from src.entrypoint.settings import AppSettings; s = AppSettings(); print(f"Debug: {s.debug}")'
        ], env=env_false_debug, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print(f"✓ Success: {result.stdout.strip()}")
        else:
            print(f"✗ Error: {result.stderr.strip()}")
    except Exception as e:
        print(f"✗ Exception: {e}")

if __name__ == "__main__":
    test_env_debug()
