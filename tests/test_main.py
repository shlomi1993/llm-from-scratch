import pytest
import subprocess
import sys
import torch


def test_basic_text_generation():
    """
    Test basic text generation with default parameters (ch04/01).
    """
    cmd = f"{sys.executable} main.py --max-new-tokens 10"
    result = subprocess.run(cmd, capture_output=True, shell=True, text=True, timeout=30)
    assert result.returncode == 0, "Process should complete successfully"
    assert "Input text: Hello, I am" in result.stdout, "Input prompt should be echoed"
    assert "Output text:" in result.stdout, "Output text should be present"
    assert "Output length:" in result.stdout, "Output length should be present"


def test_kv_cache_with_timing():
    """
    Test KV-cache with timing (ch04/03_kv-cache/gpt_ch04.py).
    """
    cmd = f"{sys.executable} main.py --use-cache --measure-time --max-new-tokens 10"
    result = subprocess.run(cmd, capture_output=True, shell=True, text=True, timeout=30)
    assert result.returncode == 0, "Process should complete successfully"
    assert "KV-Cache: Enabled" in result.stdout, "KV-cache usage should be indicated"
    assert "Time:" in result.stdout, "Timing information should be present"
    assert "tokens/sec" in result.stdout, "Throughput information should be present"
    assert "Output text:" in result.stdout, "Output text should be present"


def test_custom_architecture():
    """
    Test custom model architecture with different parameters.
    """
    cmd = f"{sys.executable} main.py --emb-dim 512 --n-heads 8 --n-layers 6 --drop-rate 0.0 --max-new-tokens 5"
    result = subprocess.run(cmd, capture_output=True, shell=True, text=True, timeout=30)
    assert result.returncode == 0, "Process should complete successfully"
    assert "Output text:" in result.stdout, "Output text should be present"


def test_deterministic_generation():
    """
    Test that same seed produces same output.
    """
    cmd1 = f"{sys.executable} main.py --seed 123 --max-new-tokens 5"
    result1 = subprocess.run(cmd1, capture_output=True, shell=True, text=True, timeout=30)
    assert result1.returncode == 0, "Process should complete successfully"

    cmd2 = f"{sys.executable} main.py --seed 123 --max-new-tokens 5"
    result2 = subprocess.run(cmd2, capture_output=True, shell=True, text=True, timeout=30)
    assert result2.returncode == 0, "Process should complete successfully"

    # Extract output text
    output1 = [line for line in result1.stdout.split('\n') if "Output text:" in line][0]
    output2 = [line for line in result2.stdout.split('\n') if "Output text:" in line][0]
    assert output1 == output2, "Same seed should produce same output"


def test_time_and_throughput_measurement():
    """
    Test timing and throughput measurement.
    """
    cmd = f"{sys.executable} main.py --measure-time --max-new-tokens 20"
    result = subprocess.run(cmd, capture_output=True, shell=True, text=True, timeout=30)
    assert result.returncode == 0, "Process should complete successfully"
    assert "Time:" in result.stdout, "Timing information should be present"
    assert "sec" in result.stdout, "Time unit should be present"
    assert "tokens/sec" in result.stdout, "Throughput information should be present"
