#!/usr/bin/env python3
"""
Core Utilization Monitor
"""

import psutil
import time
import os


def monitor_cores():
    """Monitor CPU core utilization."""
    print("=" * 70)
    print("Core Utilization Monitor")
    print("=" * 70)
    
    print(f"\nTotal cores: {os.cpu_count()}")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores: {psutil.cpu_count(logical=True)}")
    
    print("\n📊 Real-time core utilization (Ctrl+C to stop):")
    print("-" * 70)
    
    try:
        while True:
            # Get per-core utilization
            per_core = psutil.cpu_percent(percpu=True, interval=1)
            
            # Clear line and print
            print("\r", end="")
            for i, usage in enumerate(per_core):
                bar = "█" * int(usage / 10) + "░" * (10 - int(usage / 10))
                print(f"Core {i}: [{bar}] {usage:5.1f}%  ", end="")
            
            # Overall usage
            total = psutil.cpu_percent()
            print(f"  Total: {total:5.1f}%", end="", flush=True)
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")


def check_mlx():
    """Check MLX configuration."""
    try:
        import mlx.core as mx
        
        print("\n" + "=" * 70)
        print("MLX Configuration")
        print("=" * 70)
        
        print(f"\nMLX available: Yes")
        print(f"Default device: {mx.default_device()}")
        print(f"Metal available: {mx.metal.is_available()}")
        
        if mx.metal.is_available():
            print(f"GPU device: {mx.gpu}")
            print("\n✅ MLX is using GPU acceleration!")
        else:
            print("\n⚠️ MLX is using CPU only")
            
    except ImportError:
        print("\n❌ MLX not available")


if __name__ == "__main__":
    check_mlx()
    monitor_cores()
