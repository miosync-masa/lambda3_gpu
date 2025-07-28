"""
Lambda³ GPU性能ベンチマークモジュール
GPU実装の性能評価とベンチマークツール
"""

from .performance_tests import (
    Lambda3BenchmarkSuite,
    BenchmarkResult,
    run_quick_benchmark
)

__all__ = [
    'Lambda3BenchmarkSuite',
    'BenchmarkResult',
    'run_quick_benchmark'
]

__version__ = '1.0.0'
