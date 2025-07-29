"""
Lambda³ GPU Core Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPUコンピューティングの基盤となるコアモジュール群だよ〜！💕
ここには基底クラス、メモリ管理、CUDAカーネルが入ってる！

Components:
    - GPUBackend: GPU/CPU自動切り替えの基底クラス
    - GPUMemoryManager: メモリ管理システム
    - CUDAKernels: 高速カスタムカーネル集
"""

from .gpu_utils import (
    GPUBackend,
    GPUArray,
    GPUTimer,
    auto_select_device,
    get_optimal_block_size,
    check_gpu_capability
)

from .gpu_memory import (
    GPUMemoryManager,
    GPUMemoryPool,
    BatchProcessor,
    estimate_memory_usage,
    clear_gpu_cache
)

from .gpu_kernels import (
    CUDAKernels,
    residue_com_kernel,
    tension_field_kernel,
    anomaly_detection_kernel,
    distance_matrix_kernel,
    topological_charge_kernel
)

__all__ = [
    # Utils
    'GPUBackend',
    'GPUArray',
    'GPUTimer',
    'auto_select_device',
    'get_optimal_block_size',
    'check_gpu_capability',
    
    # Memory
    'GPUMemoryManager',
    'GPUMemoryPool',
    'BatchProcessor',
    'estimate_memory_usage',
    'clear_gpu_cache',
    
    # Kernels
    'CUDAKernels',
    'residue_com_kernel',
    'tension_field_kernel',
    'anomaly_detection_kernel',
    'distance_matrix_kernel',
    'topological_charge_kernel'
]

# バージョン情報
__version__ = '3.0.0'

# 初期化メッセージ
import logging
logger = logging.getLogger('lambda3_gpu.core')
logger.debug("Lambda³ GPU Core initialized")
