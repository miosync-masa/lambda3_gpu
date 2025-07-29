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

# gpu_utils.py から（実際に存在するもののみ）
from .gpu_utils import (
    GPUBackend,
    auto_select_device
)

# gpu_memory.py から（実際に存在するもののみ）
from .gpu_memory import (
    MemoryInfo,
    GPUMemoryManager,
    GPUMemoryPool,
    BatchProcessor,
    estimate_memory_usage,
    clear_gpu_cache,
    get_memory_summary,
    MemoryError
)

# gpu_kernels.py から（__all__に定義されているもの）
from .gpu_kernels import (
    CUDAKernels,
    residue_com_kernel,
    tension_field_kernel, 
    anomaly_detection_kernel,
    distance_matrix_kernel,
    topological_charge_kernel,
    compute_local_fractal_dimension_kernel,  # 新規追加
    compute_gradient_kernel,                 # 新規追加
    create_elementwise_kernel,
    benchmark_kernels',
    get_kernel_manager
)

__all__ = [
    # Utils
    'GPUBackend',
    'auto_select_device',
    
    # Memory
    'MemoryInfo',
    'GPUMemoryManager',
    'GPUMemoryPool',
    'BatchProcessor',
    'estimate_memory_usage',
    'clear_gpu_cache',
    'get_memory_summary',
    'MemoryError',
    
    # Kernels
    'CUDAKernels',
    'residue_com_kernel',
    'tension_field_kernel', 
    'anomaly_detection_kernel',
    'distance_matrix_kernel',
    'topological_charge_kernel',
    'compute_local_fractal_dimension_kernel',  # 新規追加
    'compute_gradient_kernel',                 # 新規追加
    'create_elementwise_kernel',
    'benchmark_kernels',
    'get_kernel_manager'
]

# バージョン情報
__version__ = '3.0.0'

# 初期化メッセージ
import logging
logger = logging.getLogger('lambda3_gpu.core')
logger.debug("Lambda³ GPU Core initialized")
