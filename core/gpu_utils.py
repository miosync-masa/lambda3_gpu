"""
GPU Utilities and Base Classes -
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TYPE_CHECKING
from contextlib import contextmanager
from functools import wraps
import warnings

# GPU availability check
try:
    import cupy as cp
    from cupy import cuda
    HAS_GPU = True
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    HAS_GPU = False
    GPU_AVAILABLE = False
    cp = None
    cuda = None

# 型ヒント用の定義（修正版）
if TYPE_CHECKING:
    # 型チェック時のみ（mypyなど）
    try:
        import cupy as cp
        NDArray = Union[np.ndarray, cp.ndarray]
    except ImportError:
        NDArray = Union[np.ndarray, Any]
else:
    # 実行時
    if HAS_GPU:
        NDArray = Union[np.ndarray, cp.ndarray]
    else:
        NDArray = Union[np.ndarray, Any]

logger = logging.getLogger('lambda3_gpu.core.utils')

# ===============================
# GPU Array Type
# ===============================

# 統一的な配列型定義（修正版）
ArrayType = NDArray  # 上で定義したNDArrayを使用

# ===============================
# GPU Backend Base Class
# ===============================

class GPUBackend:
    """
    GPU/CPU自動切り替えの基底クラス
    全てのGPU対応クラスはこれを継承するよ〜！
    """
    
    def __init__(self, 
                 device: Union[str, int] = 'auto',
                 force_cpu: bool = False,
                 mixed_precision: bool = False,
                 profile: bool = False):
        """
        Parameters
        ----------
        device : str or int, default='auto'
            'auto': 自動選択
            'cpu': CPU強制
            'gpu': GPU強制
            0,1,2...: 特定のGPU番号
        force_cpu : bool, default=False
            True時はGPUがあってもCPU使用
        mixed_precision : bool, default=False
            FP16使用で高速化（精度は落ちる）
        profile : bool, default=False
            プロファイリングモード
        """
        self.force_cpu = force_cpu
        self.mixed_precision = mixed_precision
        self.profile = profile
        self._timers = {}
        
        # デバイス選択
        if force_cpu or not GPU_AVAILABLE:
            self.device = 'cpu'
            self.xp = np
            self.is_gpu = False
            logger.info("Using CPU backend")
        else:
            self.device = self._select_device(device)
            self.xp = cp
            self.is_gpu = True
            logger.info(f"Using GPU backend: Device {self.device}")
            
            # Mixed precision設定
            if mixed_precision:
                self._setup_mixed_precision()
    
    def _select_device(self, device: Union[str, int]) -> int:
        """デバイス選択ロジック"""
        if device == 'auto':
            # 自動選択（メモリが一番空いてるGPU）
            return auto_select_device()
        elif device == 'cpu':
            return -1
        elif device == 'gpu':
            return 0  # デフォルトGPU
        elif isinstance(device, int):
            return device
        else:
            raise ValueError(f"Invalid device: {device}")
    
    def _setup_mixed_precision(self):
        """Mixed precision設定"""
        if self.is_gpu and HAS_GPU:
            # TensorCoreを使う設定
            cp.cuda.cublas.setMathMode(cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)
            logger.info("Mixed precision (FP16) enabled")
    
    def to_gpu(self, array: NDArray, dtype: Optional[np.dtype] = None) -> NDArray:
        """配列をGPUに転送"""
        if self.is_gpu and HAS_GPU:
            if isinstance(array, cp.ndarray):
                return array.astype(dtype) if dtype else array
            return cp.asarray(array, dtype=dtype)
        else:
            return np.asarray(array, dtype=dtype) if dtype else np.asarray(array)
    
    def to_cpu(self, array: NDArray) -> np.ndarray:
        """配列をCPUに転送"""
        if self.is_gpu and HAS_GPU and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    @contextmanager
    def timer(self, name: str):
        """タイマーコンテキスト"""
        if self.profile:
            start = time.perf_counter()
            yield
            elapsed = time.perf_counter() - start
            
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(elapsed)
            
            logger.debug(f"{name}: {elapsed:.3f}s")
        else:
            yield
    
    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """タイミング統計を取得"""
        stats = {}
        for name, times in self._timers.items():
            stats[name] = {
                'count': len(times),
                'total': sum(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': min(times),
                'max': max(times)
            }
        return stats
    
    def clear_memory(self):
        """メモリクリア"""
        if self.is_gpu and HAS_GPU:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            logger.debug("GPU memory cleared")
        else:
            import gc
            gc.collect()
            logger.debug("CPU memory garbage collected")


# ===============================
# Utility Functions
# ===============================

def auto_select_device() -> int:
    """
    自動的に最適なGPUを選択
    メモリが一番空いてるやつを選ぶよ〜！
    """
    if not GPU_AVAILABLE:
        return 0
    
    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices == 0:
        return 0
    
    best_device = 0
    max_free_memory = 0
    
    for i in range(n_devices):
        with cp.cuda.Device(i):
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            if free_mem > max_free_memory:
                max_free_memory = free_mem
                best_device = i
    
    return best_device
