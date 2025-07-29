"""
GPU Utilities and Base Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU/CPU自動切り替えとか、便利な基底クラスが入ってるよ〜！💕
by 環ちゃん
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

# 型ヒント用の定義
if TYPE_CHECKING and cp is not None:
    NDArray = cp.ndarray
else:
    NDArray = Union[np.ndarray, Any]  # Any for when cp is None

logger = logging.getLogger('lambda3_gpu.core.utils')

# ===============================
# GPU Array Type
# ===============================

# 統一的な配列型定義
if GPU_AVAILABLE:
    ArrayType = Union[np.ndarray, cp.ndarray]
else:
    ArrayType = np.ndarray

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
            # 自動選択：メモリが一番空いてるGPU
            return auto_select_device()
        elif device == 'cpu':
            self.force_cpu = True
            return -1
        elif device == 'gpu':
            return 0  # デフォルトGPU
        elif isinstance(device, int):
            if device < cp.cuda.runtime.getDeviceCount():
                cp.cuda.Device(device).use()
                return device
            else:
                logger.warning(f"GPU {device} not found, using GPU 0")
                return 0
        else:
            raise ValueError(f"Invalid device: {device}")
    
    def _setup_mixed_precision(self):
        """Mixed precision設定"""
        if self.is_gpu:
            # TensorCoreを活用
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            logger.info("Mixed precision (FP16) enabled")
    
    def to_gpu(self, 
               array: np.ndarray, 
               dtype: Optional[np.dtype] = None,
               stream: Optional[Any] = None) -> ArrayType:
        """
        配列をGPUに転送（必要なら）
        """
        if not self.is_gpu:
            return array
            
        if isinstance(array, cp.ndarray):
            return array  # 既にGPU上
            
        # データ型変換
        if dtype is None:
            dtype = cp.float16 if self.mixed_precision else cp.float32
            
        # ストリーム指定
        if stream is not None:
            with stream:
                return cp.asarray(array, dtype=dtype)
        else:
            return cp.asarray(array, dtype=dtype)
    
    def to_cpu(self, array: ArrayType) -> np.ndarray:
        """
        配列をCPUに転送（必要なら）
        """
        if isinstance(array, np.ndarray):
            return array
        elif self.is_gpu and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        else:
            return np.asarray(array)
    
    def empty(self, shape: Tuple[int, ...], dtype=None) -> ArrayType:
        """空配列作成"""
        if dtype is None:
            dtype = self.xp.float16 if self.mixed_precision else self.xp.float32
        return self.xp.empty(shape, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> ArrayType:
        """ゼロ配列作成"""
        if dtype is None:
            dtype = self.xp.float16 if self.mixed_precision else self.xp.float32
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Tuple[int, ...], dtype=None) -> ArrayType:
        """1配列作成"""
        if dtype is None:
            dtype = self.xp.float16 if self.mixed_precision else self.xp.float32
        return self.xp.ones(shape, dtype=dtype)
    
    @contextmanager
    def timer(self, name: str):
        """
        タイマーコンテキストマネージャー
        
        使い方:
            with self.timer('my_operation'):
                # 時間計測したい処理
        """
        if not self.profile:
            yield
            return
            
        start = time.perf_counter()
        if self.is_gpu:
            cp.cuda.Stream.null.synchronize()
            
        yield
        
        if self.is_gpu:
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(elapsed)
        
        logger.debug(f"{name}: {elapsed:.4f} seconds")
    
    def get_timing_stats(self) -> dict:
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
    
    def clear_cache(self):
        """GPUキャッシュクリア"""
        if self.is_gpu:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            logger.debug("GPU cache cleared")

# ===============================
# GPU Array Wrapper
# ===============================

class GPUArray:
    """
    GPU/CPU透過的な配列ラッパー
    自動的に適切なバックエンドを使うよ〜！
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, cp.ndarray, List],
                 backend: Optional[GPUBackend] = None,
                 dtype=None):
        """
        Parameters
        ----------
        data : array-like
            入力データ
        backend : GPUBackend, optional
            使用するバックエンド（Noneなら自動作成）
        dtype : dtype, optional
            データ型
        """
        if backend is None:
            backend = GPUBackend()
        self.backend = backend
        
        # データ変換
        if isinstance(data, (list, tuple)):
            data = np.array(data)
            
        if backend.is_gpu:
            self._data = backend.to_gpu(data, dtype=dtype)
        else:
            self._data = np.asarray(data, dtype=dtype or np.float32)
    
    @property
    def data(self) -> ArrayType:
        """生データアクセス"""
        return self._data
    
    @property
    def cpu_data(self) -> np.ndarray:
        """CPU版データ取得"""
        return self.backend.to_cpu(self._data)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def size(self) -> int:
        return self._data.size
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, key):
        return GPUArray(self._data[key], self.backend)
    
    def __setitem__(self, key, value):
        if isinstance(value, GPUArray):
            self._data[key] = value._data
        else:
            self._data[key] = value
    
    # 演算子オーバーロード
    def __add__(self, other):
        if isinstance(other, GPUArray):
            return GPUArray(self._data + other._data, self.backend)
        return GPUArray(self._data + other, self.backend)
    
    def __sub__(self, other):
        if isinstance(other, GPUArray):
            return GPUArray(self._data - other._data, self.backend)
        return GPUArray(self._data - other, self.backend)
    
    def __mul__(self, other):
        if isinstance(other, GPUArray):
            return GPUArray(self._data * other._data, self.backend)
        return GPUArray(self._data * other, self.backend)
    
    def __truediv__(self, other):
        if isinstance(other, GPUArray):
            return GPUArray(self._data / other._data, self.backend)
        return GPUArray(self._data / other, self.backend)
    
    def mean(self, axis=None, keepdims=False):
        """平均計算"""
        return GPUArray(self.backend.xp.mean(self._data, axis=axis, keepdims=keepdims), 
                       self.backend)
    
    def std(self, axis=None, keepdims=False):
        """標準偏差計算"""
        return GPUArray(self.backend.xp.std(self._data, axis=axis, keepdims=keepdims), 
                       self.backend)
    
    def sum(self, axis=None, keepdims=False):
        """合計計算"""
        return GPUArray(self.backend.xp.sum(self._data, axis=axis, keepdims=keepdims), 
                       self.backend)

# ===============================
# Utility Functions
# ===============================

def auto_select_device() -> int:
    """
    自動的に最適なGPUを選択
    メモリが一番空いてるやつを選ぶよ〜！
    """
    if not GPU_AVAILABLE:
        return -1
        
    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices == 0:
        return -1
    elif n_devices == 1:
        return 0
        
    # 各GPUのメモリ使用率をチェック
    best_device = 0
    max_free_memory = 0
    
    for i in range(n_devices):
        with cp.cuda.Device(i):
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            logger.debug(f"GPU {i}: {free_mem/1024**3:.1f}/{total_mem/1024**3:.1f} GB free")
            
            if free_mem > max_free_memory:
                max_free_memory = free_mem
                best_device = i
    
    logger.info(f"Auto-selected GPU {best_device} ({max_free_memory/1024**3:.1f} GB free)")
    return best_device

def get_optimal_block_size(n_threads: int, 
                          max_block_size: int = 1024) -> Tuple[int, int]:
    """
    最適なCUDAブロックサイズとグリッドサイズを計算
    """
    if not GPU_AVAILABLE:
        return 1, n_threads
        
    device = cp.cuda.Device()
    
    # デバイスの制約を考慮
    max_threads_per_block = device.attributes['MaxThreadsPerBlock']
    max_block_size = min(max_block_size, max_threads_per_block)
    
    # ワープサイズ（32）の倍数に調整
    warp_size = 32
    
    if n_threads <= max_block_size:
        # 小さい場合はワープサイズの倍数に切り上げ
        block_size = ((n_threads + warp_size - 1) // warp_size) * warp_size
        block_size = min(block_size, max_block_size)
        grid_size = 1
    else:
        # 大きい場合は最大ブロックサイズを使用
        block_size = max_block_size
        grid_size = (n_threads + block_size - 1) // block_size
    
    return grid_size, block_size

def check_gpu_capability(required_capability: Tuple[int, int] = (3, 5)) -> bool:
    """
    GPUのCompute Capabilityをチェック
    """
    if not GPU_AVAILABLE:
        return False
        
    device = cp.cuda.Device()
    capability = device.compute_capability
    
    return capability >= required_capability

# ===============================
# GPU Timer
# ===============================

class GPUTimer:
    """
    GPU処理の正確な時間計測
    CUDA Eventを使うよ〜！
    """
    
    def __init__(self, name: str = "GPU Operation"):
        self.name = name
        self.gpu_available = GPU_AVAILABLE
        
        if self.gpu_available:
            self.start_event = cp.cuda.Event()
            self.end_event = cp.cuda.Event()
    
    def __enter__(self):
        if self.gpu_available:
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_available:
            self.end_event.record()
            self.end_event.synchronize()
            elapsed = cp.cuda.get_elapsed_time(self.start_event, self.end_event) / 1000.0
        else:
            elapsed = time.perf_counter() - self.start_time
            
        logger.debug(f"{self.name}: {elapsed:.4f} seconds")
        self.elapsed = elapsed

# ===============================
# Decorators
# ===============================

def gpu_accelerated(func):
    """
    GPU加速デコレータ
    自動的にGPU/CPUを切り替えるよ〜！
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'backend') and hasattr(self.backend, 'is_gpu'):
            if self.backend.is_gpu:
                # GPU版の関数を探す
                gpu_func_name = f"{func.__name__}_gpu"
                if hasattr(self, gpu_func_name):
                    return getattr(self, gpu_func_name)(*args, **kwargs)
        
        # デフォルト（CPU版）
        return func(self, *args, **kwargs)
    
    return wrapper

def profile_gpu(func):
    """
    プロファイリングデコレータ
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'backend') and hasattr(self.backend, 'timer'):
            with self.backend.timer(func.__name__):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    
    return wrapper

# ===============================
# Error Handling
# ===============================

def handle_gpu_errors(func):
    """
    GPU関連エラーをキャッチしてCPUフォールバック
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e:
            logger.warning(f"GPU error in {func.__name__}: {e}")
            logger.warning("Falling back to CPU...")
            
            # self.backend.force_cpu = True を設定して再実行
            if args and hasattr(args[0], 'backend'):
                original_state = args[0].backend.is_gpu
                args[0].backend.is_gpu = False
                args[0].backend.xp = np
                try:
                    result = func(*args, **kwargs)
                finally:
                    args[0].backend.is_gpu = original_state
                    args[0].backend.xp = cp if original_state else np
                return result
            else:
                raise
    
    return wrapper
