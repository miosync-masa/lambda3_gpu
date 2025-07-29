"""
GPU Memory Management System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPUメモリを効率的に管理するシステムだよ〜！💕
大きなデータもバッチ処理で処理できちゃう！

by 環ちゃん
"""

import numpy as np
import logging
import gc
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
import os

# ===============================
# Imports and Setup
# ===============================

# GPU imports
try:
    import cupy as cp
    HAS_GPU = True
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    HAS_GPU = False
    GPU_AVAILABLE = False
    cp = None

# Type definitions
if TYPE_CHECKING:
    from numpy.typing import NDArray as NPArray
    if HAS_GPU:
        from cupy.typing import NDArray as CPArray
        NDArray = Union[NPArray, CPArray]
    else:
        NDArray = NPArray
else:
    # Runtime type definition
    NDArray = Union[np.ndarray, "cp.ndarray"] if HAS_GPU else np.ndarray

# Logger setup - グローバルスコープで定義
logger = logging.getLogger('lambda3_gpu.core.memory')

# ===============================
# Data Classes
# ===============================

@dataclass
class MemoryInfo:
    """メモリ情報を保持するクラス"""
    total: int  # 総メモリ（バイト）
    used: int   # 使用中メモリ
    free: int   # 空きメモリ
    
    @property
    def used_gb(self) -> float:
        """使用メモリ（GB）"""
        return self.used / 1024**3
    
    @property
    def free_gb(self) -> float:
        """空きメモリ（GB）"""
        return self.free / 1024**3
    
    @property
    def total_gb(self) -> float:
        """総メモリ（GB）"""
        return self.total / 1024**3
    
    @property
    def usage_percent(self) -> float:
        """使用率（%）"""
        return (self.used / self.total) * 100 if self.total > 0 else 0

# ===============================
# Exceptions
# ===============================

class MemoryError(Exception):
    """メモリ関連エラー"""
    pass

class GPUMemoryError(MemoryError):
    """GPU メモリ関連エラー"""
    pass

# ===============================
# GPU Memory Manager
# ===============================

class GPUMemoryManager:
    """
    GPUメモリ管理クラス
    メモリ使用量の追跡と最適化を行うよ〜！
    """
    
    def __init__(self, 
                 max_memory_gb: Optional[float] = None,
                 reserve_percent: float = 10.0,
                 enable_pooling: bool = True):
        """
        Parameters
        ----------
        max_memory_gb : float, optional
            最大使用メモリ（GB）。Noneなら利用可能な全メモリ
        reserve_percent : float, default=10.0
            予約しておくメモリの割合（%）
        enable_pooling : bool, default=True
            メモリプーリングを有効化
        """
        self.max_memory_gb = max_memory_gb
        self.reserve_percent = reserve_percent
        self.enable_pooling = enable_pooling
        self._allocations: Dict[str, int] = {}
        
        # Initialize device
        self._initialize_device()
    
    def _initialize_device(self):
        """デバイスの初期化"""
        if HAS_GPU and GPU_AVAILABLE:
            self._setup_gpu_memory()
        else:
            self._setup_cpu_memory()
    
    def _setup_gpu_memory(self):
        """GPU メモリ設定"""
        self.device_type = 'gpu'
        
        try:
            # GPU情報取得
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            self.total_memory = total_mem
            
            # 最大メモリ設定
            self._set_max_memory(total_mem)
            
            # メモリプール設定
            if self.enable_pooling:
                self._setup_memory_pool()
            
            # ログ出力
            self._log_memory_info("GPU")
            
        except Exception as e:
            logger.error(f"Failed to setup GPU memory: {e}")
            self._setup_cpu_memory()  # フォールバック
    
    def _setup_cpu_memory(self):
        """CPU メモリ設定（フォールバック）"""
        self.device_type = 'cpu'
        
        # システムメモリ情報
        mem = psutil.virtual_memory()
        self.total_memory = mem.total
        
        # 最大メモリ設定
        if self.max_memory_gb is None:
            self.max_memory = int(mem.available * 0.8)
        else:
            self.max_memory = int(min(
                self.max_memory_gb * 1024**3,
                mem.available * 0.8
            ))
        
        self._log_memory_info("CPU")
    
    def _set_max_memory(self, total_mem: int):
        """最大使用メモリを設定"""
        if self.max_memory_gb is None:
            # 予約分を引いた分を使用
            usable_memory = total_mem * (1 - self.reserve_percent / 100)
            self.max_memory = int(usable_memory)
        else:
            self.max_memory = int(min(
                self.max_memory_gb * 1024**3,
                total_mem * (1 - self.reserve_percent / 100)
            ))
    
    def _setup_memory_pool(self):
        """メモリプールの設定"""
        try:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=self.max_memory)
            logger.debug("Memory pool configured successfully")
        except Exception as e:
            logger.warning(f"Failed to configure memory pool: {e}")
    
    def _log_memory_info(self, device_type: str):
        """メモリ情報をログ出力"""
        logger.info(f"{device_type} Memory initialized:")
        logger.info(f"  Total: {self.total_memory/1024**3:.1f} GB")
        logger.info(f"  Max usable: {self.max_memory/1024**3:.1f} GB")
        if device_type == "GPU":
            logger.info(f"  Reserve: {self.reserve_percent}%")
    
    def get_memory_info(self) -> MemoryInfo:
        """現在のメモリ情報を取得"""
        if self.device_type == 'gpu' and HAS_GPU:
            try:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                used_mem = total_mem - free_mem
                return MemoryInfo(total=total_mem, used=used_mem, free=free_mem)
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}")
                # フォールバック
        
        # CPU メモリ情報
        mem = psutil.virtual_memory()
        return MemoryInfo(total=mem.total, used=mem.used, free=mem.available)
    
    def allocate(self, 
                 size_bytes: int, 
                 name: Optional[str] = None,
                 dtype: type = np.float32) -> bool:
        """
        メモリ割り当て可能かチェック
        
        Parameters
        ----------
        size_bytes : int
            必要なメモリサイズ（バイト）
        name : str, optional
            割り当て名（追跡用）
        dtype : type
            データ型
            
        Returns
        -------
        bool
            割り当て可能ならTrue
        """
        mem_info = self.get_memory_info()
        
        # 既存の割り当てを考慮
        total_allocated = sum(self._allocations.values())
        available = min(mem_info.free, self.max_memory - total_allocated)
        
        if size_bytes > available:
            logger.warning(
                f"Memory allocation failed: "
                f"requested {size_bytes/1024**3:.2f} GB, "
                f"available {available/1024**3:.2f} GB"
            )
            return False
        
        if name:
            self._allocations[name] = size_bytes
            logger.debug(f"Allocated {size_bytes/1024**3:.2f} GB for '{name}'")
        
        return True
    
    def free(self, name: str):
        """メモリ割り当てを解放"""
        if name in self._allocations:
            size = self._allocations.pop(name)
            logger.debug(f"Freed {size/1024**3:.2f} GB from '{name}'")
    
    def estimate_batch_size(self,
                           data_shape: Tuple[int, ...],
                           dtype: type = np.float32,
                           operations_multiplier: float = 3.0) -> int:
        """
        利用可能メモリから最適なバッチサイズを推定
        
        Parameters
        ----------
        data_shape : tuple
            データの形状（バッチ次元を含む）
        dtype : type
            データ型
        operations_multiplier : float
            処理に必要な追加メモリの倍率
            
        Returns
        -------
        int
            推奨バッチサイズ
        """
        # 1要素あたりのメモリ
        element_size = np.dtype(dtype).itemsize
        elements_per_batch = int(np.prod(data_shape[1:]))  # バッチ次元以外
        bytes_per_batch = elements_per_batch * element_size
        
        # 処理用の余裕を考慮
        required_per_batch = bytes_per_batch * operations_multiplier
        
        # 利用可能メモリ
        mem_info = self.get_memory_info()
        available = min(
            mem_info.free * 0.8,  # 80%まで使用
            self.max_memory - sum(self._allocations.values())
        )
        
        # バッチサイズ計算
        batch_size = max(1, int(available / required_per_batch))
        
        logger.info(
            f"Estimated batch size: {batch_size} "
            f"(available: {available/1024**3:.1f} GB, "
            f"per batch: {required_per_batch/1024**3:.3f} GB)"
        )
        
        return batch_size
    
    @contextmanager
    def temporary_allocation(self, size_bytes: int, name: str = "temp"):
        """
        一時的なメモリ割り当てコンテキスト
        
        使い方:
            with memory_manager.temporary_allocation(1024**3, "temp_buffer"):
                # メモリを使う処理
        """
        allocated = self.allocate(size_bytes, name)
        if not allocated:
            raise MemoryError(f"Failed to allocate {size_bytes/1024**3:.2f} GB")
        
        try:
            yield
        finally:
            self.free(name)
    
    def clear_cache(self):
        """キャッシュをクリア"""
        if self.device_type == 'gpu' and HAS_GPU:
            try:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")
        
        # Python GC
        gc.collect()
    
    def get_allocation_summary(self) -> str:
        """割り当て状況のサマリーを取得"""
        mem_info = self.get_memory_info()
        
        summary = [
            f"\n{'='*50}",
            f"Memory Status ({self.device_type.upper()})",
            f"{'='*50}",
            f"Total Memory: {mem_info.total_gb:.1f} GB",
            f"Used Memory: {mem_info.used_gb:.1f} GB ({mem_info.usage_percent:.1f}%)",
            f"Free Memory: {mem_info.free_gb:.1f} GB",
            f"Max Allowed: {self.max_memory/1024**3:.1f} GB",
            f"\nAllocations:"
        ]
        
        if self._allocations:
            for name, size in sorted(self._allocations.items(), 
                                    key=lambda x: x[1], reverse=True):
                summary.append(f"  {name}: {size/1024**3:.2f} GB")
        else:
            summary.append("  (none)")
        
        summary.append(f"{'='*50}\n")
        
        return '\n'.join(summary)

# ===============================
# Memory Pool
# ===============================

class GPUMemoryPool:
    """
    再利用可能なメモリプール
    頻繁な割り当て/解放のオーバーヘッドを削減するよ〜！
    """
    
    def __init__(self, 
                 block_size: int = 1024**3,  # 1GB
                 max_blocks: int = 10):
        """
        Parameters
        ----------
        block_size : int
            ブロックサイズ（バイト）
        max_blocks : int
            最大ブロック数
        """
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._pool: List[Any] = []
        self._in_use: Dict[int, Any] = {}
        
        self.is_gpu = HAS_GPU and GPU_AVAILABLE
        self.xp = cp if self.is_gpu else np
    
    def get_block(self, size: Optional[int] = None) -> Tuple[int, Any]:
        """
        ブロックを取得
        
        Returns
        -------
        block_id : int
            ブロックID
        block : array
            メモリブロック
        """
        if size is None:
            size = self.block_size
        
        # プールから探す
        for i, block in enumerate(self._pool):
            if block.size * block.itemsize >= size:
                block = self._pool.pop(i)
                block_id = id(block)
                self._in_use[block_id] = block
                logger.debug(f"Reused block {block_id} from pool")
                return block_id, block
        
        # 新規作成
        if len(self._pool) + len(self._in_use) < self.max_blocks:
            elements = size // np.dtype(np.float32).itemsize
            block = self.xp.empty(elements, dtype=np.float32)
            block_id = id(block)
            self._in_use[block_id] = block
            logger.debug(f"Created new block {block_id}")
            return block_id, block
        
        raise MemoryError("Memory pool exhausted")
    
    def release_block(self, block_id: int):
        """ブロックを解放（プールに戻す）"""
        if block_id in self._in_use:
            block = self._in_use.pop(block_id)
            if len(self._pool) < self.max_blocks:
                self._pool.append(block)
                logger.debug(f"Released block {block_id} to pool")
            else:
                logger.debug(f"Released block {block_id} (pool full)")
    
    def clear(self):
        """プールをクリア"""
        self._pool.clear()
        self._in_use.clear()
        logger.info("Memory pool cleared")

# ===============================
# Batch Processor
# ===============================

class BatchProcessor:
    """
    大規模データのバッチ処理システム
    メモリに収まらないデータも処理できるよ〜！
    """
    
    def __init__(self, 
                 memory_manager: GPUMemoryManager,
                 overlap_frames: int = 0):
        """
        Parameters
        ----------
        memory_manager : GPUMemoryManager
            メモリ管理インスタンス
        overlap_frames : int
            バッチ間のオーバーラップ（境界処理用）
        """
        self.memory_manager = memory_manager
        self.overlap_frames = overlap_frames
        self.is_gpu = memory_manager.device_type == 'gpu'
    
    def process_batched(self,
                       data: np.ndarray,
                       process_func: Callable,
                       axis: int = 0,
                       batch_size: Optional[int] = None,
                       dtype: type = np.float32,
                       return_type: str = 'concat',
                       progress_callback: Optional[Callable] = None,
                       **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        データをバッチ処理
        
        Parameters
        ----------
        data : np.ndarray
            入力データ
        process_func : callable
            各バッチに適用する関数
        axis : int
            バッチ分割する軸
        batch_size : int, optional
            バッチサイズ（Noneなら自動計算）
        dtype : type
            処理時のデータ型
        return_type : str
            'concat': 結果を結合して返す
            'list': バッチごとのリストで返す
        progress_callback : callable, optional
            進捗コールバック(current, total)
        **kwargs
            process_funcに渡す追加引数
            
        Returns
        -------
        result : array or list
            処理結果
        """
        # バッチサイズ決定
        if batch_size is None:
            batch_size = self._estimate_batch_size(data.shape, axis, dtype)
        
        # バッチ処理実行
        results = self._process_batches(
            data, process_func, axis, batch_size, dtype, 
            progress_callback, **kwargs
        )
        
        # 結果の返却
        return self._combine_results(results, return_type, axis)
    
    def _estimate_batch_size(self, shape: Tuple[int, ...], axis: int, dtype: type) -> int:
        """バッチサイズを推定"""
        batch_shape = list(shape)
        batch_shape[axis] = 1
        return self.memory_manager.estimate_batch_size(batch_shape, dtype=dtype)
    
    def _process_batches(self,
                        data: np.ndarray,
                        process_func: Callable,
                        axis: int,
                        batch_size: int,
                        dtype: type,
                        progress_callback: Optional[Callable],
                        **kwargs) -> List[np.ndarray]:
        """バッチごとに処理を実行"""
        n_samples = data.shape[axis]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logger.info(
            f"Batch processing: {n_samples} samples, "
            f"{n_batches} batches of size {batch_size}"
        )
        
        results = []
        
        for i in range(n_batches):
            # バッチデータ取得
            batch_data = self._get_batch_data(data, i, batch_size, axis, n_samples)
            
            # 処理実行
            batch_result = self._process_single_batch(
                batch_data, process_func, dtype, i, **kwargs
            )
            
            # オーバーラップ除去
            if self.overlap_frames > 0 and i < n_batches - 1:
                batch_result = self._remove_overlap(batch_result, axis)
            
            results.append(batch_result)
            
            # 進捗通知
            if progress_callback:
                progress_callback(i + 1, n_batches)
            
            # 定期的なメモリクリア
            if i % 10 == 0:
                self.memory_manager.clear_cache()
        
        return results
    
    def _get_batch_data(self, 
                       data: np.ndarray, 
                       batch_idx: int, 
                       batch_size: int,
                       axis: int,
                       n_samples: int) -> np.ndarray:
        """バッチデータを取得"""
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size + self.overlap_frames, n_samples)
        
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(start_idx, end_idx)
        
        return data[tuple(slices)]
    
    def _process_single_batch(self,
                            batch_data: np.ndarray,
                            process_func: Callable,
                            dtype: type,
                            batch_idx: int,
                            **kwargs) -> np.ndarray:
        """単一バッチを処理"""
        # GPU転送（必要なら）
        if self.is_gpu and HAS_GPU:
            batch_data = cp.asarray(batch_data, dtype=dtype)
        
        # メモリ割り当てコンテキスト内で処理
        with self.memory_manager.temporary_allocation(
            batch_data.nbytes, f"batch_{batch_idx}"
        ):
            batch_result = process_func(batch_data, **kwargs)
            
            # CPU転送（必要なら）
            if self.is_gpu and HAS_GPU and isinstance(batch_result, cp.ndarray):
                batch_result = cp.asnumpy(batch_result)
        
        return batch_result
    
    def _remove_overlap(self, 
                       batch_result: np.ndarray, 
                       axis: int) -> np.ndarray:
        """オーバーラップ部分を除去"""
        if isinstance(batch_result, np.ndarray):
            trim_slices = [slice(None)] * batch_result.ndim
            trim_slices[axis] = slice(None, -self.overlap_frames)
            return batch_result[tuple(trim_slices)]
        return batch_result
    
    def _combine_results(self,
                        results: List[np.ndarray],
                        return_type: str,
                        axis: int) -> Union[np.ndarray, List[np.ndarray]]:
        """結果を結合または返却"""
        if return_type == 'concat' and results:
            if isinstance(results[0], np.ndarray):
                return np.concatenate(results, axis=axis)
            else:
                logger.warning("Cannot concatenate non-array results")
        
        return results

# ===============================
# Utility Functions
# ===============================

def estimate_memory_usage(shape: Tuple[int, ...], 
                         dtype: type = np.float32,
                         operations: Optional[List[str]] = None) -> Dict[str, float]:
    """
    メモリ使用量を推定
    
    Parameters
    ----------
    shape : tuple
        データ形状
    dtype : type
        データ型
    operations : list of str, optional
        実行する操作のリスト
        
    Returns
    -------
    dict
        メモリ使用量の推定値（GB）
    """
    # 基本メモリ
    element_size = np.dtype(dtype).itemsize
    n_elements = int(np.prod(shape))
    base_memory = n_elements * element_size
    
    estimates = {
        'input': base_memory / 1024**3,
        'operations': 0.0,
        'output': base_memory / 1024**3,
        'total': 0.0
    }
    
    # 操作ごとの追加メモリ
    if operations:
        operation_multipliers = {
            'fft': 2.0,
            'conv': 3.0,
            'matmul': 2.5,
            'gradient': 1.5,
            'sort': 1.5,
            'cumsum': 1.0
        }
        
        for op in operations:
            multiplier = operation_multipliers.get(op, 1.0)
            estimates['operations'] += base_memory * multiplier / 1024**3
    
    estimates['total'] = sum(v for k, v in estimates.items() if k != 'total')
    
    return estimates

def clear_gpu_cache():
    """GPUキャッシュを完全にクリア"""
    if HAS_GPU and GPU_AVAILABLE:
        try:
            # CuPyメモリプール
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            # CUDA同期
            cp.cuda.Stream.null.synchronize()
            
            logger.info("GPU cache cleared completely")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    # ガベージコレクション
    gc.collect()

def get_memory_summary() -> str:
    """
    現在のメモリ状況のサマリーを取得
    """
    lines = ["\n" + "="*60, "Memory Summary", "="*60]
    
    # システムメモリ
    mem = psutil.virtual_memory()
    lines.extend([
        f"\nSystem Memory:",
        f"  Total: {mem.total/1024**3:.1f} GB",
        f"  Used: {mem.used/1024**3:.1f} GB ({mem.percent:.1f}%)",
        f"  Available: {mem.available/1024**3:.1f} GB"
    ])
    
    # GPUメモリ
    if HAS_GPU and GPU_AVAILABLE:
        try:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            used_mem = total_mem - free_mem
            lines.extend([
                f"\nGPU Memory:",
                f"  Total: {total_mem/1024**3:.1f} GB",
                f"  Used: {used_mem/1024**3:.1f} GB ({used_mem/total_mem*100:.1f}%)",
                f"  Free: {free_mem/1024**3:.1f} GB"
            ])
            
            # メモリプール状況
            mempool = cp.get_default_memory_pool()
            lines.extend([
                f"\nCuPy Memory Pool:",
                f"  Used blocks: {mempool.used_bytes()/1024**3:.2f} GB",
                f"  Total blocks: {mempool.total_bytes()/1024**3:.2f} GB"
            ])
        except Exception as e:
            lines.append(f"\nGPU Memory: Error getting info - {e}")
    
    lines.append("="*60 + "\n")
    
    return '\n'.join(lines)
