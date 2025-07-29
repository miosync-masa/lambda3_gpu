"""
Tensor Operations for Lambda³ (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

汎用的なテンソル演算のGPU実装だよ〜！💕
スライディングウィンドウとか、相関計算とか、なんでも速い！

by 環ちゃん
"""

import numpy as np
import logging
from typing import Callable, Optional, Union, Tuple, List, Any
from functools import wraps
import warnings

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy import signal as cp_signal
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cp_ndimage = None
    cp_signal = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, profile_gpu

logger = logging.getLogger('lambda3_gpu.structures.tensor_operations')

# ===============================
# Tensor Operations GPU Class
# ===============================

class TensorOperationsGPU(GPUBackend):
    """
    テンソル演算のGPU実装クラス
    Lambda³で使う色んな演算を高速化するよ〜！
    """
    
    def __init__(self, 
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        memory_manager : GPUMemoryManager
            メモリ管理
        """
        super().__init__(**kwargs)
        self.memory_manager = memory_manager or GPUMemoryManager()
        
    @profile_gpu
    def compute_gradient(self,
                        array: NDArray,
                        axis: Optional[Union[int, Tuple[int, ...]]] = None,
                        edge_order: int = 1) -> NDArray:
        """
        勾配計算（GPU版）
        
        Parameters
        ----------
        array : array-like
            入力配列
        axis : int or tuple, optional
            勾配を計算する軸
        edge_order : int
            端での精度（1 or 2）
            
        Returns
        -------
        gradient : array
            勾配
        """
        array_gpu = self.to_gpu(array)
        
        if axis is None:
            # 全軸で勾配
            gradients = []
            for i in range(array_gpu.ndim):
                grad = self.xp.gradient(array_gpu, axis=i, edge_order=edge_order)
                gradients.append(grad)
            return gradients if self.is_gpu else [self.to_cpu(g) for g in gradients]
        else:
            gradient = self.xp.gradient(array_gpu, axis=axis, edge_order=edge_order)
            return gradient if self.is_gpu else self.to_cpu(gradient)
    
    @profile_gpu
    def compute_covariance(self,
                         x: NDArray,
                         y: Optional[NDArray] = None,
                         rowvar: bool = True,
                         bias: bool = False,
                         ddof: Optional[int] = None) -> NDArray:
        """
        共分散行列計算（GPU版）
        
        Parameters
        ----------
        x : array-like
            入力データ
        y : array-like, optional
            2番目のデータ（Noneならxの自己共分散）
        rowvar : bool
            行が変数ならTrue
        bias : bool
            正規化でN-1でなくNを使うか
        ddof : int, optional
            自由度の差分
            
        Returns
        -------
        cov : array
            共分散行列
        """
        x_gpu = self.to_gpu(x)
        
        if y is not None:
            y_gpu = self.to_gpu(y)
            # スタック
            if rowvar:
                data = self.xp.vstack([x_gpu, y_gpu])
            else:
                data = self.xp.hstack([x_gpu, y_gpu])
            cov = self.xp.cov(data, rowvar=rowvar, bias=bias, ddof=ddof)
        else:
            cov = self.xp.cov(x_gpu, rowvar=rowvar, bias=bias, ddof=ddof)
        
        return cov if self.is_gpu else self.to_cpu(cov)
    
    @profile_gpu
    def compute_correlation(self,
                          x: NDArray,
                          y: Optional[NDArray] = None,
                          rowvar: bool = True) -> NDArray:
        """
        相関係数行列計算（GPU版）
        
        Parameters
        ----------
        x : array-like
            入力データ
        y : array-like, optional
            2番目のデータ
        rowvar : bool
            行が変数ならTrue
            
        Returns
        -------
        corr : array
            相関係数行列
        """
        x_gpu = self.to_gpu(x)
        
        if y is not None:
            y_gpu = self.to_gpu(y)
            if rowvar:
                data = self.xp.vstack([x_gpu, y_gpu])
            else:
                data = self.xp.hstack([x_gpu, y_gpu])
            corr = self.xp.corrcoef(data, rowvar=rowvar)
        else:
            corr = self.xp.corrcoef(x_gpu, rowvar=rowvar)
        
        return corr if self.is_gpu else self.to_cpu(corr)
    
    def sliding_window_operation(self,
                               array: NDArray,
                               window_size: int,
                               operation: Union[str, Callable],
                               axis: int = 0,
                               step: int = 1,
                               mode: str = 'valid') -> NDArray:
        """
        スライディングウィンドウ演算（GPU版）
        
        Parameters
        ----------
        array : array-like
            入力配列
        window_size : int
            ウィンドウサイズ
        operation : str or callable
            演算（'mean', 'std', 'max', 'min' または関数）
        axis : int
            ウィンドウを適用する軸
        step : int
            ウィンドウのステップサイズ
        mode : str
            端の処理（'valid', 'same'）
            
        Returns
        -------
        result : array
            演算結果
        """
        array_gpu = self.to_gpu(array)
        
        # 組み込み演算
        if isinstance(operation, str):
            op_func = self._get_operation_func(operation)
        else:
            op_func = operation
        
        # 結果の形状を計算
        if mode == 'valid':
            out_size = (array_gpu.shape[axis] - window_size + 1) // step
        else:  # 'same'
            out_size = array_gpu.shape[axis] // step
        
        # 出力配列
        out_shape = list(array_gpu.shape)
        out_shape[axis] = out_size
        result = self.xp.empty(tuple(out_shape))
        
        # スライディングウィンドウ処理
        with self.timer('sliding_window'):
            for i in range(out_size):
                start_idx = i * step
                end_idx = min(start_idx + window_size, array_gpu.shape[axis])
                
                # スライス作成
                slices = [slice(None)] * array_gpu.ndim
                slices[axis] = slice(start_idx, end_idx)
                window_data = array_gpu[tuple(slices)]
                
                # 演算実行
                window_result = op_func(window_data, axis=axis)
                
                # 結果を格納
                out_slices = [slice(None)] * result.ndim
                out_slices[axis] = i
                result[tuple(out_slices)] = window_result
        
        return result if self.is_gpu else self.to_cpu(result)
    
    def _get_operation_func(self, operation: str) -> Callable:
        """演算名から関数を取得"""
        ops = {
            'mean': self.xp.mean,
            'std': self.xp.std,
            'var': self.xp.var,
            'max': self.xp.max,
            'min': self.xp.min,
            'sum': self.xp.sum,
            'median': self.xp.median
        }
        
        if operation not in ops:
            raise ValueError(f"Unknown operation: {operation}")
        
        return ops[operation]
    
    def batch_tensor_operation(self,
                             tensors: List[NDArray],
                             operation: Callable,
                             batch_size: Optional[int] = None,
                             concat_axis: Optional[int] = 0,
                             **kwargs) -> Union[NDArray, List]:
        """
        複数テンソルのバッチ演算
        
        Parameters
        ----------
        tensors : list of arrays
            入力テンソルのリスト
        operation : callable
            各バッチに適用する演算
        batch_size : int, optional
            バッチサイズ
        concat_axis : int, optional
            結果を結合する軸（Noneならリストで返す）
        **kwargs
            operationに渡す追加引数
            
        Returns
        -------
        result : array or list
            演算結果
        """
        n_tensors = len(tensors)
        
        if batch_size is None:
            # メモリから自動決定
            sample_size = tensors[0].nbytes
            batch_size = self.memory_manager.estimate_batch_size(
                tensors[0].shape, 
                operations_multiplier=2.0
            )
            batch_size = min(batch_size, n_tensors)
        
        results = []
        
        # バッチ処理
        for i in range(0, n_tensors, batch_size):
            end = min(i + batch_size, n_tensors)
            batch_tensors = tensors[i:end]
            
            # GPU転送
            if self.is_gpu:
                batch_gpu = [self.to_gpu(t) for t in batch_tensors]
            else:
                batch_gpu = batch_tensors
            
            # 演算実行
            with self.timer(f'batch_operation_{i//batch_size}'):
                batch_results = operation(batch_gpu, **kwargs)
            
            # CPU転送（必要なら）
            if self.is_gpu and isinstance(batch_results, list):
                batch_results = [self.to_cpu(r) for r in batch_results]
            elif self.is_gpu:
                batch_results = self.to_cpu(batch_results)
            
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
            
            # メモリクリア
            if i % (batch_size * 5) == 0:
                self.clear_memory()
        
        # 結果の結合
        if concat_axis is not None and len(results) > 0:
            if isinstance(results[0], (np.ndarray, cp.ndarray)):
                return np.concatenate(results, axis=concat_axis)
        
        return results
    
    @profile_gpu
    def convolution(self,
                   array: NDArray,
                   kernel: NDArray,
                   mode: str = 'same',
                   method: str = 'auto') -> NDArray:
        """
        畳み込み演算（GPU版）
        
        Parameters
        ----------
        array : array-like
            入力配列
        kernel : array-like
            畳み込みカーネル
        mode : str
            出力サイズ（'full', 'valid', 'same'）
        method : str
            計算方法（'auto', 'direct', 'fft'）
            
        Returns
        -------
        result : array
            畳み込み結果
        """
        array_gpu = self.to_gpu(array)
        kernel_gpu = self.to_gpu(kernel)
        
        if self.is_gpu and HAS_GPU:
            # CuPyの高速畳み込み
            if array_gpu.ndim == 1:
                result = cp_signal.convolve(array_gpu, kernel_gpu, mode=mode, method=method)
            else:
                result = cp_signal.convolve2d(array_gpu, kernel_gpu, mode=mode)
        else:
            # NumPy版
            from scipy import signal
            if array_gpu.ndim == 1:
                result = signal.convolve(array_gpu, kernel_gpu, mode=mode, method=method)
            else:
                result = signal.convolve2d(array_gpu, kernel_gpu, mode=mode)
        
        return result if self.is_gpu else self.to_cpu(result)
    
    @profile_gpu
    def gaussian_filter(self,
                       array: NDArray,
                       sigma: Union[float, Tuple[float, ...]],
                       order: Union[int, Tuple[int, ...]] = 0,
                       mode: str = 'reflect',
                       truncate: float = 4.0) -> NDArray:
        """
        ガウシアンフィルタ（GPU版）
        
        Parameters
        ----------
        array : array-like
            入力配列
        sigma : float or tuple
            ガウシアンカーネルの標準偏差
        order : int or tuple
            微分の次数
        mode : str
            境界処理
        truncate : float
            カーネルの打ち切り
            
        Returns
        -------
        result : array
            フィルタ結果
        """
        array_gpu = self.to_gpu(array)
        
        if self.is_gpu and HAS_GPU:
            # CuPyのガウシアンフィルタ
            if array_gpu.ndim == 1:
                result = cp_ndimage.gaussian_filter1d(
                    array_gpu, sigma, order=order, mode=mode, truncate=truncate
                )
            else:
                result = cp_ndimage.gaussian_filter(
                    array_gpu, sigma, order=order, mode=mode, truncate=truncate
                )
        else:
            # SciPy版
            from scipy import ndimage
            if array_gpu.ndim == 1:
                result = ndimage.gaussian_filter1d(
                    array_gpu, sigma, order=order, mode=mode, truncate=truncate
                )
            else:
                result = ndimage.gaussian_filter(
                    array_gpu, sigma, order=order, mode=mode, truncate=truncate
                )
        
        return result if self.is_gpu else self.to_cpu(result)

# ===============================
# Standalone Functions
# ===============================

def compute_gradient_gpu(array: NDArray,
                       axis: Optional[Union[int, Tuple[int, ...]]] = None,
                       backend: Optional[GPUBackend] = None) -> Union[np.ndarray, List[np.ndarray]]:
    """勾配計算のスタンドアロン関数"""
    ops = TensorOperationsGPU() if backend is None else TensorOperationsGPU(device=backend.device)
    return ops.compute_gradient(array, axis)

def compute_covariance_gpu(x: NDArray,
                         y: Optional[NDArray] = None,
                         backend: Optional[GPUBackend] = None) -> np.ndarray:
    """共分散計算のスタンドアロン関数"""
    ops = TensorOperationsGPU() if backend is None else TensorOperationsGPU(device=backend.device)
    result = ops.compute_covariance(x, y)
    return ops.to_cpu(result) if ops.is_gpu else result

def compute_correlation_gpu(x: NDArray,
                          y: Optional[NDArray] = None,
                          backend: Optional[GPUBackend] = None) -> np.ndarray:
    """相関計算のスタンドアロン関数"""
    ops = TensorOperationsGPU() if backend is None else TensorOperationsGPU(device=backend.device)
    result = ops.compute_correlation(x, y)
    return ops.to_cpu(result) if ops.is_gpu else result

def sliding_window_operation_gpu(array: NDArray,
                               window_size: int,
                               operation: Union[str, Callable],
                               axis: int = 0,
                               backend: Optional[GPUBackend] = None) -> np.ndarray:
    """スライディングウィンドウ演算のスタンドアロン関数"""
    ops = TensorOperationsGPU() if backend is None else TensorOperationsGPU(device=backend.device)
    result = ops.sliding_window_operation(array, window_size, operation, axis)
    return ops.to_cpu(result) if ops.is_gpu else result

def batch_tensor_operation(tensors: List[NDArray],
                         operation: Callable,
                         batch_size: Optional[int] = None,
                         concat_axis: Optional[int] = 0,
                         backend: Optional[GPUBackend] = None,
                         **kwargs) -> Union[np.ndarray, List]:
    """バッチテンソル演算のスタンドアロン関数"""
    ops = TensorOperationsGPU() if backend is None else TensorOperationsGPU(device=backend.device)
    return ops.batch_tensor_operation(tensors, operation, batch_size, concat_axis, **kwargs)

# ===============================
# Decorators for Tensor Operations
# ===============================

def tensorize(func: Callable) -> Callable:
    """
    関数をテンソル化対応にするデコレータ
    自動的にGPU/CPU切り替えとバッチ処理を行う
    """
    @wraps(func)
    def wrapper(*args, backend: Optional[GPUBackend] = None, **kwargs):
        # バックエンド決定
        if backend is None:
            backend = GPUBackend()
        
        # 引数をGPUに転送
        gpu_args = []
        for arg in args:
            if isinstance(arg, (np.ndarray, cp.ndarray)):
                gpu_args.append(backend.to_gpu(arg))
            else:
                gpu_args.append(arg)
        
        # 関数実行
        result = func(*gpu_args, **kwargs)
        
        # 結果をCPUに戻す
        if isinstance(result, (list, tuple)):
            return type(result)(
                backend.to_cpu(r) if HAS_GPU and isinstance(r, cp.ndarray) else r 
                for r in result
            )
        elif HAS_GPU and isinstance(result, cp.ndarray):
            return backend.to_cpu(result)
        else:
            return result
    
    return wrapper

# ===============================
# Performance Testing
# ===============================

def benchmark_tensor_operations(size: int = 10000):
    """テンソル演算のベンチマーク"""
    import time
    
    if not HAS_GPU:
        logger.warning("GPU not available for benchmarking")
        return
    
    ops = TensorOperationsGPU()
    
    logger.info(f"\n{'='*60}")
    logger.info("Tensor Operations Benchmarks")
    logger.info(f"{'='*60}")
    
    # テストデータ
    data = np.random.randn(size, 100).astype(np.float32)
    
    # 1. 勾配計算
    start = time.time()
    _ = ops.compute_gradient(data, axis=0)
    gpu_time = time.time() - start
    
    start = time.time()
    _ = np.gradient(data, axis=0)
    cpu_time = time.time() - start
    
    logger.info(f"\nGradient computation:")
    logger.info(f"  CPU: {cpu_time:.4f}s")
    logger.info(f"  GPU: {gpu_time:.4f}s")
    logger.info(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    
    # 2. 相関計算
    start = time.time()
    _ = ops.compute_correlation(data[:1000])
    gpu_time = time.time() - start
    
    start = time.time()
    _ = np.corrcoef(data[:1000])
    cpu_time = time.time() - start
    
    logger.info(f"\nCorrelation computation (1000x100):")
    logger.info(f"  CPU: {cpu_time:.4f}s")
    logger.info(f"  GPU: {gpu_time:.4f}s")
    logger.info(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    
    # 3. スライディングウィンドウ
    start = time.time()
    _ = ops.sliding_window_operation(data[:1000, 0], 50, 'mean')
    gpu_time = time.time() - start
    
    logger.info(f"\nSliding window (mean, window=50):")
    logger.info(f"  GPU: {gpu_time:.4f}s")
    
    logger.info(f"\n{'='*60}\n")
