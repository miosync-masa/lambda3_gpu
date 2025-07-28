"""
Lambda³ GPU-Accelerated MD Analysis Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

環ちゃんが作った超高速GPU版Lambda³だよ〜！💕
NO TIME, NO PHYSICS, ONLY STRUCTURE... but FASTER! 🚀

Basic usage:
    >>> from lambda3_gpu import MDLambda3DetectorGPU, MDConfig
    >>> config = MDConfig()
    >>> detector = MDLambda3DetectorGPU(config)
    >>> result = detector.analyze(trajectory)

Full documentation at https://github.com/your-repo/lambda3-gpu
"""

__version__ = '3.0.0-gpu'
__author__ = 'Lambda³ Project (GPU Edition by Tamaki)'
__email__ = 'tamaki@miosync.inc'
__license__ = 'MIT'

import warnings
import logging
import sys
from typing import Optional

# ===============================
# GPU環境チェック
# ===============================

# CuPyのインポートを試みる
try:
    import cupy as cp
    HAS_CUPY = True
    GPU_AVAILABLE = cp.cuda.is_available()
    
    if GPU_AVAILABLE:
        # GPU情報を取得
        GPU_DEVICE = cp.cuda.Device()
        GPU_NAME = GPU_DEVICE.name.decode('utf-8') if hasattr(GPU_DEVICE.name, 'decode') else str(GPU_DEVICE.name)
        GPU_MEMORY = GPU_DEVICE.mem_info[1] / 1024**3  # Total memory in GB
        GPU_COMPUTE_CAPABILITY = GPU_DEVICE.compute_capability
        
        # CUDA version
        CUDA_VERSION = cp.cuda.runtime.runtimeGetVersion()
        CUDA_VERSION_STR = f"{CUDA_VERSION // 1000}.{(CUDA_VERSION % 1000) // 10}"
    else:
        GPU_NAME = None
        GPU_MEMORY = 0
        GPU_COMPUTE_CAPABILITY = None
        CUDA_VERSION_STR = None
        
except ImportError:
    HAS_CUPY = False
    GPU_AVAILABLE = False
    GPU_NAME = None
    GPU_MEMORY = 0
    GPU_COMPUTE_CAPABILITY = None
    CUDA_VERSION_STR = None
    warnings.warn(
        "CuPy not installed! GPU acceleration disabled. "
        "Install with: pip install cupy-cuda11x (replace 11x with your CUDA version)",
        ImportWarning
    )

# ===============================
# 依存関係チェック
# ===============================

# 必須パッケージ
REQUIRED_PACKAGES = {
    'numpy': '1.19.0',
    'scipy': '1.5.0',
    'numba': '0.50.0',
    'matplotlib': '3.2.0'
}

# オプショナルパッケージ
OPTIONAL_PACKAGES = {
    'cupy': '8.0.0',
    'cupyx': None,  # CuPyに含まれる
    'joblib': '0.16.0',
    'tqdm': '4.50.0'
}

def check_dependencies():
    """依存関係をチェック"""
    missing_required = []
    missing_optional = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            module = __import__(package)
            # バージョンチェック（簡易版）
            if hasattr(module, '__version__'):
                current_version = module.__version__
                # TODO: より厳密なバージョン比較
        except ImportError:
            missing_required.append(package)
    
    for package, min_version in OPTIONAL_PACKAGES.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_required)}. "
            f"Please install with: pip install {' '.join(missing_required)}"
        )
    
    if missing_optional:
        warnings.warn(
            f"Missing optional packages for full functionality: {', '.join(missing_optional)}",
            ImportWarning
        )

# 依存関係チェック実行
check_dependencies()

# ===============================
# ログ設定
# ===============================

# ログフォーマット設定
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# デフォルトロガー設定
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

# Lambda3GPU専用ロガー
logger = logging.getLogger('lambda3_gpu')
logger.setLevel(logging.INFO)

# GPU状態ログ
if GPU_AVAILABLE:
    logger.info(f"🚀 GPU Enabled: {GPU_NAME}")
    logger.info(f"   Memory: {GPU_MEMORY:.1f} GB")
    logger.info(f"   CUDA: {CUDA_VERSION_STR}")
    logger.info(f"   Compute Capability: {GPU_COMPUTE_CAPABILITY}")
else:
    logger.warning("💻 Running in CPU mode (GPU not available)")

# ===============================
# パブリックAPI
# ===============================

# 環境情報エクスポート
__all__ = [
    # バージョン情報
    '__version__',
    
    # GPU情報
    'GPU_AVAILABLE',
    'GPU_NAME',
    'GPU_MEMORY',
    'CUDA_VERSION_STR',
    
    # 設定クラス（遅延インポート）
    'MDConfig',
    'ResidueAnalysisConfig',
    
    # メインクラス（遅延インポート）
    'MDLambda3DetectorGPU',
    'TwoStageAnalyzerGPU',
    
    # ユーティリティ
    'get_gpu_info',
    'set_gpu_device',
    'enable_gpu_logging',
    'benchmark_gpu'
]

# ===============================
# 遅延インポート
# ===============================

def __getattr__(name):
    """遅延インポートで起動時間短縮"""
    
    # 設定クラス
    if name == 'MDConfig':
        from lambda3_gpu.analysis.md_lambda3_detector_gpu import MDConfig
        return MDConfig
    
    elif name == 'ResidueAnalysisConfig':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import ResidueAnalysisConfig
        return ResidueAnalysisConfig
    
    # メインクラス
    elif name == 'MDLambda3DetectorGPU':
        from lambda3_gpu.analysis.md_lambda3_detector_gpu import MDLambda3DetectorGPU
        return MDLambda3DetectorGPU
    
    elif name == 'TwoStageAnalyzerGPU':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import TwoStageAnalyzerGPU
        return TwoStageAnalyzerGPU
    
    # その他
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ===============================
# ユーティリティ関数
# ===============================

def get_gpu_info() -> dict:
    """GPU情報を取得"""
    return {
        'available': GPU_AVAILABLE,
        'name': GPU_NAME,
        'memory_gb': GPU_MEMORY,
        'cuda_version': CUDA_VERSION_STR,
        'compute_capability': GPU_COMPUTE_CAPABILITY,
        'has_cupy': HAS_CUPY
    }

def set_gpu_device(device_id: int = 0) -> bool:
    """使用するGPUデバイスを設定"""
    if not GPU_AVAILABLE:
        logger.warning("GPU not available, cannot set device")
        return False
    
    try:
        cp.cuda.Device(device_id).use()
        logger.info(f"Set GPU device to: {device_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to set GPU device: {e}")
        return False

def enable_gpu_logging(level: str = 'DEBUG') -> None:
    """GPU関連の詳細ログを有効化"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger.setLevel(numeric_level)
    
    # CuPyのログも設定
    if HAS_CUPY:
        cupy_logger = logging.getLogger('cupy')
        cupy_logger.setLevel(numeric_level)

def benchmark_gpu(size: int = 10000) -> Optional[dict]:
    """簡易GPUベンチマーク"""
    if not GPU_AVAILABLE:
        logger.warning("GPU not available for benchmarking")
        return None
    
    import time
    
    # CPU計測
    import numpy as np
    cpu_array = np.random.randn(size, size).astype(np.float32)
    
    cpu_start = time.time()
    cpu_result = np.matmul(cpu_array, cpu_array)
    cpu_time = time.time() - cpu_start
    
    # GPU計測
    gpu_array = cp.asarray(cpu_array)
    
    gpu_start = time.time()
    gpu_result = cp.matmul(gpu_array, gpu_array)
    cp.cuda.Stream.null.synchronize()  # 同期待ち
    gpu_time = time.time() - gpu_start
    
    speedup = cpu_time / gpu_time
    
    result = {
        'matrix_size': size,
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup
    }
    
    logger.info(f"Benchmark result: {speedup:.1f}x speedup on {size}x{size} matrix multiplication")
    
    return result

# ===============================
# 初期化時メッセージ
# ===============================

def _print_banner():
    """かわいいバナー表示"""
    if sys.stdout.isatty():  # ターミナルの場合のみ
        print("\n" + "="*60)
        print("🌟 Lambda³ GPU - Structural Analysis at Light Speed! 🚀")
        print("="*60)
        if GPU_AVAILABLE:
            print(f"✨ GPU Mode: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
        else:
            print("💻 CPU Mode (Install CuPy for GPU acceleration)")
        print("="*60 + "\n")

# インタラクティブモードの場合のみバナー表示
if hasattr(sys, 'ps1'):
    _print_banner()

# ===============================
# エラーハンドリング
# ===============================

class Lambda3GPUError(Exception):
    """Lambda3GPU基本例外クラス"""
    pass

class GPUMemoryError(Lambda3GPUError):
    """GPUメモリ不足エラー"""
    pass

class GPUNotAvailableError(Lambda3GPUError):
    """GPU利用不可エラー"""
    pass

# 環境変数の確認
import os

# GPUメモリ制限設定
if 'LAMBDA3_GPU_MEMORY_LIMIT' in os.environ:
    try:
        memory_limit_gb = float(os.environ['LAMBDA3_GPU_MEMORY_LIMIT'])
        logger.info(f"GPU memory limit set to: {memory_limit_gb} GB")
        # TODO: 実際のメモリ制限実装
    except ValueError:
        logger.warning("Invalid LAMBDA3_GPU_MEMORY_LIMIT value")

# デバッグモード
if os.environ.get('LAMBDA3_DEBUG', '').lower() in ('1', 'true', 'yes'):
    enable_gpu_logging('DEBUG')
    logger.debug("Debug mode enabled")
