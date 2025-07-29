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
import os
from typing import Optional, Dict, Any, Tuple

# ===============================
# Constants
# ===============================

LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

REQUIRED_PACKAGES = {
    'numpy': '1.19.0',
    'scipy': '1.5.0',
    'numba': '0.50.0',
    'matplotlib': '3.2.0'
}

OPTIONAL_PACKAGES = {
    'cupy': '8.0.0',
    'cupyx': None,
    'joblib': '0.16.0',
    'tqdm': '4.50.0'
}

# Default values for A100
DEFAULT_COMPUTE_CAPABILITY = "8.0"
DEFAULT_GPU_NAME = "NVIDIA A100"

# ===============================
# Logging Setup
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

logger = logging.getLogger('lambda3_gpu')
logger.setLevel(logging.INFO)

# ===============================
# GPU Detection and Setup
# ===============================

class GPUEnvironment:
    """GPU環境情報を管理するクラス"""
    
    def __init__(self):
        self.has_cupy = False
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_memory = 0
        self.compute_capability = None
        self.cuda_version = None
        self.device_id = 0
        
        self._detect_gpu()
    
    def _detect_gpu(self):
        """GPU環境を検出"""
        try:
            import cupy as cp
            self.has_cupy = True
            
            if cp.cuda.is_available():
                self.gpu_available = True
                self._get_gpu_info(cp)
            else:
                logger.warning("CuPy is installed but no GPU is available")
                
        except ImportError:
            warnings.warn(
                "CuPy not installed! GPU acceleration disabled. "
                "Install with: pip install cupy-cuda12x",
                ImportWarning
            )
        except Exception as e:
            self.has_cupy = True
            warnings.warn(
                f"GPU initialization error: {e}. Running in CPU mode.",
                RuntimeWarning
            )
    
    def _get_gpu_info(self, cp):
        """GPU詳細情報を取得"""
        try:
            # デバイスを選択
            cp.cuda.Device(self.device_id).use()
            
            # デバイスプロパティを取得
            props = self._get_device_properties(cp)
            
            # GPU名
            self.gpu_name = self._extract_gpu_name(props)
            
            # メモリ情報
            self.gpu_memory = self._get_gpu_memory(cp)
            
            # Compute Capability
            self.compute_capability = self._get_compute_capability(cp, props)
            
            # CUDA version
            self.cuda_version = self._get_cuda_version(cp)
            
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            self.gpu_available = False
    
    def _get_device_properties(self, cp):
        """デバイスプロパティを安全に取得"""
        try:
            return cp.cuda.runtime.getDeviceProperties(self.device_id)
        except Exception as e:
            logger.warning(f"Could not get device properties: {e}")
            return {}
    
    def _extract_gpu_name(self, props) -> str:
        """GPU名を抽出"""
        try:
            if isinstance(props, dict):
                name = props.get('name', DEFAULT_GPU_NAME)
            else:
                name = getattr(props, 'name', DEFAULT_GPU_NAME)
            
            # バイト文字列の場合はデコード
            if isinstance(name, bytes):
                return name.decode('utf-8')
            return str(name)
        except:
            return DEFAULT_GPU_NAME
    
    def _get_gpu_memory(self, cp) -> float:
        """GPUメモリ容量を取得（GB）"""
        try:
            _, total_memory = cp.cuda.runtime.memGetInfo()
            return total_memory / 1024**3
        except:
            return 0.0
    
    def _get_compute_capability(self, cp, props) -> str:
        """Compute Capabilityを取得"""
        try:
            # 方法1: propsから取得
            if isinstance(props, dict):
                major = props.get('major', 0)
                minor = props.get('minor', 0)
            else:
                major = getattr(props, 'major', 0)
                minor = getattr(props, 'minor', 0)
            
            # 取得できた場合
            if major > 0:
                return f"{major}.{minor}"
            
            # 方法2: デバイス属性から取得（古い方法）
            try:
                device = cp.cuda.Device(self.device_id)
                attrs = device.attributes
                if hasattr(attrs, 'get') and 'ComputeCapability' in attrs:
                    return attrs['ComputeCapability']
            except:
                pass
            
            # デフォルト値を返す
            logger.warning(f"Could not detect compute capability, using default: {DEFAULT_COMPUTE_CAPABILITY}")
            return DEFAULT_COMPUTE_CAPABILITY
            
        except Exception as e:
            logger.warning(f"Error getting compute capability: {e}")
            return DEFAULT_COMPUTE_CAPABILITY
    
    def _get_cuda_version(self, cp) -> str:
        """CUDAバージョンを取得"""
        try:
            version = cp.cuda.runtime.runtimeGetVersion()
            major = version // 1000
            minor = (version % 1000) // 10
            return f"{major}.{minor}"
        except:
            return "Unknown"
    
    def get_info_dict(self) -> Dict[str, Any]:
        """環境情報を辞書で返す"""
        return {
            'available': self.gpu_available,
            'name': self.gpu_name,
            'memory_gb': self.gpu_memory,
            'cuda_version': self.cuda_version,
            'compute_capability': self.compute_capability,
            'has_cupy': self.has_cupy
        }

# ===============================
# Initialize GPU Environment
# ===============================

gpu_env = GPUEnvironment()

# Export global variables for backward compatibility
HAS_CUPY = gpu_env.has_cupy
GPU_AVAILABLE = gpu_env.gpu_available
GPU_NAME = gpu_env.gpu_name
GPU_MEMORY = gpu_env.gpu_memory
GPU_COMPUTE_CAPABILITY = gpu_env.compute_capability
CUDA_VERSION_STR = gpu_env.cuda_version

# ===============================
# Dependency Checking
# ===============================

def check_dependencies():
    """依存関係をチェック"""
    missing_required = []
    missing_optional = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            __import__(package)
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
        logger.debug(f"Optional packages not installed: {', '.join(missing_optional)}")

# Run dependency check
check_dependencies()

# ===============================
# Log GPU Status
# ===============================

if GPU_AVAILABLE:
    logger.info(f"🚀 GPU Enabled: {GPU_NAME}")
    logger.info(f"   Memory: {GPU_MEMORY:.1f} GB")
    logger.info(f"   CUDA: {CUDA_VERSION_STR}")
    logger.info(f"   Compute Capability: {GPU_COMPUTE_CAPABILITY}")
else:
    logger.warning("💻 Running in CPU mode (GPU not available)")

# ===============================
# Utility Functions
# ===============================

def get_gpu_info() -> Dict[str, Any]:
    """GPU情報を取得"""
    return gpu_env.get_info_dict()

def set_gpu_device(device_id: int = 0) -> bool:
    """使用するGPUデバイスを設定"""
    if not GPU_AVAILABLE:
        logger.warning("GPU not available, cannot set device")
        return False
    
    try:
        import cupy as cp
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
    
    if HAS_CUPY:
        cupy_logger = logging.getLogger('cupy')
        cupy_logger.setLevel(numeric_level)

def benchmark_gpu(size: int = 10000) -> Optional[Dict[str, float]]:
    """簡易GPUベンチマーク"""
    if not GPU_AVAILABLE:
        logger.warning("GPU not available for benchmarking")
        return None
    
    import time
    import numpy as np
    import cupy as cp
    
    # CPU計測
    cpu_array = np.random.randn(size, size).astype(np.float32)
    
    cpu_start = time.perf_counter()
    cpu_result = np.matmul(cpu_array, cpu_array)
    cpu_time = time.perf_counter() - cpu_start
    
    # GPU計測
    gpu_array = cp.asarray(cpu_array)
    cp.cuda.Stream.null.synchronize()
    
    gpu_start = time.perf_counter()
    gpu_result = cp.matmul(gpu_array, gpu_array)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start
    
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
# Banner Display
# ===============================

def _print_banner():
    """かわいいバナー表示"""
    if sys.stdout.isatty():
        print("\n" + "="*60)
        print("🌟 Lambda³ GPU - Structural Analysis at Light Speed! 🚀")
        print("="*60)
        if GPU_AVAILABLE:
            print(f"✨ GPU Mode: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
        else:
            print("💻 CPU Mode (Install CuPy for GPU acceleration)")
        print("="*60 + "\n")

# Show banner in interactive mode
if hasattr(sys, 'ps1'):
    _print_banner()

# ===============================
# Error Classes
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

# ===============================
# Environment Variables
# ===============================

# GPU memory limit
if 'LAMBDA3_GPU_MEMORY_LIMIT' in os.environ:
    try:
        memory_limit_gb = float(os.environ['LAMBDA3_GPU_MEMORY_LIMIT'])
        logger.info(f"GPU memory limit set to: {memory_limit_gb} GB")
    except ValueError:
        logger.warning("Invalid LAMBDA3_GPU_MEMORY_LIMIT value")

# Debug mode
if os.environ.get('LAMBDA3_DEBUG', '').lower() in ('1', 'true', 'yes'):
    enable_gpu_logging('DEBUG')
    logger.debug("Debug mode enabled")

# ===============================
# Public API
# ===============================

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # GPU info
    'GPU_AVAILABLE',
    'GPU_NAME', 
    'GPU_MEMORY',
    'CUDA_VERSION_STR',
    'GPU_COMPUTE_CAPABILITY',
    
    # Classes (lazy import)
    'MDConfig',
    'ResidueAnalysisConfig',
    'MDLambda3DetectorGPU',
    'TwoStageAnalyzerGPU',
    
    # Utilities
    'get_gpu_info',
    'set_gpu_device',
    'enable_gpu_logging',
    'benchmark_gpu',
    
    # Errors
    'Lambda3GPUError',
    'GPUMemoryError',
    'GPUNotAvailableError'
]

# ===============================
# Lazy Imports
# ===============================

def __getattr__(name):
    """遅延インポートで起動時間短縮"""
    
    # Config classes
    if name == 'MDConfig':
        from lambda3_gpu.analysis.md_lambda3_detector_gpu import MDConfig
        return MDConfig
    
    elif name == 'ResidueAnalysisConfig':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import ResidueAnalysisConfig
        return ResidueAnalysisConfig
    
    # Main classes
    elif name == 'MDLambda3DetectorGPU':
        from lambda3_gpu.analysis.md_lambda3_detector_gpu import MDLambda3DetectorGPU
        return MDLambda3DetectorGPU
    
    elif name == 'TwoStageAnalyzerGPU':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import TwoStageAnalyzerGPU
        return TwoStageAnalyzerGPU
    
    # Not found
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ===============================
# Import Hook for Monkey Patching
# ===============================

def _apply_patches():
    """必要なパッチを適用"""
    try:
        # find_peaks パッチ
        from cupyx.scipy import signal
        if not hasattr(signal, 'find_peaks'):
            from .core.gpu_patches import find_peaks_gpu_fallback
            signal.find_peaks = find_peaks_gpu_fallback
            logger.debug("Applied find_peaks patch")
    except:
        pass

# Apply patches on import
_apply_patches()
