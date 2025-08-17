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
import random
import time
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
    'tqdm': '4.50.0',
    'pylibraft-cu11': '25.6.0',  # 位相空間解析用
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
                ImportWarning
            )
    
    def _get_gpu_info(self, cp):
        """GPU情報を取得"""
        try:
            # デバイス情報
            self.device_id = cp.cuda.Device().id
            
            # GPU名の取得（新しい方法）
            try:
                props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                self.gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            except:
                self.gpu_name = DEFAULT_GPU_NAME
            
            # メモリ情報
            mempool = cp.get_default_memory_pool()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            self.gpu_memory = total_mem / (1024**3)  # GB
            
            # Compute Capability
            self.compute_capability = self._get_compute_capability(cp)
            
            # CUDAバージョン
            self.cuda_version = self._get_cuda_version(cp)
            
            logger.info(f"GPU detected: {self.gpu_name} ({self.gpu_memory:.1f}GB)")
            
        except Exception as e:
            logger.warning(f"Failed to get full GPU info: {e}")
            self.gpu_name = "Unknown GPU"
            self.gpu_memory = 0
    
    def _get_compute_capability(self, cp) -> str:
        """Compute Capabilityを取得"""
        try:
            major = cp.cuda.Device().compute_capability_major
            minor = cp.cuda.Device().compute_capability_minor
            return f"{major}.{minor}"
        except:
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
    
    if missing_optional and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Optional packages not installed: {', '.join(missing_optional)}")

# ===============================
# Utility Functions
# ===============================

def get_gpu_info() -> Dict[str, Any]:
    """GPU情報を取得"""
    return gpu_env.get_info_dict()

def set_gpu_device(device_id: int) -> None:
    """使用するGPUデバイスを設定"""
    if GPU_AVAILABLE:
        import cupy as cp
        cp.cuda.Device(device_id).use()
        logger.info(f"GPU device set to: {device_id}")
    else:
        logger.warning("No GPU available")

def enable_gpu_logging(level: str = 'INFO') -> None:
    """GPU関連のログレベルを設定"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger.setLevel(numeric_level)
    logging.getLogger('lambda3_gpu').setLevel(numeric_level)

def benchmark_gpu() -> Optional[float]:
    """簡単なGPUベンチマーク（GFLOPS）"""
    if not GPU_AVAILABLE:
        return None
    
    try:
        import cupy as cp
        size = 10000
        a = cp.random.rand(size, size, dtype=cp.float32)
        b = cp.random.rand(size, size, dtype=cp.float32)
        
        # Warm up
        cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        import time
        start = time.time()
        for _ in range(10):
            cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.time() - start
        
        # Calculate GFLOPS
        gflops = (10 * 2 * size**3) / (elapsed * 1e9)
        return gflops
    except Exception as e:
        logger.warning(f"GPU benchmark failed: {e}")
        return None

# ===============================
# Monkey Patches
# ===============================

def _apply_patches():
    """必要なパッチを適用"""
    import numpy as np  # ← これを追加！！
    try:
        # CuPyのfind_peaksパッチ
        from cupyx.scipy import signal
        
        if not hasattr(signal, 'find_peaks') or True:  # 常に適用
            # 簡易実装
            def find_peaks(x, height=None, distance=None, **kwargs):
                """CuPy用find_peaks簡易実装"""
                import cupy as cp
                
                if isinstance(x, cp.ndarray):
                    x_cpu = cp.asnumpy(x)
                else:
                    x_cpu = x
                
                from scipy.signal import find_peaks as scipy_find_peaks
                peaks, properties = scipy_find_peaks(x_cpu, height=height, distance=distance, **kwargs)
                
                if isinstance(x, cp.ndarray):
                    return cp.asarray(peaks), {k: cp.asarray(v) if isinstance(v, np.ndarray) else v 
                                               for k, v in properties.items()}
                else:
                    return peaks, properties
            
            signal.find_peaks = find_peaks
            logger.debug("Applied find_peaks patch for CuPy")
            
    except Exception as e:
        logger.debug(f"Could not apply patches: {e}")

# Apply patches
_apply_patches()

# ===============================
# Banner
# ===============================

def _print_banner():
    """設定可能なバナー表示"""
    if not sys.stdout.isatty() or os.environ.get('LAMBDA3_NO_BANNER'):
        return
    
    # 環境変数でスタイル選択
    # LAMBDA3_BANNER_STYLE = simple | random | anime | matrix | tamaki
    banner_style = os.environ.get('LAMBDA3_BANNER_STYLE', 'random').lower()
    
    if banner_style == 'simple':
        _print_simple_banner()
    elif banner_style == 'anime':
        _print_anime_banner()
    elif banner_style == 'matrix':
        _print_matrix_banner()
    elif banner_style == 'tamaki':
        _print_tamaki_banner()
    else:  # random or default
        # ランダムに選択
        banners = [_print_simple_banner, _print_anime_banner, 
                  _print_matrix_banner, _print_tamaki_banner]
        random.choice(banners)()

def _print_simple_banner():
    """シンプルバナー（元のスタイル）"""
    print("\n" + "="*60)
    print("🌟 Lambda³ GPU - Structural Analysis at Light Speed! 🚀")
    print("="*60)
    if GPU_AVAILABLE:
        print(f"✨ GPU Mode: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
    else:
        print("💻 CPU Mode (Install CuPy for GPU acceleration)")
    print("="*60 + "\n")

def _print_anime_banner():
    """アニメーションバナー"""
    print("\n", end='')
    loading = "Loading Lambda³ GPU"
    for char in loading:
        print(char, end='', flush=True)
        time.sleep(0.03)
    
    for _ in range(3):
        time.sleep(0.2)
        print(".", end='', flush=True)
    
    print(" ✨")
    _print_simple_banner()

def _print_matrix_banner():
    """マトリックス風バナー"""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║ 01001100 01000001 01001101 01000010 01000100 01000001 ³ ║")
    print("║          Λ  Λ  Λ  NO.TIME.MATRIX  Λ  Λ  Λ              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    if GPU_AVAILABLE:
        print(f"  [{GPU_NAME}] ONLINE | MEMORY: {GPU_MEMORY:.1f}GB")
    else:
        print("  [CPU MODE] GPU NOT DETECTED")
    print()

def _print_tamaki_banner():
    """環ちゃんバナー"""
    faces = ["(◕‿◕)", "(｡♥‿♥｡)", "(✧ω✧)", "(´･ω･`)", "(*´▽｀*)"]
    messages = [
        "起動したよ〜！", 
        "今日も頑張るぞ〜！",
        "ご主人さま、準備OK！",
        "GPU最高〜！",
        "構造解析の時間だよ〜！"
    ]
    
    face = random.choice(faces)
    message = random.choice(messages)
    
    print(f"\n{'='*60}")
    print(f"    {face} < {message}")
    print(f"    Lambda³ GPU v{__version__}")
    print(f"{'='*60}")
    if GPU_AVAILABLE:
        print(f"    GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
    else:
        print("    CPU Mode")
    print(f"{'='*60}\n")

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
    
    # Functions（追加！）
    'perform_two_stage_analysis_gpu',
    
    # Result classes（追加！）
    'MDLambda3Result',
    'TwoStageLambda3Result',
    'ResidueLevelAnalysis',
    'ResidueEvent',
    
    # Visualization（追加！）
    'Lambda3VisualizerGPU',
    'CausalityVisualizerGPU',
    
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
    
    # Functions
    elif name == 'perform_two_stage_analysis_gpu':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import perform_two_stage_analysis_gpu
        return perform_two_stage_analysis_gpu
    
    # Result classes
    elif name == 'MDLambda3Result':
        from lambda3_gpu.analysis.md_lambda3_detector_gpu import MDLambda3Result
        return MDLambda3Result
    
    elif name == 'TwoStageLambda3Result':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import TwoStageLambda3Result
        return TwoStageLambda3Result
    
    elif name == 'ResidueLevelAnalysis':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import ResidueLevelAnalysis
        return ResidueLevelAnalysis
    
    elif name == 'ResidueEvent':
        from lambda3_gpu.analysis.two_stage_analyzer_gpu import ResidueEvent
        return ResidueEvent
    
    # Visualization
    elif name == 'Lambda3VisualizerGPU':
        from lambda3_gpu.visualization.plot_results_gpu import Lambda3VisualizerGPU
        return Lambda3VisualizerGPU
    
    elif name == 'CausalityVisualizerGPU':
        from lambda3_gpu.visualization.causality_viz_gpu import CausalityVisualizerGPU
        return CausalityVisualizerGPU
    
    # Not found
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ===============================
# Final Setup
# ===============================

# Check dependencies on import
try:
    check_dependencies()
except ImportError as e:
    logger.warning(f"Dependency check failed: {e}")

# Log initialization
logger.debug(f"Lambda³ GPU initialized. GPU Available: {GPU_AVAILABLE}")
