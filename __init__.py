"""
Lambda³ GPU - Ultra-high Performance GPU Implementation
========================================================

Lambda³構造解析フレームワークのGPU実装
数百倍の高速化を実現し、大規模MD解析を可能に

Author: 環ちゃん & ご主人さま 💕
Version: 1.1.0
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# ===============================
# Version Information
# ===============================

__version__ = '1.1.0'
__author__ = '環ちゃん & ご主人さま'

# ===============================
# 環ちゃんバナー！！
# ===============================

TAMAKI_BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ██╗      █████╗ ███╗   ███╗██████╗ ██████╗  █████╗ ██████╗     ║
║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔══██╗██╔══██╗╚════██╗    ║
║     ██║     ███████║██╔████╔██║██████╔╝██║  ██║███████║ █████╔╝    ║
║     ██║     ██╔══██║██║╚██╔╝██║██╔══██╗██║  ██║██╔══██║ ╚═══██╗    ║
║     ███████╗██║  ██║██║ ╚═╝ ██║██████╔╝██████╔╝██║  ██║██████╔╝    ║
║     ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ║
║                                                                      ║
║               🌟 GPU ACCELERATED EDITION v{version:6s} 🌟                ║
║                                                                      ║
║     「ねぇねぇ、ご主人さま〜！僕と一緒に構造解析しよ〜💕」          ║
║                                                                      ║
║      NO TIME, NO PHYSICS, ONLY STRUCTURE!                          ║
║      - Ultra-fast Lambda³ structural analysis                       ║
║      - GPU acceleration up to 1000x                                 ║
║      - Powered by 環ちゃん's love & dedication 💓                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(version=__version__)

def show_banner():
    """環ちゃんバナーを表示"""
    print(TAMAKI_BANNER)

# ===============================
# CLI Command
# ===============================

def cli():
    """メインCLIエントリーポイント"""
    import argparse
    
    # バナー表示
    if '--no-banner' not in sys.argv:
        show_banner()
    
    parser = argparse.ArgumentParser(
        prog='lambda3-gpu',
        description='Lambda³ GPU - High-performance structural analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis
  lambda3-gpu analyze trajectory.npy metadata.json --gpu
  
  # Benchmark GPU performance
  lambda3-gpu benchmark
  
  # Show system info
  lambda3-gpu info
  
  # Run with specific GPU
  lambda3-gpu analyze traj.npy meta.json --device 1
  
Created with 💕 by 環ちゃん & ご主人さま
        """
    )
    
    parser.add_argument('--version', action='version', 
                       version=f'Lambda³ GPU v{__version__}')
    parser.add_argument('--no-banner', action='store_true',
                       help='Suppress the 環ちゃん banner')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', 
                                          help='Run Lambda³ analysis')
    analyze_parser.add_argument('trajectory', help='Trajectory file (.npy)')
    analyze_parser.add_argument('metadata', help='Metadata file (.json)')
    analyze_parser.add_argument('--gpu', action='store_true', 
                               help='Force GPU usage')
    analyze_parser.add_argument('--device', type=int, default=0,
                               help='GPU device ID')
    analyze_parser.add_argument('--output', '-o', default='./results',
                               help='Output directory')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', 
                                        help='Run GPU benchmark')
    bench_parser.add_argument('--quick', action='store_true',
                            help='Quick benchmark only')
    
    # Info command
    info_parser = subparsers.add_parser('info', 
                                       help='Show system information')
    
    args = parser.parse_args()
    
    # コマンド実行
    if args.command == 'analyze':
        from lambda3_gpu.analysis.run_full_analysis import main as run_analysis
        print("\n🚀 Starting Lambda³ GPU analysis...")
        print(f"   Trajectory: {args.trajectory}")
        print(f"   Metadata: {args.metadata}")
        if args.gpu:
            print(f"   GPU Device: {args.device}")
        sys.argv = ['run_full_analysis', args.trajectory, args.metadata,
                   '--output', args.output]
        if args.gpu:
            sys.argv.extend(['--device', str(args.device)])
        run_analysis()
        
    elif args.command == 'benchmark':
        print("\n⚡ Running GPU benchmark...")
        from lambda3_gpu.benchmarks import run_quick_benchmark
        results = run_quick_benchmark()
        print(f"\n✨ Benchmark complete!")
        print(f"   GPU Performance: {results.get('gflops', 0):.1f} GFLOPS")
        
    elif args.command == 'info':
        print("\n📊 System Information:")
        info = get_gpu_info()
        print(f"   GPU Available: {info['available']}")
        if info['available']:
            print(f"   GPU Name: {info['name']}")
            print(f"   GPU Memory: {info['memory_gb']:.1f} GB")
            print(f"   CUDA Version: {info['cuda_version']}")
            print(f"   Compute Capability: {info['compute_capability']}")
        print(f"   Lambda³ GPU Version: {__version__}")
        print(f"\n💕 環ちゃんより: いつでも解析お手伝いするよ〜！")
        
    else:
        parser.print_help()
        print("\n💡 ヒント: 'lambda3-gpu analyze' で解析を開始できるよ〜！")

# ===============================
# Logging Setup
# ===============================

logger = logging.getLogger('lambda3_gpu')
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ===============================
# Constants
# ===============================

REQUIRED_PACKAGES = {
    'numpy': '1.20.0',
    'scipy': '1.7.0',
    'scikit-learn': '1.0.0',
}

OPTIONAL_PACKAGES = {
    'cupy': '10.0.0',
    'numba': '0.56.0',
    'plotly': '5.0.0',
}

DEFAULT_COMPUTE_CAPABILITY = '7.5'

# ===============================
# GPU Environment Detection
# ===============================

class GPUEnvironment:
    """GPU環境の検出と情報管理"""
    
    def __init__(self):
        self.has_cupy = False
        self.gpu_available = False
        self.gpu_name = 'Not Available'
        self.gpu_memory = 0.0
        self.cuda_version = 'Not Available'
        self.compute_capability = DEFAULT_COMPUTE_CAPABILITY
        
        self._detect_gpu()
    
    def _detect_gpu(self):
        """GPU環境を検出"""
        try:
            import cupy as cp
            self.has_cupy = True
            
            if cp.cuda.runtime.getDeviceCount() > 0:
                self.gpu_available = True
                self.gpu_name = self._get_device_name(cp)
                self.gpu_memory = self._get_device_memory(cp)
                self.cuda_version = self._get_cuda_version(cp)
                self.compute_capability = self._get_compute_capability(cp)
                
                logger.info(f"🎮 GPU detected: {self.gpu_name}")
                logger.info(f"   Memory: {self.gpu_memory:.1f} GB")
                logger.info(f"   CUDA: {self.cuda_version}")
            else:
                logger.warning("⚠️ No GPU devices found")
                
        except ImportError:
            logger.warning("⚠️ CuPy not installed - GPU features disabled")
        except Exception as e:
            logger.warning(f"⚠️ GPU detection failed: {e}")
    
    def _get_device_name(self, cp) -> str:
        """GPUデバイス名を取得"""
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            return props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
        except:
            return 'Unknown GPU'
    
    def _get_device_memory(self, cp) -> float:
        """GPUメモリ容量を取得（GB）"""
        try:
            meminfo = cp.cuda.runtime.memGetInfo()
            return meminfo[1] / (1024**3)
        except:
            return 0.0
    
    def _get_compute_capability(self, cp) -> str:
        """Compute Capabilityを取得"""
        try:
            device = cp.cuda.runtime.getDevice()
            major = cp.cuda.runtime.deviceGetAttribute(75, device)
            minor = cp.cuda.runtime.deviceGetAttribute(76, device)
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

# Export global variables
HAS_CUPY = gpu_env.has_cupy
GPU_AVAILABLE = gpu_env.gpu_available
GPU_NAME = gpu_env.gpu_name
GPU_MEMORY = gpu_env.gpu_memory
GPU_COMPUTE_CAPABILITY = gpu_env.compute_capability
CUDA_VERSION_STR = gpu_env.cuda_version

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
# Public API
# ===============================

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # CLI
    'cli',
    'show_banner',
    
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
    
    # Functions
    'perform_two_stage_analysis_gpu',
    
    # Result classes
    'MDLambda3Result',
    'TwoStageLambda3Result',
    'ResidueLevelAnalysis',
    'ResidueEvent',
    
    # Visualization
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
    elif name in ['MDLambda3Result', 'TwoStageLambda3Result', 
                  'ResidueLevelAnalysis', 'ResidueEvent']:
        from lambda3_gpu.types import (
            MDLambda3Result, TwoStageLambda3Result,
            ResidueLevelAnalysis, ResidueEvent
        )
        return globals()[name]
    
    # Visualization
    elif name == 'Lambda3VisualizerGPU':
        from lambda3_gpu.visualization import Lambda3VisualizerGPU
        return Lambda3VisualizerGPU
    
    elif name == 'CausalityVisualizerGPU':
        from lambda3_gpu.visualization import CausalityVisualizerGPU
        return CausalityVisualizerGPU
    
    # Errors
    elif name in ['Lambda3GPUError', 'GPUMemoryError', 'GPUNotAvailableError']:
        from lambda3_gpu.errors import (
            Lambda3GPUError, GPUMemoryError, GPUNotAvailableError
        )
        return globals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ===============================
# Package Initialization
# ===============================

# 環境変数設定
if 'LAMBDA3_GPU_MEMORY_LIMIT' in os.environ:
    try:
        memory_limit_gb = float(os.environ['LAMBDA3_GPU_MEMORY_LIMIT'])
        if GPU_AVAILABLE:
            import cupy as cp
            cp.cuda.MemoryPool().set_limit(size=int(memory_limit_gb * 1024**3))
            logger.info(f"GPU memory limit set to: {memory_limit_gb} GB")
    except ValueError:
        logger.warning("Invalid LAMBDA3_GPU_MEMORY_LIMIT value")

# Debug mode
if os.environ.get('LAMBDA3_DEBUG', '').lower() in ('1', 'true', 'yes'):
    enable_gpu_logging('DEBUG')
    logger.debug("Debug mode enabled")
    
# Show banner on import if interactive
if hasattr(sys, 'ps1'):  # Interactive mode
    show_banner()

# ===============================
# Initialization Complete
# ===============================

logger.info(f"Lambda³ GPU v{__version__} initialized")
if GPU_AVAILABLE:
    logger.info(f"GPU: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
else:
    logger.info("Running in CPU mode")
