"""
Lambda³ GPU Material Analysis Package
======================================

材料解析用Lambda³ GPUパッケージ
転位・亀裂・相変態の階層的解析

トポロジカル欠陥解析とCUDAカーネル最適化対応

Version: 2.0.0
Authors: 環ちゃん
"""

__version__ = '2.0.0'
__author__ = '環ちゃん'

# ========================================
# Main Pipeline
# ========================================
from .material_full_analysis import (
    run_material_analysis_pipeline,
    get_material_parameters,
    create_spatial_clusters
)

# ========================================
# Macro Analysis
# ========================================
from .material_lambda3_detector import (
    MaterialLambda3DetectorGPU,
    MaterialLambda3Result,
    MaterialConfig,
    detect_material_events
)

# ========================================
# CUDA Kernels (v2.0 新機能)
# ========================================
try:
    from .cuda_kernels import (
        MaterialCUDAKernels,
        STRAIN_TENSOR_KERNEL_CODE,
        COORDINATION_NUMBER_KERNEL_CODE,
        DAMAGE_SCORE_KERNEL_CODE
    )
    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False
    MaterialCUDAKernels = None

# ========================================
# Two-Stage Analysis
# ========================================
from .material_two_stage_analyzer import (
    MaterialTwoStageAnalyzerGPU,
    MaterialTwoStageResult,
    ClusterAnalysisConfig,
    ClusterEvent,
    ClusterLevelAnalysis,
    perform_material_two_stage_analysis_gpu
)

# ========================================
# Impact Analysis
# ========================================
from .material_impact_analytics import (
    MaterialImpactAnalyzer,
    MaterialImpactResult,
    MaterialDefectNetwork,
    MaterialDefectNetworkGPU,
    AtomicDefectTrace,
    DefectOrigin,
    run_material_impact_analysis
)

# ========================================
# Report Generation
# ========================================
from .material_report_generator import (
    generate_material_report_from_results,
    prepare_vtk_export_data,
    generate_material_comparison_report
)

# ========================================
# Material Database
# ========================================
MATERIAL_DATABASE = {
    'SUJ2': {
        'name': 'Bearing Steel (SUJ2)',
        'E': 210.0,      # GPa
        'nu': 0.3,       # Poisson's ratio
        'yield': 1.5,    # GPa
        'ultimate': 2.0, # GPa
        'K_IC': 30.0,    # MPa√m
        'density': 7.85, # g/cm³
        'crystal': 'BCC'
    },
    'AL7075': {
        'name': 'Aluminum Alloy 7075',
        'E': 71.7,
        'nu': 0.33,
        'yield': 0.503,
        'ultimate': 0.572,
        'K_IC': 23.0,
        'density': 2.81,
        'crystal': 'FCC'
    },
    'Ti6Al4V': {
        'name': 'Titanium Alloy (Ti-6Al-4V)',
        'E': 113.8,
        'nu': 0.342,
        'yield': 0.88,
        'ultimate': 0.95,
        'K_IC': 75.0,
        'density': 4.43,
        'crystal': 'HCP'
    },
    'SS316L': {
        'name': 'Stainless Steel 316L',
        'E': 193.0,
        'nu': 0.3,
        'yield': 0.205,
        'ultimate': 0.515,
        'K_IC': 112.0,
        'density': 8.0,
        'crystal': 'FCC'
    }
}

# ========================================
# Package Information
# ========================================
def get_package_info():
    """パッケージ情報を取得"""
    cuda_status = "Enabled" if HAS_CUDA_KERNELS else "Disabled"
    
    return {
        'name': 'material_analysis',
        'version': __version__,
        'description': 'Lambda³ GPU Material Analysis Package',
        'features': [
            'Macro material event detection',
            'Two-stage cluster analysis',
            'Atomic-level defect analysis',
            'Topological charge analysis (v2.0)',
            'Structural coherence detection (v2.0)',
            'CUDA kernel optimization (v2.0)',
            'Comprehensive report generation',
            'GPU acceleration support'
        ],
        'cuda_kernels': cuda_status,
        'materials': list(MATERIAL_DATABASE.keys()),
        'author': __author__
    }

def list_available_materials():
    """利用可能な材料リスト"""
    print("\n💎 Available Materials:")
    print("="*50)
    for key, props in MATERIAL_DATABASE.items():
        print(f"\n{key}: {props['name']}")
        print(f"  - Elastic modulus: {props['E']} GPa")
        print(f"  - Yield strength: {props['yield']} GPa")
        print(f"  - Fracture toughness: {props['K_IC']} MPa√m")
        print(f"  - Crystal structure: {props['crystal']}")

def check_cuda_status():
    """CUDAカーネルの状態確認"""
    print("\n🚀 CUDA Kernel Status:")
    print("="*50)
    
    if HAS_CUDA_KERNELS:
        print("✅ CUDA kernels are available")
        
        try:
            import cupy as cp
            kernels = MaterialCUDAKernels()
            if kernels.compiled:
                print("✅ All kernels compiled successfully")
                print("  - Strain tensor kernel: Ready")
                print("  - Coordination number kernel: Ready")
                print("  - Damage score kernel: Ready")
            else:
                print("⚠️ Kernel compilation failed")
        except Exception as e:
            print(f"❌ Runtime error: {e}")
    else:
        print("❌ CUDA kernels not available")
        print("   Install CuPy to enable GPU acceleration")

# ========================================
# Quick Start Function
# ========================================
def quick_analysis(trajectory_path, atom_types_path, material_type='SUJ2', **kwargs):
    """
    クイック材料解析
    
    最小限のパラメータで解析を実行
    
    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイルパス
    atom_types_path : str
        原子タイプファイルパス
    material_type : str
        材料タイプ
    **kwargs
        追加パラメータ
        - use_cuda_kernels : bool (default: True)
        - use_topological : bool (default: True)
    
    Returns
    -------
    dict
        解析結果
    """
    from pathlib import Path
    
    # メタデータ自動生成
    metadata = {
        'system_name': f'{material_type}_quick_analysis',
        'material_type': material_type,
        'temperature': kwargs.get('temperature', 300.0),
        'strain_rate': kwargs.get('strain_rate', 1e-3),
        'loading_type': kwargs.get('loading_type', 'tensile'),
        'use_cuda_kernels': kwargs.get('use_cuda_kernels', True),
        'use_topological': kwargs.get('use_topological', True)
    }
    
    # 一時メタデータファイル作成
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metadata, f)
        metadata_path = f.name
    
    try:
        # パイプライン実行
        results = run_material_analysis_pipeline(
            trajectory_path=trajectory_path,
            metadata_path=metadata_path,
            atom_types_path=atom_types_path,
            material_type=material_type,
            output_dir=kwargs.get('output_dir', f'./{material_type}_results'),
            verbose=kwargs.get('verbose', False)
        )
        return results
    finally:
        # 一時ファイル削除
        Path(metadata_path).unlink(missing_ok=True)

# ========================================
# Export List
# ========================================
__all__ = [
    # Main pipeline
    'run_material_analysis_pipeline',
    'quick_analysis',
    
    # Detectors and analyzers
    'MaterialLambda3DetectorGPU',
    'MaterialTwoStageAnalyzerGPU',
    'MaterialImpactAnalyzer',
    
    # Results
    'MaterialLambda3Result',
    'MaterialTwoStageResult',
    'MaterialImpactResult',
    
    # Configurations
    'MaterialConfig',
    'ClusterAnalysisConfig',
    
    # Data structures
    'ClusterEvent',
    'ClusterLevelAnalysis',
    'AtomicDefectTrace',
    'DefectOrigin',
    'MaterialDefectNetwork',
    
    # CUDA kernels (v2.0)
    'MaterialCUDAKernels',
    'HAS_CUDA_KERNELS',
    
    # Functions
    'detect_material_events',
    'perform_material_two_stage_analysis_gpu',
    'run_material_impact_analysis',
    'generate_material_report_from_results',
    'get_material_parameters',
    'create_spatial_clusters',
    'prepare_vtk_export_data',
    'check_cuda_status',
    
    # Database and info
    'MATERIAL_DATABASE',
    'get_package_info',
    'list_available_materials',
    
    # Version
    '__version__'
]

# ========================================
# Package Initialization Message
# ========================================
if __name__ != '__main__':
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Lambda³ GPU Material Analysis Package v{__version__} loaded")
    logger.info(f"Available materials: {', '.join(MATERIAL_DATABASE.keys())}")
    
    if HAS_CUDA_KERNELS:
        logger.info("🚀 CUDA kernels enabled for acceleration")
    else:
        logger.info("⚠️ CUDA kernels not available - using standard GPU/CPU computation")
    
    # v2.0 新機能のアナウンス
    logger.info("✨ New in v2.0: Topological charge analysis & Structural coherence detection")
