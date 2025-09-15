#!/usr/bin/env python3
"""
Lambda³ GPU Material Analysis Package
======================================

材料解析用Lambda³ GPUパッケージ
転位・亀裂・相変態の階層的解析

トポロジカル欠陥解析とCUDAカーネル最適化対応
欠陥領域自動検出機能追加（v3.0）

Version: 3.0.0
Authors: 環ちゃん
"""

__version__ = '3.0.0'
__author__ = '環ちゃん'
__email__ = 'tamaki@lambda3.ai'

# ========================================
# Main Pipeline
# ========================================
from .material_full_analysis import (
    run_material_analysis_pipeline,
    get_material_parameters,
    create_spatial_clusters
)

# ========================================
# MD Features for Materials (v3.0 新機能)
# ========================================
try:
    from .material_md_features_gpu import (
        MaterialMDFeaturesGPU,
        MaterialMDFeatureConfig,
        extract_material_md_features,
        get_defect_region_indices
    )
except ImportError:
    # material/サブディレクトリにある場合
    from ..material.material_md_features_gpu import (
        MaterialMDFeaturesGPU,
        MaterialMDFeatureConfig,
        extract_material_md_features,
        get_defect_region_indices
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
# CUDA Kernels (v2.0+)
# ========================================
try:
    from .cuda_kernels import (
        MaterialCUDAKernels,
        STRAIN_TENSOR_KERNEL_CODE,
        COORDINATION_NUMBER_KERNEL_CODE,
        DAMAGE_SCORE_KERNEL_CODE,
        compile_material_kernels,
        check_cuda_status
    )
    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False
    MaterialCUDAKernels = None
    
    # ダミー関数
    def check_cuda_status():
        return {'available': False, 'reason': 'CUDA kernels module not found'}

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
# Material Analytics (optional)
# ========================================
try:
    from .material_analytics_gpu import (
        MaterialAnalyticsGPU,
        CrystalDefectResult,
        MaterialState,
        compute_crystal_defect_charge,
        compute_structural_coherence
    )
except ImportError:
    MaterialAnalyticsGPU = None
    CrystalDefectResult = None
    MaterialState = None

# ========================================
# Material Features (optional)
# ========================================
try:
    from .material_features_gpu import (
        MaterialFeaturesGPU,
        extract_material_features,
        compute_coordination_numbers_gpu,
        compute_local_strain_gpu
    )
except ImportError:
    MaterialFeaturesGPU = None
    
    # ダミー関数
    def extract_material_features(*args, **kwargs):
        raise NotImplementedError("Material features module not available")

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
    'TI6AL4V': {
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
# Convenience Functions
# ========================================

def quick_analysis(trajectory, atom_types, material_type='SUJ2', **kwargs):
    """
    クイック材料解析
    
    Parameters
    ----------
    trajectory : np.ndarray
        トラジェクトリ (n_frames, n_atoms, 3)
    atom_types : np.ndarray
        原子タイプ配列
    material_type : str
        材料タイプ
        
    Returns
    -------
    dict
        解析結果
    """
    import tempfile
    import numpy as np
    from pathlib import Path
    
    # 一時ファイル作成
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        metadata_path = f.name
        
    metadata = {
        'material': material_type,
        'properties': MATERIAL_DATABASE.get(material_type, {}),
        'n_frames': trajectory.shape[0],
        'n_atoms': trajectory.shape[1]
    }
    
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f)
    
    try:
        # パイプライン実行
        results = run_material_analysis_pipeline(
            trajectory_path=trajectory,  # numpy配列を直接渡す
            metadata_path=metadata_path,
            atom_types_path=atom_types,  
            material_type=material_type,
            enable_two_stage=kwargs.get('enable_two_stage', True),
            enable_impact=kwargs.get('enable_impact', False),
            enable_visualization=kwargs.get('enable_visualization', False),
            output_dir=kwargs.get('output_dir', './quick_results')
        )
        
        # v3.0: 欠陥検出情報を追加
        if hasattr(results.get('macro_result'), 'md_features'):
            md_features = results['macro_result'].md_features
            if 'defect_indices' in md_features:
                n_defects = len(md_features['defect_indices'])
                density = n_defects / trajectory.shape[1]
                print(f"\n🎯 Defect Detection Results:")
                print(f"   Defect atoms: {n_defects}")
                print(f"   Defect density: {density:.1%}")
        
        return results
        
    finally:
        # 一時ファイル削除
        Path(metadata_path).unlink(missing_ok=True)

def analyze_with_defect_detection(trajectory, atom_types, material_type='SUJ2', 
                                 crystal_structure=None, **kwargs):
    """
    欠陥自動検出付き解析（v3.0新機能）
    
    Parameters
    ----------
    trajectory : np.ndarray
        トラジェクトリ (n_frames, n_atoms, 3)
    atom_types : np.ndarray
        原子タイプ配列
    material_type : str
        材料タイプ
    crystal_structure : str, optional
        結晶構造（None の場合は材料タイプから推定）
    
    Returns
    -------
    dict
        解析結果
    """
    import numpy as np
    
    # 結晶構造の推定
    if crystal_structure is None:
        crystal_structure = MATERIAL_DATABASE.get(material_type, {}).get('crystal', 'BCC')
    
    # MD特徴抽出設定
    md_config = MaterialMDFeatureConfig(
        crystal_structure=crystal_structure,
        auto_detect_defects=True,
        defect_expansion_radius=kwargs.get('expansion_radius', 5.0),
        coordination_cutoff=kwargs.get('cutoff', 3.5)
    )
    
    # 検出器設定
    config = MaterialConfig()
    config.material_type = material_type
    config.use_material_analytics = True
    
    # 検出器初期化
    detector = MaterialLambda3DetectorGPU(config)
    
    # 解析実行（欠陥自動検出）
    result = detector.analyze(
        trajectory=trajectory,
        backbone_indices=None,  # 自動検出！
        atom_types=atom_types
    )
    
    return {
        'result': result,
        'defect_info': {
            'n_defect_atoms': result.md_features.get('n_defect_atoms', 0),
            'defect_density': result.md_features.get('defect_density', 0),
            'defect_indices': result.md_features.get('defect_indices', [])
        },
        'material_type': material_type,
        'crystal_structure': crystal_structure
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
            'Auto defect detection (v3.0)',
            'Multi-scale analysis',
            'CUDA acceleration',
            'Material database'
        ],
        'cuda_kernels': cuda_status,
        'materials': list(MATERIAL_DATABASE.keys())
    }

def list_available_materials():
    """利用可能な材料リスト"""
    materials = []
    for key, props in MATERIAL_DATABASE.items():
        materials.append({
            'code': key,
            'name': props['name'],
            'crystal': props['crystal'],
            'yield_strength': props['yield'],
            'fracture_toughness': props['K_IC']
        })
    return materials

# ========================================
# Export List (v3.0更新)
# ========================================

__all__ = [
    # Main pipeline
    'run_material_analysis_pipeline',
    'quick_analysis',
    'analyze_with_defect_detection',  # v3.0新機能
    
    # MD Features (v3.0新機能)
    'MaterialMDFeaturesGPU',
    'MaterialMDFeatureConfig',
    'extract_material_md_features',
    'get_defect_region_indices',
    
    # Detectors and analyzers
    'MaterialLambda3DetectorGPU',
    'MaterialTwoStageAnalyzerGPU',
    'MaterialImpactAnalyzer',
    
    # Analytics and Features
    'MaterialAnalyticsGPU',
    'MaterialFeaturesGPU',
    
    # Results
    'MaterialLambda3Result',
    'MaterialTwoStageResult',
    'MaterialImpactResult',
    'CrystalDefectResult',
    'MaterialState',
    
    # Configurations
    'MaterialConfig',
    'ClusterAnalysisConfig',
    
    # Data structures
    'ClusterEvent',
    'ClusterLevelAnalysis',
    'AtomicDefectTrace',
    'DefectOrigin',
    'MaterialDefectNetwork',
    'MaterialDefectNetworkGPU',
    
    # CUDA kernels
    'MaterialCUDAKernels',
    'HAS_CUDA_KERNELS',
    'check_cuda_status',
    
    # Functions
    'detect_material_events',
    'perform_material_two_stage_analysis_gpu',
    'run_material_impact_analysis',
    'generate_material_report_from_results',
    'prepare_vtk_export_data',
    'generate_material_comparison_report',
    'get_material_parameters',
    'create_spatial_clusters',
    'compute_crystal_defect_charge',
    'compute_structural_coherence',
    'extract_material_features',
    'compute_coordination_numbers_gpu',
    'compute_local_strain_gpu',
    
    # Database and info
    'MATERIAL_DATABASE',
    'get_package_info',
    'list_available_materials',
    
    # Version
    '__version__',
    '__author__'
]

# ========================================
# Package Initialization
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
    
    # v3.0 新機能のアナウンス
    logger.info("✨ New in v3.0: Automatic defect region detection with CUDA acceleration")
    logger.info("   - MaterialMDFeaturesGPU for efficient feature extraction")
    logger.info("   - Reduced computation from 50000 to ~2000 atoms automatically")
    logger.info("   - Crystal structure aware defect detection")
