"""
Lambda³ GPU Material Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用Lambda³構造計算のGPU実装！
金属・セラミックス・ポリマーの疲労・破壊を高速解析！💎

Components:
    - MaterialMDFeaturesGPU: 材料用MD特徴抽出（欠陥領域自動検出）
    - ClusterStructuresGPU: クラスター構造計算（歪み・配位数）
    - ClusterNetworkGPU: ネットワーク解析（転位・亀裂伝播）
    - MaterialCausalityAnalyzerGPU: 因果関係解析（歪み伝播）
    - MaterialConfidenceAnalyzerGPU: 統計的信頼性（ワイブル統計）

by 環ちゃん - Material Edition v2.0
"""

# ===============================
# Material MD Features (NEW!)
# ===============================
from .material_md_features_gpu import (
    # Classes
    MaterialMDFeaturesGPU,
    
    # Config
    MaterialMDFeatureConfig,
    
    # Functions
    extract_material_md_features,
    get_defect_region_indices,
)

# ===============================
# Cluster Structures
# ===============================
from .cluster_structures_gpu import (
    # Classes
    ClusterStructuresGPU,
    ClusterStructureBatchProcessor,
    
    # Data Classes
    ClusterStructureResult,
    
    # Main Functions
    compute_cluster_structures_gpu,
    compute_cluster_lambda_f_gpu,
    compute_cluster_rho_t_gpu,
    compute_cluster_coupling_gpu,
    
    # Utility Functions
    create_nearest_neighbor_clusters,
    create_grid_based_clusters,
)

# ===============================
# Cluster Network
# ===============================
from .cluster_network_gpu import (
    # Classes
    ClusterNetworkGPU,
    
    # Data Classes
    MaterialNetworkLink,
    ClusterNetworkResult,
    
    # Main Functions
    analyze_cluster_network_gpu,
)

# ===============================
# Causality Analysis
# ===============================
from .cluster_causality_analysis_gpu import (
    # Classes
    MaterialCausalityAnalyzerGPU,
    
    # Data Classes
    MaterialCausalityResult,
    
    # Main Functions
    calculate_dislocation_causality_gpu,
    calculate_strain_causality_gpu,
    calculate_crack_causality_gpu,
    detect_material_causal_pairs_gpu,
)

# ===============================
# Confidence Analysis
# ===============================
from .cluster_confidence_analysis_gpu import (
    # Classes
    MaterialConfidenceAnalyzerGPU,
    
    # Data Classes
    MaterialConfidenceResult,
    
    # Main Functions
    analyze_material_reliability_gpu,
    bootstrap_strain_confidence_gpu,
    estimate_weibull_parameters_gpu,
    compute_material_reliability_index,
)

# ===============================
# Material Analytics
# ===============================
from .material_analytics_gpu import (
    # Classes
    MaterialAnalyticsGPU,
    
    # Data Classes
    CrystalDefectResult,
    MaterialState,
    
    # Functions
    compute_crystal_defect_charge,
    compute_structural_coherence,
)

# ===============================
# Material Features
# ===============================
from .material_features_gpu import (
    # Classes
    MaterialFeaturesGPU,
    
    # Functions
    extract_material_features,
    compute_coordination_numbers_gpu,
    compute_local_strain_gpu,
)

# ===============================
# Material Lambda3 Detector
# ===============================
from .material_lambda3_detector_gpu import (
    # Classes
    MaterialLambda3DetectorGPU,
    
    # Config
    MaterialConfig,
    
    # Result
    MaterialLambda3Result,
)

# ===============================
# Material Properties (Constants)
# ===============================

# SUJ2 (高炭素クロム軸受鋼) の材料定数
SUJ2_PROPERTIES = {
    'elastic_modulus': 210.0,        # GPa
    'poisson_ratio': 0.3,
    'yield_strength': 1.5,            # GPa
    'ultimate_strength': 2.0,         # GPa
    'fatigue_strength': 0.7,          # GPa
    'fracture_toughness': 30.0,       # MPa√m
    'density': 7.85,                  # g/cm³
}

# アルミニウム合金 (A7075-T6)
AL7075_PROPERTIES = {
    'elastic_modulus': 71.7,          # GPa
    'poisson_ratio': 0.33,
    'yield_strength': 0.503,          # GPa
    'ultimate_strength': 0.572,       # GPa
    'fatigue_strength': 0.159,        # GPa
    'fracture_toughness': 23.0,       # MPa√m
    'density': 2.81,                  # g/cm³
}

# チタン合金 (Ti-6Al-4V)
TI6AL4V_PROPERTIES = {
    'elastic_modulus': 113.8,         # GPa
    'poisson_ratio': 0.342,
    'yield_strength': 0.88,           # GPa
    'ultimate_strength': 0.95,        # GPa
    'fatigue_strength': 0.51,         # GPa
    'fracture_toughness': 75.0,       # MPa√m
    'density': 4.43,                  # g/cm³
}

# ステンレス鋼 (SUS316L)
SS316L_PROPERTIES = {
    'elastic_modulus': 193.0,         # GPa
    'poisson_ratio': 0.3,
    'yield_strength': 0.205,          # GPa
    'ultimate_strength': 0.515,       # GPa
    'fatigue_strength': 0.175,        # GPa
    'fracture_toughness': 112.0,      # MPa√m
    'density': 8.0,                   # g/cm³
}

# ===============================
# Module Information
# ===============================

__version__ = '2.0.0'
__author__ = '環ちゃん'

__all__ = [
    # ===== Material MD Features (NEW!) =====
    'MaterialMDFeaturesGPU',
    'MaterialMDFeatureConfig',
    'extract_material_md_features',
    'get_defect_region_indices',
    
    # ===== Cluster Structures =====
    'ClusterStructuresGPU',
    'ClusterStructureBatchProcessor',
    'ClusterStructureResult',
    'compute_cluster_structures_gpu',
    'compute_cluster_lambda_f_gpu',
    'compute_cluster_rho_t_gpu',
    'compute_cluster_coupling_gpu',
    'create_nearest_neighbor_clusters',
    'create_grid_based_clusters',
    
    # ===== Cluster Network =====
    'ClusterNetworkGPU',
    'MaterialNetworkLink',
    'ClusterNetworkResult',
    'analyze_cluster_network_gpu',
    
    # ===== Causality Analysis =====
    'MaterialCausalityAnalyzerGPU',
    'MaterialCausalityResult',
    'calculate_dislocation_causality_gpu',
    'calculate_strain_causality_gpu',
    'calculate_crack_causality_gpu',
    'detect_material_causal_pairs_gpu',
    
    # ===== Confidence Analysis =====
    'MaterialConfidenceAnalyzerGPU',
    'MaterialConfidenceResult',
    'analyze_material_reliability_gpu',
    'bootstrap_strain_confidence_gpu',
    'estimate_weibull_parameters_gpu',
    'compute_material_reliability_index',
    
    # ===== Material Analytics =====
    'MaterialAnalyticsGPU',
    'CrystalDefectResult',
    'MaterialState',
    'compute_crystal_defect_charge',
    'compute_structural_coherence',
    
    # ===== Material Features =====
    'MaterialFeaturesGPU',
    'extract_material_features',
    'compute_coordination_numbers_gpu',
    'compute_local_strain_gpu',
    
    # ===== Material Lambda3 Detector =====
    'MaterialLambda3DetectorGPU',
    'MaterialConfig',
    'MaterialLambda3Result',
    
    # ===== Material Properties =====
    'SUJ2_PROPERTIES',
    'AL7075_PROPERTIES',
    'TI6AL4V_PROPERTIES',
    'SS316L_PROPERTIES',
]

# ===============================
# Initialization
# ===============================

import logging
logger = logging.getLogger('lambda3_gpu.material')

# GPU availability check
try:
    import cupy as cp
    HAS_GPU = True
    GPU_INFO = {
        'available': True,
        'device_count': cp.cuda.runtime.getDeviceCount(),
        'current_device': cp.cuda.runtime.getDevice(),
    }
    logger.info(f"Lambda³ GPU Material module initialized with {GPU_INFO['device_count']} GPU(s)")
except ImportError:
    HAS_GPU = False
    GPU_INFO = {
        'available': False,
        'device_count': 0,
        'current_device': None,
    }
    logger.warning("CuPy not available - Material module will run in CPU mode")

# ===============================
# Quick Start Functions
# ===============================

def quick_analyze_material(trajectory, atom_types, material='SUJ2', 
                          cluster_definition_path=None,
                          verbose=False):
    """
    材料の高速解析（欠陥領域自動検出版）
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ (n_frames, n_atoms, 3)
    atom_types : np.ndarray
        原子タイプ配列
    material : str
        材料名 ('SUJ2', 'AL7075', 'TI6AL4V', 'SS316L')
    cluster_definition_path : str, optional
        クラスター定義ファイル（Noneの場合は自動検出）
    verbose : bool
        詳細出力
    
    Returns
    -------
    dict
        解析結果の辞書
    """
    import numpy as np
    
    # 材料定数選択
    material_props = {
        'SUJ2': SUJ2_PROPERTIES,
        'AL7075': AL7075_PROPERTIES,
        'TI6AL4V': TI6AL4V_PROPERTIES,
        'SS316L': SS316L_PROPERTIES,
    }
    props = material_props.get(material, SUJ2_PROPERTIES)
    
    # 設定
    config = MaterialConfig()
    config.material_type = material
    config.use_material_analytics = True
    config.adaptive_window = True
    
    # 検出器初期化
    detector = MaterialLambda3DetectorGPU(config)
    
    # 解析実行（欠陥領域自動検出）
    result = detector.analyze(
        trajectory=trajectory,
        backbone_indices=None,  # 自動検出！
        atom_types=atom_types,
        cluster_definition_path=cluster_definition_path
    )
    
    # 結果整理
    analysis_summary = {
        'material': material,
        'n_frames': result.n_frames,
        'n_atoms': result.n_atoms,
        'defect_density': result.md_features.get('defect_density', 0),
        'n_defect_atoms': result.md_features.get('n_defect_atoms', 0),
        'computation_time': result.computation_time,
        'gpu_used': result.gpu_info.get('device_name', 'CPU'),
    }
    
    # イベント情報
    if hasattr(result, 'material_events'):
        analysis_summary['n_events'] = len(result.material_events)
        analysis_summary['event_types'] = list(set(e[2] for e in result.material_events if len(e) > 2))
    
    # 破壊予測
    if hasattr(result, 'failure_prediction') and result.failure_prediction:
        analysis_summary['failure_probability'] = result.failure_prediction.get('failure_probability', 0)
    
    if verbose:
        logger.info(f"\n=== Quick Analysis Results ===")
        for key, value in analysis_summary.items():
            logger.info(f"  {key}: {value}")
    
    return {
        'summary': analysis_summary,
        'full_result': result,
        'material_properties': props,
    }

def detect_fatigue_damage_gpu(trajectory, atom_types, material='SUJ2',
                             n_cycles=1e6, cluster_definition_path=None):
    """
    疲労損傷検出（GPU高速版）
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ
    atom_types : np.ndarray
        原子タイプ
    material : str
        材料名
    n_cycles : float
        サイクル数
    cluster_definition_path : str, optional
        クラスター定義ファイル
    
    Returns
    -------
    dict
        疲労損傷解析結果
    """
    # 材料解析実行
    analysis = quick_analyze_material(
        trajectory, atom_types, material,
        cluster_definition_path, verbose=False
    )
    
    result = analysis['full_result']
    props = analysis['material_properties']
    
    # 欠陥領域の歪み
    if 'local_strain' in result.md_features:
        strain = result.md_features['local_strain']
        
        # von Mises応力推定
        von_mises_stress = props['elastic_modulus'] * np.mean(np.abs(strain))
        
        # 疲労損傷計算（Palmgren-Miner則）
        damage = 0
        if von_mises_stress > props['fatigue_strength']:
            # Basquin則
            N_f = 1e6 * (von_mises_stress / props['ultimate_strength']) ** (-3)
            damage = n_cycles / N_f
    else:
        damage = 0
        von_mises_stress = 0
    
    return {
        'damage': float(damage),
        'von_mises_stress': float(von_mises_stress),
        'failure_probability': float(damage > 1.0),
        'remaining_life': float(max(0, 1.0 - damage)),
        'defect_density': analysis['summary']['defect_density'],
    }

# ===============================
# Module Test
# ===============================

def test_module():
    """モジュールテスト（欠陥自動検出版）"""
    import numpy as np
    
    logger.info("Testing Lambda³ GPU Material module v2.0...")
    
    # ダミーデータ
    n_frames = 100
    n_atoms = 1000
    trajectory = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    
    # 原子タイプ
    atom_types = np.array(['Fe'] * 900 + ['C'] * 50 + ['Cr'] * 50)
    
    # MD特徴抽出テスト（欠陥自動検出）
    try:
        from .material_md_features_gpu import MaterialMDFeaturesGPU
        
        md_config = MaterialMDFeatureConfig(
            crystal_structure='BCC',
            auto_detect_defects=True
        )
        extractor = MaterialMDFeaturesGPU(config=md_config)
        
        features = extractor.extract_md_features(
            trajectory,
            backbone_indices=None,  # 自動検出
            atom_types=atom_types
        )
        
        n_defects = features.get('n_defect_atoms', 0)
        logger.info(f"✓ MD features: {n_defects} defect atoms detected")
    except Exception as e:
        logger.error(f"✗ MD feature extraction failed: {e}")
        return False
    
    # 高速解析テスト
    try:
        result = quick_analyze_material(
            trajectory, atom_types, 'SUJ2',
            verbose=True
        )
        logger.info(f"✓ Quick analysis: {result['summary']['computation_time']:.2f}s")
    except Exception as e:
        logger.error(f"✗ Quick analysis failed: {e}")
        return False
    
    # 疲労損傷テスト
    try:
        damage_result = detect_fatigue_damage_gpu(
            trajectory, atom_types, 'SUJ2', n_cycles=1e6
        )
        logger.info(f"✓ Fatigue damage: {damage_result['damage']:.3f}")
    except Exception as e:
        logger.error(f"✗ Fatigue damage detection failed: {e}")
        return False
    
    logger.info("All tests passed! 💎")
    return True

if __name__ == '__main__':
    test_module()
