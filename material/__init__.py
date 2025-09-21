#!/usr/bin/env python3
"""
Lambda³ GPU Material Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用Lambda³構造計算のGPU実装！
金属・セラミックスの疲労・破壊を高速解析！💎

Components:
    - ClusterStructuresGPU: クラスター構造計算（歪み・配位数）
    - ClusterNetworkGPU: ネットワーク解析（転位・亀裂伝播）
    - MaterialCausalityAnalyzerGPU: 因果関係解析（歪み伝播）
    - MaterialConfidenceAnalyzerGPU: 統計的信頼性（ワイブル統計）
    - MaterialMDFeaturesGPU: MD特徴抽出（欠陥領域フォーカス）
    - MaterialFailurePhysicsGPU: 物理原理による破損予測（RMSF発散・相転移）

by 環ちゃん - Material Edition v1.1.0
"""

# ===============================
# Version and Metadata
# ===============================

__version__ = '1.1.0'  # Updated for physics integration
__author__ = '環ちゃん'
__email__ = 'tamaki@lambda3.ai'

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
# Material MD Features
# ===============================

from .material_md_features_gpu import (
    # Classes
    MaterialMDFeaturesGPU,
    MaterialMDFeatureConfig,
    
    # Main Functions
    extract_material_md_features,
    get_defect_region_indices,
)

# ===============================
# Material Failure Physics
# ===============================

from .material_failure_physics_gpu import (
    # Classes
    MaterialFailurePhysicsGPU,
    
    # Data Classes
    FailurePhysicsResult,
    RMSFAnalysisResult,
    EnergyBalanceResult,
    DamageNucleusResult,
    FatigueCycleResult,
    
    # Main Functions
    detect_failure_precursor,
    predict_fatigue_life  
)

# ===============================
# Material Properties Database
# ===============================
from .material_database import (
    MATERIAL_DATABASE,
    get_material_parameters,
    K_B,
    AMU_TO_KG,
    J_TO_EV
)

# 互換性のため、個別の材料プロパティも公開
SUJ2_PROPERTIES = MATERIAL_DATABASE['SUJ2']
AL7075_PROPERTIES = MATERIAL_DATABASE['AL7075']
TI6AL4V_PROPERTIES = MATERIAL_DATABASE['Ti6Al4V']
SS316L_PROPERTIES = MATERIAL_DATABASE['SS316L']  # 新規追加！

# ===============================
# Quick Analysis Functions
# ===============================

def quick_analyze_suj2(trajectory, cluster_atoms, atom_types, 
                       start_frame=0, end_frame=-1, 
                       window_size=50):
    """
    SUJ2鋼の高速解析
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ (n_frames, n_atoms, 3)
    cluster_atoms : Dict[int, List[int]]
        クラスター定義
    atom_types : np.ndarray
        原子タイプ ('Fe', 'C', 'Cr', ...)
    start_frame, end_frame : int
        解析範囲
    window_size : int
        時間窓サイズ
    
    Returns
    -------
    dict
        解析結果の辞書
    """
    if end_frame == -1:
        end_frame = trajectory.shape[0] - 1

    material_props = MATERIAL_DATABASE['SUJ2']
    # 構造計算
    structures = ClusterStructuresGPU()
    result = structures.compute_cluster_structures(
        trajectory, start_frame, end_frame,
        cluster_atoms, atom_types, window_size
    )
    
    # ネットワーク解析
    network = ClusterNetworkGPU()
    net_result = network.analyze_network(
        {i: result.cluster_lambda_f_mag[:, i] for i in range(result.n_clusters)},
        result.cluster_coupling,
        result.cluster_centers,
        result.coordination_numbers,
        result.local_strain
    )
    
    return {
        'structures': result,
        'network': net_result,
        'critical_clusters': net_result.critical_clusters,
        'material_type': 'SUJ2'
    }

def quick_fatigue_analysis(trajectory, cluster_atoms, atom_types,
                          stress_history, material_type='SUJ2',
                          n_cycles=1000):
    """
    疲労解析の高速実行
    
    Parameters
    ----------
    trajectory : np.ndarray
        トラジェクトリ
    cluster_atoms : Dict[int, List[int]]
        クラスター定義
    atom_types : np.ndarray
        原子タイプ
    stress_history : np.ndarray
        応力履歴 (n_frames, n_clusters) or (n_frames,)
    material_type : str
        材料タイプ
    n_cycles : int
        サイクル数
        
    Returns
    -------
    dict
        疲労解析結果
    """
    import numpy as np
    
    # 材料プロパティ取得
    props = globals().get(f'{material_type}_PROPERTIES', SUJ2_PROPERTIES)
    
    # 構造解析
    structures = ClusterStructuresGPU(**props)
    result = structures.compute_cluster_structures(
        trajectory, 0, -1, cluster_atoms, atom_types
    )
    
    # 損傷累積（Miner則）
    n_clusters = result.n_clusters
    damage = np.zeros(n_clusters)
    
    for cluster_id in range(n_clusters):
        # 各クラスターの応力
        if stress_history.ndim == 1:
            stress = stress_history
        else:
            stress = stress_history[:, cluster_id]
        
        # S-N曲線による寿命予測
        for s in stress:
            if s > props['fatigue_strength']:
                # Basquin則
                N_f = 1e6 * (s / props['ultimate_strength']) ** (-3)
                damage[cluster_id] += n_cycles / N_f
    
    # 臨界クラスター検出
    critical_clusters = np.where(damage > 0.5)[0].tolist()
    
    return {
        'damage': damage,
        'critical_clusters': critical_clusters,
        'max_damage': float(np.max(damage)),
        'failure_probability': float(np.mean(damage > 1.0)),
        'material_type': material_type
    }

def quick_physics_analysis(trajectory, stress_history=None,
                         material_type='SUJ2', temperature=300.0):
    """
    物理ベース破損解析の高速実行
    
    水の沸点研究から生まれた理論による破損予測！
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ (n_frames, n_atoms, 3)
    stress_history : np.ndarray, optional
        応力履歴
    material_type : str
        材料タイプ
    temperature : float
        基礎温度 (K)
        
    Returns
    -------
    dict
        物理解析結果
    """
    # 物理解析器初期化
    physics = MaterialFailurePhysicsGPU(material_type=material_type)
    
    # 解析実行
    result = physics.analyze_failure_physics(
        trajectory=trajectory,
        stress_history=stress_history,
        temperature=temperature
    )
    
    return {
        'mechanism': result.failure_mechanism,
        'time_to_failure': result.time_to_failure,
        'critical_atoms': result.failure_location,
        'lindemann_ratio': result.rmsf_analysis.lindemann_ratio,
        'phase_state': result.energy_balance.phase_state,
        'confidence': result.confidence,
        'material_type': material_type
    }

# ===============================
# Module Export
# ===============================

__all__ = [
    # ===== Version Info =====
    '__version__',
    '__author__',
    
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
    
    # ===== Material MD Features =====
    'MaterialMDFeaturesGPU',
    'MaterialMDFeatureConfig',
    'extract_material_md_features',
    'get_defect_region_indices',
    
    # ===== Material Failure Physics =====
    'MaterialFailurePhysicsGPU',
    'FailurePhysicsResult',
    'RMSFAnalysisResult',
    'EnergyBalanceResult',
    'DamageNucleusResult',
    'FatigueCycleResult',
    'detect_failure_precursor',
    'predict_fatigue_life',
    
    # ===== Material Database =====
    'MATERIAL_DATABASE',
    'get_material_parameters',
    'K_B',
    'AMU_TO_KG', 
    'J_TO_EV',
    
    # ===== Material Properties =====
    'SUJ2_PROPERTIES',
    'AL7075_PROPERTIES',
    'TI6AL4V_PROPERTIES',
    'SS316L_PROPERTIES',
    
    # ===== Quick Analysis =====
    'quick_analyze_suj2',
    'quick_fatigue_analysis',
    'quick_physics_analysis',  # NEW!
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
    logger.info(f"Lambda³ GPU Material module v{__version__} initialized with {GPU_INFO['device_count']} GPU(s)")
    logger.info("Physics-based failure prediction module loaded 💫")
except ImportError:
    HAS_GPU = False
    GPU_INFO = {
        'available': False,
        'device_count': 0,
        'current_device': None,
    }
    logger.warning("CuPy not available - Material module will run in CPU mode")

# ===============================
# Module Test
# ===============================

def test_module():
    """モジュールテスト（物理解析を含む）"""
    import numpy as np
    
    logger.info(f"Testing Lambda³ GPU Material module v{__version__}...")
    
    # ダミーデータ
    n_frames = 100
    n_atoms = 1000
    trajectory = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    
    # クラスター定義（最近接）
    positions = trajectory[0]
    clusters = create_nearest_neighbor_clusters(positions, cutoff=3.0)
    
    # 原子タイプ
    atom_types = np.array(['Fe'] * 900 + ['C'] * 50 + ['Cr'] * 50)
    
    # 構造計算テスト
    try:
        result = compute_cluster_structures_gpu(
            trajectory, 0, 10, clusters, atom_types
        )
        logger.info(f"✓ Structure computation: {result.n_clusters} clusters")
    except Exception as e:
        logger.error(f"✗ Structure computation failed: {e}")
        return False
    
    # ネットワーク解析テスト
    try:
        anomaly_scores = {i: np.random.randn(n_frames) for i in range(len(clusters))}
        net_result = analyze_cluster_network_gpu(
            anomaly_scores,
            result.cluster_coupling,
            result.cluster_centers,
            result.coordination_numbers,
            result.local_strain
        )
        logger.info(f"✓ Network analysis: {net_result.n_strain_links} strain links detected")
    except Exception as e:
        logger.error(f"✗ Network analysis failed: {e}")
        return False
    
    # 物理解析テスト（NEW!）
    try:
        physics_result = detect_failure_precursor(
            trajectory[:50],  # Short trajectory for test
            material_type='SUJ2',
            temperature=300.0
        )
        logger.info(f"✓ Physics analysis: {physics_result['mechanism']} mechanism detected")
        logger.info(f"  Lindemann ratio: {physics_result['lindemann_ratio']:.3f}")
    except Exception as e:
        logger.error(f"✗ Physics analysis failed: {e}")
        return False
    
    logger.info("All tests passed! 💎✨")
    return True

if __name__ == '__main__':
    test_module()
