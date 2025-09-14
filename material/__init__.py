"""
Lambda³ GPU Material Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用Lambda³構造計算のGPU実装！
金属・セラミックス・ポリマーの疲労・破壊を高速解析！💎

Components:
    - ClusterStructuresGPU: クラスター構造計算（歪み・配位数）
    - ClusterNetworkGPU: ネットワーク解析（転位・亀裂伝播）
    - MaterialCausalityAnalyzerGPU: 因果関係解析（歪み伝播）
    - MaterialConfidenceAnalyzerGPU: 統計的信頼性（ワイブル統計）

by 環ちゃん - Material Edition v1.0
"""

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

# ===============================
# Module Information
# ===============================

__version__ = '1.0.0'
__author__ = '環ちゃん'

__all__ = [
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
    
    # ===== Material Properties =====
    'SUJ2_PROPERTIES',
    'AL7075_PROPERTIES',
    'TI6AL4V_PROPERTIES',
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
    
    # 構造計算
    structures = ClusterStructuresGPU(**SUJ2_PROPERTIES)
    result = structures.compute_cluster_structures(
        trajectory, start_frame, end_frame,
        cluster_atoms, atom_types, window_size,
        cutoff=3.0  # BCC/FCC用
    )
    
    # ネットワーク解析
    network = ClusterNetworkGPU(**SUJ2_PROPERTIES)
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
        'max_strain': result.get_summary_stats()['max_strain'],
    }

def detect_fatigue_damage(trajectory, cluster_atoms, atom_types,
                         material='SUJ2', n_cycles=1e6):
    """
    疲労損傷検出
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ
    cluster_atoms : Dict[int, List[int]]
        クラスター定義
    atom_types : np.ndarray
        原子タイプ
    material : str
        材料名 ('SUJ2', 'AL7075', 'TI6AL4V')
    n_cycles : float
        サイクル数
    
    Returns
    -------
    dict
        疲労損傷解析結果
    """
    # 材料定数選択
    if material == 'SUJ2':
        props = SUJ2_PROPERTIES
    elif material == 'AL7075':
        props = AL7075_PROPERTIES
    elif material == 'TI6AL4V':
        props = TI6AL4V_PROPERTIES
    else:
        props = SUJ2_PROPERTIES
        logger.warning(f"Unknown material {material}, using SUJ2 properties")
    
    # 構造計算
    structures = ClusterStructuresGPU(**props)
    result = structures.compute_cluster_structures(
        trajectory, 0, len(trajectory)-1,
        cluster_atoms, atom_types
    )
    
    # von Mises応力推定
    von_mises_stress = props['elastic_modulus'] * np.mean(
        np.abs(result.local_strain), axis=(2, 3)
    )
    
    # 疲労損傷計算（Palmgren-Miner則）
    damage = np.zeros(result.n_clusters)
    for cluster_id in range(result.n_clusters):
        stress = von_mises_stress[:, cluster_id]
        
        # S-N曲線
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
    }

# ===============================
# Module Test
# ===============================

def test_module():
    """モジュールテスト"""
    import numpy as np
    
    logger.info("Testing Lambda³ GPU Material module...")
    
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
        logger.info(f"✓ Network analysis: {net_result.n_strain_links} strain links")
    except Exception as e:
        logger.error(f"✗ Network analysis failed: {e}")
        return False
    
    logger.info("All tests passed! 💎")
    return True

if __name__ == '__main__':
    test_module()
