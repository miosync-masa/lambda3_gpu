#!/usr/bin/env python3
"""
Cluster Network Analysis for Materials (GPU Version) - v4.0 Design Unified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

クラスター間ネットワーク解析のGPU実装 - 材料版！
金属の転位伝播・歪み場・亀裂進展を解析💎

Version: 4.0-material
by 環ちゃん - Material Edition
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist as cp_cdist
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors
from .cluster_id_mapping import ClusterIDMapper

logger = logging.getLogger('lambda3_gpu.material.network')

# ===============================
# Data Classes (Material版)
# ===============================

@dataclass
class MaterialNetworkLink:
    """材料ネットワークリンク"""
    from_cluster: int
    to_cluster: int
    strength: float
    lag: int = 0
    distance: Optional[float] = None
    strain_correlation: Optional[float] = None  # 歪み相関
    link_type: str = 'elastic'  # 'elastic', 'plastic', 'fracture', 'dislocation'
    confidence: float = 1.0
    damage_signature: Optional[str] = None  # 'crack_initiation', 'void_growth', etc.
    element_bridge: Optional[Tuple[str, str]] = None  # 元素間ブリッジ（Fe-C等）

@dataclass
class ClusterNetworkResult:
    """クラスターネットワーク解析結果"""
    strain_network: List[MaterialNetworkLink]  # 歪み伝播ネットワーク
    dislocation_network: List[MaterialNetworkLink]  # 転位ネットワーク
    damage_network: List[MaterialNetworkLink]  # 損傷ネットワーク
    spatial_constraints: Dict[Tuple[int, int], float]
    adaptive_windows: Dict[int, int]
    network_stats: Dict[str, Any]
    critical_clusters: List[int]  # 臨界クラスター（破壊起点候補）
    
    @property
    def n_strain_links(self) -> int:
        return len(self.strain_network)
    
    @property
    def n_dislocation_links(self) -> int:
        return len(self.dislocation_network)
    
    @property
    def n_damage_links(self) -> int:
        return len(self.damage_network)

# ===============================
# CUDA Kernels (Material版)
# ===============================

STRAIN_CORRELATION_KERNEL = r'''
extern "C" __global__
void compute_strain_correlation_kernel(
    const float* __restrict__ strain_tensors,  // (n_clusters, 3, 3)
    float* __restrict__ correlations,         // (n_pairs,)
    const int* __restrict__ pair_indices,     // (n_pairs, 2)
    const int n_pairs,
    const int n_clusters
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_pairs) return;
    
    const int cluster_i = pair_indices[idx * 2 + 0];
    const int cluster_j = pair_indices[idx * 2 + 1];
    
    // 歪みテンソルの相関計算
    float correlation = 0.0f;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            const int idx_i = cluster_i * 9 + i * 3 + j;
            const int idx_j = cluster_j * 9 + i * 3 + j;
            
            correlation += strain_tensors[idx_i] * strain_tensors[idx_j];
        }
    }
    
    // Frobenius内積を正規化
    correlations[idx] = correlation / 9.0f;
}
'''

DISLOCATION_DETECTION_KERNEL = r'''
extern "C" __global__
void detect_dislocation_kernel(
    const float* __restrict__ coordination,   // (n_clusters,)
    const float* __restrict__ strain,        // (n_clusters, 3, 3)
    bool* __restrict__ is_dislocation,       // (n_clusters,)
    float* __restrict__ dislocation_strength, // (n_clusters,)
    const int n_clusters,
    const float coord_threshold,
    const float strain_threshold
) {
    const int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    // 配位数欠陥チェック
    float coord_defect = fabs(coordination[cluster_id] - 12.0f);  // FCC理想値
    
    // 歪みのトレース（体積歪み）
    float strain_trace = 0.0f;
    for (int i = 0; i < 3; i++) {
        strain_trace += strain[cluster_id * 9 + i * 3 + i];
    }
    
    // 転位判定
    if (coord_defect > coord_threshold && fabs(strain_trace) > strain_threshold) {
        is_dislocation[cluster_id] = true;
        dislocation_strength[cluster_id] = coord_defect * fabs(strain_trace);
    } else {
        is_dislocation[cluster_id] = false;
        dislocation_strength[cluster_id] = 0.0f;
    }
}
'''

# ===============================
# ClusterNetworkGPU Class
# ===============================

class ClusterNetworkGPU(GPUBackend):
    """
    材料クラスターネットワーク解析のGPU実装
    歪み伝播・転位移動・亀裂進展を追跡
    """
    
    def __init__(self,
                 max_interaction_distance: float = 10.0,  # 材料は短い
                 strain_threshold: float = 0.01,  # 歪み閾値
                 coord_defect_threshold: float = 1.0,  # 配位数欠陥閾値
                 min_damage_strength: float = 0.3,
                 max_damage_links: int = 200,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_interaction_distance = max_interaction_distance
        self.strain_threshold = strain_threshold
        self.coord_defect_threshold = coord_defect_threshold
        self.min_damage_strength = min_damage_strength
        self.max_damage_links = max_damage_links
        self.memory_manager = memory_manager or GPUMemoryManager()

        # 🆕 追加
        self.id_mapper = None
        
        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.strain_corr_kernel = cp.RawKernel(
                STRAIN_CORRELATION_KERNEL, 'compute_strain_correlation_kernel'
            )
            self.dislocation_kernel = cp.RawKernel(
                DISLOCATION_DETECTION_KERNEL, 'detect_dislocation_kernel'
            )
            logger.debug("Material network kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.strain_corr_kernel = None
            self.dislocation_kernel = None
    
    @handle_gpu_errors
    def analyze_network(self,
                       cluster_anomaly_scores: Dict[int, np.ndarray],
                       cluster_coupling: np.ndarray,
                       cluster_centers: Optional[np.ndarray] = None,
                       coordination_numbers: Optional[np.ndarray] = None,
                       local_strain: Optional[np.ndarray] = None,
                       element_composition: Optional[Dict] = None,
                       lag_window: int = 100) -> ClusterNetworkResult:
        """
        材料クラスターネットワークを解析
        
        Parameters
        ----------
        cluster_anomaly_scores : 各クラスターの異常スコア時系列
        cluster_coupling : クラスター間カップリング
        cluster_centers : クラスター重心座標
        coordination_numbers : 配位数
        local_strain : 局所歪みテンソル
        element_composition : 元素組成
        lag_window : ラグ窓サイズ
        """
        with self.timer('analyze_material_network'):
            logger.info("⚙️ Analyzing material cluster network")
            
            # フレーム数確認
            if not cluster_anomaly_scores:
                logger.warning("No anomaly scores provided")
                return self._create_empty_result()
            
            first_score = next(iter(cluster_anomaly_scores.values()))
            n_frames = len(first_score)
            
            if n_frames <= 0:
                logger.warning("No frames to analyze")
                return self._create_empty_result()
            
            cluster_ids = sorted(cluster_anomaly_scores.keys())
            
            # 🆕 IDマッパー初期化
            dummy_atoms = {cid: [] for cid in cluster_ids}
            self.id_mapper = ClusterIDMapper(dummy_atoms)
            
            # ========================================
            # 材料特有のパターン判定
            # ========================================
            
            material_pattern = self._detect_material_pattern(
                cluster_anomaly_scores,
                coordination_numbers,
                local_strain,
                n_frames
            )
            
            logger.info(f"   Material pattern: {material_pattern}")
            
            # パターン別解析
            if material_pattern == 'elastic_deformation':
                return self._analyze_elastic_pattern(
                    cluster_anomaly_scores, cluster_coupling, 
                    cluster_centers, local_strain
                )
            elif material_pattern == 'plastic_deformation':
                return self._analyze_plastic_pattern(
                    cluster_anomaly_scores, cluster_coupling,
                    cluster_centers, coordination_numbers, local_strain
                )
            elif material_pattern == 'fracture_initiation':
                return self._analyze_fracture_pattern(
                    cluster_anomaly_scores, cluster_coupling,
                    cluster_centers, coordination_numbers, 
                    local_strain, element_composition
                )
            else:
                # デフォルト解析
                return self._analyze_general_pattern(
                    cluster_anomaly_scores, cluster_coupling,
                    cluster_centers, lag_window
                )
    
    def _detect_material_pattern(self,
                                anomaly_scores: Dict[int, np.ndarray],
                                coordination_numbers: Optional[np.ndarray],
                                local_strain: Optional[np.ndarray],
                                n_frames: int) -> str:
        """材料変形パターンを検出"""
        
        # 歪みレベルチェック
        if local_strain is not None and local_strain.size > 0:
            max_strain = np.max(np.abs(local_strain))
            
            if max_strain < 0.01:  # 1%未満
                return 'elastic_deformation'
            elif max_strain < 0.05:  # 5%未満
                # 配位数欠陥チェック
                if coordination_numbers is not None:
                    coord_defects = np.abs(coordination_numbers - 12.0)  # FCC理想
                    if np.max(coord_defects) > 2:
                        return 'plastic_deformation'
            elif max_strain > 0.1:  # 10%以上
                return 'fracture_initiation'
        
        # 異常スコアパターン
        max_anomaly = max(np.max(scores) for scores in anomaly_scores.values())
        if max_anomaly > 3.0:
            return 'fracture_initiation'
        
        return 'general_deformation'
    
    # _analyze_elastic_pattern メソッドの修正（重要！）
    def _analyze_elastic_pattern(self,
                                anomaly_scores: Dict[int, np.ndarray],
                                cluster_coupling: np.ndarray,
                                cluster_centers: np.ndarray,
                                local_strain: np.ndarray) -> ClusterNetworkResult:
        """弾性変形パターンの解析"""
        
        cluster_ids = sorted(anomaly_scores.keys())
        
        # 歪み相関ネットワーク
        strain_links = []
        
        if local_strain is not None and local_strain.shape[0] > 0:
            # 最新フレームの歪み
            current_strain = local_strain[-1] if local_strain.ndim == 4 else local_strain
            
            for i, cluster_i in enumerate(cluster_ids):
                for j, cluster_j in enumerate(cluster_ids[i+1:], i+1):
                    # 🆕 IDをインデックスに変換
                    idx_i = self.id_mapper.to_idx(cluster_i)
                    idx_j = self.id_mapper.to_idx(cluster_j)
                    
                    if idx_i >= len(current_strain) or idx_j >= len(current_strain):
                        continue
                    
                    # 🆕 インデックスで配列アクセス
                    strain_i = current_strain[idx_i]
                    strain_j = current_strain[idx_j]
                    
                    correlation = np.sum(strain_i * strain_j) / 9.0
                    
                    if abs(correlation) > 0.1:
                        link = MaterialNetworkLink(
                            from_cluster=cluster_i,  # 実際のIDを保存
                            to_cluster=cluster_j,    # 実際のIDを保存
                            strength=abs(correlation),
                            strain_correlation=correlation,
                            link_type='elastic',
                            damage_signature='elastic_coupling'
                        )
                        strain_links.append(link)
        
        # 空間制約
        spatial_constraints = self._compute_spatial_constraints(
            cluster_ids, cluster_centers
        ) if cluster_centers is not None else {}
        
        # 統計
        network_stats = {
            'pattern': 'elastic_deformation',
            'n_strain_links': len(strain_links),
            'max_strain': float(np.max(np.abs(local_strain))) if local_strain is not None else 0
        }
        
        return ClusterNetworkResult(
            strain_network=strain_links,
            dislocation_network=[],
            damage_network=[],
            spatial_constraints=spatial_constraints,
            adaptive_windows={},
            network_stats=network_stats,
            critical_clusters=[]
        )

    # _analyze_plastic_pattern メソッドの修正
    def _analyze_plastic_pattern(self,
                                anomaly_scores: Dict[int, np.ndarray],
                                cluster_coupling: np.ndarray,
                                cluster_centers: np.ndarray,
                                coordination_numbers: np.ndarray,
                                local_strain: np.ndarray) -> ClusterNetworkResult:
        """塑性変形パターンの解析（転位検出）"""
        
        cluster_ids = sorted(anomaly_scores.keys())
        
        # 転位検出
        dislocation_clusters = []
        dislocation_links = []
        
        if coordination_numbers is not None and local_strain is not None:
            # 最新フレーム
            current_coord = coordination_numbers[-1] if coordination_numbers.ndim == 2 else coordination_numbers
            current_strain = local_strain[-1] if local_strain.ndim == 4 else local_strain
            
            # 転位判定
            for cluster_id in cluster_ids:
                # 🆕 IDをインデックスに変換
                idx = self.id_mapper.to_idx(cluster_id)
                
                if idx >= len(current_coord):
                    continue
                
                # 🆕 インデックスで配列アクセス
                coord_defect = abs(current_coord[idx] - 12.0)
                strain_trace = np.trace(current_strain[idx])
                
                if coord_defect > self.coord_defect_threshold and abs(strain_trace) > self.strain_threshold:
                    dislocation_clusters.append(cluster_id) 
                
            # 転位ネットワーク構築
            for i, cluster_i in enumerate(dislocation_clusters):
                for j, cluster_j in enumerate(dislocation_clusters[i+1:], i+1):
                    # 転位間相互作用
                    distance = self._compute_distance(
                        cluster_centers, cluster_i, cluster_j
                    ) if cluster_centers is not None else 10.0
                    
                    if distance < self.max_interaction_distance:
                        strength = 1.0 / (1.0 + distance)
                        
                        link = MaterialNetworkLink(
                            from_cluster=cluster_i,
                            to_cluster=cluster_j,
                            strength=strength,
                            distance=distance,
                            link_type='plastic',
                            damage_signature='dislocation_interaction'
                        )
                        dislocation_links.append(link)
        
        # 歪みネットワークも計算
        strain_links = self._compute_strain_network(
            cluster_ids, local_strain
        ) if local_strain is not None else []
        
        # 統計
        network_stats = {
            'pattern': 'plastic_deformation',
            'n_dislocations': len(dislocation_clusters),
            'n_dislocation_links': len(dislocation_links)
        }
        
        return ClusterNetworkResult(
            strain_network=strain_links,
            dislocation_network=dislocation_links,
            damage_network=[],
            spatial_constraints={},
            adaptive_windows={},
            network_stats=network_stats,
            critical_clusters=dislocation_clusters
        )
    
    def _analyze_fracture_pattern(self,
                                 anomaly_scores: Dict[int, np.ndarray],
                                 cluster_coupling: np.ndarray,
                                 cluster_centers: np.ndarray,
                                 coordination_numbers: np.ndarray,
                                 local_strain: np.ndarray,
                                 element_composition: Optional[Dict]) -> ClusterNetworkResult:
        """破壊開始パターンの解析"""
        
        cluster_ids = sorted(anomaly_scores.keys())
        
        # 臨界クラスター検出
        critical_clusters = []
        damage_links = []
        
        # 高異常クラスター
        for cluster_id, scores in anomaly_scores.items():
            if np.max(scores) > 3.0:
                critical_clusters.append(cluster_id)
        
        logger.info(f"   Found {len(critical_clusters)} critical clusters")
        
        # 損傷ネットワーク構築
        for i, cluster_i in enumerate(critical_clusters):
            for j, cluster_j in enumerate(critical_clusters[i+1:], i+1):
                # 損傷伝播強度
                score_i = np.max(anomaly_scores[cluster_i])
                score_j = np.max(anomaly_scores[cluster_j])
                strength = np.sqrt(score_i * score_j) / 3.0
                
                # 元素ブリッジチェック
                element_bridge = None
                if element_composition:
                    comp_i = element_composition.get(cluster_i, {})
                    comp_j = element_composition.get(cluster_j, {})
                    
                    # 主要元素
                    main_i = max(comp_i, key=comp_i.get) if comp_i else 'Fe'
                    main_j = max(comp_j, key=comp_j.get) if comp_j else 'Fe'
                    
                    if main_i != main_j:
                        element_bridge = (main_i, main_j)
                        # 異種元素界面は弱い
                        strength *= 1.5
                
                link = MaterialNetworkLink(
                    from_cluster=cluster_i,
                    to_cluster=cluster_j,
                    strength=strength,
                    link_type='fracture',
                    damage_signature='crack_propagation',
                    element_bridge=element_bridge
                )
                damage_links.append(link)
        
        # 統計
        network_stats = {
            'pattern': 'fracture_initiation',
            'n_critical_clusters': len(critical_clusters),
            'n_damage_links': len(damage_links)
        }
        
        return ClusterNetworkResult(
            strain_network=[],
            dislocation_network=[],
            damage_network=damage_links,
            spatial_constraints={},
            adaptive_windows={},
            network_stats=network_stats,
            critical_clusters=critical_clusters
        )
    
    def _analyze_general_pattern(self,
                                anomaly_scores: Dict[int, np.ndarray],
                                cluster_coupling: np.ndarray,
                                cluster_centers: np.ndarray,
                                lag_window: int) -> ClusterNetworkResult:
        """一般的なパターンの解析"""
        # 簡易版実装
        return self._create_empty_result()
    
    # ========================================
    # ヘルパーメソッド
    # ========================================
    def _compute_strain_network(self,
                               cluster_ids: List[int],
                               local_strain: np.ndarray) -> List[MaterialNetworkLink]:
        """歪みネットワーク計算"""
        strain_links = []
        
        if local_strain is None or local_strain.size == 0:
            return strain_links
        
        # 最新フレーム
        current_strain = local_strain[-1] if local_strain.ndim == 4 else local_strain
        
        for i, cluster_i in enumerate(cluster_ids):
            for j, cluster_j in enumerate(cluster_ids[i+1:], i+1):
                # 🆕 IDをインデックスに変換
                idx_i = self.id_mapper.to_idx(cluster_i)
                idx_j = self.id_mapper.to_idx(cluster_j)
                
                if idx_i >= len(current_strain) or idx_j >= len(current_strain):
                    continue
                
                # 🆕 インデックスで配列アクセス
                strain_i = current_strain[idx_i].flatten()
                strain_j = current_strain[idx_j].flatten()
                
                try:
                    correlation = np.corrcoef(strain_i, strain_j)[0, 1]
                    
                    if abs(correlation) > 0.3:
                        link = MaterialNetworkLink(
                            from_cluster=cluster_i,  # 実際のIDを保存
                            to_cluster=cluster_j,    # 実際のIDを保存
                            strength=abs(correlation),
                            strain_correlation=correlation,
                            link_type='elastic'
                        )
                        strain_links.append(link)
                except:
                    continue
        
        return strain_links
    
    def _compute_distance(self,
                         centers: np.ndarray,
                         cluster_i: int,
                         cluster_j: int) -> float:
        """クラスター間距離計算"""
        # 🆕 IDをインデックスに変換
        idx_i = self.id_mapper.to_idx(cluster_i)
        idx_j = self.id_mapper.to_idx(cluster_j)
        
        if centers.ndim == 3:  # (frames, clusters, 3)
            center_i = centers[-1, idx_i]
            center_j = centers[-1, idx_j]
        else:  # (clusters, 3)
            center_i = centers[idx_i]
            center_j = centers[idx_j]
        
        return float(np.linalg.norm(center_i - center_j))
    
    def _compute_spatial_constraints(self,
                                    cluster_ids: List[int],
                                    cluster_centers: np.ndarray) -> Dict:
        """空間制約計算"""
        spatial_constraints = {}
        
        for i, cluster_i in enumerate(cluster_ids):
            for j, cluster_j in enumerate(cluster_ids[i+1:], i+1):
                distance = self._compute_distance(
                    cluster_centers, cluster_i, cluster_j
                )
                
                if distance < self.max_interaction_distance:
                    spatial_constraints[(cluster_i, cluster_j)] = distance
        
        return spatial_constraints
    
    def _create_empty_result(self) -> ClusterNetworkResult:
        """空の結果を生成"""
        return ClusterNetworkResult(
            strain_network=[],
            dislocation_network=[],
            damage_network=[],
            spatial_constraints={},
            adaptive_windows={},
            network_stats={'error': 'No data to analyze'},
            critical_clusters=[]
        )

# ===============================
# Standalone Functions
# ===============================

def analyze_cluster_network_gpu(cluster_anomaly_scores: Dict[int, np.ndarray],
                               cluster_coupling: np.ndarray,
                               cluster_centers: Optional[np.ndarray] = None,
                               coordination_numbers: Optional[np.ndarray] = None,
                               local_strain: Optional[np.ndarray] = None,
                               **kwargs) -> ClusterNetworkResult:
    """クラスターネットワーク解析のスタンドアロン関数"""
    analyzer = ClusterNetworkGPU(**kwargs)
    return analyzer.analyze_network(
        cluster_anomaly_scores, cluster_coupling, cluster_centers,
        coordination_numbers, local_strain
    )
