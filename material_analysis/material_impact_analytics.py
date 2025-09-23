#!/usr/bin/env python3
"""
Material Impact Analytics v1.0 - GPU Atomic Network Edition
============================================================

材料の原子レベル欠陥核の高速検出 + 高度なネットワーク解析
転位核・亀裂先端・相変態核を原子レベルで追跡！💎

Third Impact v3.0を材料解析用に最適化

Version: 1.0.0 - Material Edition
Authors: 環ちゃん
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import Counter, defaultdict

# 既存のインポートの後に追加
from lambda3_gpu.material.material_database import MATERIAL_DATABASE, get_material_parameters

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist as cp_cdist
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

logger = logging.getLogger('lambda3_gpu.material_analysis.impact')

# ============================================
# Data Classes (Material Edition)
# ============================================

@dataclass
class AtomicDefectTrace:
    """原子レベル欠陥痕跡（材料版）"""
    atom_id: int
    cluster_id: int
    atom_type: str  # 'Fe', 'C', 'Cr', etc.
    
    # 統計的異常度
    displacement_zscore: float = 0.0
    coordination_change: float = 0.0  # 配位数変化
    strain_magnitude: float = 0.0  # 局所歪み
    
    # 欠陥シグネチャー
    defect_signature: str = "unknown"  # 'vacancy', 'interstitial', 'dislocation_core', etc.
    confidence: float = 0.0
    
    # ネットワーク特性
    connectivity_degree: int = 0
    is_hub: bool = False
    is_bridge: bool = False  # クラスター間ブリッジ
    
    # 材料特有
    burgers_vector: Optional[np.ndarray] = None  # バーガースベクトル
    stress_concentration: float = 0.0  # 応力集中係数

@dataclass
class DefectNetworkLink:
    """欠陥ネットワークリンク"""
    from_atom: int
    to_atom: int
    from_cluster: int
    to_cluster: int
    link_type: str  # 'elastic', 'plastic', 'fracture'
    strength: float
    lag: int = 0
    distance: Optional[float] = None
    stress_transfer: float = 0.0  # 応力伝達

@dataclass
class ClusterBridge:
    """クラスター間ブリッジ（転位伝播経路）"""
    from_cluster: int
    to_cluster: int
    bridge_atoms: List[Tuple[int, int]]
    total_strength: float
    bridge_type: str  # 'dislocation', 'crack', 'grain_boundary'
    stress_path: List[float] = field(default_factory=list)  # 応力経路

@dataclass
class DefectOrigin:
    """欠陥起源情報"""
    nucleation_atoms: List[int] = field(default_factory=list)  # 核生成原子
    propagation_front: List[int] = field(default_factory=list)  # 伝播フロント
    
    # ネットワーク起源
    stress_concentrators: List[int] = field(default_factory=list)  # 応力集中点
    
    # 統計情報
    mean_displacement: float = 0.0
    std_displacement: float = 0.0
    critical_strain: float = 0.0
    von_mises_stress: float = 0.0

@dataclass
class MaterialDefectNetwork:
    """材料欠陥ネットワーク"""
    elastic_network: List[DefectNetworkLink] = field(default_factory=list)
    plastic_network: List[DefectNetworkLink] = field(default_factory=list)
    fracture_network: List[DefectNetworkLink] = field(default_factory=list)
    
    cluster_bridges: List[ClusterBridge] = field(default_factory=list)
    adaptive_windows: Dict[int, int] = field(default_factory=dict)
    
    network_pattern: str = "unknown"  # 'elastic', 'plastic_flow', 'crack_propagation'
    stress_concentrators: List[int] = field(default_factory=list)
    dislocation_cores: List[int] = field(default_factory=list)
    crack_tips: List[int] = field(default_factory=list)

@dataclass
class MaterialImpactResult:
    """Material Impact解析結果"""
    event_name: str
    cluster_id: int
    event_type: str  # 'dislocation', 'crack', 'phase_transition'
    
    # 起源情報
    origin: DefectOrigin = field(default_factory=DefectOrigin)
    
    # 原子欠陥痕跡
    defect_atoms: Dict[int, AtomicDefectTrace] = field(default_factory=dict)
    
    # 欠陥ネットワーク
    defect_network: Optional[MaterialDefectNetwork] = None
    
    # 統計
    n_defect_atoms: int = 0
    n_network_links: int = 0
    n_cluster_bridges: int = 0
    dominant_defect: str = ""
    max_stress_concentration: float = 0.0
    
    # 補強ターゲット
    reinforcement_points: List[int] = field(default_factory=list)
    critical_atoms: List[int] = field(default_factory=list)
    
    # 材料パラメータ
    estimated_k_ic: Optional[float] = None  # 推定破壊靭性
    plastic_zone_size: Optional[float] = None  # 塑性域サイズ

# ============================================
# GPU Material Defect Network Analyzer
# ============================================

class MaterialDefectNetworkGPU:
    """
    材料欠陥ネットワーク解析（GPU高速化版）
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.5,
                 sync_threshold: float = 0.7,
                 max_lag: int = 3,  # 材料は短い時間スケール
                 distance_cutoff: float = 4.0,  # BCC/FCCの第一近接
                 elastic_modulus: float = 210.0):  # GPa（デフォルト値、SUJ2相当）
        """
        Parameters
        ----------
        correlation_threshold : float
            相関閾値
        sync_threshold : float
            同期判定閾値
        max_lag : int
            最大ラグ
        distance_cutoff : float
            原子間距離カットオフ（Å）
        elastic_modulus : float
            弾性係数（GPa）
        """
        self.correlation_threshold = correlation_threshold
        self.sync_threshold = sync_threshold
        self.max_lag = max_lag
        self.distance_cutoff = distance_cutoff
        self.elastic_modulus = elastic_modulus
        
        self.xp = cp if HAS_GPU else np
        logger.info(f"💎 MaterialDefectNetworkGPU initialized (GPU: {HAS_GPU})")
    
    def analyze_network(self,
                       trajectory: np.ndarray,
                       defect_atoms: List[int],
                       cluster_mapping: Dict[int, List[int]],
                       atom_types: np.ndarray,
                       start_frame: int,
                       end_frame: int,
                       strain_field: Optional[np.ndarray] = None) -> MaterialDefectNetwork:
        """
        材料欠陥ネットワーク解析
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子トラジェクトリ
        defect_atoms : List[int]
            欠陥原子リスト
        cluster_mapping : Dict[int, List[int]]
            クラスター→原子マッピング
        atom_types : np.ndarray
            原子タイプ配列
        strain_field : np.ndarray, optional
            歪み場データ
        """
        if not defect_atoms:
            return MaterialDefectNetwork()
        
        n_frames = end_frame - start_frame + 1
        
        # 単一フレーム
        if n_frames <= 1:
            return self._analyze_instantaneous_defects(
                trajectory[start_frame], defect_atoms,
                cluster_mapping, atom_types, strain_field
            )
        
        # 複数フレーム
        return self._analyze_temporal_defects(
            trajectory[start_frame:end_frame+1],
            defect_atoms, cluster_mapping,
            atom_types, strain_field
        )
    
    def _analyze_instantaneous_defects(self,
                                      frame: np.ndarray,
                                      atoms: List[int],
                                      cluster_mapping: Dict,
                                      atom_types: np.ndarray,
                                      strain_field: Optional[np.ndarray]) -> MaterialDefectNetwork:
        """単一フレームの瞬間的欠陥ネットワーク"""
        result = MaterialDefectNetwork(network_pattern="elastic")
        
        coords = frame[atoms]
        n_atoms = len(atoms)
        
        # 距離行列
        if HAS_GPU:
            coords_gpu = cp.asarray(coords)
            dist_matrix = cp_cdist(coords_gpu, coords_gpu)
            dist_matrix = cp.asnumpy(dist_matrix)
        else:
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(coords, coords)
        
        # 結晶構造に基づく近接判定
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if dist_matrix[i, j] < self.distance_cutoff:
                    # 応力伝達の推定
                    stress_transfer = self._estimate_stress_transfer(
                        dist_matrix[i, j], atom_types[atoms[i]], atom_types[atoms[j]]
                    )
                    
                    link = DefectNetworkLink(
                        from_atom=atoms[i],
                        to_atom=atoms[j],
                        from_cluster=self._get_cluster_id(atoms[i], cluster_mapping),
                        to_cluster=self._get_cluster_id(atoms[j], cluster_mapping),
                        link_type='elastic',
                        strength=1.0 / (1.0 + dist_matrix[i, j]),
                        distance=dist_matrix[i, j],
                        stress_transfer=stress_transfer
                    )
                    result.elastic_network.append(link)
        
        # 応力集中点の特定
        if strain_field is not None:
            stress_conc = self._find_stress_concentrators(
                atoms, strain_field[atoms] if len(strain_field.shape) == 1 else strain_field
            )
            result.stress_concentrators = stress_conc
        
        # クラスター間ブリッジ
        result.cluster_bridges = self._detect_cluster_bridges(
            result.elastic_network
        )
        
        return result
    
    def _analyze_temporal_defects(self,
                                 trajectory: np.ndarray,
                                 atoms: List[int],
                                 cluster_mapping: Dict,
                                 atom_types: np.ndarray,
                                 strain_field: Optional[np.ndarray]) -> MaterialDefectNetwork:
        """複数フレームの時間的欠陥解析"""
        result = MaterialDefectNetwork()
        
        # GPU転送
        if HAS_GPU:
            traj_gpu = cp.asarray(trajectory[:, atoms])
        else:
            traj_gpu = trajectory[:, atoms]
        
        n_frames, n_atoms = traj_gpu.shape[0], len(atoms)
        
        # 1. 変位解析で欠陥タイプ判定
        defect_types = self._classify_defects(traj_gpu, atoms, atom_types)
        
        # 2. 適応的窓
        adaptive_windows = self._compute_adaptive_windows_material(
            traj_gpu, atoms, defect_types
        )
        result.adaptive_windows = adaptive_windows
        
        # 3. 相関計算
        correlations = self._compute_defect_correlations(
            traj_gpu, adaptive_windows
        )
        
        # 4. ネットワーク構築
        networks = self._build_defect_networks(
            correlations, atoms, trajectory[0],
            cluster_mapping, defect_types
        )
        
        result.elastic_network = networks['elastic']
        result.plastic_network = networks['plastic']
        result.fracture_network = networks['fracture']
        
        # 5. パターン識別
        result.network_pattern = self._identify_defect_pattern(networks)
        
        # 6. 欠陥コア特定
        result.dislocation_cores = self._find_dislocation_cores(
            networks['plastic'], defect_types
        )
        result.crack_tips = self._find_crack_tips(
            networks['fracture'], strain_field
        )
        
        # 7. クラスター間ブリッジ
        result.cluster_bridges = self._detect_cluster_bridges(
            networks['plastic'] + networks['fracture']
        )
        
        return result
    
    def _classify_defects(self,
                        traj_gpu: Any,
                        atoms: List[int],
                        atom_types: np.ndarray) -> Dict[int, str]:
        """欠陥タイプの分類"""
        defect_types = {}
        n_frames = traj_gpu.shape[0]
        
        for i, atom in enumerate(atoms):
            # 変位の大きさ
            if HAS_GPU:
                displacements = cp.diff(traj_gpu[:, i], axis=0)
                total_disp = float(cp.sum(cp.linalg.norm(displacements, axis=1)))
            else:
                displacements = np.diff(traj_gpu[:, i], axis=0)
                total_disp = float(np.sum(np.linalg.norm(displacements, axis=1)))
            
            # 簡易分類
            if total_disp > n_frames * 0.5:  # 大変位
                defect_types[atom] = 'dislocation'
            elif total_disp > n_frames * 0.2:  # 中変位
                defect_types[atom] = 'plastic'
            else:
                defect_types[atom] = 'elastic'
        
        return defect_types
    
    def _compute_adaptive_windows_material(self,
                                         traj_gpu: Any,
                                         atoms: List[int],
                                         defect_types: Dict) -> Dict[int, int]:
        """材料用適応的窓サイズ"""
        n_frames = traj_gpu.shape[0]
        windows = {}
        
        for i, atom in enumerate(atoms):
            defect_type = defect_types.get(atom, 'elastic')
            
            # 欠陥タイプに応じた窓サイズ
            if defect_type == 'dislocation':
                windows[atom] = min(5, n_frames)  # 転位は速い
            elif defect_type == 'plastic':
                windows[atom] = min(10, n_frames)
            else:
                windows[atom] = min(20, n_frames)  # 弾性は遅い
        
        return windows
    
    def _compute_defect_correlations(self,
                                   traj_gpu: Any,
                                   windows: Dict) -> Dict:
        """欠陥相関の計算"""
        n_frames, n_atoms = traj_gpu.shape[:2]
        
        correlations = {
            'sync': np.zeros((n_atoms, n_atoms)),
            'lagged': np.zeros((n_atoms, n_atoms, self.max_lag + 1)),
            'stress_coupling': np.zeros((n_atoms, n_atoms))
        }
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                window = min(
                    windows.get(list(windows.keys())[i], 10),
                    windows.get(list(windows.keys())[j], 10)
                )
                
                # 変位の時系列
                if HAS_GPU:
                    ts_i = cp.linalg.norm(traj_gpu[:, i], axis=1)
                    ts_j = cp.linalg.norm(traj_gpu[:, j], axis=1)
                else:
                    ts_i = np.linalg.norm(traj_gpu[:, i], axis=1)
                    ts_j = np.linalg.norm(traj_gpu[:, j], axis=1)
                
                # 同期相関
                if HAS_GPU:
                    sync_corr = float(cp.corrcoef(ts_i[:window], ts_j[:window])[0, 1])
                else:
                    sync_corr = np.corrcoef(ts_i[:window], ts_j[:window])[0, 1]
                
                if not np.isnan(sync_corr):
                    correlations['sync'][i, j] = sync_corr
                    correlations['sync'][j, i] = sync_corr
                
                # ラグ付き（転位伝播）
                for lag in range(1, min(self.max_lag + 1, window)):
                    if HAS_GPU:
                        lagged_corr = float(cp.corrcoef(ts_i[:-lag], ts_j[lag:])[0, 1])
                    else:
                        lagged_corr = np.corrcoef(ts_i[:-lag], ts_j[lag:])[0, 1]
                    
                    if not np.isnan(lagged_corr):
                        correlations['lagged'][i, j, lag] = lagged_corr
                
                # 応力結合（距離ベース）
                if HAS_GPU:
                    mean_dist = float(cp.mean(cp.linalg.norm(
                        traj_gpu[:, i] - traj_gpu[:, j], axis=1
                    )))
                else:
                    mean_dist = np.mean(np.linalg.norm(
                        traj_gpu[:, i] - traj_gpu[:, j], axis=1
                    ))
                
                if mean_dist < self.distance_cutoff * 1.5:
                    correlations['stress_coupling'][i, j] = 1.0 / (1.0 + mean_dist)
        
        return correlations
    
    def _build_defect_networks(self,
                             correlations: Dict,
                             atoms: List[int],
                             first_frame: np.ndarray,
                             cluster_mapping: Dict,
                             defect_types: Dict) -> Dict:
        """欠陥ネットワーク構築"""
        networks = {'elastic': [], 'plastic': [], 'fracture': []}
        n_atoms = len(atoms)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                atom_i, atom_j = atoms[i], atoms[j]
                cluster_i = self._get_cluster_id(atom_i, cluster_mapping)
                cluster_j = self._get_cluster_id(atom_j, cluster_mapping)
                
                dist = np.linalg.norm(first_frame[atom_i] - first_frame[atom_j])
                
                # 欠陥タイプ
                type_i = defect_types.get(atom_i, 'elastic')
                type_j = defect_types.get(atom_j, 'elastic')
                
                # 同期ネットワーク（弾性）
                sync_corr = correlations['sync'][i, j]
                if abs(sync_corr) > self.sync_threshold and \
                   type_i == 'elastic' and type_j == 'elastic':
                    networks['elastic'].append(DefectNetworkLink(
                        from_atom=atom_i,
                        to_atom=atom_j,
                        from_cluster=cluster_i,
                        to_cluster=cluster_j,
                        link_type='elastic',
                        strength=abs(sync_corr),
                        distance=dist,
                        stress_transfer=correlations['stress_coupling'][i, j]
                    ))
                
                # 塑性ネットワーク
                if (type_i == 'plastic' or type_j == 'plastic'):
                    max_lag_corr = 0.0
                    best_lag = 0
                    for lag in range(1, self.max_lag + 1):
                        lag_corr = correlations['lagged'][i, j, lag]
                        if abs(lag_corr) > abs(max_lag_corr):
                            max_lag_corr = lag_corr
                            best_lag = lag
                    
                    if abs(max_lag_corr) > self.correlation_threshold:
                        networks['plastic'].append(DefectNetworkLink(
                            from_atom=atom_i,
                            to_atom=atom_j,
                            from_cluster=cluster_i,
                            to_cluster=cluster_j,
                            link_type='plastic',
                            strength=abs(max_lag_corr),
                            lag=best_lag,
                            distance=dist,
                            stress_transfer=correlations['stress_coupling'][i, j]
                        ))
                
                # 破壊ネットワーク
                if (type_i == 'dislocation' or type_j == 'dislocation') and \
                   correlations['stress_coupling'][i, j] > 0.5:
                    networks['fracture'].append(DefectNetworkLink(
                        from_atom=atom_i,
                        to_atom=atom_j,
                        from_cluster=cluster_i,
                        to_cluster=cluster_j,
                        link_type='fracture',
                        strength=correlations['stress_coupling'][i, j],
                        distance=dist,
                        stress_transfer=correlations['stress_coupling'][i, j] * self.elastic_modulus
                    ))
        
        return networks
    
    def _identify_defect_pattern(self, networks: Dict) -> str:
        """欠陥パターン識別"""
        n_elastic = len(networks['elastic'])
        n_plastic = len(networks['plastic'])
        n_fracture = len(networks['fracture'])
        
        if n_fracture > n_plastic:
            return "crack_propagation"
        elif n_plastic > n_elastic:
            return "plastic_flow"
        else:
            return "elastic_deformation"
    
    def _find_dislocation_cores(self,
                               plastic_network: List,
                               defect_types: Dict) -> List[int]:
        """転位コアの特定"""
        cores = []
        degree_count = Counter()
        
        for link in plastic_network:
            degree_count[link.from_atom] += 1
            degree_count[link.to_atom] += 1
        
        # 高次数かつ転位タイプ
        for atom, degree in degree_count.most_common(10):
            if defect_types.get(atom) == 'dislocation' and degree >= 3:
                cores.append(atom)
        
        return cores
    
    def _find_crack_tips(self,
                        fracture_network: List,
                        strain_field: Optional[np.ndarray]) -> List[int]:
        """亀裂先端の特定"""
        tips = []
        
        if not fracture_network:
            return tips
        
        # 応力伝達が最大の原子
        max_stress_atoms = sorted(
            fracture_network,
            key=lambda l: l.stress_transfer,
            reverse=True
        )[:5]
        
        for link in max_stress_atoms:
            tips.append(link.from_atom)
            tips.append(link.to_atom)
        
        return list(set(tips))[:5]
    
    def _find_stress_concentrators(self,
                                  atoms: List[int],
                                  strain_values: np.ndarray) -> List[int]:
        """応力集中点の特定"""
        if strain_values is None or len(strain_values) == 0:
            return []
        
        # 歪みの大きい原子
        threshold = np.mean(strain_values) + 2 * np.std(strain_values)
        concentrators = []
        
        for i, atom in enumerate(atoms):
            if i < len(strain_values) and strain_values[i] > threshold:
                concentrators.append(atom)
        
        return concentrators[:10]
    
    def _detect_cluster_bridges(self,
                              links: List[DefectNetworkLink]) -> List[ClusterBridge]:
        """クラスター間ブリッジ検出"""
        bridges_dict = defaultdict(list)
        
        for link in links:
            if link.from_cluster != link.to_cluster:
                key = (link.from_cluster, link.to_cluster)
                bridges_dict[key].append(
                    (link.from_atom, link.to_atom, link.strength, link.stress_transfer)
                )
        
        bridges = []
        for (from_c, to_c), atom_data in bridges_dict.items():
            total_strength = sum(s for _, _, s, _ in atom_data)
            stress_path = [st for _, _, _, st in atom_data]
            
            # ブリッジタイプ判定
            if max(stress_path) > self.elastic_modulus * 0.01:  # 1%歪み相当
                bridge_type = 'crack'
            elif total_strength > 2.0:
                bridge_type = 'dislocation'
            else:
                bridge_type = 'grain_boundary'
            
            bridges.append(ClusterBridge(
                from_cluster=from_c,
                to_cluster=to_c,
                bridge_atoms=[(a1, a2) for a1, a2, _, _ in atom_data],
                total_strength=total_strength,
                bridge_type=bridge_type,
                stress_path=stress_path
            ))
        
        return sorted(bridges, key=lambda b: b.total_strength, reverse=True)
    
    def _estimate_stress_transfer(self,
                                distance: float,
                                atom_type_i: str,
                                atom_type_j: str) -> float:
        """応力伝達の推定"""
        # 簡易モデル：距離の逆数と原子タイプ
        base_transfer = 1.0 / (1.0 + distance)
        
        # 原子タイプによる補正（例：Feは強い結合）
        if atom_type_i == 'Fe' and atom_type_j == 'Fe':
            return base_transfer * 1.2
        elif 'C' in [atom_type_i, atom_type_j]:
            return base_transfer * 1.5  # 炭素は強い
        else:
            return base_transfer
    
    def _get_cluster_id(self, atom_id: int, mapping: Dict) -> int:
        """原子IDからクラスターID取得"""
        for cluster_id, atom_list in mapping.items():
            if atom_id in atom_list:
                return cluster_id
        return -1

# ============================================
# Material Impact Analyzer
# ============================================

class MaterialImpactAnalyzer:
    """
    Material Impact Analytics v1.0
    材料の原子レベル欠陥解析
    """
    
    def __init__(self,
                 cluster_mapping: Optional[Dict[int, List[int]]] = None,
                 sigma_threshold: float = 2.5,  # 材料は低め
                 use_network_analysis: bool = True,
                 use_gpu: bool = True,
                 material_type: str = 'SUJ2'):
        """
        Parameters
        ----------
        cluster_mapping : Dict[int, List[int]]
            クラスターID -> 原子IDリスト
        sigma_threshold : float
            統計的有意性閾値
        use_network_analysis : bool
            ネットワーク解析使用
        use_gpu : bool
            GPU加速
        material_type : str
            材料タイプ
        """
        self.cluster_mapping = cluster_mapping
        self.sigma_threshold = sigma_threshold
        self.use_network_analysis = use_network_analysis
        self.material_type = material_type
        
        # 材料パラメータ
        self._set_material_params()
        
        # ネットワーク解析器
        if use_network_analysis and use_gpu and HAS_GPU:
            self.defect_network = MaterialDefectNetworkGPU(
                elastic_modulus=self.elastic_modulus
            )
            logger.info("💎 GPU-accelerated defect network analysis enabled")
        elif use_network_analysis:
            self.defect_network = MaterialDefectNetworkGPU(
                elastic_modulus=self.elastic_modulus
            )
            logger.info("💎 CPU defect network analysis enabled")
        else:
            self.defect_network = None
        
        logger.info(f"💎 Material Impact Analyzer v1.0 initialized ({material_type})")

    def _set_material_params(self):
        """材料パラメータ設定（統一データベースから）"""
        params = get_material_parameters(self.material_type)
        
        # 材料プロパティ取得（互換性キーも考慮）
        self.elastic_modulus = params.get('elastic_modulus', params.get('E', 210.0))
        self.yield_strength = params.get('yield_strength', params.get('yield', 1.5))
        self.k_ic = params.get('fracture_toughness', params.get('K_IC', 30.0))
        
        # 追加パラメータも保存（後で使うかも）
        self.material_params = params
    
    def analyze_critical_clusters(self,
                                 macro_result: Any,
                                 two_stage_result: Any,
                                 trajectory: np.ndarray,
                                 atom_types: np.ndarray,
                                 strain_field: Optional[np.ndarray] = None,
                                 top_n: int = 3,
                                 **kwargs) -> Dict[str, MaterialImpactResult]:
        """
        異常クラスターの原子レベル解析
        """
        logger.info("\n" + "="*60)
        logger.info("💎 MATERIAL IMPACT v1.0 - Defect Network Analysis")
        logger.info("="*60)
        
        start_time = time.time()
        results = {}
        
        # Top Nクラスターの特定
        top_clusters = self._identify_critical_clusters(two_stage_result, top_n)
        logger.info(f"Analyzing top {len(top_clusters)} critical clusters")
        
        # 各イベントの解析
        for event_name, cluster_analysis in two_stage_result.cluster_analyses.items():
            for cluster_event in cluster_analysis.cluster_events:
                if cluster_event.cluster_id not in top_clusters:
                    continue
                
                cluster_id = cluster_event.cluster_id
                start_frame = cluster_event.start_frame
                end_frame = cluster_event.end_frame
                
                # イベントタイプ判定
                if cluster_event.dislocation_density and cluster_event.dislocation_density > 1e12:
                    event_type = "dislocation"
                elif cluster_event.peak_damage > 0.5:
                    event_type = "crack"
                else:
                    event_type = "elastic"
                
                logger.info(f"\n⚙️ {event_name} - Cluster {cluster_id} ({event_type})")
                
                # クラスターの原子取得
                cluster_atoms = self._get_cluster_atoms(cluster_id, trajectory.shape[1])
                
                # 基本解析
                result = self._analyze_defect_event(
                    cluster_id, event_name, event_type,
                    trajectory, atom_types,
                    start_frame, end_frame,
                    cluster_atoms, strain_field
                )
                
                # ネットワーク解析
                if self.defect_network and result.defect_atoms:
                    logger.info("   🌐 Analyzing defect network...")
                    network_result = self.defect_network.analyze_network(
                        trajectory=trajectory,
                        defect_atoms=list(result.defect_atoms.keys()),
                        cluster_mapping=self.cluster_mapping,
                        atom_types=atom_types,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        strain_field=strain_field
                    )
                    
                    result.defect_network = network_result
                    result.n_network_links = (
                        len(network_result.elastic_network) +
                        len(network_result.plastic_network) +
                        len(network_result.fracture_network)
                    )
                    result.n_cluster_bridges = len(network_result.cluster_bridges)
                    
                    # ネットワーク情報反映
                    self._update_defect_traces_with_network(result)
                
                # 補強ポイント特定
                result.reinforcement_points = self._identify_reinforcement_points(result)
                result.critical_atoms = self._identify_critical_atoms(result)
                
                # 材料パラメータ推定
                self._estimate_material_params(result, cluster_event)
                
                results[f"{event_name}_cluster{cluster_id}"] = result
        
        computation_time = time.time() - start_time
        logger.info(f"\n💎 Analysis complete in {computation_time:.2f}s")
        
        self._print_summary(results)
        return results
    
    def _analyze_defect_event(self,
                            cluster_id: int,
                            event_name: str,
                            event_type: str,
                            trajectory: np.ndarray,
                            atom_types: np.ndarray,
                            start_frame: int,
                            end_frame: int,
                            cluster_atoms: List[int],
                            strain_field: Optional[np.ndarray]) -> MaterialImpactResult:
        """欠陥イベント解析"""
        result = MaterialImpactResult(
            event_name=event_name,
            cluster_id=cluster_id,
            event_type=event_type
        )
        
        if start_frame == 0:
            return result
        
        # 変位解析
        displacements = trajectory[start_frame] - trajectory[start_frame-1]
        distances = np.linalg.norm(displacements, axis=1)
        
        # 統計
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        threshold = mean_d + self.sigma_threshold * std_d
        
        result.origin.mean_displacement = mean_d
        result.origin.std_displacement = std_d
        
        # 異常原子特定
        for atom_id in cluster_atoms:
            if atom_id >= len(distances):
                continue
            
            z_score = (distances[atom_id] - mean_d) / (std_d + 1e-10)
            
            if z_score > self.sigma_threshold:
                trace = AtomicDefectTrace(
                    atom_id=atom_id,
                    cluster_id=cluster_id,
                    atom_type=str(atom_types[atom_id]) if atom_id < len(atom_types) else 'Unknown',
                    displacement_zscore=z_score
                )
                
                # 歪み
                if strain_field is not None and atom_id < len(strain_field):
                    trace.strain_magnitude = float(strain_field[atom_id])
                
                # 欠陥シグネチャー
                trace.defect_signature = self._classify_defect(
                    z_score, event_type, trace.strain_magnitude
                )
                trace.confidence = min(z_score / 4.0, 1.0)
                
                # 応力集中
                trace.stress_concentration = trace.strain_magnitude * self.elastic_modulus
                
                result.defect_atoms[atom_id] = trace
                result.origin.nucleation_atoms.append(atom_id)
        
        # 統計更新
        result.n_defect_atoms = len(result.defect_atoms)
        if result.defect_atoms:
            stress_concs = [t.stress_concentration for t in result.defect_atoms.values()]
            result.max_stress_concentration = max(stress_concs)
            
            signatures = [t.defect_signature for t in result.defect_atoms.values()]
            if signatures:
                result.dominant_defect = Counter(signatures).most_common(1)[0][0]
        
        return result
    
    def _update_defect_traces_with_network(self, result: MaterialImpactResult):
        """ネットワーク情報で欠陥痕跡更新"""
        if not result.defect_network:
            return
        
        # 接続度計算
        degree_count = Counter()
        all_links = (result.defect_network.elastic_network +
                    result.defect_network.plastic_network +
                    result.defect_network.fracture_network)
        
        for link in all_links:
            degree_count[link.from_atom] += 1
            degree_count[link.to_atom] += 1
        
        # 応力集中点
        stress_conc = set(result.defect_network.stress_concentrators)
        
        # 転位コア
        disl_cores = set(result.defect_network.dislocation_cores)
        
        # 欠陥痕跡更新
        for atom_id, trace in result.defect_atoms.items():
            trace.connectivity_degree = degree_count.get(atom_id, 0)
            trace.is_hub = atom_id in stress_conc
            
            # バーガースベクトル（簡易）
            if atom_id in disl_cores:
                trace.burgers_vector = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)  # BCC <111>
                result.origin.stress_concentrators.append(atom_id)
    
    def _classify_defect(self,
                        z_score: float,
                        event_type: str,
                        strain: float) -> str:
        """欠陥シグネチャー分類"""
        if event_type == "dislocation":
            if z_score > 4.0:
                return "dislocation_core"
            else:
                return "dislocation_cloud"
        elif event_type == "crack":
            if strain > 0.02:  # 2%歪み
                return "crack_tip"
            else:
                return "process_zone"
        else:
            if z_score > 3.0:
                return "point_defect"
            else:
                return "elastic_distortion"
    
    def _identify_reinforcement_points(self,
                                      result: MaterialImpactResult) -> List[int]:
        """補強ポイント特定"""
        points = []
        
        # 応力集中点を優先
        for atom_id, trace in result.defect_atoms.items():
            if trace.is_hub and trace.stress_concentration > self.yield_strength:
                points.append(atom_id)
        
        # 転位コア
        if result.defect_network:
            points.extend(result.defect_network.dislocation_cores[:3])
        
        return list(set(points))[:10]
    
    def _identify_critical_atoms(self,
                                result: MaterialImpactResult) -> List[int]:
        """臨界原子特定"""
        critical = []
        
        # 亀裂先端
        if result.defect_network:
            critical.extend(result.defect_network.crack_tips)
        
        # 高応力集中
        for atom_id, trace in result.defect_atoms.items():
            if trace.stress_concentration > self.yield_strength * 0.8:
                critical.append(atom_id)
        
        return list(set(critical))[:10]
    
    def _estimate_material_params(self,
                                 result: MaterialImpactResult,
                                 cluster_event: Any):
        """材料パラメータ推定"""
        # 塑性域サイズ（Irwinモデル）
        if result.max_stress_concentration > 0:
            result.plastic_zone_size = (
                (self.k_ic / result.max_stress_concentration) ** 2 / (2 * np.pi)
            )
        
        # 推定破壊靭性（逆算）
        if cluster_event.von_mises_stress:
            result.estimated_k_ic = (
                cluster_event.von_mises_stress * 
                np.sqrt(np.pi * cluster_event.peak_strain * 10.0)  # 仮定：亀裂長10Å
            )
    
    def _identify_critical_clusters(self,
                                   two_stage_result: Any,
                                   top_n: int) -> Set[int]:
        """臨界クラスター特定"""
        if not hasattr(two_stage_result, 'critical_clusters'):
            return set()
        
        return set(two_stage_result.critical_clusters[:top_n])
    
    def _get_cluster_atoms(self, cluster_id: int, n_atoms: int) -> List[int]:
        """クラスター原子取得（細分化ID対応版）"""
        if self.cluster_mapping and cluster_id in self.cluster_mapping:
            atoms = self.cluster_mapping[cluster_id]
            if not atoms:
                logger.warning(f"Cluster {cluster_id} has empty atom list in mapping")
            return atoms
        
        # 細分化クラスターの場合の処理
        if cluster_id > 1000:  # 細分化クラスターID
            parent_cluster = cluster_id // 1000  # 3017 → 3
            subdivision = cluster_id % 1000      # 3017 → 17
            
            logger.warning(f"Cluster {cluster_id} not in mapping, using fallback (parent={parent_cluster}, sub={subdivision})")
            
            # 親クラスターの推定範囲を細分化
            atoms_per_parent = 1000  # 親クラスターごと
            atoms_per_sub = 50       # 細分化ごと
            
            start_atom = (parent_cluster - 1) * atoms_per_parent + subdivision * atoms_per_sub
            end_atom = min(start_atom + atoms_per_sub, n_atoms)
            
            if start_atom < n_atoms:
                return list(range(start_atom, end_atom))
            else:
                logger.warning(f"Cluster {cluster_id}: computed range {start_atom}-{end_atom} exceeds n_atoms={n_atoms}")
                return []
        
        # 通常のフォールバック（健全領域など）
        atoms_per_cluster = 500  # もっと大きく
        start_atom = min(cluster_id * atoms_per_cluster, n_atoms - 100)
        end_atom = min(start_atom + atoms_per_cluster, n_atoms)
        return list(range(max(0, start_atom), end_atom))

    def _print_summary(self, results: Dict[str, MaterialImpactResult]):
        """解析サマリー"""
        print("\n" + "="*60)
        print("💎 MATERIAL IMPACT v1.0 SUMMARY")
        print("="*60)
        
        total_nucleation = sum(len(r.origin.nucleation_atoms) for r in results.values())
        total_defects = sum(r.n_defect_atoms for r in results.values())
        total_links = sum(r.n_network_links for r in results.values())
        total_bridges = sum(r.n_cluster_bridges for r in results.values())
        
        print(f"\n📊 Statistics:")
        print(f"  - Events analyzed: {len(results)}")
        print(f"  - Nucleation atoms: {total_nucleation}")
        print(f"  - Defect atoms: {total_defects}")
        print(f"  - Network links: {total_links}")
        print(f"  - Cluster bridges: {total_bridges}")
        
        for event_key, result in results.items():
            print(f"\n⚙️ {event_key} ({result.event_type}):")
            print(f"  - Cluster: {result.cluster_id}")
            print(f"  - Nucleation atoms: {result.origin.nucleation_atoms[:5]}")
            
            if result.defect_network:
                print(f"  - Network pattern: {result.defect_network.network_pattern}")
                
                if result.defect_network.dislocation_cores:
                    print(f"  - Dislocation cores: {result.defect_network.dislocation_cores[:3]}")
                
                if result.defect_network.crack_tips:
                    print(f"  - Crack tips: {result.defect_network.crack_tips[:3]}")
                
                if result.defect_network.cluster_bridges:
                    bridge = result.defect_network.cluster_bridges[0]
                    print(f"  - Main bridge: C{bridge.from_cluster}→C{bridge.to_cluster}")
                    print(f"    Type: {bridge.bridge_type}")
            
            if result.reinforcement_points:
                print(f"  - Reinforcement points: {result.reinforcement_points[:5]}")
            
            if result.plastic_zone_size:
                print(f"  - Plastic zone: {result.plastic_zone_size:.2f} Å")

# ============================================
# Integration Functions
# ============================================

def run_material_impact_analysis(macro_result: Any,
                                two_stage_result: Any,
                                trajectory: np.ndarray,
                                atom_types: np.ndarray,
                                cluster_mapping: Optional[Dict[int, List[int]]] = None,
                                strain_field: Optional[np.ndarray] = None,
                                material_type: str = 'SUJ2',
                                output_dir: Optional[Path] = None,
                                **kwargs) -> Dict[str, MaterialImpactResult]:
    """
    Material Impact解析の実行
    """
    logger.info("💎 Starting Material Impact Analysis v1.0...")
    
    # アナライザー初期化
    analyzer = MaterialImpactAnalyzer(
        cluster_mapping=cluster_mapping,
        sigma_threshold=kwargs.get('sigma_threshold', 2.5),
        use_network_analysis=kwargs.get('use_network_analysis', True),
        use_gpu=kwargs.get('use_gpu', True),
        material_type=material_type
    )
    
    # 解析実行
    results = analyzer.analyze_critical_clusters(
        macro_result=macro_result,
        two_stage_result=two_stage_result,
        trajectory=trajectory,
        atom_types=atom_types,
        strain_field=strain_field,
        top_n=kwargs.get('top_n', 3)
    )
    
    # 結果保存
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON保存
        save_material_results_json(results, output_path)
        
        # レポート生成
        report = generate_material_impact_report(results, material_type)
        with open(output_path / 'material_impact_report.txt', 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to {output_path}")
    
    return results

def save_material_results_json(results: Dict[str, MaterialImpactResult], output_path: Path):
    """結果をJSON保存"""
    json_data = {}
    
    for event_key, result in results.items():
        json_data[event_key] = {
            'event_name': result.event_name,
            'cluster_id': result.cluster_id,
            'event_type': result.event_type,
            'n_defect_atoms': result.n_defect_atoms,
            'n_network_links': result.n_network_links,
            'n_cluster_bridges': result.n_cluster_bridges,
            'nucleation_atoms': result.origin.nucleation_atoms,
            'dominant_defect': result.dominant_defect,
            'max_stress_concentration': float(result.max_stress_concentration),
            'reinforcement_points': result.reinforcement_points,
            'critical_atoms': result.critical_atoms
        }
        
        if result.plastic_zone_size:
            json_data[event_key]['plastic_zone_size'] = float(result.plastic_zone_size)
        
        if result.estimated_k_ic:
            json_data[event_key]['estimated_k_ic'] = float(result.estimated_k_ic)
    
    with open(output_path / 'material_impact.json', 'w') as f:
        json.dump(json_data, f, indent=2)

def generate_material_impact_report(results: Dict[str, MaterialImpactResult],
                                   material_type: str) -> str:
    """レポート生成"""
    report = f"""
================================================================================
💎 MATERIAL IMPACT ANALYSIS v1.0 - {material_type}
================================================================================

EXECUTIVE SUMMARY
-----------------
Events Analyzed: {len(results)}
Total Defect Atoms: {sum(r.n_defect_atoms for r in results.values())}
Total Network Links: {sum(r.n_network_links for r in results.values())}

DETAILED ANALYSIS
-----------------
"""
    
    for event_key, result in results.items():
        report += f"\n{event_key} ({result.event_type})\n"
        report += "=" * len(event_key) + "\n"
        report += f"Cluster: {result.cluster_id}\n"
        report += f"Dominant Defect: {result.dominant_defect}\n"
        report += f"Max Stress Concentration: {result.max_stress_concentration:.2f} GPa\n"
        
        if result.defect_network:
            report += f"\nNetwork Analysis:\n"
            report += f"  Pattern: {result.defect_network.network_pattern}\n"
            report += f"  Dislocation cores: {len(result.defect_network.dislocation_cores)}\n"
            report += f"  Crack tips: {len(result.defect_network.crack_tips)}\n"
        
        if result.plastic_zone_size:
            report += f"\nPlastic Zone Size: {result.plastic_zone_size:.2f} Å\n"
        
        if result.reinforcement_points:
            report += f"Reinforcement Points: {result.reinforcement_points}\n"
    
    return report
