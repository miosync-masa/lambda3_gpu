#!/usr/bin/env python3
"""
Material Analytics GPU - Advanced Material-Specific Analysis
============================================================

材料解析に特化した高度な解析機能群
結晶欠陥定量化、構造一貫性評価、破壊予測など

Lambda³ GPU既存モジュールと連携して材料特有の現象を解析

Version: 1.0.0
Author: 環ちゃん
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import logging

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

# Lambda³ Core imports
from ..core.gpu_utils import GPUBackend

# Logger設定
logger = logging.getLogger(__name__)

# ===============================
# Material Properties Database
# ===============================

MATERIAL_PROPERTIES = {
    'SUJ2': {
        'name': 'Bearing Steel (SUJ2)',
        'elastic_modulus': 210.0,  # GPa
        'yield_strength': 1.5,      # GPa
        'ultimate_strength': 2.0,   # GPa
        'fatigue_strength': 0.7,    # GPa
        'fracture_toughness': 30.0, # MPa√m
        'crystal_structure': 'BCC',
        'ideal_coordination': 8
    },
    'AL7075': {
        'name': 'Aluminum Alloy 7075',
        'elastic_modulus': 71.7,
        'yield_strength': 0.503,
        'ultimate_strength': 0.572,
        'fatigue_strength': 0.159,
        'fracture_toughness': 23.0,
        'crystal_structure': 'FCC',
        'ideal_coordination': 12
    },
    'TI6AL4V': {
        'name': 'Titanium Alloy Ti-6Al-4V',
        'elastic_modulus': 113.8,
        'yield_strength': 0.880,
        'ultimate_strength': 0.950,
        'fatigue_strength': 0.510,
        'fracture_toughness': 75.0,
        'crystal_structure': 'HCP',
        'ideal_coordination': 12
    }
}

# ===============================
# Data Classes
# ===============================

@dataclass
class DefectAnalysisResult:
    """結晶欠陥解析結果"""
    defect_charge: np.ndarray           # 欠陥チャージ (n_frames, n_clusters)
    cumulative_charge: np.ndarray       # 累積チャージ (n_frames,)
    burgers_vectors: Optional[np.ndarray] = None  # Burgers vectors
    coordination_defects: Optional[np.ndarray] = None  # 配位数欠陥

@dataclass
class CrystalDefectResult:
    """結晶欠陥解析結果（簡易版）"""
    defect_charge: np.ndarray           # 欠陥チャージ
    cumulative_charge: np.ndarray       # 累積チャージ
    max_charge: float                   # 最大チャージ
    critical_frame: int                 # 臨界フレーム

@dataclass
class MaterialState:
    """材料状態情報"""
    state: str                          # 'healthy', 'damaged', 'critical', 'failed'
    health_index: float                 # 健全性指標 (0.0-1.0)
    max_damage: float                   # 最大損傷値
    damage_distribution: Optional[np.ndarray] = None  # クラスター別損傷分布
    critical_clusters: Optional[List[int]] = None     # 臨界クラスターリスト
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（後方互換性のため）"""
        return {
            'state': self.state,
            'health_index': self.health_index,
            'max_damage': self.max_damage,
            'damage_distribution': self.damage_distribution,
            'critical_clusters': self.critical_clusters
        }

@dataclass
class FailurePredictionResult:
    """破壊予測結果"""
    failure_probability: float
    reliability_index: float
    remaining_life_cycles: Optional[float] = None
    critical_clusters: Optional[List[int]] = None
    failure_mode: Optional[str] = None

# ===============================
# Main Class
# ===============================

class MaterialAnalyticsGPU(GPUBackend):
    """
    材料特有の高度な解析機能を提供するGPUクラス
    
    既存のLambda³検出器と連携して、材料科学的な観点から
    結晶欠陥、構造安定性、破壊リスクなどを評価
    
    Parameters
    ----------
    material_type : str, optional
        材料タイプ ('SUJ2', 'AL7075', 'TI6AL4V')
    force_cpu : bool, optional
        CPUモードを強制するかどうか
    """
    
    def __init__(self, material_type: str = 'SUJ2', force_cpu: bool = False):
        """
        Parameters
        ----------
        material_type : str
            材料タイプ
        force_cpu : bool
            CPUモードを強制
        """
        super().__init__(force_cpu=force_cpu)
        
        # 材料プロパティ設定
        self.material_type = material_type
        self.material_props = MATERIAL_PROPERTIES.get(
            material_type, 
            MATERIAL_PROPERTIES['SUJ2']
        )
        
        # 解析パラメータ
        self.coherence_window = 50
        self.defect_threshold = 0.1
        
        logger.info(f"MaterialAnalyticsGPU initialized for {material_type}")
        logger.info(f"Crystal structure: {self.material_props['crystal_structure']}")
    
    # ===============================
    # Crystal Defect Analysis
    # ===============================
    
    def compute_crystal_defect_charge(self,
                                     trajectory: np.ndarray,
                                     cluster_atoms: Dict[int, List[int]],
                                     cutoff: float = 3.5) -> DefectAnalysisResult:
        """
        結晶欠陥のトポロジカルチャージを計算
        
        転位、空孔、格子間原子などの結晶欠陥を定量化。
        既存のtopology_breaks_gpu.pyとは異なり、結晶構造の欠陥に特化。
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子トラジェクトリ (n_frames, n_atoms, 3)
        cluster_atoms : Dict[int, List[int]]
            クラスター定義
        cutoff : float
            カットオフ距離 [Å]
            
        Returns
        -------
        DefectAnalysisResult
            欠陥解析結果
        """
        trajectory = self.to_gpu(trajectory)
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        # 欠陥チャージ配列
        defect_charge = self.zeros((n_frames - 1, n_clusters))
        burgers_list = []
        
        for frame in range(n_frames - 1):
            frame_burgers = []
            
            for cid, atoms in cluster_atoms.items():
                if cid >= n_clusters or len(atoms) < 10:
                    continue
                
                # Burgers回路解析
                burgers = self._compute_burgers_circuit(
                    self.to_cpu(trajectory[frame]), atoms, cutoff
                )
                frame_burgers.append(burgers)
                
                # 配位数変化
                coord_change = self._compute_coordination_change(
                    self.to_cpu(trajectory[frame]),
                    self.to_cpu(trajectory[frame + 1]),
                    atoms, cutoff
                )
                
                # 欠陥チャージ = Burgers大きさ + 配位数変化
                defect_charge[frame, cid] = (
                    np.linalg.norm(burgers) / max(1, len(atoms)) +
                    abs(coord_change) * 0.1
                )
            
            if frame_burgers:
                burgers_list.append(np.array(frame_burgers))
        
        # 累積チャージ（欠陥の蓄積）
        cumulative_charge = self.xp.cumsum(self.xp.sum(defect_charge, axis=1))
        
        return DefectAnalysisResult(
            defect_charge=self.to_cpu(defect_charge),
            cumulative_charge=self.to_cpu(cumulative_charge),
            burgers_vectors=np.array(burgers_list) if burgers_list else None,
            coordination_defects=None
        )
    
    # ===============================
    # Material State Evaluation
    # ===============================
    
    def evaluate_material_state(self,
                               damage_scores: Optional[np.ndarray] = None,
                               defect_charge: Optional[np.ndarray] = None,
                               strain_history: Optional[np.ndarray] = None,
                               critical_clusters: Optional[List[int]] = None) -> MaterialState:
        """
        材料の現在状態を評価
        
        Parameters
        ----------
        damage_scores : np.ndarray, optional
            損傷スコア (n_frames,) or (n_frames, n_clusters)
        defect_charge : np.ndarray, optional
            欠陥チャージ
        strain_history : np.ndarray, optional
            歪み履歴
        critical_clusters : List[int], optional
            臨界クラスターリスト
            
        Returns
        -------
        MaterialState
            材料状態
        """
        # 初期値
        state = 'healthy'
        health_index = 1.0
        max_damage = 0.0
        damage_distribution = None
        
        # 損傷評価
        if damage_scores is not None:
            damage_scores = self.to_gpu(damage_scores)
            
            if damage_scores.ndim > 1:
                # クラスター別損傷
                max_damage_per_cluster = self.xp.max(damage_scores, axis=0)
                max_damage = float(self.xp.max(max_damage_per_cluster))
                damage_distribution = self.to_cpu(max_damage_per_cluster)
            else:
                max_damage = float(self.xp.max(damage_scores))
            
            # 健全性指標
            health_index = max(0.0, 1.0 - max_damage)
        
        # 欠陥蓄積評価
        if defect_charge is not None:
            defect_charge = self.to_gpu(defect_charge)
            cumulative_defect = float(self.xp.sum(defect_charge))
            
            # 欠陥による健全性低下
            defect_penalty = min(0.3, cumulative_defect / 100.0)
            health_index = max(0.0, health_index - defect_penalty)
        
        # 歪み評価
        if strain_history is not None:
            strain_history = self.to_gpu(strain_history)
            max_strain = float(self.xp.max(self.xp.abs(strain_history)))
            
            # 材料プロパティ
            E = self.material_props['elastic_modulus']
            yield_strain = self.material_props['yield_strength'] / E
            
            if max_strain > yield_strain:
                # 塑性変形による健全性低下
                plastic_penalty = min(0.4, (max_strain - yield_strain) * 5.0)
                health_index = max(0.0, health_index - plastic_penalty)
        
        # 状態分類
        if health_index > 0.8:
            state = 'healthy'
        elif health_index > 0.5:
            state = 'damaged'
        elif health_index > 0.2:
            state = 'critical'
        else:
            state = 'failed'
        
        # 臨界クラスター情報を追加
        if critical_clusters is None and damage_distribution is not None:
            # 損傷の高いクラスターを臨界とみなす
            threshold = max_damage * 0.7
            critical_indices = np.where(damage_distribution > threshold)[0]
            critical_clusters = critical_indices.tolist() if len(critical_indices) > 0 else None
        
        return MaterialState(
            state=state,
            health_index=health_index,
            max_damage=max_damage,
            damage_distribution=damage_distribution,
            critical_clusters=critical_clusters
        )
    
    # ===============================
    # Structural Coherence Analysis
    # ===============================
    
    def compute_structural_coherence(self,
                                    coordination: np.ndarray,
                                    strain: np.ndarray,
                                    window: Optional[int] = None) -> np.ndarray:
        """
        構造一貫性の計算 - 熱ゆらぎと永続的構造変化を区別
        
        既存のboundary_detection_gpu.pyの境界検出とは異なり、
        構造の時間的安定性を評価。
        
        Parameters
        ----------
        coordination : np.ndarray
            配位数時系列 (n_frames, n_clusters)
        strain : np.ndarray
            歪み時系列 (n_frames, n_clusters)
        window : int, optional
            時間平均ウィンドウサイズ
            
        Returns
        -------
        np.ndarray
            構造一貫性スコア (n_frames,)
        """
        coordination = self.to_gpu(coordination)
        strain = self.to_gpu(strain)
        
        n_frames, n_clusters = coordination.shape
        coherence = self.ones(n_frames)
        
        # ウィンドウサイズ決定
        if window is None:
            window = min(self.coherence_window, n_frames // 4)
        
        # 理想配位数
        ideal_coord = self.material_props['ideal_coordination']
        
        for i in range(window, n_frames - window):
            cluster_coherence = self.zeros(n_clusters)
            
            for c in range(n_clusters):
                # 局所的な配位数統計
                local_coord = coordination[i-window:i+window, c]
                coord_mean = self.xp.mean(local_coord)
                coord_std = self.xp.std(local_coord)
                
                # 構造安定性評価
                if coord_std < 0.5 and abs(coord_mean - ideal_coord) < 1:
                    # 安定して理想構造を保持
                    cluster_coherence[c] = 1.0
                elif coord_std > 2.0:
                    # 激しく揺らいでいる（熱的）
                    cluster_coherence[c] = 0.3
                else:
                    # 構造変化中
                    cluster_coherence[c] = 1.0 - self.xp.minimum(coord_std / 3.0, 1.0)
                
                # 歪みによる補正
                if strain[i, c] > 0.05:  # 5%以上の歪み
                    cluster_coherence[c] *= (1.0 - self.xp.minimum(strain[i, c], 1.0))
            
            coherence[i] = self.xp.mean(cluster_coherence)
        
        # エッジ処理
        coherence[:window] = coherence[window]
        coherence[-window:] = coherence[-window-1]
        
        return self.to_cpu(coherence)
    
    # ===============================
    # Material Event Classification
    # ===============================
    
    def classify_material_events(self,
                                critical_events: List[Tuple[int, int]],
                                anomaly_scores: Dict[str, np.ndarray],
                                structural_coherence: np.ndarray,
                                defect_charge: Optional[np.ndarray] = None) -> List[Tuple[int, int, str]]:
        """
        材料イベントの物理的意味を分類
        
        Parameters
        ----------
        critical_events : List[Tuple[int, int]]
            臨界イベントのリスト (start, end)
        anomaly_scores : Dict[str, np.ndarray]
            異常スコア辞書
        structural_coherence : np.ndarray
            構造一貫性スコア
        defect_charge : np.ndarray, optional
            欠陥チャージ
            
        Returns
        -------
        List[Tuple[int, int, str]]
            分類されたイベント (start, end, event_type)
        """
        classified_events = []
        
        for event in critical_events:
            if not isinstance(event, tuple) or len(event) < 2:
                continue
                
            start, end = int(event[0]), int(event[1])
            
            # イベント期間の統計量
            coherence_mean = np.mean(structural_coherence[start:end+1])
            
            # 各異常スコアの最大値
            strain_max = 0.0
            if 'strain' in anomaly_scores and len(anomaly_scores['strain']) > start:
                strain_max = np.max(anomaly_scores['strain'][start:min(end+1, len(anomaly_scores['strain']))])
            
            damage_max = 0.0
            if 'damage' in anomaly_scores and len(anomaly_scores['damage']) > start:
                damage_max = np.max(anomaly_scores['damage'][start:min(end+1, len(anomaly_scores['damage']))])
            
            # 欠陥チャージの加速度
            defect_acceleration = 0.0
            if defect_charge is not None and len(defect_charge) > start:
                local_charge = defect_charge[start:min(end+1, len(defect_charge))]
                if len(local_charge) > 2:
                    defect_acceleration = np.max(np.abs(np.gradient(local_charge)))
            
            # イベント分類ロジック
            if coherence_mean < 0.4:  # 構造一貫性が低い
                if damage_max > 0.7:
                    event_type = 'crack_initiation'
                elif strain_max > 2.0:
                    event_type = 'plastic_deformation'
                elif defect_acceleration > 0.5:
                    event_type = 'dislocation_avalanche'
                else:
                    event_type = 'dislocation_nucleation'
                    
            elif coherence_mean > 0.8:  # 構造一貫性が高い
                if strain_max < 0.5:
                    event_type = 'elastic_deformation'
                else:
                    event_type = 'uniform_deformation'
                    
            else:  # 中間的な一貫性
                if defect_acceleration > 0.3:
                    event_type = 'defect_migration'
                else:
                    event_type = 'transition_state'
            
            classified_events.append((start, end, event_type))
        
        return classified_events
    
    # ===============================
    # Failure Prediction
    # ===============================
    
    def predict_failure(self,
                       strain_history: np.ndarray,
                       damage_history: Optional[np.ndarray] = None,
                       defect_charge: Optional[np.ndarray] = None,
                       loading_cycles: Optional[int] = None) -> FailurePredictionResult:
        """
        材料の破壊確率と信頼性を予測
        
        Parameters
        ----------
        strain_history : np.ndarray
            歪み履歴 (n_frames, n_clusters) or (n_frames,)
        damage_history : np.ndarray, optional
            損傷履歴
        defect_charge : np.ndarray, optional
            欠陥チャージ履歴
        loading_cycles : int, optional
            負荷サイクル数
            
        Returns
        -------
        FailurePredictionResult
            破壊予測結果
        """
        strain_history = self.to_gpu(strain_history)
        
        # 歪みの統計量
        if strain_history.ndim > 1:
            mean_strain = self.xp.mean(self.xp.abs(strain_history))
            max_strain = self.xp.max(self.xp.abs(strain_history))
            std_strain = self.xp.std(self.xp.abs(strain_history))
        else:
            mean_strain = self.xp.mean(self.xp.abs(strain_history))
            max_strain = self.xp.max(self.xp.abs(strain_history))
            std_strain = self.xp.std(self.xp.abs(strain_history))
        
        # 材料プロパティ
        E = self.material_props['elastic_modulus']
        yield_strain = self.material_props['yield_strength'] / E
        ultimate_strain = self.material_props['ultimate_strength'] / E
        
        # 破壊確率計算
        if max_strain > ultimate_strain:
            failure_prob = 1.0
            failure_mode = 'immediate_fracture'
        elif max_strain > yield_strain:
            # 塑性域での破壊確率
            failure_prob = float((max_strain - yield_strain) / (ultimate_strain - yield_strain))
            failure_mode = 'plastic_failure'
        else:
            # 弾性域（疲労考慮）
            if loading_cycles is not None and loading_cycles > 0:
                # Basquin則による疲労寿命推定
                fatigue_strength = self.material_props['fatigue_strength'] / E
                if mean_strain > fatigue_strength:
                    Nf = 1e6 * (fatigue_strength / mean_strain) ** 12  # S-N曲線
                    failure_prob = min(1.0, loading_cycles / Nf)
                    failure_mode = 'fatigue_failure'
                else:
                    failure_prob = 0.0
                    failure_mode = 'safe'
            else:
                failure_prob = 0.0
                failure_mode = 'elastic_safe'
        
        # 信頼性指標（β指標）
        if std_strain > 1e-10:
            beta = float((ultimate_strain - mean_strain) / std_strain)
        else:
            beta = 5.0  # 高信頼性
        
        # 損傷による補正
        if damage_history is not None:
            damage_history = self.to_gpu(damage_history)
            max_damage = float(self.xp.max(damage_history))
            failure_prob = min(1.0, failure_prob + max_damage * 0.3)
        
        # 欠陥蓄積による補正
        if defect_charge is not None:
            defect_charge = self.to_gpu(defect_charge)
            defect_rate = float(self.xp.mean(self.xp.gradient(defect_charge)))
            if defect_rate > 0.1:
                failure_prob = min(1.0, failure_prob + defect_rate * 0.2)
        
        # 残存寿命推定
        remaining_life = None
        if failure_mode == 'fatigue_failure' and loading_cycles is not None:
            remaining_life = max(0, Nf - loading_cycles)
        
        # 臨界クラスター特定
        critical_clusters = None
        if strain_history.ndim > 1:
            cluster_max_strain = self.xp.max(self.xp.abs(strain_history), axis=0)
            critical_threshold = yield_strain * 0.8
            critical_indices = self.xp.where(cluster_max_strain > critical_threshold)[0]
            critical_clusters = self.to_cpu(critical_indices).tolist()
        
        return FailurePredictionResult(
            failure_probability=float(failure_prob),
            reliability_index=float(beta),
            remaining_life_cycles=remaining_life,
            critical_clusters=critical_clusters,
            failure_mode=failure_mode
        )
    
    # ===============================
    # Helper Methods
    # ===============================
    
    def _compute_burgers_circuit(self,
                                positions: np.ndarray,
                                atoms: List[int],
                                cutoff: float) -> np.ndarray:
        """
        Burgers回路による転位検出
        
        Parameters
        ----------
        positions : np.ndarray
            原子位置 (n_atoms, 3)
        atoms : List[int]
            クラスター内原子インデックス
        cutoff : float
            カットオフ距離
            
        Returns
        -------
        np.ndarray
            Burgers vector (3,)
        """
        if len(atoms) < 10:
            return np.zeros(3)
        
        # クラスター中心原子
        center_idx = len(atoms) // 2
        center_atom = atoms[center_idx]
        center_pos = positions[center_atom]
        
        # 近傍原子探索
        atom_positions = positions[atoms]
        distances = np.linalg.norm(atom_positions - center_pos, axis=1)
        neighbor_mask = (distances > 0) & (distances < cutoff)
        neighbor_indices = np.array(atoms)[neighbor_mask]
        
        if len(neighbor_indices) < 6:
            return np.zeros(3)
        
        # 最近傍6原子で閉回路を構成
        sorted_indices = neighbor_indices[np.argsort(
            np.linalg.norm(positions[neighbor_indices] - center_pos, axis=1)
        )][:6]
        
        # Burgers回路計算
        burgers = np.zeros(3)
        for i in range(len(sorted_indices)):
            j = (i + 1) % len(sorted_indices)
            burgers += positions[sorted_indices[j]] - positions[sorted_indices[i]]
        
        return burgers
    
    def _compute_coordination_change(self,
                                    pos1: np.ndarray,
                                    pos2: np.ndarray,
                                    atoms: List[int],
                                    cutoff: float) -> float:
        """
        配位数変化の計算
        
        Parameters
        ----------
        pos1 : np.ndarray
            フレーム1の原子位置
        pos2 : np.ndarray
            フレーム2の原子位置
        atoms : List[int]
            クラスター内原子インデックス
        cutoff : float
            カットオフ距離
            
        Returns
        -------
        float
            平均配位数変化
        """
        if len(atoms) < 5:
            return 0.0
        
        # サンプリング（計算効率のため）
        sample_size = min(10, len(atoms))
        sample_atoms = atoms[:sample_size]
        
        coord_changes = []
        for atom in sample_atoms:
            # フレーム1の配位数
            dist1 = np.linalg.norm(pos1 - pos1[atom], axis=1)
            coord1 = np.sum((dist1 > 0) & (dist1 < cutoff))
            
            # フレーム2の配位数
            dist2 = np.linalg.norm(pos2 - pos2[atom], axis=1)
            coord2 = np.sum((dist2 > 0) & (dist2 < cutoff))
            
            coord_changes.append(abs(coord2 - coord1))
        
        return np.mean(coord_changes)
    
    # ===============================
    # Utility Methods
    # ===============================
    
    def set_material_type(self, material_type: str):
        """
        材料タイプを変更
        
        Parameters
        ----------
        material_type : str
            新しい材料タイプ
        """
        if material_type not in MATERIAL_PROPERTIES:
            warnings.warn(f"Unknown material type: {material_type}, using SUJ2 as default")
            material_type = 'SUJ2'
        
        self.material_type = material_type
        self.material_props = MATERIAL_PROPERTIES[material_type]
        logger.info(f"Material type changed to {material_type}")
    
    def get_material_info(self) -> Dict[str, Any]:
        """
        現在の材料情報を取得
        
        Returns
        -------
        Dict[str, Any]
            材料プロパティ
        """
        return {
            'type': self.material_type,
            'properties': self.material_props.copy(),
            'device': str(self.device),
            'gpu_enabled': self.is_gpu
        }

# ===============================
# Export Functions
# ===============================

def compute_crystal_defect_charge(trajectory: np.ndarray,
                                 cluster_atoms: Dict[int, List[int]],
                                 material_type: str = 'SUJ2',
                                 cutoff: float = 3.5,
                                 use_gpu: bool = True) -> CrystalDefectResult:
    """
    結晶欠陥チャージを計算する便利関数
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ
    cluster_atoms : Dict[int, List[int]]
        クラスター定義
    material_type : str
        材料タイプ
    cutoff : float
        カットオフ距離
    use_gpu : bool
        GPU使用フラグ
        
    Returns
    -------
    CrystalDefectResult
        結晶欠陥解析結果
    """
    analyzer = MaterialAnalyticsGPU(material_type, force_cpu=not use_gpu)
    result = analyzer.compute_crystal_defect_charge(trajectory, cluster_atoms, cutoff)
    
    # CrystalDefectResult形式に変換
    max_charge = float(np.max(result.cumulative_charge))
    critical_frame = int(np.argmax(result.cumulative_charge))
    
    return CrystalDefectResult(
        defect_charge=result.defect_charge,
        cumulative_charge=result.cumulative_charge,
        max_charge=max_charge,
        critical_frame=critical_frame
    )

def compute_structural_coherence(coordination: np.ndarray,
                                strain: np.ndarray,
                                material_type: str = 'SUJ2',
                                window: Optional[int] = None,
                                use_gpu: bool = True) -> np.ndarray:
    """
    構造一貫性を計算する便利関数
    
    Parameters
    ----------
    coordination : np.ndarray
        配位数時系列
    strain : np.ndarray
        歪み時系列
    material_type : str
        材料タイプ
    window : int, optional
        時間窓サイズ
    use_gpu : bool
        GPU使用フラグ
        
    Returns
    -------
    np.ndarray
        構造一貫性スコア
    """
    analyzer = MaterialAnalyticsGPU(material_type, force_cpu=not use_gpu)
    return analyzer.compute_structural_coherence(coordination, strain, window)
