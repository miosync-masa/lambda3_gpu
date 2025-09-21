#!/usr/bin/env python3
"""
Material Analytics GPU - Advanced Material-Specific Analysis (REFACTORED)
==========================================================================

材料解析に特化した高度な解析機能群
結晶欠陥定量化、構造一貫性評価、破壊予測など

Lambda³ GPU既存モジュールと連携して材料特有の現象を解析

Version: 2.0.0 - Refactored with unified material database
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

# Material Database import (統一データベース)
from .material_database import MATERIAL_DATABASE, get_material_parameters

# Logger設定
logger = logging.getLogger(__name__)

# ===============================
# Data Classes (変更なし)
# ===============================

@dataclass
class DefectAnalysisResult:
    """結晶欠陥解析結果"""
    defect_charge: np.ndarray           # 欠陥チャージ (n_frames, n_clusters)
    cumulative_charge: np.ndarray       # 累積チャージ (n_frames,)
    burgers_vectors: Optional[np.ndarray] = None  # Burgers vectors
    coordination_defects: Optional[np.ndarray] = None  # 配位数欠陥
    defect_density: Optional[float] = None  # 欠陥密度

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
    failure_mode: Optional[str] = None  # 破壊モード
    reliability_beta: Optional[float] = None  # 信頼性β指標
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（後方互換性のため）"""
        return {
            'state': self.state,
            'health_index': self.health_index,
            'max_damage': self.max_damage,
            'damage_distribution': self.damage_distribution,
            'critical_clusters': self.critical_clusters,
            'failure_mode': self.failure_mode,
            'reliability_beta': self.reliability_beta
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
# Main Class (リファクタリング版)
# ===============================

class MaterialAnalyticsGPU(GPUBackend):
    """
    材料特有の高度な解析機能を提供するGPUクラス（リファクタリング版）
    
    統一材料データベース(MATERIAL_DATABASE)を使用
    物理的に正確な欠陥解析と材料状態評価を実装
    
    Parameters
    ----------
    material_type : str, optional
        材料タイプ ('SUJ2', 'AL7075', 'Ti6Al4V', 'SS316L')
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
        
        # 材料データベースから取得（統一版）
        self.material_type = material_type
        self.material_props = get_material_parameters(material_type)
        
        # 解析パラメータ
        self.coherence_window = 50
        self.defect_threshold = 0.1
        
        logger.info(f"MaterialAnalyticsGPU initialized for {material_type}")
        logger.info(f"Crystal structure: {self.material_props['crystal_structure']}")
    
    # ===============================
    # Crystal Defect Analysis (正規化版)
    # ===============================
    
    def compute_normalized_defect_charge(self,
                                        trajectory: np.ndarray,
                                        cluster_atoms: Dict[int, List[int]],
                                        cutoff: float = 3.5) -> DefectAnalysisResult:
        """
        正規化された欠陥チャージ計算（改善版）
        
        転位、空孔、格子間原子などの結晶欠陥を定量化し、
        値を0-1範囲に正規化して数値の爆発を防ぐ
        
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
            欠陥解析結果（正規化済み）
        """
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        # 材料定数
        lattice = self.material_props.get('lattice_constant', 2.87)
        ideal_coord = self.material_props.get('ideal_coordination', 8)
        
        # 欠陥チャージ配列
        defect_charge = np.zeros((n_frames - 1, n_clusters))
        
        for frame in range(n_frames - 1):
            pos_current = trajectory[frame]
            pos_next = trajectory[frame + 1]
            
            for cid, atoms in cluster_atoms.items():
                if cid >= n_clusters or len(atoms) < 5:
                    continue
                
                # 1. Burgersベクトル計算（簡略版）
                cluster_center = np.mean(pos_current[atoms], axis=0)
                cluster_displacements = pos_next[atoms] - pos_current[atoms]
                burgers_magnitude = np.linalg.norm(np.sum(cluster_displacements, axis=0))
                burgers_normalized = burgers_magnitude / (lattice * len(atoms))
                
                # 2. 配位数変化の正規化計算
                coord_changes = []
                sample_size = min(10, len(atoms))
                sample_atoms = atoms[:sample_size] if isinstance(atoms, list) else list(atoms)[:sample_size]
                
                for atom in sample_atoms:
                    # 現在フレーム配位数
                    distances_current = np.linalg.norm(pos_current - pos_current[atom], axis=1)
                    coord_current = np.sum((distances_current > 0) & (distances_current < cutoff))
                    
                    # 次フレーム配位数
                    distances_next = np.linalg.norm(pos_next - pos_next[atom], axis=1)
                    coord_next = np.sum((distances_next > 0) & (distances_next < cutoff))
                    
                    # 理想配位数との差を正規化
                    coord_change = abs(coord_next - coord_current) / ideal_coord
                    coord_changes.append(coord_change)
                
                coord_change_normalized = np.mean(coord_changes) if coord_changes else 0.0
                
                # 3. 欠陥チャージ（0-1範囲に正規化）
                # tanhで飽和させて爆発を防ぐ
                raw_charge = 0.3 * burgers_normalized + 0.2 * coord_change_normalized
                defect_charge[frame, cid] = np.tanh(raw_charge)
        
        # 累積チャージ（平均化して爆発を防ぐ）
        cumulative_charge = np.cumsum(np.mean(defect_charge, axis=1))
        
        # 全体の欠陥密度
        total_defect_atoms = sum(
            len(atoms) for cid, atoms in cluster_atoms.items() 
            if cid != 0  # クラスター0（健全領域）を除外
        )
        defect_density = total_defect_atoms / n_atoms if n_atoms > 0 else 0.0
        
        return DefectAnalysisResult(
            defect_charge=defect_charge,
            cumulative_charge=cumulative_charge,
            burgers_vectors=None,  # 簡略化のため省略
            coordination_defects=None,
            defect_density=defect_density
        )
    
    # 旧メソッドとの互換性保持（内部で正規化版を呼ぶ）
    def compute_crystal_defect_charge(self,
                                     trajectory: np.ndarray,
                                     cluster_atoms: Dict[int, List[int]],
                                     cutoff: float = 3.5) -> DefectAnalysisResult:
        """旧メソッド名との互換性保持"""
        return self.compute_normalized_defect_charge(trajectory, cluster_atoms, cutoff)
    
    # ===============================
    # Material State Evaluation (物理的改善版)
    # ===============================
    
    def evaluate_material_state(self,
                               stress_history: Optional[np.ndarray] = None,
                               defect_charge: Optional[np.ndarray] = None,
                               energy_balance: Optional[Dict] = None,
                               damage_scores: Optional[np.ndarray] = None,
                               strain_history: Optional[np.ndarray] = None,
                               critical_clusters: Optional[List[int]] = None) -> MaterialState:
        """
        材料の健全性評価（物理的改善版）
        
        応力、欠陥、エネルギー状態を総合的に評価
        
        Parameters
        ----------
        stress_history : np.ndarray, optional
            応力履歴 (n_frames, 1) or (n_frames,)
        defect_charge : np.ndarray, optional
            欠陥チャージ (n_frames, n_clusters)
        energy_balance : Dict, optional
            エネルギーバランス結果
        damage_scores : np.ndarray, optional
            損傷スコア（後方互換性）
        strain_history : np.ndarray, optional
            歪み履歴（後方互換性）
        critical_clusters : List[int], optional
            臨界クラスターリスト
            
        Returns
        -------
        MaterialState
            材料状態
        """
        # 初期化
        health_index = 1.0
        state = 'healthy'
        failure_probability = 0.0
        failure_mode = None
        
        # 材料パラメータ取得
        E = self.material_props.get('elastic_modulus', 210.0)
        sigma_y = self.material_props.get('yield_strength', 1.5)
        sigma_u = self.material_props.get('ultimate_strength', 2.0)
        
        # 1. 応力ベース評価
        if stress_history is not None:
            stress_history = np.asarray(stress_history)
            if stress_history.ndim == 2 and stress_history.shape[1] == 1:
                stress_history = stress_history.squeeze()
            
            max_stress = float(np.max(np.abs(stress_history)))
            mean_stress = float(np.mean(np.abs(stress_history)))
            
            if max_stress > sigma_u:
                health_index *= 0.1
                failure_probability = 1.0
                state = 'failed'
                failure_mode = 'immediate_fracture'
            elif max_stress > sigma_y:
                plastic_factor = (max_stress - sigma_y) / (sigma_u - sigma_y)
                health_index *= (1 - 0.5 * plastic_factor)
                failure_probability = plastic_factor
                failure_mode = 'plastic_failure'
                if health_index < 0.5:
                    state = 'critical'
                else:
                    state = 'damaged'
        else:
            max_stress = 0.0
            mean_stress = 0.0
        
        # 2. 欠陥ベース評価
        if defect_charge is not None and defect_charge.size > 0:
            max_defect = float(np.max(defect_charge))
            mean_defect = float(np.mean(defect_charge))
            
            # 欠陥による健全性低下（最大30%）
            defect_penalty = min(0.3, mean_defect)
            health_index *= (1 - defect_penalty)
            
            # 欠陥加速度（危険度指標）
            if len(defect_charge) > 2:
                if defect_charge.ndim == 2:
                    defect_evolution = np.mean(defect_charge, axis=1)
                else:
                    defect_evolution = defect_charge
                
                defect_acceleration = np.max(np.abs(np.gradient(defect_evolution)))
                if defect_acceleration > 0.1:
                    health_index *= 0.8
                    if state == 'healthy':
                        state = 'damaged'
                    if failure_mode is None:
                        failure_mode = 'defect_driven_failure'
        
        # 3. エネルギー/相状態評価
        if energy_balance is not None:
            phase = energy_balance.get('phase_state', 'solid')
            lindemann = energy_balance.get('lindemann_parameter', 0.0)
            lindemann_criterion = self.material_props.get('lindemann_criterion', 0.1)
            
            if phase == 'liquid' or lindemann > lindemann_criterion:
                health_index *= 0.2
                state = 'failed'
                failure_probability = 1.0
                if failure_mode is None:
                    failure_mode = 'melting'
            elif phase == 'transition':
                health_index *= 0.7
                if state == 'healthy':
                    state = 'damaged'
        
        # 4. 後方互換性：damage_scoresの処理
        if damage_scores is not None:
            damage_scores = np.asarray(damage_scores)
            max_damage = float(np.max(damage_scores))
            health_index *= max(0.0, 1.0 - max_damage)
        
        # 5. 後方互換性：strain_historyの処理
        if strain_history is not None and stress_history is None:
            # 歪みから応力を推定（簡易版）
            strain_history = np.asarray(strain_history)
            max_strain = float(np.max(np.abs(strain_history)))
            
            yield_strain = sigma_y / E
            if max_strain > yield_strain:
                plastic_penalty = min(0.4, (max_strain - yield_strain) * 5.0)
                health_index *= (1 - plastic_penalty)
        
        # 6. 最終的な健全性指標の正規化
        health_index = max(0.0, min(1.0, health_index))
        
        # 7. 信頼性指標（β指標）
        if mean_stress > 0:
            reliability_beta = (sigma_u - mean_stress) / (sigma_u * 0.1)  # 簡略化
        else:
            reliability_beta = 5.0
        
        # 8. 臨界クラスターの決定
        if critical_clusters is None and defect_charge is not None and defect_charge.ndim == 2:
            # 欠陥の高いクラスターを臨界とみなす
            cluster_max_defect = np.max(defect_charge, axis=0)
            threshold = np.mean(cluster_max_defect) + 2 * np.std(cluster_max_defect)
            critical_indices = np.where(cluster_max_defect > threshold)[0]
            critical_clusters = critical_indices.tolist() if len(critical_indices) > 0 else None
        
        return MaterialState(
            state=state,
            health_index=health_index,
            max_damage=1.0 - health_index,
            damage_distribution=None,  # 簡略化
            critical_clusters=critical_clusters,
            failure_mode=failure_mode,
            reliability_beta=reliability_beta
        )
    
    # ===============================
    # Structural Coherence Analysis (変更なし)
    # ===============================
    
    def compute_structural_coherence(self,
                                    coordination: np.ndarray,
                                    strain: np.ndarray,
                                    window: Optional[int] = None) -> np.ndarray:
        """
        構造一貫性の計算 - 熱ゆらぎと永続的構造変化を区別
        """
        coordination = self.to_gpu(coordination)
        strain = self.to_gpu(strain)
        
        n_frames, n_clusters = coordination.shape
        coherence = self.ones(n_frames)
        
        # ウィンドウサイズ決定
        if window is None:
            window = min(self.coherence_window, n_frames // 4)
        
        # 理想配位数
        ideal_coord = self.material_props.get('ideal_coordination', 8)
        
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
    # Material Event Classification (変更なし)
    # ===============================
    
    def classify_material_events(self,
                                critical_events: List[Tuple[int, int]],
                                anomaly_scores: Dict[str, np.ndarray],
                                structural_coherence: np.ndarray,
                                defect_charge: Optional[np.ndarray] = None) -> List[Tuple[int, int, str]]:
        """材料イベントの物理的意味を分類"""
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
    # Failure Prediction (材料データベース使用版)
    # ===============================
    
    def predict_failure(self,
                       strain_history: np.ndarray,
                       damage_history: Optional[np.ndarray] = None,
                       defect_charge: Optional[np.ndarray] = None,
                       loading_cycles: Optional[int] = None) -> FailurePredictionResult:
        """材料の破壊確率と信頼性を予測"""
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
        
        # 材料プロパティ（統一データベースから）
        E = self.material_props.get('elastic_modulus', 210.0)
        yield_strain = self.material_props.get('yield_strength', 1.5) / E
        ultimate_strain = self.material_props.get('ultimate_strength', 2.0) / E
        fatigue_strength = self.material_props.get('fatigue_strength', 0.7)
        
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
                fatigue_strain = fatigue_strength / E
                if mean_strain > fatigue_strain:
                    Nf = 1e6 * (fatigue_strain / mean_strain) ** 12  # S-N曲線
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
            if defect_charge.ndim == 2:
                defect_evolution = self.xp.mean(defect_charge, axis=1)
            else:
                defect_evolution = defect_charge
            
            defect_rate = float(self.xp.mean(self.xp.gradient(defect_evolution)))
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
    # Helper Methods（変更なし）
    # ===============================
    
    def _compute_burgers_circuit(self,
                                positions: np.ndarray,
                                atoms: List[int],
                                cutoff: float) -> np.ndarray:
        """Burgers回路による転位検出"""
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
        """配位数変化の計算"""
        if len(atoms) < 5:
            return 0.0
        
        # サンプリング（計算効率のため）
        sample_size = min(10, len(atoms))
        sample_atoms = atoms[:sample_size] if isinstance(atoms, list) else list(atoms)[:sample_size]
        
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
        """材料タイプを変更"""
        self.material_type = material_type
        self.material_props = get_material_parameters(material_type)
        logger.info(f"Material type changed to {material_type}")
    
    def get_material_info(self) -> Dict[str, Any]:
        """現在の材料情報を取得"""
        return {
            'type': self.material_type,
            'properties': self.material_props.copy(),
            'device': str(self.device),
            'gpu_enabled': self.is_gpu
        }

# ===============================
# Export Functions（統一データベース使用版）
# ===============================

def compute_crystal_defect_charge(trajectory: np.ndarray,
                                 cluster_atoms: Dict[int, List[int]],
                                 material_type: str = 'SUJ2',
                                 cutoff: float = 3.5,
                                 use_gpu: bool = True) -> CrystalDefectResult:
    """
    結晶欠陥チャージを計算する便利関数（正規化版）
    """
    analyzer = MaterialAnalyticsGPU(material_type, force_cpu=not use_gpu)
    result = analyzer.compute_normalized_defect_charge(trajectory, cluster_atoms, cutoff)
    
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
    """構造一貫性を計算する便利関数"""
    analyzer = MaterialAnalyticsGPU(material_type, force_cpu=not use_gpu)
    return analyzer.compute_structural_coherence(coordination, strain, window)
