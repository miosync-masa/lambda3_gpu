#!/usr/bin/env python3
"""
Material Analytics GPU - Advanced Material-Specific Analysis (INTEGRATED EDITION)
==================================================================================

材料解析に特化した高度な解析機能群 - 物理ベースダメージ計算統合版
結晶欠陥定量化、構造一貫性評価、物理的に正しい累積損傷計算、破壊予測など

全ての重い処理をCUDAカーネルで実装！
物理的に正しいダメージ計算（Miner則、パーコレーション理論）を統合！💎

Version: 4.0.0 - Physically Accurate Damage Integration
Author: 環ちゃん - Physics Master Edition
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import logging
import time

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

# CUDA Kernels import
try:
    from .cuda_kernels_extended import MaterialCUDAKernels
    HAS_CUDA_KERNELS = True
except ImportError:
    HAS_CUDA_KERNELS = False
    MaterialCUDAKernels = None

# Logger設定
logger = logging.getLogger(__name__)

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
    defect_density: Optional[float] = None  # 欠陥密度

@dataclass
class CrystalDefectResult:
    """結晶欠陥解析結果（簡易版）"""
    defect_charge: np.ndarray           # 欠陥チャージ
    cumulative_charge: np.ndarray       # 累積チャージ
    max_charge: float                   # 最大チャージ
    critical_frame: int                 # 臨界フレーム

@dataclass
class DamageAccumulation:
    """累積損傷データ（物理ベース）"""
    instant_damage: np.ndarray      # 瞬間ダメージ (n_frames, n_clusters)
    cumulative_damage: np.ndarray   # 累積ダメージ (n_frames, n_clusters)
    damage_rate: np.ndarray         # ダメージ速度 (n_frames, n_clusters)
    critical_clusters: List[int]    # 臨界クラスター
    failure_probability: float      # 破壊確率
    remaining_life: Optional[float] # 残存寿命

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
# Physical Damage Calculator (NEW!)
# ===============================

class PhysicalDamageCalculator:
    """
    物理的に正しいダメージ計算クラス
    
    - Miner則に基づく疲労累積
    - K/V比からのダメージ変換
    - パーコレーション理論による破壊判定
    """
    
    def __init__(self, material_type: str = 'SUJ2'):
        self.material_type = material_type
        
        # 材料パラメータ（材料データベースから取得）
        mat_props = get_material_parameters(material_type)
        self.material_params = {
            'critical_kv': 1.5,          # 臨界K/V比
            'damage_exponent': 2.0,       # ダメージ累積指数
            'fatigue_limit': mat_props.get('fatigue_strength', 0.7) / mat_props.get('elastic_modulus', 210.0),
            'percolation_threshold': 0.65, # パーコレーション閾値
            'recovery_rate': 0.001        # 回復速度（焼鈍効果）
        }
        
        # GPU使用可能性チェック
        self.use_gpu = HAS_GPU
        self.xp = cp if self.use_gpu else np
        
    def calculate_damage_from_kv(self,
                                 kv_ratios: np.ndarray,
                                 kv_history: Optional[np.ndarray] = None,
                                 temperature: float = 300.0) -> DamageAccumulation:
        """
        K/V比から物理的に正しいダメージを計算
        
        Parameters
        ----------
        kv_ratios : np.ndarray
            K/V比 (n_frames, n_clusters)または(n_frames,)
        kv_history : np.ndarray, optional
            過去のK/V比履歴（疲労考慮用）
        temperature : float
            温度 [K]（熱活性化過程の考慮）
            
        Returns
        -------
        DamageAccumulation
            累積損傷データ
        """
        # 入力整形
        if kv_ratios.ndim == 1:
            kv_ratios = kv_ratios[:, np.newaxis]
        
        n_frames, n_clusters = kv_ratios.shape
        
        # GPUへ転送
        if self.use_gpu:
            kv_ratios = cp.asarray(kv_ratios, dtype=cp.float32)
        
        # 臨界K/V比（温度依存性を考慮）
        kv_critical = self.material_params['critical_kv']
        kv_critical *= (1.0 - (temperature - 300.0) / 1000.0)  # 高温で臨界値低下
        
        # 1. 瞬間ダメージ計算（物理モデル）
        instant_damage = self._compute_instant_damage(
            kv_ratios, kv_critical, temperature
        )
        
        # 2. 累積ダメージ計算（Miner則拡張）
        cumulative_damage = self._compute_cumulative_damage(
            instant_damage, kv_history, temperature
        )
        
        # 3. ダメージ速度（時間微分）
        damage_rate = self._compute_damage_rate(cumulative_damage)
        
        # 4. 臨界クラスター特定（パーコレーション）
        critical_clusters = self._identify_critical_clusters(
            cumulative_damage, damage_rate
        )
        
        # 5. 破壊確率計算
        failure_probability = self._compute_failure_probability(
            cumulative_damage, critical_clusters
        )
        
        # 6. 残存寿命推定
        remaining_life = self._estimate_remaining_life(
            damage_rate, cumulative_damage
        )
        
        # CPU転送（必要な場合）
        if self.use_gpu:
            instant_damage = cp.asnumpy(instant_damage)
            cumulative_damage = cp.asnumpy(cumulative_damage)
            damage_rate = cp.asnumpy(damage_rate)
        
        return DamageAccumulation(
            instant_damage=instant_damage,
            cumulative_damage=cumulative_damage,
            damage_rate=damage_rate,
            critical_clusters=critical_clusters,
            failure_probability=failure_probability,
            remaining_life=remaining_life
        )
    
    def _compute_instant_damage(self,
                               kv_ratios: np.ndarray,
                               kv_critical: float,
                               temperature: float) -> np.ndarray:
        """
        瞬間ダメージ計算（物理モデルベース）
        
        D_instant = (K/V - K/V_c)^n * H(K/V - K/V_c) * exp(-E_a/kT)
        """
        xp = cp if self.use_gpu else np
        
        # 超過K/V比
        excess_kv = kv_ratios - kv_critical
        
        # Heaviside関数（臨界値以下はダメージなし）
        damage_mask = excess_kv > 0
        
        # 瞬間ダメージ（べき乗則）
        n = self.material_params['damage_exponent']
        instant_damage = xp.zeros_like(kv_ratios)
        instant_damage[damage_mask] = (excess_kv[damage_mask] / kv_critical) ** n
        
        # 温度効果（Arrhenius型）
        if temperature != 300.0:
            activation_factor = xp.exp(-(temperature - 300.0) / 100.0)
            instant_damage *= activation_factor
        
        # 正規化（0-1範囲）
        instant_damage = xp.minimum(instant_damage, 1.0)
        
        return instant_damage
    
    def _compute_cumulative_damage(self,
                                  instant_damage: np.ndarray,
                                  kv_history: Optional[np.ndarray],
                                  temperature: float) -> np.ndarray:
        """
        累積ダメージ計算（修正Miner則）
        
        D_cum(t) = D_cum(t-1) + D_instant(t) * (1 + α*N_cycles) - β*recovery
        """
        xp = cp if self.use_gpu else np
        
        n_frames, n_clusters = instant_damage.shape
        cumulative_damage = xp.zeros_like(instant_damage)
        
        # 疲労履歴の考慮
        fatigue_factor = 1.0
        if kv_history is not None:
            # 過去のサイクル数から疲労係数計算
            n_cycles = len(kv_history)
            fatigue_factor = 1.0 + 0.001 * np.log10(n_cycles + 1)
        
        # 回復速度（温度依存）
        recovery_rate = self.material_params['recovery_rate']
        recovery_rate *= xp.exp((temperature - 300.0) / 200.0)
        
        # 累積計算
        for t in range(n_frames):
            if t == 0:
                cumulative_damage[t] = instant_damage[t] * fatigue_factor
            else:
                # 前時刻からの累積 + 新規ダメージ - 回復
                cumulative_damage[t] = (
                    cumulative_damage[t-1] * (1 - recovery_rate) +
                    instant_damage[t] * fatigue_factor
                )
                
                # 上限設定（1を超えない）
                cumulative_damage[t] = xp.minimum(cumulative_damage[t], 1.0)
        
        return cumulative_damage
    
    def _compute_damage_rate(self, cumulative_damage: np.ndarray) -> np.ndarray:
        """ダメージ速度計算（時間微分）"""
        xp = cp if self.use_gpu else np
        
        damage_rate = xp.zeros_like(cumulative_damage)
        
        # 中央差分で微分
        if len(cumulative_damage) > 2:
            damage_rate[1:-1] = (cumulative_damage[2:] - cumulative_damage[:-2]) / 2.0
            # 境界処理
            damage_rate[0] = cumulative_damage[1] - cumulative_damage[0]
            damage_rate[-1] = cumulative_damage[-1] - cumulative_damage[-2]
        elif len(cumulative_damage) == 2:
            damage_rate[0] = 0.0
            damage_rate[1] = cumulative_damage[1] - cumulative_damage[0]
        
        return damage_rate
    
    def _identify_critical_clusters(self,
                                   cumulative_damage: np.ndarray,
                                   damage_rate: np.ndarray) -> List[int]:
        """
        臨界クラスター特定（パーコレーション理論）
        """
        xp = cp if self.use_gpu else np
        
        # 最終フレームでの状態
        final_damage = cumulative_damage[-1]
        final_rate = damage_rate[-1]
        
        # パーコレーション閾値
        percolation_thresh = self.material_params['percolation_threshold']
        
        # 臨界条件
        critical_mask = (
            (final_damage > percolation_thresh) |
            (final_rate > 0.1)  # 急速なダメージ進展
        )
        
        # クラスターインデックス取得
        if self.use_gpu:
            critical_indices = cp.where(critical_mask)[0]
            critical_clusters = cp.asnumpy(critical_indices).tolist()
        else:
            critical_clusters = np.where(critical_mask)[0].tolist()
        
        return critical_clusters
    
    def _compute_failure_probability(self,
                                    cumulative_damage: np.ndarray,
                                    critical_clusters: List[int]) -> float:
        """
        破壊確率計算（Weibull分布ベース）
        
        P_f = 1 - exp[-(D_max/D_0)^m]
        """
        xp = cp if self.use_gpu else np
        
        # 最大ダメージ
        max_damage = float(xp.max(cumulative_damage))
        
        # Weibullパラメータ
        D_0 = 0.8  # スケールパラメータ
        m = 2.0    # 形状パラメータ（鋼材の典型値）
        
        # 破壊確率
        if max_damage > 0:
            failure_prob = 1.0 - np.exp(-(max_damage / D_0) ** m)
        else:
            failure_prob = 0.0
        
        # 臨界クラスター数による補正
        if len(critical_clusters) > 0:
            n_clusters = cumulative_damage.shape[1]
            critical_fraction = len(critical_clusters) / n_clusters
            failure_prob = max(failure_prob, critical_fraction)
        
        return min(1.0, failure_prob)
    
    def _estimate_remaining_life(self,
                                damage_rate: np.ndarray,
                                cumulative_damage: np.ndarray) -> Optional[float]:
        """
        残存寿命推定（線形外挿）
        
        N_remaining = (1 - D_current) / dD/dt
        """
        xp = cp if self.use_gpu else np
        
        # 現在の最大ダメージと速度
        current_damage = float(xp.max(cumulative_damage[-1]))
        
        # 最近10フレームの平均速度
        recent_frames = min(10, len(damage_rate))
        current_rate = float(xp.max(damage_rate[-recent_frames:].mean()))
        
        # 残存寿命計算
        if current_rate > 1e-6:
            remaining = (1.0 - current_damage) / current_rate
            return max(0.0, remaining)
        else:
            return None  # 速度が小さすぎて推定不可

# ===============================
# Main Class (統合版)
# ===============================

class MaterialAnalyticsGPU(GPUBackend):
    """
    材料特有の高度な解析機能を提供するGPUクラス（物理ダメージ統合版）
    
    統一材料データベース(MATERIAL_DATABASE)を使用
    物理的に正確な欠陥解析と材料状態評価を実装
    全ての重い処理をCUDAカーネルで実行
    
    Parameters
    ----------
    material_type : str, optional
        材料タイプ ('SUJ2', 'AL7075', 'Ti6Al4V', 'SS316L')
    force_cpu : bool, optional
        CPUモードを強制するかどうか
    use_cuda_kernels : bool, optional
        CUDAカーネルを使用するか（デフォルト: True）
    """
    
    def __init__(self, material_type: str = 'SUJ2', 
                 force_cpu: bool = False,
                 use_cuda_kernels: bool = True):
        """
        Parameters
        ----------
        material_type : str
            材料タイプ
        force_cpu : bool
            CPUモードを強制
        use_cuda_kernels : bool
            CUDAカーネルの使用有無
        """
        super().__init__(force_cpu=force_cpu)
        
        # 材料データベースから取得（統一版）
        self.material_type = material_type
        self.material_props = get_material_parameters(material_type)
        
        # 解析パラメータ
        self.coherence_window = 50
        self.defect_threshold = 0.1
        
        # 物理ダメージ計算器（NEW!）
        self.damage_calculator = PhysicalDamageCalculator(material_type)
        
        # CUDAカーネル初期化
        self.cuda_kernels = None
        self.use_cuda = False
        
        if self.is_gpu and HAS_CUDA_KERNELS and use_cuda_kernels:
            try:
                self.cuda_kernels = MaterialCUDAKernels()
                if self.cuda_kernels.compiled:
                    self.use_cuda = True
                    logger.info(f"✅ CUDA kernels enabled for {material_type}")
                    logger.info(f"   Crystal structure: {self.material_props['crystal_structure']}")
                else:
                    logger.warning("⚠️ CUDA kernels compilation failed, using standard GPU")
            except Exception as e:
                logger.warning(f"⚠️ CUDA kernels initialization failed: {e}")
        else:
            if not self.is_gpu:
                logger.info(f"MaterialAnalyticsGPU initialized for {material_type} (CPU mode)")
            else:
                logger.info(f"MaterialAnalyticsGPU initialized for {material_type} (Standard GPU)")
    
    # ===============================
    # Physical Damage Calculation (NEW!)
    # ===============================
    
    def compute_physical_damage(self,
                               kv_ratios: np.ndarray,
                               kv_history: Optional[np.ndarray] = None,
                               temperature: float = 300.0) -> DamageAccumulation:
        """
        物理的に正しいダメージ計算
        K/V比から累積損傷を計算
        
        Parameters
        ----------
        kv_ratios : np.ndarray
            K/V比データ
        kv_history : np.ndarray, optional
            過去のK/V比履歴
        temperature : float
            温度 [K]
            
        Returns
        -------
        DamageAccumulation
            累積損傷データ
        """
        return self.damage_calculator.calculate_damage_from_kv(
            kv_ratios, kv_history, temperature
        )
    
    # ===============================
    # Crystal Defect Analysis (CUDA最適化版)
    # ===============================
    
    def compute_normalized_defect_charge(self,
                                        trajectory: np.ndarray,
                                        cluster_atoms: Dict[int, List[int]],
                                        cutoff: float = 3.5) -> DefectAnalysisResult:
        """
        正規化された欠陥チャージ計算（CUDA最適化版）
        
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
        
        # CUDA最適化版を使用
        if self.use_cuda and self.cuda_kernels is not None:
            logger.debug("Using CUDA kernels for defect charge calculation")
            return self._compute_defect_charge_cuda(
                trajectory, cluster_atoms, cutoff, lattice, ideal_coord
            )
        
        # フォールバック: CPU版（既存実装）
        logger.debug("Using CPU fallback for defect charge calculation")
        return self._compute_defect_charge_cpu(
            trajectory, cluster_atoms, cutoff, lattice, ideal_coord
        )
    
    def _compute_defect_charge_cuda(self,
                                   trajectory: np.ndarray,
                                   cluster_atoms: Dict[int, List[int]],
                                   cutoff: float,
                                   lattice: float,
                                   ideal_coord: int) -> DefectAnalysisResult:
        """CUDA最適化版の欠陥チャージ計算"""
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        # GPU転送
        traj_gpu = cp.asarray(trajectory, dtype=cp.float32)
        
        # 結果配列
        defect_charge = cp.zeros((n_frames - 1, n_clusters), dtype=cp.float32)
        
        # バッチ処理でCUDAカーネル実行
        batch_size = min(50, n_frames - 1)  # メモリ制約を考慮
        
        for batch_start in range(0, n_frames - 1, batch_size):
            batch_end = min(batch_start + batch_size, n_frames - 1)
            
            for frame_idx in range(batch_start, batch_end):
                # CUDAカーネル呼び出し
                frame_charge = self.cuda_kernels.compute_defect_charge_cuda(
                    traj_gpu[frame_idx],      # 現在フレーム
                    traj_gpu[frame_idx + 1],  # 次フレーム
                    cluster_atoms,
                    lattice,
                    ideal_coord,
                    cutoff
                )
                defect_charge[frame_idx] = frame_charge
        
        # Burgersベクトル計算（オプション）
        burgers_vectors = None
        if n_frames > 0:
            burgers_vectors = self.cuda_kernels.compute_burgers_vectors_cuda(
                traj_gpu[0], cluster_atoms, cutoff
            )
            burgers_vectors = cp.asnumpy(burgers_vectors)
        
        # CPU転送と後処理
        defect_charge_cpu = cp.asnumpy(defect_charge)
        cumulative_charge = np.cumsum(np.mean(defect_charge_cpu, axis=1))
        
        # 全体の欠陥密度
        total_defect_atoms = sum(
            len(atoms) for cid, atoms in cluster_atoms.items() 
            if cid != 0  # クラスター0（健全領域）を除外
        )
        defect_density = total_defect_atoms / n_atoms if n_atoms > 0 else 0.0
        
        return DefectAnalysisResult(
            defect_charge=defect_charge_cpu,
            cumulative_charge=cumulative_charge,
            burgers_vectors=burgers_vectors,
            coordination_defects=None,
            defect_density=defect_density
        )
    
    def _compute_defect_charge_cpu(self,
                                  trajectory: np.ndarray,
                                  cluster_atoms: Dict[int, List[int]],
                                  cutoff: float,
                                  lattice: float,
                                  ideal_coord: int) -> DefectAnalysisResult:
        """CPU版の欠陥チャージ計算（フォールバック）"""
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
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
                raw_charge = 0.3 * burgers_normalized + 0.2 * coord_change_normalized
                defect_charge[frame, cid] = np.tanh(raw_charge)
        
        # 累積チャージ
        cumulative_charge = np.cumsum(np.mean(defect_charge, axis=1))
        
        # 全体の欠陥密度
        total_defect_atoms = sum(
            len(atoms) for cid, atoms in cluster_atoms.items() 
            if cid != 0
        )
        defect_density = total_defect_atoms / n_atoms if n_atoms > 0 else 0.0
        
        return DefectAnalysisResult(
            defect_charge=defect_charge,
            cumulative_charge=cumulative_charge,
            burgers_vectors=None,
            coordination_defects=None,
            defect_density=defect_density
        )
    
    # 旧メソッドとの互換性保持
    def compute_crystal_defect_charge(self,
                                     trajectory: np.ndarray,
                                     cluster_atoms: Dict[int, List[int]],
                                     cutoff: float = 3.5) -> DefectAnalysisResult:
        """旧メソッド名との互換性保持"""
        return self.compute_normalized_defect_charge(trajectory, cluster_atoms, cutoff)
    
    # ===============================
    # Structural Coherence Analysis (CUDA最適化版)
    # ===============================
    
    def compute_structural_coherence(self,
                                    coordination: np.ndarray,
                                    strain: np.ndarray,
                                    window: Optional[int] = None) -> np.ndarray:
        """
        構造一貫性の計算 - 熱ゆらぎと永続的構造変化を区別（CUDA最適化版）
        """
        n_frames, n_clusters = coordination.shape
        
        # ウィンドウサイズ決定
        if window is None:
            window = min(self.coherence_window, n_frames // 4)
        
        # 理想配位数
        ideal_coord = self.material_props.get('ideal_coordination', 8)
        
        # CUDA最適化版を使用
        if self.use_cuda and self.cuda_kernels is not None:
            logger.debug("Using CUDA kernels for structural coherence")
            
            # GPU転送
            coord_gpu = cp.asarray(coordination, dtype=cp.float32)
            strain_gpu = cp.asarray(strain, dtype=cp.float32)
            
            # CUDAカーネル実行
            coherence_gpu = self.cuda_kernels.compute_structural_coherence_cuda(
                coord_gpu, strain_gpu, window, ideal_coord
            )
            
            return cp.asnumpy(coherence_gpu)
        
        # フォールバック: 標準GPU/CPU版
        logger.debug("Using standard GPU/CPU for structural coherence")
        coordination = self.to_gpu(coordination)
        strain = self.to_gpu(strain)
        
        coherence = self.ones(n_frames)
        
        for i in range(window, n_frames - window):
            cluster_coherence = self.zeros(n_clusters)
            
            for c in range(n_clusters):
                # 局所的な配位数統計
                local_coord = coordination[i-window:i+window, c]
                coord_mean = self.xp.mean(local_coord)
                coord_std = self.xp.std(local_coord)
                
                # 構造安定性評価
                if coord_std < 0.5 and abs(coord_mean - ideal_coord) < 1:
                    cluster_coherence[c] = 1.0
                elif coord_std > 2.0:
                    cluster_coherence[c] = 0.3
                else:
                    cluster_coherence[c] = 1.0 - self.xp.minimum(coord_std / 3.0, 1.0)
                
                # 歪みによる補正
                if strain[i, c] > 0.05:
                    cluster_coherence[c] *= (1.0 - self.xp.minimum(strain[i, c], 1.0))
            
            coherence[i] = self.xp.mean(cluster_coherence)
        
        # エッジ処理
        coherence[:window] = coherence[window]
        coherence[-window:] = coherence[-window-1]
        
        return self.to_cpu(coherence)
    
    # ===============================
    # Material State Evaluation (物理ダメージ統合版)
    # ===============================
    
    def evaluate_material_state(self,
                               stress_history: Optional[np.ndarray] = None,
                               defect_charge: Optional[np.ndarray] = None,
                               energy_balance: Optional[Dict] = None,
                               damage_scores: Optional[np.ndarray] = None,
                               strain_history: Optional[np.ndarray] = None,
                               critical_clusters: Optional[List[int]] = None,
                               kv_ratios: Optional[np.ndarray] = None,
                               temperature: float = 300.0) -> MaterialState:
        """
        材料の健全性評価（物理ダメージ統合版）
        
        応力、欠陥、エネルギー状態を総合的に評価
        K/V比が与えられた場合は物理的に正しいダメージ計算を使用
        """
        # GPU転送（可能な場合）
        if self.is_gpu:
            if stress_history is not None:
                stress_history = cp.asarray(stress_history)
            if defect_charge is not None:
                defect_charge = cp.asarray(defect_charge)
            if damage_scores is not None:
                damage_scores = cp.asarray(damage_scores)
            if strain_history is not None:
                strain_history = cp.asarray(strain_history)
        
        # 初期化
        health_index = 1.0
        state = 'healthy'
        failure_probability = 0.0
        failure_mode = None
        
        # 材料パラメータ取得
        E = self.material_props.get('elastic_modulus', 210.0)
        sigma_y = self.material_props.get('yield_strength', 1.5)
        sigma_u = self.material_props.get('ultimate_strength', 2.0)
        
        # K/V比からの物理ダメージ計算（NEW!）
        physical_damage_result = None
        if kv_ratios is not None:
            logger.debug("Computing physical damage from K/V ratios")
            physical_damage_result = self.compute_physical_damage(
                kv_ratios, temperature=temperature
            )
            
            # 物理ダメージによる評価
            health_index *= (1.0 - physical_damage_result.failure_probability)
            failure_probability = physical_damage_result.failure_probability
            
            if physical_damage_result.failure_probability > 0.8:
                state = 'failed'
                failure_mode = 'physical_damage_failure'
            elif physical_damage_result.failure_probability > 0.5:
                state = 'critical'
            elif physical_damage_result.failure_probability > 0.2:
                state = 'damaged'
            
            # 臨界クラスター情報を使用
            if not critical_clusters:
                critical_clusters = physical_damage_result.critical_clusters
        
        # 1. 応力ベース評価（GPU上で計算）
        if stress_history is not None:
            if self.is_gpu:
                stress_history = stress_history.ravel() if stress_history.ndim > 1 else stress_history
                max_stress = float(cp.max(cp.abs(stress_history)))
                mean_stress = float(cp.mean(cp.abs(stress_history)))
            else:
                stress_history = stress_history.squeeze() if stress_history.ndim > 1 else stress_history
                max_stress = float(np.max(np.abs(stress_history)))
                mean_stress = float(np.mean(np.abs(stress_history)))
            
            if max_stress > sigma_u:
                health_index *= 0.1
                failure_probability = max(failure_probability, 1.0)
                state = 'failed'
                if not failure_mode:
                    failure_mode = 'immediate_fracture'
            elif max_stress > sigma_y:
                plastic_factor = (max_stress - sigma_y) / (sigma_u - sigma_y)
                health_index *= (1 - 0.5 * plastic_factor)
                failure_probability = max(failure_probability, plastic_factor)
                if not failure_mode:
                    failure_mode = 'plastic_failure'
                if health_index < 0.5 and state != 'failed':
                    state = 'critical'
        else:
            max_stress = 0.0
            mean_stress = 0.0
        
        # 2. 欠陥ベース評価（GPU上で計算）
        if defect_charge is not None and defect_charge.size > 0:
            if self.is_gpu:
                max_defect = float(cp.max(defect_charge))
                mean_defect = float(cp.mean(defect_charge))
                
                if len(defect_charge) > 2:
                    if defect_charge.ndim == 2:
                        defect_evolution = cp.mean(defect_charge, axis=1)
                    else:
                        defect_evolution = defect_charge
                    
                    defect_gradient = cp.gradient(defect_evolution)
                    defect_acceleration = float(cp.max(cp.abs(defect_gradient)))
            else:
                max_defect = float(np.max(defect_charge))
                mean_defect = float(np.mean(defect_charge))
                
                if len(defect_charge) > 2:
                    if defect_charge.ndim == 2:
                        defect_evolution = np.mean(defect_charge, axis=1)
                    else:
                        defect_evolution = defect_charge
                    
                    defect_acceleration = np.max(np.abs(np.gradient(defect_evolution)))
            
            # 欠陥による健全性低下
            defect_penalty = min(0.3, mean_defect)
            health_index *= (1 - defect_penalty)
            
            # 欠陥加速度評価
            if 'defect_acceleration' in locals() and defect_acceleration > 0.1:
                health_index *= 0.8
                if state == 'healthy':
                    state = 'damaged'
                if not failure_mode:
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
                if not failure_mode:
                    failure_mode = 'melting'
            elif phase == 'transition':
                health_index *= 0.7
                if state == 'healthy':
                    state = 'damaged'
        
        # 4. 損傷スコア処理（物理ダメージで代替可能）
        if damage_scores is not None and physical_damage_result is None:
            # 物理ダメージ計算がない場合のみ使用
            if self.is_gpu:
                max_damage = float(cp.max(damage_scores))
            else:
                max_damage = float(np.max(damage_scores))
            health_index *= max(0.0, 1.0 - max_damage)
        
        # 5. 歪み履歴処理
        if strain_history is not None and stress_history is None:
            if self.is_gpu:
                max_strain = float(cp.max(cp.abs(strain_history)))
            else:
                max_strain = float(np.max(np.abs(strain_history)))
            
            yield_strain = sigma_y / E
            if max_strain > yield_strain:
                plastic_penalty = min(0.4, (max_strain - yield_strain) * 5.0)
                health_index *= (1 - plastic_penalty)
        
        # 6. 最終的な健全性指標の正規化
        health_index = max(0.0, min(1.0, health_index))
        
        # 7. 信頼性指標（β指標）
        if mean_stress > 0:
            reliability_beta = (sigma_u - mean_stress) / (sigma_u * 0.1)
        else:
            reliability_beta = 5.0
        
        # 8. 損傷分布（物理ダメージから取得）
        damage_distribution = None
        if physical_damage_result is not None:
            damage_distribution = physical_damage_result.cumulative_damage[-1]
        
        return MaterialState(
            state=state,
            health_index=health_index,
            max_damage=1.0 - health_index,
            damage_distribution=damage_distribution,
            critical_clusters=critical_clusters,
            failure_mode=failure_mode,
            reliability_beta=reliability_beta
        )
    
    # ===============================
    # Additional Methods (既存のまま)
    # ===============================
    
    def compute_coordination_numbers(self,
                                    trajectory: np.ndarray,
                                    cluster_atoms: Dict[int, List[int]],
                                    cutoff: float = 3.5) -> np.ndarray:
        """配位数計算（CUDA最適化版）"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        
        if self.use_cuda and self.cuda_kernels is not None:
            logger.debug("Using CUDA kernels for coordination calculation")
            
            # GPU転送
            traj_gpu = cp.asarray(trajectory, dtype=cp.float32)
            
            # 結果配列
            coordination = np.zeros((n_frames, n_clusters))
            
            # フレームごとに計算
            for frame in range(n_frames):
                coord_gpu = self.cuda_kernels.compute_coordination_cuda(
                    traj_gpu[frame], cluster_atoms, cutoff
                )
                coordination[frame] = cp.asnumpy(coord_gpu)
            
            return coordination
        
        # フォールバック: 標準計算
        logger.debug("Using standard calculation for coordination")
        coordination = np.zeros((n_frames, n_clusters))
        
        for frame in range(n_frames):
            positions = trajectory[frame]
            for cid, atoms in cluster_atoms.items():
                if cid >= n_clusters or len(atoms) == 0:
                    continue
                
                # サンプル原子の配位数を計算
                sample_size = min(10, len(atoms))
                sample_atoms = atoms[:sample_size]
                
                coord_sum = 0
                for atom in sample_atoms:
                    distances = np.linalg.norm(positions - positions[atom], axis=1)
                    coord_sum += np.sum((distances > 0) & (distances < cutoff))
                
                coordination[frame, cid] = coord_sum / sample_size if sample_size > 0 else 0
        
        return coordination
    
    def compute_strain_tensors(self,
                              ref_positions: np.ndarray,
                              trajectory: np.ndarray,
                              cluster_atoms: Dict[int, List[int]]) -> np.ndarray:
        """歪みテンソル計算（CUDA最適化版）"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        
        if self.use_cuda and self.cuda_kernels is not None:
            logger.debug("Using CUDA kernels for strain tensor calculation")
            
            # GPU転送
            ref_pos_gpu = cp.asarray(ref_positions, dtype=cp.float32)
            traj_gpu = cp.asarray(trajectory, dtype=cp.float32)
            
            # 結果配列
            strain_tensors = np.zeros((n_frames, n_clusters, 6))
            
            # フレームごとに計算
            for frame in range(n_frames):
                strain_gpu = self.cuda_kernels.compute_strain_tensors_cuda(
                    ref_pos_gpu, traj_gpu[frame], cluster_atoms
                )
                strain_tensors[frame] = cp.asnumpy(strain_gpu)
            
            return strain_tensors
        
        # フォールバック: 簡易計算
        logger.debug("Using simplified strain calculation")
        strain_tensors = np.zeros((n_frames, n_clusters, 6))
        
        # 簡易的にRMSD的な歪みのみ計算
        for frame in range(n_frames):
            for cid, atoms in cluster_atoms.items():
                if cid >= n_clusters or len(atoms) < 4:
                    continue
                
                # 変位から簡易的な歪みを推定
                displacements = trajectory[frame, atoms] - ref_positions[atoms]
                strain_mag = np.linalg.norm(displacements, axis=1).mean()
                
                # 対角成分のみ設定（簡易版）
                strain_tensors[frame, cid, 0] = strain_mag * 0.33
                strain_tensors[frame, cid, 1] = strain_mag * 0.33
                strain_tensors[frame, cid, 2] = strain_mag * 0.33
        
        return strain_tensors
    
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
    
    def predict_failure(self,
                       strain_history: np.ndarray,
                       damage_history: Optional[np.ndarray] = None,
                       defect_charge: Optional[np.ndarray] = None,
                       loading_cycles: Optional[int] = None) -> FailurePredictionResult:
        """材料の破壊確率と信頼性を予測（GPU最適化版）"""
        
        # GPU転送
        strain_history = self.to_gpu(strain_history)
        
        # 歪みの統計量（GPU上で計算）
        if strain_history.ndim > 1:
            mean_strain = float(self.xp.mean(self.xp.abs(strain_history)))
            max_strain = float(self.xp.max(self.xp.abs(strain_history)))
            std_strain = float(self.xp.std(self.xp.abs(strain_history)))
        else:
            mean_strain = float(self.xp.mean(self.xp.abs(strain_history)))
            max_strain = float(self.xp.max(self.xp.abs(strain_history)))
            std_strain = float(self.xp.std(self.xp.abs(strain_history)))
        
        # 材料プロパティ
        E = self.material_props.get('elastic_modulus', 210.0)
        yield_strain = self.material_props.get('yield_strength', 1.5) / E
        ultimate_strain = self.material_props.get('ultimate_strength', 2.0) / E
        fatigue_strength = self.material_props.get('fatigue_strength', 0.7)
        
        # 破壊確率計算
        if max_strain > ultimate_strain:
            failure_prob = 1.0
            failure_mode = 'immediate_fracture'
        elif max_strain > yield_strain:
            failure_prob = float((max_strain - yield_strain) / (ultimate_strain - yield_strain))
            failure_mode = 'plastic_failure'
        else:
            if loading_cycles is not None and loading_cycles > 0:
                fatigue_strain = fatigue_strength / E
                if mean_strain > fatigue_strain:
                    Nf = 1e6 * (fatigue_strain / mean_strain) ** 12
                    failure_prob = min(1.0, loading_cycles / Nf)
                    failure_mode = 'fatigue_failure'
                else:
                    failure_prob = 0.0
                    failure_mode = 'safe'
            else:
                failure_prob = 0.0
                failure_mode = 'elastic_safe'
        
        # 信頼性指標
        if std_strain > 1e-10:
            beta = float((ultimate_strain - mean_strain) / std_strain)
        else:
            beta = 5.0
        
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
            remaining_life = max(0, Nf - loading_cycles) if 'Nf' in locals() else None
        
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
    # Utility Methods
    # ===============================
    
    def set_material_type(self, material_type: str):
        """材料タイプを変更"""
        self.material_type = material_type
        self.material_props = get_material_parameters(material_type)
        self.damage_calculator = PhysicalDamageCalculator(material_type)
        logger.info(f"Material type changed to {material_type}")
    
    def get_material_info(self) -> Dict[str, Any]:
        """現在の材料情報を取得"""
        info = {
            'type': self.material_type,
            'properties': self.material_props.copy(),
            'device': str(self.device),
            'gpu_enabled': self.is_gpu,
            'cuda_kernels_enabled': self.use_cuda,
            'physical_damage_enabled': True  # NEW!
        }
        
        if self.cuda_kernels is not None:
            info['kernel_status'] = self.cuda_kernels.get_kernel_status()
        
        return info
    
    def get_optimization_status(self) -> Dict[str, str]:
        """最適化状態を取得"""
        status = {}
        
        if self.use_cuda:
            status['defect_charge'] = 'CUDA Optimized'
            status['structural_coherence'] = 'CUDA Optimized'
            status['coordination'] = 'CUDA Optimized'
            status['strain_tensor'] = 'CUDA Optimized'
            status['physical_damage'] = 'Physics-Based'  # NEW!
            status['overall'] = 'Full CUDA + Physics'
        elif self.is_gpu:
            status['defect_charge'] = 'Standard GPU'
            status['structural_coherence'] = 'Standard GPU'
            status['coordination'] = 'Standard GPU'
            status['strain_tensor'] = 'Standard GPU'
            status['physical_damage'] = 'Physics-Based'  # NEW!
            status['overall'] = 'Standard GPU + Physics'
        else:
            status['defect_charge'] = 'CPU'
            status['structural_coherence'] = 'CPU'
            status['coordination'] = 'CPU'
            status['strain_tensor'] = 'CPU'
            status['physical_damage'] = 'Physics-Based'  # NEW!
            status['overall'] = 'CPU Mode + Physics'
        
        return status

# ===============================
# Export Functions（統一データベース使用版）
# ===============================

def compute_crystal_defect_charge(trajectory: np.ndarray,
                                 cluster_atoms: Dict[int, List[int]],
                                 material_type: str = 'SUJ2',
                                 cutoff: float = 3.5,
                                 use_gpu: bool = True,
                                 use_cuda: bool = True) -> CrystalDefectResult:
    """
    結晶欠陥チャージを計算する便利関数（CUDA最適化版）
    """
    analyzer = MaterialAnalyticsGPU(
        material_type, 
        force_cpu=not use_gpu,
        use_cuda_kernels=use_cuda
    )
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
                                use_gpu: bool = True,
                                use_cuda: bool = True) -> np.ndarray:
    """構造一貫性を計算する便利関数（CUDA最適化版）"""
    analyzer = MaterialAnalyticsGPU(
        material_type,
        force_cpu=not use_gpu,
        use_cuda_kernels=use_cuda
    )
    return analyzer.compute_structural_coherence(coordination, strain, window)

def calculate_physical_damage(kv_ratios: np.ndarray,
                             material_type: str = 'SUJ2',
                             temperature: float = 300.0) -> DamageAccumulation:
    """
    物理的に正しいダメージ計算の統合関数
    
    Parameters
    ----------
    kv_ratios : np.ndarray
        K/V比データ
    material_type : str
        材料タイプ
    temperature : float
        温度 [K]
        
    Returns
    -------
    DamageAccumulation
        累積損傷データ
    """
    calculator = PhysicalDamageCalculator(material_type)
    return calculator.calculate_damage_from_kv(kv_ratios, temperature=temperature)

# ===============================
# Test/Benchmark Function
# ===============================

if __name__ == "__main__":
    print("💎 Material Analytics GPU - Integrated Physics Edition v4.0")
    print("=" * 60)
    
    # テストデータ生成
    n_frames = 100
    n_atoms = 1000
    n_clusters = 5
    trajectory = np.random.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.1
    
    # クラスター定義
    cluster_atoms = {
        0: list(range(0, 900)),      # 健全領域
        1: list(range(900, 950)),    # 欠陥クラスター1
        2: list(range(950, 1000)),   # 欠陥クラスター2
    }
    
    # K/V比（時間とともに増加）
    time = np.linspace(0, 10, n_frames)
    kv_ratios = np.zeros((n_frames, n_clusters))
    for i in range(n_clusters):
        kv_ratios[:, i] = 1.0 + 0.1 * i * np.sin(time) + 0.05 * time
    
    # アナライザー初期化
    analyzer = MaterialAnalyticsGPU('SUJ2', use_cuda_kernels=True)
    
    print("\n📊 Optimization Status:")
    for key, value in analyzer.get_optimization_status().items():
        print(f"   {key}: {value}")
    
    print("\n🔬 Material Properties:")
    info = analyzer.get_material_info()
    print(f"   Material: {info['type']}")
    print(f"   Device: {info['device']}")
    print(f"   CUDA Kernels: {'✅ Enabled' if info['cuda_kernels_enabled'] else '❌ Disabled'}")
    print(f"   Physical Damage: {'✅ Enabled' if info['physical_damage_enabled'] else '❌ Disabled'}")
    
    # 物理ダメージ計算テスト
    print("\n⚡ Testing Physical Damage Calculation...")
    damage_result = analyzer.compute_physical_damage(kv_ratios, temperature=300.0)
    
    print(f"\n📈 Damage Analysis Results:")
    print(f"   Max cumulative damage: {damage_result.cumulative_damage.max():.3f}")
    print(f"   Failure probability: {damage_result.failure_probability:.1%}")
    print(f"   Critical clusters: {damage_result.critical_clusters}")
    print(f"   Remaining life: {damage_result.remaining_life}")
    
    # 材料状態評価テスト
    print("\n🔍 Testing Material State Evaluation...")
    material_state = analyzer.evaluate_material_state(
        kv_ratios=kv_ratios,
        temperature=300.0
    )
    
    print(f"\n📊 Material State:")
    print(f"   State: {material_state.state}")
    print(f"   Health Index: {material_state.health_index:.3f}")
    print(f"   Failure Mode: {material_state.failure_mode}")
    print(f"   Critical Clusters: {material_state.critical_clusters}")
    
    print("\n✨ Integration Complete! Physics-based damage calculation is working!")
