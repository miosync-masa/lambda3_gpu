#!/usr/bin/env python3
"""
Material Failure Physics GPU Module (REFACTORED)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

物理原理に基づく材料破損予測
- RMSF発散による破損前兆検出
- エネルギーバランスによる相転移判定
- 逆核形成理論による破損核追跡
- 疲労サイクルでの液化→再結晶→欠陥生成の追跡

水の沸点研究から生まれた統一理論の実装

Version: 2.0.0 - Refactored with unified material database
Author: 環ちゃん - Physics Edition
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import logging
from scipy import stats
from scipy.optimize import curve_fit

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
from .material_database import (
    MATERIAL_DATABASE, 
    get_material_parameters,
    K_B,           # Boltzmann定数
    AMU_TO_KG,     # 原子質量単位→kg変換
    J_TO_EV        # J→eV変換
)

# Logger設定
logger = logging.getLogger(__name__)

# ===============================
# Data Classes (変更なし)
# ===============================

@dataclass
class RMSFAnalysisResult:
    """RMSF解析結果"""
    rmsf_field: np.ndarray              # RMSF時間発展 (n_windows, n_atoms)
    critical_atoms: List[int]           # 臨界原子リスト
    max_rmsf: float                      # 最大RMSF値
    lindemann_ratio: float               # Lindemann比
    divergence_time: Optional[int]      # 発散開始時刻
    divergence_rate: float               # 発散速度
    lindemann_parameter: Optional[float] = None  # 実際のLindemann値

@dataclass
class EnergyBalanceResult:
    """エネルギーバランス解析結果"""
    kinetic_energy: np.ndarray          # 運動エネルギー分布
    binding_energy: np.ndarray          # 束縛エネルギー分布
    energy_ratio: float                  # 運動/束縛エネルギー比
    local_temperature: np.ndarray       # 局所温度場
    phase_state: str                     # 'solid', 'liquid', 'transition'
    critical_temperature: float          # 臨界温度
    lindemann_parameter: Optional[float] = None  # Lindemann値

@dataclass
class DamageNucleusResult:
    """破損核解析結果"""
    nuclei_locations: List[np.ndarray]  # 核位置
    nuclei_sizes: np.ndarray            # 核サイズ
    critical_nucleus_size: float         # 臨界核サイズ
    growth_rates: np.ndarray            # 成長速度
    coalescence_time: Optional[float]   # 合体時刻

@dataclass
class FatigueCycleResult:
    """疲労サイクル解析結果"""
    melting_regions: List[np.ndarray]   # 局所融解領域
    new_defects: List[np.ndarray]       # 新規欠陥
    defect_accumulation: np.ndarray     # 欠陥蓄積曲線
    stress_concentration: np.ndarray    # 応力集中場
    cycles_to_failure: float            # 破損までのサイクル数

@dataclass
class FailurePhysicsResult:
    """統合物理予測結果"""
    # RMSF解析
    rmsf_analysis: RMSFAnalysisResult
    
    # エネルギーバランス
    energy_balance: EnergyBalanceResult
    
    # 破損核
    damage_nuclei: DamageNucleusResult
    
    # 疲労（オプション）
    fatigue_analysis: Optional[FatigueCycleResult] = None
    
    # 統合予測
    failure_mechanism: str = 'unknown'    # 破損メカニズム
    time_to_failure: float = np.inf       # 破損までの時間 (ps)
    failure_location: List[int] = field(default_factory=list)
    confidence: float = 0.0                # 予測信頼度

# ===============================
# CUDA Kernels (変更なし)
# ===============================

# RMSF時間発展カーネル
RMSF_EVOLUTION_KERNEL = r'''
extern "C" __global__
void compute_rmsf_evolution(
    const float* __restrict__ positions,  // (n_frames, n_atoms, 3)
    float* __restrict__ rmsf_field,       // (n_windows, n_atoms)
    const int n_frames,
    const int n_atoms,
    const int window_size,
    const int stride
) {
    int atom_id = blockIdx.x * blockDim.x + threadIdx.x;
    int window_id = blockIdx.y;
    
    if (atom_id >= n_atoms) return;
    
    int start = window_id * stride;
    int end = min(start + window_size, n_frames);
    
    if (start >= n_frames) return;
    
    // 平均位置計算
    float mean[3] = {0.0f, 0.0f, 0.0f};
    int count = end - start;
    
    for (int t = start; t < end; t++) {
        for (int d = 0; d < 3; d++) {
            mean[d] += positions[t * n_atoms * 3 + atom_id * 3 + d];
        }
    }
    for (int d = 0; d < 3; d++) mean[d] /= count;
    
    // RMSF計算
    float rmsf = 0.0f;
    for (int t = start; t < end; t++) {
        float dist_sq = 0.0f;
        for (int d = 0; d < 3; d++) {
            float delta = positions[t * n_atoms * 3 + atom_id * 3 + d] - mean[d];
            dist_sq += delta * delta;
        }
        rmsf += dist_sq;
    }
    rmsf = sqrtf(rmsf / count);
    
    rmsf_field[window_id * n_atoms + atom_id] = rmsf;
}
'''

# 局所温度計算カーネル（応力→熱変換）
LOCAL_TEMPERATURE_KERNEL = r'''
extern "C" __global__
void compute_local_temperature(
    const float* __restrict__ stress_field,     // (n_atoms,) 応力場
    const float* __restrict__ strain_rate,      // (n_atoms,) 歪み速度
    float* __restrict__ local_temp,             // (n_atoms,) 局所温度
    const float base_temp,                       // 基礎温度
    const float conversion_factor,               // 応力→熱変換係数
    const int n_atoms
) {
    int atom_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (atom_id >= n_atoms) return;
    
    // 塑性仕事 = 応力 × 歪み速度
    float plastic_work = stress_field[atom_id] * strain_rate[atom_id];
    
    // Taylor-Quinney係数で熱に変換
    float heat_generated = conversion_factor * plastic_work;
    
    // 局所温度上昇
    local_temp[atom_id] = base_temp + heat_generated;
}
'''

# 破損核検出カーネル
DAMAGE_NUCLEUS_KERNEL = r'''
extern "C" __global__
void detect_damage_nuclei(
    const float* __restrict__ rmsf_field,       // (n_atoms,) RMSF場
    int* __restrict__ nucleus_mask,             // (n_atoms,) 核マスク
    float* __restrict__ nucleus_sizes,          // (max_nuclei,) 核サイズ
    const float critical_rmsf,                  // 臨界RMSF
    const float lattice_constant,               // 格子定数
    const int n_atoms,
    const int max_nuclei
) {
    int atom_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (atom_id >= n_atoms) return;
    
    // Lindemann基準でのローカル融解判定
    float lindemann_ratio = rmsf_field[atom_id] / lattice_constant;
    
    if (lindemann_ratio > 0.1f) {  // Lindemann criterion
        nucleus_mask[atom_id] = 1;
        
        // 簡易的な核サイズ推定（実際はクラスタリングが必要）
        int nucleus_id = atom_id % max_nuclei;
        atomicAdd(&nucleus_sizes[nucleus_id], 1.0f);
    } else {
        nucleus_mask[atom_id] = 0;
    }
}
'''

# ===============================
# Main Class (リファクタリング版)
# ===============================

class MaterialFailurePhysicsGPU(GPUBackend):
    """
    物理原理に基づく材料破損予測（リファクタリング版）
    
    統一材料データベース(MATERIAL_DATABASE)を使用
    水の沸点研究から発見した普遍原理を材料破損に適用
    """
    
    def __init__(self,
                 material_type: str = 'SUJ2',
                 lindemann_criterion: float = None,
                 force_cpu: bool = False):
        """
        Parameters
        ----------
        material_type : str
            材料タイプ ('SUJ2', 'AL7075', 'Ti6Al4V', 'SS316L')
        lindemann_criterion : float, optional
            Lindemann基準値（Noneの場合材料デフォルト）
        force_cpu : bool
            CPUモードを強制
        """
        super().__init__(force_cpu=force_cpu)
        
        # 材料パラメータ取得（統一データベースから）
        self.material_type = material_type
        self.material_params = get_material_parameters(material_type)
        
        # 必要なパラメータ抽出
        self.lattice_constant = self.material_params.get('lattice_constant', 2.87)
        self.melting_temp = self.material_params.get('melting_temp', 1811)
        
        # Lindemann基準
        if lindemann_criterion is not None:
            self.lindemann_criterion = lindemann_criterion
        else:
            self.lindemann_criterion = self.material_params.get('lindemann_criterion', 0.10)
        
        # CUDAカーネル初期化
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
        
        logger.info(f"MaterialFailurePhysicsGPU initialized for {material_type}")
        logger.info(f"Lindemann criterion: {self.lindemann_criterion}")
    
    def _compile_kernels(self):
        """CUDAカーネルのコンパイル"""
        try:
            self.rmsf_evolution_kernel = cp.RawKernel(
                RMSF_EVOLUTION_KERNEL, 'compute_rmsf_evolution'
            )
            self.local_temp_kernel = cp.RawKernel(
                LOCAL_TEMPERATURE_KERNEL, 'compute_local_temperature'
            )
            self.damage_nucleus_kernel = cp.RawKernel(
                DAMAGE_NUCLEUS_KERNEL, 'detect_damage_nuclei'
            )
            logger.debug("Physics kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile kernels: {e}")
            self.rmsf_evolution_kernel = None
            self.local_temp_kernel = None
            self.damage_nucleus_kernel = None
    
    # ===============================
    # Main Analysis Method (変更なし)
    # ===============================
    
    def analyze_failure_physics(self,
                               trajectory: np.ndarray,
                               stress_history: Optional[np.ndarray] = None,
                               temperature: float = 300.0,
                               loading_cycles: Optional[List[Dict]] = None) -> FailurePhysicsResult:
        """
        物理原理による破損解析
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子軌道 (n_frames, n_atoms, 3)
        stress_history : np.ndarray, optional
            応力履歴 (n_frames, n_atoms) or (n_frames,)
        temperature : float
            基礎温度 (K)
        loading_cycles : List[Dict], optional
            負荷サイクル情報
            
        Returns
        -------
        FailurePhysicsResult
            統合解析結果
        """
        n_frames, n_atoms, _ = trajectory.shape
        logger.info(f"Analyzing failure physics: {n_frames} frames, {n_atoms} atoms")
        
        # 1. RMSF発散検出
        logger.debug("Detecting RMSF divergence...")
        rmsf_result = self._detect_rmsf_divergence(trajectory)
        
        # 2. エネルギーバランス解析
        logger.debug("Analyzing energy balance...")
        energy_result = self._analyze_energy_balance(
            trajectory, stress_history, temperature
        )
        
        # 3. 破損核追跡
        logger.debug("Tracking damage nucleation...")
        nuclei_result = self._track_damage_nucleation(
            rmsf_result.rmsf_field, stress_history
        )
        
        # 4. 疲労解析（オプション）
        fatigue_result = None
        if loading_cycles is not None:
            logger.debug("Analyzing fatigue cycles...")
            fatigue_result = self._analyze_fatigue_cycles(
                trajectory, loading_cycles, rmsf_result.rmsf_field
            )
        
        # 5. 統合予測
        logger.debug("Integrating predictions...")
        mechanism, ttf, location, confidence = self._integrate_predictions(
            rmsf_result, energy_result, nuclei_result, fatigue_result
        )
        
        return FailurePhysicsResult(
            rmsf_analysis=rmsf_result,
            energy_balance=energy_result,
            damage_nuclei=nuclei_result,
            fatigue_analysis=fatigue_result,
            failure_mechanism=mechanism,
            time_to_failure=ttf,
            failure_location=location,
            confidence=confidence
        )
    
    # ===============================
    # RMSF Divergence Detection (改善版)
    # ===============================
    
    def _detect_rmsf_divergence(self, trajectory: np.ndarray) -> RMSFAnalysisResult:
        """
        RMSF発散による破損前兆検出（バッチ処理対応版）
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        # バッチサイズ制限
        max_atoms_per_batch = 10000
        
        # ウィンドウパラメータ
        window_size = min(100, n_frames // 10)
        stride = max(1, window_size // 2)  # 50%オーバーラップ
        n_windows = (n_frames - window_size) // stride + 1
        
        # 結果配列を初期化
        rmsf_field = np.zeros((n_windows, n_atoms))
        
        # 大規模システムの場合はバッチ処理
        if n_atoms > max_atoms_per_batch:
            logger.info(f"Large system ({n_atoms} atoms) - using batch processing with batch size {max_atoms_per_batch}")
            
            n_batches = (n_atoms + max_atoms_per_batch - 1) // max_atoms_per_batch
            
            for batch_idx in range(n_batches):
                start_atom = batch_idx * max_atoms_per_batch
                end_atom = min((batch_idx + 1) * max_atoms_per_batch, n_atoms)
                batch_size = end_atom - start_atom
                
                logger.debug(f"Processing batch {batch_idx+1}/{n_batches}: atoms {start_atom}-{end_atom}")
                
                # バッチトラジェクトリ
                batch_trajectory = trajectory[:, start_atom:end_atom, :]
                
                if self.is_gpu and HAS_GPU and self.rmsf_evolution_kernel is not None:
                    # GPU計算（バッチ版）
                    traj_gpu = cp.asarray(batch_trajectory.reshape(n_frames, -1), dtype=cp.float32)
                    rmsf_batch_gpu = cp.zeros((n_windows, batch_size), dtype=cp.float32)
                    
                    blocks = (256,)
                    grids = ((batch_size + 255) // 256, n_windows)
                    
                    self.rmsf_evolution_kernel(
                        grids, blocks,
                        (traj_gpu, rmsf_batch_gpu, n_frames, batch_size, window_size, stride)
                    )
                    
                    rmsf_field[:, start_atom:end_atom] = cp.asnumpy(rmsf_batch_gpu)
                else:
                    # CPU計算（バッチ版）
                    for w in range(n_windows):
                        start_frame = w * stride
                        end_frame = min(start_frame + window_size, n_frames)
                        
                        for atom_idx in range(batch_size):
                            atom_traj = batch_trajectory[start_frame:end_frame, atom_idx, :]
                            mean_pos = np.mean(atom_traj, axis=0)
                            deviations = atom_traj - mean_pos
                            rmsf_field[w, start_atom + atom_idx] = np.sqrt(np.mean(deviations**2))
        
        else:
            # 小規模システムは従来通り一括処理
            if self.is_gpu and HAS_GPU and self.rmsf_evolution_kernel is not None:
                # GPU計算
                traj_gpu = cp.asarray(trajectory.reshape(n_frames, -1), dtype=cp.float32)
                rmsf_field_gpu = cp.zeros((n_windows, n_atoms), dtype=cp.float32)
                
                blocks = (256,)
                grids = ((n_atoms + 255) // 256, n_windows)
                
                self.rmsf_evolution_kernel(
                    grids, blocks,
                    (traj_gpu, rmsf_field_gpu, n_frames, n_atoms, window_size, stride)
                )
                
                rmsf_field = cp.asnumpy(rmsf_field_gpu)
            else:
                # CPU計算
                for w in range(n_windows):
                    start = w * stride
                    end = min(start + window_size, n_frames)
                    
                    for atom in range(n_atoms):
                        atom_traj = trajectory[start:end, atom, :]
                        mean_pos = np.mean(atom_traj, axis=0)
                        deviations = atom_traj - mean_pos
                        rmsf_field[w, atom] = np.sqrt(np.mean(deviations**2))
        
        # Lindemann基準での臨界原子特定
        critical_threshold = self.lindemann_criterion * self.lattice_constant
        max_rmsf_per_atom = np.max(rmsf_field, axis=0)
        critical_atoms = np.where(max_rmsf_per_atom > critical_threshold)[0].tolist()
        
        # メモリ節約のため、臨界原子が多すぎる場合は制限
        if len(critical_atoms) > 1000:
            logger.warning(f"Too many critical atoms ({len(critical_atoms)}), limiting to top 1000")
            # RMSFの高い順に1000個選ぶ
            top_indices = np.argsort(max_rmsf_per_atom)[-1000:]
            critical_atoms = top_indices.tolist()
        
        # 全体のLindemann値計算
        mean_rmsf = np.mean(rmsf_field)
        lindemann_parameter = mean_rmsf / self.lattice_constant
        
        # 発散検出
        divergence_time = None
        divergence_rate = 0.0
        
        if len(critical_atoms) > 0:
            # 最も危険な原子のRMSF時間発展
            worst_atom = critical_atoms[0]
            rmsf_evolution = rmsf_field[:, worst_atom]
            
            # 発散開始点を検出（勾配が急増する点）
            gradient = np.gradient(rmsf_evolution)
            threshold = np.mean(gradient) + 2 * np.std(gradient)
            divergence_indices = np.where(gradient > threshold)[0]
            
            if len(divergence_indices) > 0:
                divergence_time = divergence_indices[0] * stride
                divergence_rate = float(np.max(gradient))
        
        return RMSFAnalysisResult(
            rmsf_field=rmsf_field,
            critical_atoms=critical_atoms,
            max_rmsf=float(np.max(rmsf_field)),
            lindemann_ratio=float(np.max(rmsf_field) / self.lattice_constant),
            divergence_time=divergence_time,
            divergence_rate=divergence_rate,
            lindemann_parameter=lindemann_parameter
        )
    
    # ===============================
    # Energy Balance Analysis (物理的改善版)
    # ===============================
    
    def _analyze_energy_balance(self,
                           trajectory: np.ndarray,
                           stress_history: Optional[np.ndarray],
                           base_temperature: float) -> EnergyBalanceResult:
        """
        エネルギーバランス解析（統一データベース版）
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        # 材料の原子質量を取得
        atomic_mass = self.material_params.get('atomic_mass', 55.845)  # デフォルトFe
        mass = atomic_mass * AMU_TO_KG
        
        # 1. 運動エネルギー（ちゃんと質量考慮）
        if n_frames > 1:
            dt = 1e-15  # 1 fs timestep（MD標準）
            velocities = np.diff(trajectory, axis=0) / dt  # m/s
            # KE = 1/2 * m * v^2 （eVに変換）
            kinetic_energy = 0.5 * mass * np.sum(velocities**2, axis=-1) * J_TO_EV
            kinetic_energy = np.vstack([kinetic_energy, kinetic_energy[-1]])
        else:
            kinetic_energy = np.zeros((n_frames, n_atoms))
        
        # 2. 応力から温度上昇（物理的に）
        local_temperature = np.full((n_frames, n_atoms), base_temperature)
        
        if stress_history is not None:
            stress_history = np.asarray(stress_history)
            
            # 形状処理（2次元に統一）
            if stress_history.ndim == 1:
                stress_history = stress_history.reshape(-1, 1)
            
            # ブロードキャスト
            if stress_history.shape[1] == 1:
                stress_history = np.broadcast_to(stress_history, (n_frames, n_atoms))
            
            # 歪み速度推定（変位から）
            if n_frames > 1:
                strain_rate = np.diff(trajectory, axis=0) / (self.lattice_constant * dt)
                strain_rate = np.abs(np.mean(strain_rate, axis=-1))  # (n_frames-1, n_atoms)
                strain_rate = np.vstack([strain_rate, strain_rate[-1]])
            else:
                strain_rate = 1e-3 * np.ones((n_frames, n_atoms))  # デフォルト値
            
            # Taylor-Quinney係数（材料から取得）
            beta = self.material_params.get('taylor_quinney', 0.9)
            
            # 塑性仕事 → 熱変換
            # ΔT = β × σ × ε̇ / (ρ × Cp)
            rho = self.material_params.get('density', 7850)  # kg/m³
            cp = self.material_params.get('specific_heat', 460)  # J/(kg·K)
            
            # 応力はGPa、歪み速度は1/s
            plastic_work = np.abs(stress_history) * 1e9 * strain_rate  # Pa·s⁻¹ = W/m³
            delta_T = beta * plastic_work / (rho * cp)
            
            # 温度場更新
            local_temperature = base_temperature + delta_T
            
            # 物理的制限（融点以下）
            local_temperature = np.clip(local_temperature, base_temperature, self.melting_temp * 1.1)
        
        # 3. 束縛エネルギー（Debye-Waller因子から）
        # Debye温度の推定（金属の経験則）
        debye_temp = self.melting_temp * 0.4
        
        # <u²> = (3ℏ²T) / (mk_Bθ_D²) でDebye-Waller因子
        # E_binding ≈ k_B × θ_D × exp(-2W)
        mean_temp = np.mean(local_temperature)
        debye_waller = (3 * mean_temp) / (debye_temp**2) 
        binding_energy = K_B * debye_temp * np.exp(-2 * debye_waller) * np.ones_like(kinetic_energy)
        
        # 4. エネルギー比とLindemann判定
        energy_ratio = float(np.mean(kinetic_energy) / np.mean(binding_energy)) if np.mean(binding_energy) > 0 else np.inf
        
        # Lindemann基準：<u²>^0.5 / a > 0.1で融解
        rms_displacement = np.sqrt(np.mean((trajectory - np.mean(trajectory, axis=0))**2))
        lindemann_parameter = rms_displacement / self.lattice_constant
        
        # 相状態判定（Lindemann基準も考慮）
        if lindemann_parameter > self.lindemann_criterion:
            phase_state = 'liquid'
        elif mean_temp < 0.7 * self.melting_temp:
            phase_state = 'solid'
        elif mean_temp > 0.9 * self.melting_temp:
            phase_state = 'liquid'
        else:
            phase_state = 'transition'
        
        # 臨界温度（Lindemann基準から逆算）
        critical_temperature = debye_temp * (self.lindemann_criterion / 0.1)**2
        
        return EnergyBalanceResult(
            kinetic_energy=np.mean(kinetic_energy, axis=1),
            binding_energy=np.mean(binding_energy, axis=1),
            energy_ratio=energy_ratio,
            local_temperature=local_temperature,
            phase_state=phase_state,
            critical_temperature=float(critical_temperature),
            lindemann_parameter=lindemann_parameter
        )
    
    # ===============================
    # Damage Nucleation Tracking (統一データベース版)
    # ===============================
    
    def _track_damage_nucleation(self,
                                rmsf_field: np.ndarray,
                                stress_history: Optional[np.ndarray]) -> DamageNucleusResult:
        """
        破損核の形成と成長を追跡（逆核形成理論）
        """
        n_windows, n_atoms = rmsf_field.shape
        
        # 臨界RMSF（Lindemann基準）
        critical_rmsf = self.lindemann_criterion * self.lattice_constant
        
        # 各時間窓での核検出
        nuclei_locations = []
        nuclei_sizes = []
        growth_rates = []
        
        for w in range(n_windows):
            # 臨界領域の検出
            critical_mask = rmsf_field[w] > critical_rmsf
            
            if np.any(critical_mask):
                # 連結成分解析（簡易版）
                critical_indices = np.where(critical_mask)[0]
                
                # クラスタリング（簡易的に連続領域を1つの核とする）
                clusters = []
                current_cluster = [critical_indices[0]]
                
                for i in range(1, len(critical_indices)):
                    if critical_indices[i] - critical_indices[i-1] <= 10:  # 近接判定
                        current_cluster.append(critical_indices[i])
                    else:
                        clusters.append(np.array(current_cluster))
                        current_cluster = [critical_indices[i]]
                clusters.append(np.array(current_cluster))
                
                nuclei_locations.append(clusters)
                nuclei_sizes.append([len(c) for c in clusters])
        
        # 成長速度計算
        if len(nuclei_sizes) > 1:
            for i in range(1, len(nuclei_sizes)):
                if len(nuclei_sizes[i]) > 0 and len(nuclei_sizes[i-1]) > 0:
                    # 最大核の成長速度
                    max_size_current = max(nuclei_sizes[i])
                    max_size_prev = max(nuclei_sizes[i-1])
                    growth_rate = max_size_current - max_size_prev
                    growth_rates.append(growth_rate)
        
        # 臨界核サイズ（古典核形成理論から）
        if stress_history is not None:
            mean_stress = np.mean(np.abs(stress_history))
            # 表面エネルギー（材料データベースから、デフォルト値も用意）
            surface_energy = self.material_params.get('surface_energy', 2.0)
            # r* = 2γ/σ （簡略化）
            critical_nucleus_size = 2 * surface_energy / (mean_stress * 1e9)  # GPa→Pa
        else:
            critical_nucleus_size = 10.0  # デフォルト値（原子数）
        
        # 合体時間推定
        coalescence_time = None
        if len(growth_rates) > 0 and np.mean(growth_rates) > 0:
            # 平均成長速度から推定
            mean_growth = np.mean(growth_rates)
            system_size = n_atoms
            if len(nuclei_sizes) > 0 and len(nuclei_sizes[-1]) > 0:
                current_max_size = max(nuclei_sizes[-1])
                # パーコレーション閾値（系の～30%）
                percolation_threshold = 0.3 * system_size
                remaining_growth = percolation_threshold - current_max_size
                if remaining_growth > 0 and mean_growth > 0:
                    coalescence_time = remaining_growth / mean_growth
        
        return DamageNucleusResult(
            nuclei_locations=nuclei_locations,
            nuclei_sizes=np.array([max(sizes) if sizes else 0 for sizes in nuclei_sizes]),
            critical_nucleus_size=critical_nucleus_size,
            growth_rates=np.array(growth_rates) if growth_rates else np.array([0]),
            coalescence_time=coalescence_time
        )
    
    # ===============================
    # Fatigue Cycle Analysis (変更なし)
    # ===============================
    
    def _analyze_fatigue_cycles(self,
                               trajectory: np.ndarray,
                               loading_cycles: List[Dict],
                               rmsf_field: np.ndarray) -> FatigueCycleResult:
        """
        疲労サイクル解析（液化→再結晶→欠陥生成）
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        melting_regions = []
        new_defects = []
        defect_accumulation = []
        stress_concentration = np.zeros(n_atoms)
        
        cumulative_defects = 0
        
        for cycle in loading_cycles:
            start = cycle.get('start', 0)
            end = cycle.get('end', n_frames)
            is_loading = cycle.get('loading', True)
            
            if is_loading:
                # 負荷フェーズ：局所融解検出
                cycle_rmsf = np.mean(rmsf_field[start//10:end//10], axis=0)
                melting_mask = cycle_rmsf > self.lindemann_criterion * self.lattice_constant
                melting_indices = np.where(melting_mask)[0]
                melting_regions.append(melting_indices)
                
                # 応力集中更新
                stress_concentration[melting_indices] += 0.1
            else:
                # 除荷フェーズ：再結晶欠陥生成
                if len(melting_regions) > 0:
                    prev_melting = melting_regions[-1]
                    
                    # 再結晶での欠陥生成（確率的）
                    defect_probability = 0.1  # 10%の確率で欠陥
                    n_defects = int(len(prev_melting) * defect_probability)
                    if n_defects > 0:
                        defect_indices = np.random.choice(prev_melting, n_defects, replace=False)
                        new_defects.append(defect_indices)
                        cumulative_defects += n_defects
            
            defect_accumulation.append(cumulative_defects)
        
        # 破損までのサイクル数推定（Paris則）
        if len(defect_accumulation) > 1:
            defect_rate = np.gradient(defect_accumulation)
            if np.mean(defect_rate) > 0:
                critical_defects = 0.1 * n_atoms  # 10%で破損
                remaining_defects = critical_defects - cumulative_defects
                cycles_to_failure = remaining_defects / np.mean(defect_rate)
            else:
                cycles_to_failure = np.inf
        else:
            cycles_to_failure = np.inf
        
        return FatigueCycleResult(
            melting_regions=melting_regions,
            new_defects=new_defects,
            defect_accumulation=np.array(defect_accumulation),
            stress_concentration=stress_concentration,
            cycles_to_failure=float(cycles_to_failure)
        )
    
    # ===============================
    # Integration and Prediction (変更なし)
    # ===============================
    
    def _integrate_predictions(self,
                             rmsf_result: RMSFAnalysisResult,
                             energy_result: EnergyBalanceResult,
                             nuclei_result: DamageNucleusResult,
                             fatigue_result: Optional[FatigueCycleResult]) -> Tuple[str, float, List[int], float]:
        """
        各解析結果を統合して最終予測
        """
        # 破損メカニズムの判定
        mechanism = 'unknown'
        confidence = 0.0
        
        # RMSF基準
        if rmsf_result.lindemann_ratio > self.lindemann_criterion:
            mechanism = 'local_melting'
            confidence = min(1.0, rmsf_result.lindemann_ratio / (self.lindemann_criterion * 2))
        
        # エネルギー基準
        if energy_result.phase_state == 'liquid':
            mechanism = 'phase_transition'
            confidence = max(confidence, 0.9)
        elif energy_result.phase_state == 'transition':
            if mechanism == 'unknown':
                mechanism = 'incipient_melting'
            confidence = max(confidence, 0.6)
        
        # 疲労基準
        if fatigue_result is not None and fatigue_result.cycles_to_failure < 1000:
            mechanism = 'fatigue_failure'
            confidence = max(confidence, 0.7)
        
        # 破損時間の推定
        time_estimates = []
        
        # RMSF発散時間
        if rmsf_result.divergence_time is not None and rmsf_result.divergence_rate > 0:
            # 臨界RMSFまでの時間
            remaining_rmsf = self.lindemann_criterion * self.lattice_constant - rmsf_result.max_rmsf
            if remaining_rmsf > 0:
                ttf_rmsf = remaining_rmsf / rmsf_result.divergence_rate
                time_estimates.append(ttf_rmsf)
        
        # 核合体時間
        if nuclei_result.coalescence_time is not None:
            time_estimates.append(nuclei_result.coalescence_time)
        
        # 疲労寿命
        if fatigue_result is not None:
            # サイクル→時間変換（1サイクル=10ps仮定）
            ttf_fatigue = fatigue_result.cycles_to_failure * 10.0
            time_estimates.append(ttf_fatigue)
        
        # 最小時間を採用
        if time_estimates:
            time_to_failure = float(min(time_estimates))
        else:
            time_to_failure = np.inf
        
        # 破損位置
        failure_location = rmsf_result.critical_atoms[:10]  # Top 10原子
        
        return mechanism, time_to_failure, failure_location, confidence
    
    # ===============================
    # Utility Methods
    # ===============================
    
    def estimate_critical_rmsf(self, temperature: float) -> float:
        """
        温度依存のLindemann基準を推定
        """
        # 温度補正（Debye-Waller因子的なアプローチ）
        temp_ratio = temperature / self.melting_temp
        
        # 高温ほどLindemann基準が緩和
        adjusted_lindemann = self.lindemann_criterion * (1 + 0.5 * temp_ratio)
        
        return adjusted_lindemann * self.lattice_constant
    
    def get_physics_parameters(self) -> Dict[str, Any]:
        """
        現在の物理パラメータを取得
        """
        return {
            'material_type': self.material_type,
            'lattice_constant': self.lattice_constant,
            'melting_temp': self.melting_temp,
            'lindemann_criterion': self.lindemann_criterion,
            'material_params': self.material_params.copy(),
            'device': str(self.device),
            'gpu_enabled': self.is_gpu
        }

# ===============================
# Convenience Functions
# ===============================

def detect_failure_precursor(trajectory: np.ndarray,
                            material_type: str = 'SUJ2',
                            temperature: float = 300.0,
                            use_gpu: bool = True) -> Dict[str, Any]:
    """
    破損前兆を検出する便利関数
    """
    analyzer = MaterialFailurePhysicsGPU(
        material_type=material_type,
        force_cpu=not use_gpu
    )
    
    result = analyzer.analyze_failure_physics(
        trajectory=trajectory,
        temperature=temperature
    )
    
    return {
        'mechanism': result.failure_mechanism,
        'time_to_failure': result.time_to_failure,
        'critical_atoms': result.failure_location,
        'lindemann_ratio': result.rmsf_analysis.lindemann_ratio,
        'phase_state': result.energy_balance.phase_state,
        'confidence': result.confidence
    }

def predict_fatigue_life(trajectory: np.ndarray,
                        loading_cycles: List[Dict],
                        material_type: str = 'SUJ2',
                        use_gpu: bool = True) -> float:
    """
    疲労寿命を予測する便利関数
    """
    analyzer = MaterialFailurePhysicsGPU(
        material_type=material_type,
        force_cpu=not use_gpu
    )
    
    result = analyzer.analyze_failure_physics(
        trajectory=trajectory,
        loading_cycles=loading_cycles
    )
    
    if result.fatigue_analysis is not None:
        return result.fatigue_analysis.cycles_to_failure
    else:
        return np.inf
