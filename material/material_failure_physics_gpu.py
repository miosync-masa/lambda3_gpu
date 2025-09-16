#!/usr/bin/env python3
"""
Material Failure Physics GPU Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

物理原理に基づく材料破損予測
- RMSF発散による破損前兆検出
- エネルギーバランスによる相転移判定
- 逆核形成理論による破損核追跡
- 疲労サイクルでの液化→再結晶→欠陥生成の追跡

水の沸点研究から生まれた統一理論の実装

Version: 1.0.0
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

# Logger設定
logger = logging.getLogger(__name__)

# ===============================
# Physical Constants
# ===============================

# Boltzmann定数 (eV/K)
K_B = 8.617333e-5

# 材料定数データベース
MATERIAL_CONSTANTS = {
    'SUJ2': {
        'lattice_constant': 2.87,      # BCC Fe (Å)
        'melting_temp': 1811,          # Fe melting point (K)
        'surface_energy': 2.0,         # J/m²
        'elastic_modulus': 210.0,      # GPa
        'thermal_expansion': 1.2e-5,   # /K
        'specific_heat': 460,          # J/(kg·K)
        'density': 7850,               # kg/m³
        'lindemann_criterion': 0.10    # Lindemann融解基準
    },
    'AL7075': {
        'lattice_constant': 4.05,      # FCC Al (Å)
        'melting_temp': 933,           # Al melting point (K)
        'surface_energy': 1.1,         # J/m²
        'elastic_modulus': 71.7,       # GPa
        'thermal_expansion': 2.4e-5,   # /K
        'specific_heat': 900,          # J/(kg·K)
        'density': 2810,               # kg/m³
        'lindemann_criterion': 0.12    # FCC用に調整
    },
    'TI6AL4V': {
        'lattice_constant': 3.23,      # HCP Ti (Å)
        'melting_temp': 1941,          # Ti melting point (K)
        'surface_energy': 2.1,         # J/m²
        'elastic_modulus': 113.8,      # GPa
        'thermal_expansion': 8.6e-6,   # /K
        'specific_heat': 520,          # J/(kg·K)
        'density': 4430,               # kg/m³
        'lindemann_criterion': 0.11    # HCP用に調整
    }
}

# ===============================
# Data Classes
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

@dataclass
class EnergyBalanceResult:
    """エネルギーバランス解析結果"""
    kinetic_energy: np.ndarray          # 運動エネルギー分布
    binding_energy: np.ndarray          # 束縛エネルギー分布
    energy_ratio: float                  # 運動/束縛エネルギー比
    local_temperature: np.ndarray       # 局所温度場
    phase_state: str                     # 'solid', 'liquid', 'transition'
    critical_temperature: float          # 臨界温度

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
# CUDA Kernels
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
    
    // 90%が熱に変換（Taylor-Quinney係数）
    float heat_generated = 0.9f * plastic_work;
    
    // 局所温度上昇
    float delta_T = heat_generated * conversion_factor;
    
    local_temp[atom_id] = base_temp + delta_T;
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
# Main Class
# ===============================

class MaterialFailurePhysicsGPU(GPUBackend):
    """
    物理原理に基づく材料破損予測
    
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
            材料タイプ
        lindemann_criterion : float, optional
            Lindemann基準値（Noneの場合材料デフォルト）
        force_cpu : bool
            CPUモードを強制
        """
        super().__init__(force_cpu=force_cpu)
        
        # 材料パラメータ設定
        self.material_type = material_type
        if material_type not in MATERIAL_CONSTANTS:
            logger.warning(f"Unknown material {material_type}, using SUJ2")
            material_type = 'SUJ2'
        
        self.material_params = MATERIAL_CONSTANTS[material_type]
        self.lattice_constant = self.material_params['lattice_constant']
        
        # Lindemann基準
        if lindemann_criterion is not None:
            self.lindemann_criterion = lindemann_criterion
        else:
            self.lindemann_criterion = self.material_params['lindemann_criterion']
        
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
    # Main Analysis Method
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
    # RMSF Divergence Detection
    # ===============================
    
    def _detect_rmsf_divergence(self, trajectory: np.ndarray) -> RMSFAnalysisResult:
        """
        RMSF発散による破損前兆検出
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子軌道 (n_frames, n_atoms, 3)
            
        Returns
        -------
        RMSFAnalysisResult
            RMSF解析結果
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        # ウィンドウパラメータ
        window_size = min(100, n_frames // 10)
        stride = max(1, window_size // 2)  # 50%オーバーラップ
        n_windows = (n_frames - window_size) // stride + 1
        
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
            rmsf_field = np.zeros((n_windows, n_atoms))
            
            for w in range(n_windows):
                start = w * stride
                end = min(start + window_size, n_frames)
                
                # 各原子のRMSF計算
                for atom in range(n_atoms):
                    atom_traj = trajectory[start:end, atom, :]
                    mean_pos = np.mean(atom_traj, axis=0)
                    deviations = atom_traj - mean_pos
                    rmsf_field[w, atom] = np.sqrt(np.mean(deviations**2))
        
        # Lindemann基準での臨界原子特定
        critical_threshold = self.lindemann_criterion * self.lattice_constant
        max_rmsf_per_atom = np.max(rmsf_field, axis=0)
        critical_atoms = np.where(max_rmsf_per_atom > critical_threshold)[0].tolist()
        
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
            divergence_rate=divergence_rate
        )
    
    # ===============================
    # Energy Balance Analysis
    # ===============================
    
    def _analyze_energy_balance(self,
                               trajectory: np.ndarray,
                               stress_history: Optional[np.ndarray],
                               base_temperature: float) -> EnergyBalanceResult:
        """
        エネルギーバランス解析（応力→熱変換と相転移）
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子軌道
        stress_history : np.ndarray, optional
            応力履歴
        base_temperature : float
            基礎温度 (K)
            
        Returns
        -------
        EnergyBalanceResult
            エネルギーバランス結果
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        # 運動エネルギー計算（簡略化：速度から）
        if n_frames > 1:
            velocities = np.diff(trajectory, axis=0)  # 簡易速度
            kinetic_energy = 0.5 * np.sum(velocities**2, axis=-1)  # (n_frames-1, n_atoms)
            # 最終フレーム分を複製
            kinetic_energy = np.vstack([kinetic_energy, kinetic_energy[-1]])
        else:
            kinetic_energy = np.zeros((n_frames, n_atoms))
        
        # 局所温度計算
        local_temperature = np.zeros((n_frames, n_atoms))
        
        if stress_history is not None:
            # 応力から温度上昇を推定
            stress_history = np.atleast_2d(stress_history)
            if stress_history.shape[0] != n_frames:
                stress_history = np.broadcast_to(
                    stress_history, (n_frames, n_atoms)
                )
            
            # Taylor-Quinney係数（塑性仕事の90%が熱に）
            heat_conversion = 0.9
            
            # 材料パラメータ
            rho = self.material_params['density']
            cp = self.material_params['specific_heat']
            
            # ΔT = (η * σ * ε̇ * Δt) / (ρ * Cp)
            # 簡略化：応力に比例した温度上昇
            delta_T = heat_conversion * np.abs(stress_history) * 10.0  # スケーリング係数
            local_temperature = base_temperature + delta_T
        else:
            local_temperature[:] = base_temperature
        
        # 束縛エネルギー推定（Debye温度から）
        # E_binding ≈ 3kT_Debye
        debye_temp = self.material_params['melting_temp'] * 0.5  # 簡略化
        binding_energy = 3 * K_B * debye_temp * np.ones_like(kinetic_energy)
        
        # エネルギー比
        energy_ratio = float(np.mean(kinetic_energy) / np.mean(binding_energy))
        
        # 相状態判定
        mean_temp = np.mean(local_temperature)
        melting_temp = self.material_params['melting_temp']
        
        if mean_temp < 0.8 * melting_temp:
            phase_state = 'solid'
        elif mean_temp > 0.95 * melting_temp:
            phase_state = 'liquid'
        else:
            phase_state = 'transition'
        
        # 臨界温度（エネルギーバランスから）
        critical_temperature = binding_energy[0, 0] / (1.5 * K_B)
        
        return EnergyBalanceResult(
            kinetic_energy=np.mean(kinetic_energy, axis=1),
            binding_energy=np.mean(binding_energy, axis=1),
            energy_ratio=energy_ratio,
            local_temperature=local_temperature,
            phase_state=phase_state,
            critical_temperature=float(critical_temperature)
        )
    
    # ===============================
    # Damage Nucleation Tracking
    # ===============================
    
    def _track_damage_nucleation(self,
                                rmsf_field: np.ndarray,
                                stress_history: Optional[np.ndarray]) -> DamageNucleusResult:
        """
        破損核の形成と成長を追跡（逆核形成理論）
        
        Parameters
        ----------
        rmsf_field : np.ndarray
            RMSF場 (n_windows, n_atoms)
        stress_history : np.ndarray, optional
            応力履歴
            
        Returns
        -------
        DamageNucleusResult
            破損核解析結果
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
            surface_energy = self.material_params['surface_energy']
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
    # Fatigue Cycle Analysis
    # ===============================
    
    def _analyze_fatigue_cycles(self,
                               trajectory: np.ndarray,
                               loading_cycles: List[Dict],
                               rmsf_field: np.ndarray) -> FatigueCycleResult:
        """
        疲労サイクル解析（液化→再結晶→欠陥生成）
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子軌道
        loading_cycles : List[Dict]
            負荷サイクル情報
        rmsf_field : np.ndarray
            RMSF場
            
        Returns
        -------
        FatigueCycleResult
            疲労解析結果
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
    # Integration and Prediction
    # ===============================
    
    def _integrate_predictions(self,
                             rmsf_result: RMSFAnalysisResult,
                             energy_result: EnergyBalanceResult,
                             nuclei_result: DamageNucleusResult,
                             fatigue_result: Optional[FatigueCycleResult]) -> Tuple[str, float, List[int], float]:
        """
        各解析結果を統合して最終予測
        
        Returns
        -------
        Tuple[str, float, List[int], float]
            (破損メカニズム, 破損時間, 破損位置, 信頼度)
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
        
        Parameters
        ----------
        temperature : float
            温度 (K)
            
        Returns
        -------
        float
            臨界RMSF値 (Å)
        """
        # 温度補正（Debye-Waller因子的なアプローチ）
        melting_temp = self.material_params['melting_temp']
        temp_ratio = temperature / melting_temp
        
        # 高温ほどLindemann基準が緩和
        adjusted_lindemann = self.lindemann_criterion * (1 + 0.5 * temp_ratio)
        
        return adjusted_lindemann * self.lattice_constant
    
    def get_physics_parameters(self) -> Dict[str, Any]:
        """
        現在の物理パラメータを取得
        
        Returns
        -------
        Dict[str, Any]
            物理パラメータ辞書
        """
        return {
            'material_type': self.material_type,
            'lattice_constant': self.lattice_constant,
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
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子軌道
    material_type : str
        材料タイプ
    temperature : float
        温度 (K)
    use_gpu : bool
        GPU使用フラグ
        
    Returns
    -------
    Dict[str, Any]
        破損前兆情報
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
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子軌道
    loading_cycles : List[Dict]
        負荷サイクル情報
    material_type : str
        材料タイプ
    use_gpu : bool
        GPU使用フラグ
        
    Returns
    -------
    float
        疲労寿命（サイクル数）
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
