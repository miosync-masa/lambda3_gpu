#!/usr/bin/env python3
"""
Causality Analysis for Material Clusters (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

クラスター間の因果関係をGPUで高速解析！
転位伝播、歪み伝播、亀裂進展の因果性を定量化！💎

残基版をベースに材料解析に特化

by 環ちゃん - Material Edition
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# GPU imports
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors, profile_gpu

logger = logging.getLogger('lambda3_gpu.material.causality')

# ===============================
# Data Classes (Material版)
# ===============================

@dataclass
class MaterialCausalityResult:
    """材料因果関係解析の結果"""
    pair: Tuple[int, int]           # (from_cluster, to_cluster)
    causality_strength: float       # 因果強度
    optimal_lag: int               # 最適ラグ（転位伝播時間）
    causality_profile: np.ndarray  # ラグごとの因果強度
    transfer_entropy: Optional[float] = None  # Transfer Entropy
    granger_causality: Optional[float] = None  # Granger因果
    confidence: float = 1.0        # 信頼度
    p_value: Optional[float] = None  # p値
    
    # 材料特有の追加フィールド
    propagation_type: str = 'elastic'  # 'elastic', 'plastic', 'fracture'
    propagation_velocity: Optional[float] = None  # 伝播速度 (Å/frame)
    strain_coupling: Optional[float] = None  # 歪み結合強度
    damage_correlation: Optional[float] = None  # 損傷相関

# ===============================
# CUDA Kernels (Material特化版)
# ===============================

# 転位伝播因果性カーネル
DISLOCATION_CAUSALITY_KERNEL = r'''
extern "C" __global__
void compute_dislocation_causality_kernel(
    const float* __restrict__ coord_defect_i,  // (n_frames,) 配位数欠陥
    const float* __restrict__ coord_defect_j,  // (n_frames,) 
    const float* __restrict__ strain_i,        // (n_frames,) 歪み
    const float* __restrict__ strain_j,        // (n_frames,)
    float* __restrict__ causality_profile,     // (n_lags,)
    const int n_frames,
    const int n_lags,
    const float defect_threshold,
    const float strain_threshold
) {
    const int lag = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lag >= n_lags || lag == 0) return;
    
    // 転位イベント検出
    int dislocation_events_i = 0;
    int dislocation_events_j = 0;
    int propagation_events = 0;
    
    for (int t = 0; t < n_frames - lag; t++) {
        // 転位判定：配位数欠陥 + 歪み
        bool disl_i = (coord_defect_i[t] > defect_threshold) && 
                      (fabsf(strain_i[t]) > strain_threshold);
        bool disl_j = (coord_defect_j[t + lag] > defect_threshold) && 
                      (fabsf(strain_j[t + lag]) > strain_threshold);
        
        if (disl_i) {
            dislocation_events_i++;
            if (disl_j) {
                propagation_events++;
            }
        }
        if (disl_j) {
            dislocation_events_j++;
        }
    }
    
    // 転位伝播確率
    if (dislocation_events_i > 0) {
        float propagation_prob = (float)propagation_events / dislocation_events_i;
        
        // 歪み場の影響を考慮
        float strain_factor = 1.0f;
        for (int t = 0; t < n_frames - lag; t++) {
            if (coord_defect_i[t] > defect_threshold) {
                strain_factor += fabsf(strain_i[t]) * 0.1f;
            }
        }
        
        causality_profile[lag] = propagation_prob * strain_factor;
    } else {
        causality_profile[lag] = 0.0f;
    }
}
'''

# 歪み伝播因果性カーネル
STRAIN_PROPAGATION_KERNEL = r'''
extern "C" __global__
void compute_strain_propagation_kernel(
    const float* __restrict__ strain_tensor_i,  // (n_frames, 9) フラット化
    const float* __restrict__ strain_tensor_j,  // (n_frames, 9)
    float* __restrict__ causality_profile,      // (n_lags,)
    const int n_frames,
    const int n_lags
) {
    const int lag = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lag >= n_lags || lag == 0) return;
    
    float correlation_sum = 0.0f;
    float energy_transfer = 0.0f;
    int valid_samples = 0;
    
    for (int t = 0; t < n_frames - lag; t++) {
        // 歪みテンソルのFrobenius内積
        float dot_product = 0.0f;
        float norm_i = 0.0f;
        float norm_j = 0.0f;
        
        for (int k = 0; k < 9; k++) {
            float s_i = strain_tensor_i[t * 9 + k];
            float s_j = strain_tensor_j[(t + lag) * 9 + k];
            
            dot_product += s_i * s_j;
            norm_i += s_i * s_i;
            norm_j += s_j * s_j;
        }
        
        if (norm_i > 1e-6f && norm_j > 1e-6f) {
            correlation_sum += dot_product / (sqrtf(norm_i) * sqrtf(norm_j));
            
            // エネルギー移動の推定
            if (t > 0) {
                float energy_i_prev = 0.0f;
                float energy_i_curr = 0.0f;
                float energy_j_prev = 0.0f;
                float energy_j_curr = 0.0f;
                
                for (int k = 0; k < 9; k++) {
                    energy_i_prev += strain_tensor_i[(t-1) * 9 + k] * 
                                    strain_tensor_i[(t-1) * 9 + k];
                    energy_i_curr += strain_tensor_i[t * 9 + k] * 
                                    strain_tensor_i[t * 9 + k];
                    energy_j_prev += strain_tensor_j[(t+lag-1) * 9 + k] * 
                                    strain_tensor_j[(t+lag-1) * 9 + k];
                    energy_j_curr += strain_tensor_j[(t+lag) * 9 + k] * 
                                    strain_tensor_j[(t+lag) * 9 + k];
                }
                
                // エネルギー変化の相関
                float delta_i = energy_i_curr - energy_i_prev;
                float delta_j = energy_j_curr - energy_j_prev;
                
                if (delta_i < 0 && delta_j > 0) {
                    // iからjへのエネルギー移動
                    energy_transfer += fminf(-delta_i, delta_j);
                }
            }
            
            valid_samples++;
        }
    }
    
    if (valid_samples > 0) {
        causality_profile[lag] = (correlation_sum / valid_samples) * 
                                 (1.0f + energy_transfer / valid_samples);
    } else {
        causality_profile[lag] = 0.0f;
    }
}
'''

# 亀裂伝播因果性カーネル
CRACK_PROPAGATION_KERNEL = r'''
extern "C" __global__
void compute_crack_propagation_kernel(
    const float* __restrict__ damage_i,         // (n_frames,) 損傷度
    const float* __restrict__ damage_j,         // (n_frames,)
    const float* __restrict__ stress_intensity, // (n_frames,) 応力拡大係数
    float* __restrict__ causality_profile,      // (n_lags,)
    const int n_frames,
    const int n_lags,
    const float damage_threshold,
    const float k_ic  // 破壊靭性
) {
    const int lag = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lag >= n_lags || lag == 0) return;
    
    int crack_init_events = 0;
    int crack_prop_events = 0;
    float stress_factor_sum = 0.0f;
    
    for (int t = 0; t < n_frames - lag; t++) {
        // 亀裂開始判定
        bool crack_i = damage_i[t] > damage_threshold;
        bool crack_j = damage_j[t + lag] > damage_threshold;
        
        if (crack_i) {
            crack_init_events++;
            
            // 応力拡大係数の影響
            float k_factor = (stress_intensity[t] > 0) ? 
                           stress_intensity[t] / k_ic : 0.0f;
            
            if (crack_j) {
                crack_prop_events++;
                stress_factor_sum += k_factor;
            }
        }
    }
    
    if (crack_init_events > 0) {
        float prop_prob = (float)crack_prop_events / crack_init_events;
        float avg_stress_factor = stress_factor_sum / crack_init_events;
        
        // Paris則的な関係：da/dN ∝ (ΔK)^m
        float paris_factor = powf(fmaxf(avg_stress_factor, 0.1f), 2.0f);
        
        causality_profile[lag] = prop_prob * paris_factor;
    } else {
        causality_profile[lag] = 0.0f;
    }
}
'''

# ===============================
# Material Causality Analyzer GPU Class
# ===============================

class MaterialCausalityAnalyzerGPU(GPUBackend):
    """
    材料クラスター間の因果関係解析のGPU実装
    
    転位伝播、歪み伝播、亀裂進展の因果性を高速計算！
    """
    
    def __init__(self,
                 # 基本パラメータ
                 event_threshold: float = 1.0,
                 min_lag: int = 1,
                 max_lag: int = 100,  # 材料は短い時間スケール
                 history_length: int = 3,  # 短い履歴
                 n_bins: int = 8,  # 少なめのビン
                 
                 # 材料特有パラメータ
                 defect_threshold: float = 1.0,  # 配位数欠陥閾値
                 strain_threshold: float = 0.01,  # 歪み閾値
                 damage_threshold: float = 0.5,  # 損傷閾値
                 k_ic: float = 1.0,  # 破壊靭性（MPa√m相当）
                 
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        event_threshold : float
            イベント判定閾値
        min_lag : int
            最小ラグ
        max_lag : int
            最大ラグ（材料は短め）
        history_length : int
            履歴長（Transfer Entropy用）
        n_bins : int
            離散化のビン数
        defect_threshold : float
            配位数欠陥の閾値
        strain_threshold : float
            歪みの閾値
        damage_threshold : float
            損傷度の閾値
        k_ic : float
            破壊靭性
        """
        super().__init__(**kwargs)
        self.event_threshold = event_threshold
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.history_length = history_length
        self.n_bins = n_bins
        
        # 材料パラメータ
        self.defect_threshold = defect_threshold
        self.strain_threshold = strain_threshold
        self.damage_threshold = damage_threshold
        self.k_ic = k_ic
        
        self.memory_manager = memory_manager or GPUMemoryManager()
        
        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.dislocation_kernel = cp.RawKernel(
                DISLOCATION_CAUSALITY_KERNEL, 'compute_dislocation_causality_kernel'
            )
            self.strain_kernel = cp.RawKernel(
                STRAIN_PROPAGATION_KERNEL, 'compute_strain_propagation_kernel'
            )
            self.crack_kernel = cp.RawKernel(
                CRACK_PROPAGATION_KERNEL, 'compute_crack_propagation_kernel'
            )
            logger.debug("Material causality kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.dislocation_kernel = None
            self.strain_kernel = None
            self.crack_kernel = None
    
    @handle_gpu_errors
    @profile_gpu
    def calculate_dislocation_causality(self,
                                       coord_defect_i: np.ndarray,
                                       coord_defect_j: np.ndarray,
                                       strain_i: np.ndarray,
                                       strain_j: np.ndarray,
                                       lag_window: Optional[int] = None) -> MaterialCausalityResult:
        """
        転位伝播の因果性を計算
        
        Parameters
        ----------
        coord_defect_i, coord_defect_j : np.ndarray
            配位数欠陥の時系列
        strain_i, strain_j : np.ndarray
            歪みの時系列
        lag_window : int, optional
            ラグウィンドウ
        """
        if lag_window is None:
            lag_window = self.max_lag
        
        n_lags = min(lag_window, len(coord_defect_i) // 2)
        
        # GPU転送
        coord_i_gpu = self.to_gpu(coord_defect_i)
        coord_j_gpu = self.to_gpu(coord_defect_j)
        strain_i_gpu = self.to_gpu(strain_i)
        strain_j_gpu = self.to_gpu(strain_j)
        
        if self.is_gpu and self.dislocation_kernel is not None:
            # カスタムカーネル使用
            causality_profile = self.zeros(n_lags, dtype=cp.float32)
            
            block_size = 256
            grid_size = (n_lags + block_size - 1) // block_size
            
            self.dislocation_kernel(
                (grid_size,), (block_size,),
                (coord_i_gpu, coord_j_gpu, strain_i_gpu, strain_j_gpu,
                 causality_profile, len(coord_defect_i), n_lags,
                 self.defect_threshold, self.strain_threshold)
            )
        else:
            # 汎用版
            causality_profile = self._calculate_dislocation_causality_generic(
                coord_i_gpu, coord_j_gpu, strain_i_gpu, strain_j_gpu, n_lags
            )
        
        # 最大値と最適ラグ
        max_causality = float(self.xp.max(causality_profile))
        optimal_lag = int(self.xp.argmax(causality_profile))
        
        # 伝播速度の推定（Å/frame）
        # 典型的な転位速度を仮定
        propagation_velocity = None
        if optimal_lag > 0:
            propagation_velocity = 5.0 / optimal_lag  # 5Å距離を仮定
        
        # Transfer Entropy
        transfer_entropy = None
        if optimal_lag > 0:
            transfer_entropy = self.calculate_transfer_entropy(
                coord_i_gpu, coord_j_gpu, optimal_lag, k_history=2, l_history=2
            )
        
        result = MaterialCausalityResult(
            pair=(0, 1),
            causality_strength=max_causality,
            optimal_lag=optimal_lag,
            causality_profile=self.to_cpu(causality_profile),
            transfer_entropy=transfer_entropy,
            propagation_type='plastic',
            propagation_velocity=propagation_velocity,
            strain_coupling=float(self.xp.mean(self.xp.abs(strain_i_gpu)))
        )
        
        return result
    
    def _calculate_dislocation_causality_generic(self,
                                                coord_i: Union[np.ndarray, cp.ndarray],
                                                coord_j: Union[np.ndarray, cp.ndarray],
                                                strain_i: Union[np.ndarray, cp.ndarray],
                                                strain_j: Union[np.ndarray, cp.ndarray],
                                                n_lags: int) -> Union[np.ndarray, cp.ndarray]:
        """転位因果性の汎用計算"""
        causality_profile = self.zeros(n_lags)
        
        # 転位イベント検出
        disl_events_i = (coord_i > self.defect_threshold) & (self.xp.abs(strain_i) > self.strain_threshold)
        disl_events_j = (coord_j > self.defect_threshold) & (self.xp.abs(strain_j) > self.strain_threshold)
        
        for lag in range(1, n_lags):
            cause = disl_events_i[:-lag]
            effect = disl_events_j[lag:]
            
            # 転位伝播確率
            cause_mask = cause > 0
            if self.xp.sum(cause_mask) > 0:
                propagation_prob = self.xp.mean(effect[cause_mask])
                
                # 歪み場の影響
                strain_factor = 1.0 + self.xp.mean(self.xp.abs(strain_i[:-lag][cause_mask]))
                
                causality_profile[lag] = propagation_prob * strain_factor
        
        return causality_profile
    
    @handle_gpu_errors
    @profile_gpu
    def calculate_strain_causality(self,
                                  strain_tensor_i: np.ndarray,
                                  strain_tensor_j: np.ndarray,
                                  lag_window: Optional[int] = None) -> MaterialCausalityResult:
        """
        歪み伝播の因果性を計算
        
        Parameters
        ----------
        strain_tensor_i, strain_tensor_j : np.ndarray
            歪みテンソルの時系列 (n_frames, 3, 3)
        lag_window : int, optional
            ラグウィンドウ
        """
        if lag_window is None:
            lag_window = self.max_lag
        
        n_frames = strain_tensor_i.shape[0]
        n_lags = min(lag_window, n_frames // 2)
        
        # フラット化してGPU転送
        strain_i_flat = strain_tensor_i.reshape(n_frames, -1)
        strain_j_flat = strain_tensor_j.reshape(n_frames, -1)
        
        strain_i_gpu = self.to_gpu(strain_i_flat)
        strain_j_gpu = self.to_gpu(strain_j_flat)
        
        if self.is_gpu and self.strain_kernel is not None:
            # カスタムカーネル使用
            causality_profile = self.zeros(n_lags, dtype=cp.float32)
            
            block_size = 256
            grid_size = (n_lags + block_size - 1) // block_size
            
            self.strain_kernel(
                (grid_size,), (block_size,),
                (strain_i_gpu, strain_j_gpu, causality_profile, n_frames, n_lags)
            )
        else:
            # 汎用版
            causality_profile = self._calculate_strain_causality_generic(
                strain_i_gpu, strain_j_gpu, n_lags
            )
        
        # 最大値と最適ラグ
        max_causality = float(self.xp.max(causality_profile))
        optimal_lag = int(self.xp.argmax(causality_profile))
        
        # 歪み相関
        strain_corr = float(self.xp.corrcoef(
            strain_i_gpu.flatten(), strain_j_gpu.flatten()
        )[0, 1])
        
        result = MaterialCausalityResult(
            pair=(0, 1),
            causality_strength=max_causality,
            optimal_lag=optimal_lag,
            causality_profile=self.to_cpu(causality_profile),
            propagation_type='elastic',
            strain_coupling=abs(strain_corr)
        )
        
        return result
    
    def _calculate_strain_causality_generic(self,
                                          strain_i: Union[np.ndarray, cp.ndarray],
                                          strain_j: Union[np.ndarray, cp.ndarray],
                                          n_lags: int) -> Union[np.ndarray, cp.ndarray]:
        """歪み因果性の汎用計算"""
        causality_profile = self.zeros(n_lags)
        
        for lag in range(1, n_lags):
            # Frobenius内積による相関
            correlation = 0.0
            for t in range(len(strain_i) - lag):
                dot_product = self.xp.sum(strain_i[t] * strain_j[t + lag])
                norm_i = self.xp.linalg.norm(strain_i[t])
                norm_j = self.xp.linalg.norm(strain_j[t + lag])
                
                if norm_i > 1e-6 and norm_j > 1e-6:
                    correlation += dot_product / (norm_i * norm_j)
            
            causality_profile[lag] = correlation / (len(strain_i) - lag)
        
        return causality_profile
    
    @handle_gpu_errors
    @profile_gpu
    def calculate_crack_causality(self,
                                 damage_i: np.ndarray,
                                 damage_j: np.ndarray,
                                 stress_intensity: Optional[np.ndarray] = None,
                                 lag_window: Optional[int] = None) -> MaterialCausalityResult:
        """
        亀裂伝播の因果性を計算
        
        Parameters
        ----------
        damage_i, damage_j : np.ndarray
            損傷度の時系列
        stress_intensity : np.ndarray, optional
            応力拡大係数
        lag_window : int, optional
            ラグウィンドウ
        """
        if lag_window is None:
            lag_window = self.max_lag
        
        n_frames = len(damage_i)
        n_lags = min(lag_window, n_frames // 2)
        
        # GPU転送
        damage_i_gpu = self.to_gpu(damage_i)
        damage_j_gpu = self.to_gpu(damage_j)
        
        if stress_intensity is None:
            # デフォルト値
            stress_intensity = np.ones(n_frames) * 0.5
        stress_gpu = self.to_gpu(stress_intensity)
        
        if self.is_gpu and self.crack_kernel is not None:
            # カスタムカーネル使用
            causality_profile = self.zeros(n_lags, dtype=cp.float32)
            
            block_size = 256
            grid_size = (n_lags + block_size - 1) // block_size
            
            self.crack_kernel(
                (grid_size,), (block_size,),
                (damage_i_gpu, damage_j_gpu, stress_gpu, causality_profile,
                 n_frames, n_lags, self.damage_threshold, self.k_ic)
            )
        else:
            # 汎用版
            causality_profile = self._calculate_crack_causality_generic(
                damage_i_gpu, damage_j_gpu, stress_gpu, n_lags
            )
        
        # 最大値と最適ラグ
        max_causality = float(self.xp.max(causality_profile))
        optimal_lag = int(self.xp.argmax(causality_profile))
        
        # 損傷相関
        damage_corr = float(self.xp.corrcoef(damage_i_gpu, damage_j_gpu)[0, 1])
        
        result = MaterialCausalityResult(
            pair=(0, 1),
            causality_strength=max_causality,
            optimal_lag=optimal_lag,
            causality_profile=self.to_cpu(causality_profile),
            propagation_type='fracture',
            damage_correlation=abs(damage_corr)
        )
        
        return result
    
    def _calculate_crack_causality_generic(self,
                                         damage_i: Union[np.ndarray, cp.ndarray],
                                         damage_j: Union[np.ndarray, cp.ndarray],
                                         stress_intensity: Union[np.ndarray, cp.ndarray],
                                         n_lags: int) -> Union[np.ndarray, cp.ndarray]:
        """亀裂因果性の汎用計算"""
        causality_profile = self.zeros(n_lags)
        
        # 亀裂イベント
        crack_i = damage_i > self.damage_threshold
        crack_j = damage_j > self.damage_threshold
        
        for lag in range(1, n_lags):
            cause = crack_i[:-lag]
            effect = crack_j[lag:]
            
            # 亀裂伝播確率
            cause_mask = cause > 0
            if self.xp.sum(cause_mask) > 0:
                prop_prob = self.xp.mean(effect[cause_mask])
                
                # 応力拡大係数の影響（Paris則）
                avg_k = self.xp.mean(stress_intensity[:-lag][cause_mask])
                paris_factor = (avg_k / self.k_ic) ** 2 if avg_k > 0 else 0.1
                
                causality_profile[lag] = prop_prob * paris_factor
        
        return causality_profile
    
    # ========================================
    # 継承メソッド（基本機能は残基版と同じ）
    # ========================================
    
    def calculate_transfer_entropy(self,
                                  source: Union[np.ndarray, cp.ndarray],
                                  target: Union[np.ndarray, cp.ndarray],
                                  lag: int = 1,
                                  k_history: Optional[int] = None,
                                  l_history: Optional[int] = None) -> float:
        """Transfer Entropy計算（残基版と同じロジック）"""
        if k_history is None:
            k_history = min(self.history_length, 2)  # 材料は短め
        if l_history is None:
            l_history = min(self.history_length, 2)
        
        # GPU変換
        if not isinstance(source, (cp.ndarray if HAS_GPU else np.ndarray)):
            source = self.to_gpu(source)
        if not isinstance(target, (cp.ndarray if HAS_GPU else np.ndarray)):
            target = self.to_gpu(target)
        
        n_frames = min(len(source), len(target))
        max_history = max(k_history, l_history)
        
        # データが短すぎる場合
        if n_frames < max_history + lag + 10:
            return 0.0
        
        # 信号の離散化
        source_discrete = self._discretize_signal(source, self.n_bins)
        target_discrete = self._discretize_signal(target, self.n_bins)
        
        # 簡易計算
        te = 0.0
        for t in range(max_history, n_frames - lag):
            # 予測誤差の改善を評価
            y_past = target_discrete[t - 1]
            x_past = source_discrete[t - lag - 1] if t >= lag + 1 else 0
            y_future = target_discrete[t]
            
            # 条件付きエントロピーの近似
            if y_past != y_future:
                te += 0.1  # 簡易スコア
            if x_past == y_future:
                te += 0.2
        
        return te / (n_frames - max_history - lag)
    
    def _discretize_signal(self, 
                          signal: Union[np.ndarray, cp.ndarray], 
                          n_bins: int) -> Union[np.ndarray, cp.ndarray]:
        """信号を離散化（残基版と同じ）"""
        percentiles = self.xp.linspace(0, 100, n_bins + 1)
        bin_edges = self.xp.percentile(signal, percentiles)
        bin_edges = self.xp.unique(bin_edges)
        
        if len(bin_edges) < 2:
            return self.zeros_like(signal, dtype=self.xp.int32)
        
        discrete = self.xp.digitize(signal, bin_edges[1:-1])
        return discrete.astype(self.xp.int32)
    
    @handle_gpu_errors
    def detect_causal_pairs(self,
                          anomaly_scores: Dict[int, np.ndarray],
                          coordination_numbers: Optional[Dict[int, np.ndarray]] = None,
                          local_strain: Optional[Dict[int, np.ndarray]] = None,
                          damage_scores: Optional[Dict[int, np.ndarray]] = None,
                          threshold: float = 0.2,
                          batch_size: int = 50) -> List[MaterialCausalityResult]:
        """
        複数クラスターペアの因果関係を検出（材料版）
        
        Parameters
        ----------
        anomaly_scores : dict
            クラスターID -> 異常スコア
        coordination_numbers : dict, optional
            クラスターID -> 配位数
        local_strain : dict, optional
            クラスターID -> 局所歪み
        damage_scores : dict, optional
            クラスターID -> 損傷度
        threshold : float
            因果強度閾値
        batch_size : int
            バッチサイズ
        """
        cluster_ids = sorted(anomaly_scores.keys())
        n_clusters = len(cluster_ids)
        causal_pairs = []
        
        logger.info(f"⚙️ Detecting causal pairs among {n_clusters} clusters")
        
        # 因果タイプを判定
        if coordination_numbers is not None and local_strain is not None:
            causality_type = 'dislocation'
        elif local_strain is not None:
            causality_type = 'strain'
        elif damage_scores is not None:
            causality_type = 'crack'
        else:
            causality_type = 'general'
        
        logger.info(f"   Causality type: {causality_type}")
        
        # ペア処理
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i = cluster_ids[i]
                cluster_j = cluster_ids[j]
                
                result = None
                
                if causality_type == 'dislocation':
                    # 転位因果性
                    result = self.calculate_dislocation_causality(
                        coordination_numbers[cluster_i] - 12.0,  # 欠陥
                        coordination_numbers[cluster_j] - 12.0,
                        local_strain[cluster_i].flatten(),
                        local_strain[cluster_j].flatten()
                    )
                elif causality_type == 'strain':
                    # 歪み因果性
                    result = self.calculate_strain_causality(
                        local_strain[cluster_i],
                        local_strain[cluster_j]
                    )
                elif causality_type == 'crack':
                    # 亀裂因果性
                    result = self.calculate_crack_causality(
                        damage_scores[cluster_i],
                        damage_scores[cluster_j]
                    )
                else:
                    # 一般的な因果性（anomaly_scoresベース）
                    result = self._calculate_general_causality(
                        anomaly_scores[cluster_i],
                        anomaly_scores[cluster_j]
                    )
                
                if result and result.causality_strength >= threshold:
                    result.pair = (cluster_i, cluster_j)
                    causal_pairs.append(result)
        
        # 強度でソート
        causal_pairs.sort(key=lambda x: x.causality_strength, reverse=True)
        
        logger.info(f"   Found {len(causal_pairs)} causal relationships")
        
        return causal_pairs
    
    def _calculate_general_causality(self,
                                    anomaly_i: np.ndarray,
                                    anomaly_j: np.ndarray) -> MaterialCausalityResult:
        """一般的な因果性計算（フォールバック）"""
        n_lags = min(self.max_lag, len(anomaly_i) // 2)
        causality_profile = self.zeros(n_lags)
        
        # 簡易相関ベース
        for lag in range(1, n_lags):
            if lag < len(anomaly_i):
                corr = self.xp.corrcoef(
                    anomaly_i[:-lag], 
                    anomaly_j[lag:]
                )[0, 1]
                causality_profile[lag] = abs(corr)
        
        max_causality = float(self.xp.max(causality_profile))
        optimal_lag = int(self.xp.argmax(causality_profile))
        
        return MaterialCausalityResult(
            pair=(0, 1),
            causality_strength=max_causality,
            optimal_lag=optimal_lag,
            causality_profile=self.to_cpu(causality_profile),
            propagation_type='general'
        )

# ===============================
# Standalone Functions (Material版)
# ===============================

def calculate_dislocation_causality_gpu(coord_defect_i: np.ndarray,
                                       coord_defect_j: np.ndarray,
                                       strain_i: np.ndarray,
                                       strain_j: np.ndarray,
                                       lag_window: int = 100,
                                       **kwargs) -> MaterialCausalityResult:
    """転位因果性計算のスタンドアロン関数"""
    analyzer = MaterialCausalityAnalyzerGPU(**kwargs)
    return analyzer.calculate_dislocation_causality(
        coord_defect_i, coord_defect_j, strain_i, strain_j, lag_window
    )

def calculate_strain_causality_gpu(strain_tensor_i: np.ndarray,
                                  strain_tensor_j: np.ndarray,
                                  lag_window: int = 100,
                                  **kwargs) -> MaterialCausalityResult:
    """歪み因果性計算のスタンドアロン関数"""
    analyzer = MaterialCausalityAnalyzerGPU(**kwargs)
    return analyzer.calculate_strain_causality(
        strain_tensor_i, strain_tensor_j, lag_window
    )

def calculate_crack_causality_gpu(damage_i: np.ndarray,
                                 damage_j: np.ndarray,
                                 stress_intensity: Optional[np.ndarray] = None,
                                 lag_window: int = 100,
                                 **kwargs) -> MaterialCausalityResult:
    """亀裂因果性計算のスタンドアロン関数"""
    analyzer = MaterialCausalityAnalyzerGPU(**kwargs)
    return analyzer.calculate_crack_causality(
        damage_i, damage_j, stress_intensity, lag_window
    )

def detect_material_causal_pairs_gpu(anomaly_scores: Dict[int, np.ndarray],
                                    coordination_numbers: Optional[Dict[int, np.ndarray]] = None,
                                    local_strain: Optional[Dict[int, np.ndarray]] = None,
                                    damage_scores: Optional[Dict[int, np.ndarray]] = None,
                                    threshold: float = 0.2,
                                    **kwargs) -> List[MaterialCausalityResult]:
    """材料因果ペア検出のスタンドアロン関数"""
    analyzer = MaterialCausalityAnalyzerGPU(**kwargs)
    return analyzer.detect_causal_pairs(
        anomaly_scores, coordination_numbers, local_strain, 
        damage_scores, threshold
    )
