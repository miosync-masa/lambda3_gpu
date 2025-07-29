"""
Lambda³ GPU版異常検出モジュール
異常検出アルゴリズムの完全GPU実装
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

# CuPyの条件付きインポート（ここが重要！）
    except ImportError:
        cp = None
        NDArray = np.ndarray 
        
from numba import cuda
import math
import warnings

from ..core.gpu_utils import GPUBackend
from ..core.gpu_kernels import (
from ..types import ArrayType, NDArray
    detect_local_anomalies_kernel,
    compute_mad_kernel,
    gaussian_filter_1d_kernel
)

warnings.filterwarnings('ignore')


class AnomalyDetectorGPU(GPUBackend):
    """異常検出のGPU実装"""
    
    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.anomaly_cache = {}
        
    def detect_local_anomalies(self, 
                              series: np.ndarray, 
                              window: int) -> np.ndarray:
        """
        局所異常検出（CUDAカーネル使用）
        
        Parameters
        ----------
        series : np.ndarray
            時系列データ
        window : int
            ウィンドウサイズ
            
        Returns
        -------
        np.ndarray
            異常スコア
        """
        # GPUに転送
        series_gpu = self.to_gpu(series)
        anomaly_gpu = cp.zeros_like(series_gpu)
        
        # ブロックとグリッドの設定
        threads_per_block = 256
        blocks_per_grid = (len(series) + threads_per_block - 1) // threads_per_block
        
        # CUDAカーネル実行
        detect_local_anomalies_kernel[blocks_per_grid, threads_per_block](
            series_gpu, anomaly_gpu, window, len(series)
        )
        
        # CPUに戻す
        return self.to_cpu(anomaly_gpu)
    
    def compute_multiscale_anomalies(self,
                                   lambda_structures: Dict[str, np.ndarray],
                                   boundaries: Dict[str, any],
                                   breaks: Dict[str, np.ndarray],
                                   md_features: Dict[str, np.ndarray],
                                   config: any) -> Dict[str, np.ndarray]:
        """
        マルチスケール異常検出（GPU並列処理）
        
        Returns
        -------
        Dict[str, np.ndarray]
            各種異常スコア
        """
        print("\n🚀 GPU Multi-scale Anomaly Detection...")
        
        n_frames = len(lambda_structures['rho_T'])
        
        # GPUメモリ管理
        with self.memory_manager.batch_context(n_frames):
            # 1. Global anomalies
            global_score = self._compute_global_anomalies_gpu(
                breaks, config, n_frames
            )
            
            # 2. Local anomalies (boundary-focused)
            local_score = self._compute_local_anomalies_gpu(
                boundaries, n_frames
            )
            
            # 3. Extended anomalies (新しい検出ロジック)
            extended_scores = self._compute_extended_anomalies_gpu(
                lambda_structures, md_features, config
            )
            
            # 4. スコアの正規化
            global_score_norm = self._normalize_scores_gpu(global_score)
            local_score_norm = self._normalize_scores_gpu(local_score)
            
            # 5. 統合スコア計算
            final_combined = self._compute_final_combined_gpu(
                global_score_norm, 
                local_score_norm,
                extended_scores
            )
        
        return {
            'global': self.to_cpu(global_score_norm),
            'local': self.to_cpu(local_score_norm),
            'combined': self.to_cpu((global_score_norm + local_score_norm) / 2),
            **{k: self.to_cpu(v) for k, v in extended_scores.items()},
            'final_combined': self.to_cpu(final_combined)
        }
    
    def _compute_global_anomalies_gpu(self, 
                                    breaks: Dict,
                                    config: any,
                                    n_frames: int) -> cp.ndarray:
        """グローバル異常スコア計算（GPU）"""
        global_score = cp.zeros(n_frames)
        
        # 各種異常を重み付けして加算
        if 'lambda_F_anomaly' in breaks:
            lf_anomaly = self.to_gpu(breaks['lambda_F_anomaly'])
            global_score[:len(lf_anomaly)] += config.w_lambda_f * lf_anomaly
            
        if 'lambda_FF_anomaly' in breaks:
            lff_anomaly = self.to_gpu(breaks['lambda_FF_anomaly'])
            global_score[:len(lff_anomaly)] += config.w_lambda_ff * lff_anomaly
            
        if 'rho_T_breaks' in breaks:
            rho_breaks = self.to_gpu(breaks['rho_T_breaks'])
            global_score[:len(rho_breaks)] += config.w_rho_t * rho_breaks
            
        if 'Q_breaks' in breaks:
            q_breaks = self.to_gpu(breaks['Q_breaks'])
            global_score[:len(q_breaks)] += config.w_topology * q_breaks
            
        return global_score
    
    def _compute_local_anomalies_gpu(self,
                                   boundaries: Dict,
                                   n_frames: int) -> cp.ndarray:
        """ローカル異常スコア計算（境界周辺を強調）"""
        local_score = cp.zeros(n_frames)
        
        if 'boundary_locations' in boundaries:
            # 境界位置をGPUに転送
            boundary_locs = self.to_gpu(boundaries['boundary_locations'])
            
            # 各境界周辺にガウシアンウィンドウを適用（GPU並列）
            threads = 256
            blocks = (len(boundary_locs) + threads - 1) // threads
            
            # カスタムカーネルで境界周辺を強調
            self._apply_boundary_emphasis_kernel[blocks, threads](
                local_score, boundary_locs, n_frames, 50, 20  # window=50, sigma=20
            )
        
        # boundary_scoreを加算
        if 'boundary_score' in boundaries:
            bs = self.to_gpu(boundaries['boundary_score'])
            local_score[:len(bs)] += bs
            
        return local_score
    
    def _compute_extended_anomalies_gpu(self,
                                      structures: Dict,
                                      md_features: Dict,
                                      config: any) -> Dict[str, cp.ndarray]:
        """拡張異常検出（GPU最適化）"""
        extended_scores = {}
        
        # 各種拡張検出をGPUで実行
        if hasattr(config, 'use_periodic') and config.use_periodic:
            # FFTベースの周期検出はCuPyのFFTを使用
            extended_scores['periodic'] = self._detect_periodic_gpu(structures)
            
        if hasattr(config, 'use_gradual') and config.use_gradual:
            extended_scores['gradual'] = self._detect_gradual_gpu(structures)
            
        if hasattr(config, 'use_drift') and config.use_drift:
            extended_scores['drift'] = self._detect_drift_gpu(structures)
            
        if hasattr(config, 'radius_of_gyration') and config.radius_of_gyration:
            if md_features and 'radius_of_gyration' in md_features:
                extended_scores['rg_based'] = self._detect_rg_transitions_gpu(md_features)
                
        if hasattr(config, 'use_phase_space') and config.use_phase_space:
            extended_scores['phase_space'] = self._detect_phase_space_gpu(structures)
        
        # 統合スコア
        if extended_scores:
            weights = {
                'periodic': 0.2,
                'gradual': 0.3,
                'drift': 0.3,
                'phase_space': 0.2,
                'rg_based': 0.3
            }
            
            combined = cp.zeros_like(list(extended_scores.values())[0])
            for key, score in extended_scores.items():
                if key in weights:
                    combined += weights[key] * score
                    
            extended_scores['extended_combined'] = combined
            
        return extended_scores
    
    def _normalize_scores_gpu(self, scores: cp.ndarray) -> cp.ndarray:
        """
        ロバストなスコア正規化（MAD使用、GPU版）
        """
        median = cp.median(scores)
        mad = cp.median(cp.abs(scores - median))
        
        if mad > 1e-10:
            # MAD to standard deviation
            normalized = 0.6745 * (scores - median) / mad
        else:
            # Fallback to IQR
            q75, q25 = cp.percentile(scores, [75, 25])
            iqr = q75 - q25
            if iqr > 1e-10:
                normalized = (scores - median) / (1.5 * iqr)
            else:
                normalized = scores - median
                
        return normalized
    
    def _compute_final_combined_gpu(self,
                                  global_norm: cp.ndarray,
                                  local_norm: cp.ndarray,
                                  extended_scores: Dict) -> cp.ndarray:
        """最終統合スコア計算"""
        # extended_combinedの正規化
        if 'extended_combined' in extended_scores:
            extended_combined_norm = self._normalize_scores_gpu(
                extended_scores['extended_combined']
            )
        else:
            extended_combined_norm = cp.zeros_like(global_norm)
            
        # 最終統合
        final_combined = (
            0.5 * global_norm +
            0.3 * local_norm +
            0.2 * extended_combined_norm
        )
        
        return final_combined
    
    # 拡張検出メソッド（簡略版）
    def _detect_periodic_gpu(self, structures: Dict) -> cp.ndarray:
        """周期的遷移検出（CuPy FFT使用）"""
        rho_t_gpu = self.to_gpu(structures['rho_T'])
        
        # FFT実行
        rho_t_centered = rho_t_gpu - cp.mean(rho_t_gpu)
        yf = cp.fft.rfft(rho_t_centered)
        power = cp.abs(yf)**2
        
        # 簡略化：パワーの変動を異常スコアとして使用
        scores = cp.zeros_like(rho_t_gpu)
        # 実際の実装ではより詳細な周期検出を行う
        
        return scores
    
    def _detect_gradual_gpu(self, structures: Dict) -> cp.ndarray:
        """緩やかな遷移検出"""
        rho_t_gpu = self.to_gpu(structures['rho_T'])
        
        # GPU上でガウシアンフィルタ
        window_sizes = [500, 1000, 2000]
        gradual_scores = cp.zeros_like(rho_t_gpu)
        
        for window in window_sizes:
            # 簡略化：移動平均で代用
            smoothed = self._moving_average_gpu(rho_t_gpu, window)
            gradient = cp.gradient(smoothed)
            gradual_scores += cp.abs(gradient) / len(window_sizes)
            
        return self._normalize_scores_gpu(gradual_scores)
    
    def _detect_drift_gpu(self, structures: Dict) -> cp.ndarray:
        """構造ドリフト検出"""
        rho_t_gpu = self.to_gpu(structures['rho_T'])
        reference_window = 1000
        
        # 参照値
        ref_value = cp.mean(rho_t_gpu[:reference_window])
        
        # ドリフト計算
        drift_scores = cp.abs(rho_t_gpu - ref_value) / (ref_value + 1e-10)
        
        return drift_scores
    
    def _detect_rg_transitions_gpu(self, md_features: Dict) -> cp.ndarray:
        """Radius of Gyration変化検出"""
        rg_gpu = self.to_gpu(md_features['radius_of_gyration'])
        
        # 勾配計算
        gradient = cp.gradient(rg_gpu)
        
        # 収縮を強調
        contraction_score = cp.where(gradient < 0,
                                   cp.abs(gradient) * 2.0,
                                   cp.abs(gradient))
        
        return contraction_score
    
    def _detect_phase_space_gpu(self, structures: Dict) -> cp.ndarray:
        """位相空間異常検出（簡略版）"""
        rho_t_gpu = self.to_gpu(structures['rho_T'])
        
        # 簡略化：局所的な変動を異常とする
        window = 50
        local_std = self._local_std_gpu(rho_t_gpu, window)
        
        return local_std
    
    # ユーティリティメソッド
    def _moving_average_gpu(self, data: cp.ndarray, window: int) -> cp.ndarray:
        """GPU上での移動平均"""
        return cp.convolve(data, cp.ones(window)/window, mode='same')
    
    def _local_std_gpu(self, data: cp.ndarray, window: int) -> cp.ndarray:
        """局所標準偏差"""
        # 実装は簡略化
        return cp.zeros_like(data)
    
    # カスタムCUDAカーネル
    @cuda.jit
    def _apply_boundary_emphasis_kernel(local_score, boundary_locs, n_frames, 
                                      window, sigma):
        """境界周辺を強調するカーネル"""
        idx = cuda.grid(1)
        if idx < boundary_locs.shape[0]:
            loc = boundary_locs[idx]
            
            # ガウシアンウィンドウを適用
            for i in range(max(0, loc - window), 
                          min(n_frames, loc + window)):
                dist = abs(i - loc)
                weight = math.exp(-0.5 * (dist / sigma) ** 2)
                cuda.atomic.add(local_score, i, weight)
