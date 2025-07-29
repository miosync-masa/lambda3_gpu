"""
Lambda³ GPU版トポロジカル破れ検出モジュール
構造フローのトポロジカルな破れをGPUで高速検出
"""

import numpy as np
import cupy as cp
from typing import Dict, List, Tuple, Optional
from numba import cuda
from cupyx.scipy.ndimage import gaussian_filter1d as gaussian_filter1d_gpu

from ..types import ArrayType, NDArray
from ..core.gpu_utils import GPUBackend
from ..core.gpu_kernels import (
    anomaly_detection_kernel  # 正しい名前でインポート
)

class TopologyBreaksDetectorGPU(GPUBackend):
    """トポロジカル破れ検出のGPU実装"""
    
    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.breaks_cache = {}
        
    def detect_topological_breaks(self,
                                structures: Dict[str, np.ndarray],
                                window_steps: int) -> Dict[str, np.ndarray]:
        """
        トポロジカル破れの検出（完全GPU版）
        
        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Lambda構造辞書
        window_steps : int
            ウィンドウサイズ
            
        Returns
        -------
        Dict[str, np.ndarray]
            各種破れの検出結果
        """
        print("\n💥 Detecting topological breaks on GPU...")
        
        n_frames = len(structures['rho_T'])
        
        with self.memory_manager.batch_context(n_frames):
            # 1. ΛF異常（構造フロー破れ）
            lambda_f_anomaly = self._detect_flow_anomalies_gpu(
                structures['lambda_F_mag'], window_steps
            )
            
            # 2. ΛFF異常（加速度破れ）
            lambda_ff_anomaly = self._detect_acceleration_anomalies_gpu(
                structures['lambda_FF_mag'], window_steps // 2
            )
            
            # 3. テンション場ジャンプ
            rho_t_breaks = self._detect_tension_field_jumps_gpu(
                structures['rho_T'], window_steps
            )
            
            # 4. トポロジカルチャージ異常
            q_breaks = self._detect_topological_charge_breaks_gpu(
                structures['Q_lambda']
            )
            
            # 5. 位相コヒーレンス破れ（新規追加）
            phase_coherence_breaks = self._detect_phase_coherence_breaks_gpu(
                structures
            )
            
            # 6. 構造的特異点検出（新規追加）
            singularities = self._detect_structural_singularities_gpu(
                structures, window_steps
            )
            
            # 7. 統合異常スコア
            combined_anomaly = self._combine_topological_anomalies_gpu(
                lambda_f_anomaly,
                lambda_ff_anomaly,
                rho_t_breaks,
                q_breaks,
                phase_coherence_breaks,
                singularities
            )
        
        return {
            'lambda_F_anomaly': self.to_cpu(lambda_f_anomaly),
            'lambda_FF_anomaly': self.to_cpu(lambda_ff_anomaly),
            'rho_T_breaks': self.to_cpu(rho_t_breaks),
            'Q_breaks': self.to_cpu(q_breaks),
            'phase_coherence_breaks': self.to_cpu(phase_coherence_breaks),
            'singularities': self.to_cpu(singularities),
            'combined_anomaly': self.to_cpu(combined_anomaly)
        }
    
    def _detect_flow_anomalies_gpu(self,
                                 lambda_f_mag: np.ndarray,
                                 window: int) -> NDArray:
        """構造フローの異常検出（GPU最適化）"""
        lf_mag_gpu = self.to_gpu(lambda_f_mag)
        
        # 適応的z-scoreによる異常検出（カーネル使用）
        anomaly_gpu = anomaly_detection_kernel(lf_mag_gpu, window)
        
        # 追加: 急激な変化の検出
        gradient = cp.abs(cp.gradient(lf_mag_gpu))
        sudden_changes = self._detect_sudden_changes_gpu(gradient, window)
        
        # 両方の異常を統合
        return cp.maximum(anomaly_gpu, sudden_changes)
    
    def _detect_acceleration_anomalies_gpu(self,
                                         lambda_ff_mag: np.ndarray,
                                         window: int) -> NDArray:
        """加速度異常の検出"""
        lff_mag_gpu = self.to_gpu(lambda_ff_mag)
        
        # 基本的な異常検出（カーネル使用）
        anomaly_gpu = anomaly_detection_kernel(lff_mag_gpu, window)
        
        # 加速度特有の処理：符号変化の検出
        if 'lambda_FF' in self.breaks_cache:
            lambda_ff = self.to_gpu(self.breaks_cache['lambda_FF'])
            sign_changes = self._detect_sign_changes_gpu(lambda_ff)
            anomaly_gpu = cp.maximum(anomaly_gpu, sign_changes)
        
        return anomaly_gpu
    
    def _detect_tension_field_jumps_gpu(self,
                                      rho_t: np.ndarray,
                                      window_steps: int) -> NDArray:
        """テンション場のジャンプ検出（改良版）"""
        rho_t_gpu = self.to_gpu(rho_t)
        
        # マルチスケールスムージング
        sigmas = [window_steps/6, window_steps/3, window_steps/2]
        jumps_multiscale = cp.zeros_like(rho_t_gpu)
        
        for sigma in sigmas:
            # ガウシアンフィルタ
            rho_t_smooth = gaussian_filter1d_gpu(rho_t_gpu, sigma=sigma)
            
            # ジャンプ検出
            jumps = cp.abs(rho_t_gpu - rho_t_smooth)
            
            # 正規化
            jumps_norm = jumps / (cp.std(jumps) + 1e-10)
            
            jumps_multiscale += jumps_norm / len(sigmas)
        
        return jumps_multiscale
    
    def _detect_topological_charge_breaks_gpu(self,
                                            q_lambda: np.ndarray) -> NDArray:
        """トポロジカルチャージの破れ検出"""
        q_lambda_gpu = self.to_gpu(q_lambda)
        breaks = cp.zeros_like(q_lambda_gpu)
        
        # 位相差の計算
        phase_diff = cp.abs(cp.diff(q_lambda_gpu))
        
        # 閾値以上の急激な変化を検出
        threshold = 0.1  # 0.1 * 2π radians
        breaks[1:] = cp.where(phase_diff > threshold, phase_diff, 0)
        
        # 累積的な破れの検出
        cumulative_breaks = self._detect_cumulative_breaks_gpu(q_lambda_gpu)
        
        return cp.maximum(breaks, cumulative_breaks)
    
    def _detect_phase_coherence_breaks_gpu(self,
                                         structures: Dict) -> NDArray:
        """位相コヒーレンスの破れ検出（新機能）"""
        if 'structural_coherence' not in structures:
            return cp.zeros(len(structures['rho_T']))
        
        coherence_gpu = self.to_gpu(structures['structural_coherence'])
        
        # コヒーレンスの急激な低下を検出
        coherence_gradient = cp.gradient(coherence_gpu)
        
        # 負の勾配（コヒーレンス低下）を強調
        breaks = cp.where(coherence_gradient < 0,
                         -coherence_gradient * 2.0,
                         cp.abs(coherence_gradient))
        
        # 閾値処理
        threshold = cp.mean(breaks) + 2 * cp.std(breaks)
        breaks = cp.where(breaks > threshold, breaks, 0)
        
        return breaks
    
    def _detect_structural_singularities_gpu(self,
                                           structures: Dict,
                                           window: int) -> NDArray:
        """構造的特異点の検出（新機能）"""
        n_frames = len(structures['rho_T'])
        singularities = cp.zeros(n_frames)
        
        # 複数の指標から特異点を検出
        rho_t_gpu = self.to_gpu(structures['rho_T'])
        lf_mag_gpu = self.to_gpu(structures['lambda_F_mag'])
        
        # 1. テンション場の局所極値
        tension_extrema = self._find_local_extrema_gpu(rho_t_gpu, window)
        
        # 2. フロー場の発散/収束
        if len(structures['lambda_F'].shape) == 2:  # ベクトル場の場合
            lambda_f_gpu = self.to_gpu(structures['lambda_F'])
            divergence = self._compute_divergence_gpu(lambda_f_gpu)
            div_anomaly = cp.abs(divergence) > cp.std(divergence) * 3
            singularities += div_anomaly.astype(cp.float32)
        
        # 3. 位相空間での異常軌道
        phase_anomaly = self._detect_phase_space_singularities_gpu(
            lf_mag_gpu, rho_t_gpu, window
        )
        
        singularities += tension_extrema + phase_anomaly
        
        return singularities / 3.0  # 正規化
    
    def _detect_sudden_changes_gpu(self,
                                 gradient: NDArray,
                                 window: int) -> NDArray:
        """急激な変化の検出"""
        # 移動標準偏差
        moving_std = self._moving_std_gpu(gradient, window)
        
        # 外れ値検出
        threshold = 3.0
        sudden_changes = cp.where(
            gradient > moving_std * threshold,
            gradient / (moving_std + 1e-10),
            0
        )
        
        return sudden_changes
    
    def _detect_sign_changes_gpu(self, vector_field: NDArray) -> NDArray:
        """符号変化の検出"""
        if len(vector_field.shape) == 1:
            # スカラー場
            sign_diff = cp.diff(cp.sign(vector_field))
            changes = cp.abs(sign_diff) / 2.0
            return cp.pad(changes, (1, 0), mode='constant')
        else:
            # ベクトル場
            changes = cp.zeros(len(vector_field))
            for i in range(vector_field.shape[1]):
                component = vector_field[:, i]
                sign_diff = cp.diff(cp.sign(component))
                changes[1:] += cp.abs(sign_diff) / (2.0 * vector_field.shape[1])
            return changes
    
    def _detect_cumulative_breaks_gpu(self, q_lambda: NDArray) -> NDArray:
        """累積的な破れの検出"""
        # 累積和
        q_cumsum = cp.cumsum(q_lambda)
        
        # 期待される線形成長からの乖離
        x = cp.arange(len(q_lambda))
        slope = (q_cumsum[-1] - q_cumsum[0]) / (len(q_lambda) - 1)
        expected = q_cumsum[0] + slope * x
        
        deviation = cp.abs(q_cumsum - expected)
        
        # 急激な乖離を検出
        deviation_gradient = cp.abs(cp.gradient(deviation))
        
        return deviation_gradient / (cp.max(deviation_gradient) + 1e-10)
    
    def _find_local_extrema_gpu(self,
                               data: NDArray,
                               window: int) -> NDArray:
        """局所極値の検出"""
        extrema = cp.zeros_like(data)
        
        # GPU並列処理で局所最大/最小を検出
        threads = 256
        blocks = (len(data) + threads - 1) // threads
        
        self._local_extrema_kernel[blocks, threads](
            data, extrema, window, len(data)
        )
        
        return extrema
    
    def _compute_divergence_gpu(self, vector_field: NDArray) -> NDArray:
        """ベクトル場の発散を計算"""
        if len(vector_field.shape) != 2:
            return cp.zeros(len(vector_field))
        
        # 各成分の偏微分
        div = cp.zeros(len(vector_field) - 1)
        for i in range(vector_field.shape[1]):
            component_grad = cp.gradient(vector_field[:, i])
            div += component_grad[:-1]
        
        return cp.pad(div, (0, 1), mode='edge')
    
    def _detect_phase_space_singularities_gpu(self,
                                            lf_mag: NDArray,
                                            rho_t: NDArray,
                                            window: int) -> NDArray:
        """位相空間での特異点検出"""
        n = len(lf_mag)
        singularities = cp.zeros(n)
        
        # 簡易的な位相空間埋め込み
        for i in range(window, n - window):
            # 局所的な軌道の異常性
            local_lf = lf_mag[i-window:i+window]
            local_rho = rho_t[i-window:i+window]
            
            # 相関の急激な変化
            if len(local_lf) > 5:
                corr = cp.corrcoef(local_lf, local_rho)[0, 1]
                if cp.isnan(corr):
                    corr = 0
                
                # 相関の絶対値が低い = 特異的
                singularities[i] = 1 - cp.abs(corr)
        
        return singularities
    
    def _moving_std_gpu(self, data: NDArray, window: int) -> NDArray:
        """移動標準偏差の計算"""
        # 簡易実装（より効率的な実装も可能）
        std_array = cp.zeros_like(data)
        
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            
            if end - start > 1:
                std_array[i] = cp.std(data[start:end])
        
        return std_array
    
    def _combine_topological_anomalies_gpu(self, *anomalies) -> NDArray:
        """トポロジカル異常の統合"""
        # 全ての長さを揃える
        min_len = min(len(a) for a in anomalies)
        
        # 重み（新しい破れタイプも含む）
        weights = [1.0, 0.8, 0.6, 1.2, 0.9, 1.1]
        
        combined = cp.zeros(min_len)
        
        for i, (anomaly, weight) in enumerate(zip(anomalies, weights)):
            if i < len(weights):
                combined += weight * anomaly[:min_len]
        
        combined /= sum(weights[:len(anomalies)])
        
        return combined
    
    # カスタムCUDAカーネル（@cuda.jitは残す）
    @cuda.jit
    def _local_extrema_kernel(data, extrema, window, n):
        """局所極値検出カーネル"""
        idx = cuda.grid(1)
        
        if idx >= window and idx < n - window:
            center = data[idx]
            is_max = True
            is_min = True
            
            # 局所範囲で比較
            for i in range(idx - window, idx + window + 1):
                if i != idx:
                    if data[i] >= center:
                        is_max = False
                    if data[i] <= center:
                        is_min = False
            
            if is_max or is_min:
                extrema[idx] = 1.0
