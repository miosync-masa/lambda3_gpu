"""
Lambda³ GPU版構造境界検出モジュール
構造境界（ΔΛC）検出のGPU最適化実装
"""
import numpy as np
import cupy as cp
from typing import Dict, List, Tuple, Any
from numba import cuda
from cupyx.scipy.signal import find_peaks as find_peaks_gpu
from cupyx.scipy.ndimage import gaussian_filter1d as gaussian_filter1d_gpu

from ..types import ArrayType, NDArray
from ..core.gpu_utils import GPUBackend

class BoundaryDetectorGPU(GPUBackend):
    """構造境界検出のGPU実装"""
    
    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.boundary_cache = {}
        
    def detect_structural_boundaries(self,
                                   structures: Dict[str, np.ndarray],
                                   window_steps: int) -> Dict[str, Any]:
        """
        構造境界検出（ΔΛC - 意味の結晶化モーメント）
        
        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Lambda構造辞書
        window_steps : int
            ウィンドウサイズ
            
        Returns
        -------
        Dict[str, Any]
            境界情報
        """
        print("\n🔍 Detecting structural boundaries (ΔΛC) on GPU...")
        
        n_steps = len(structures['rho_T'])
        
        # GPUメモリコンテキスト（memory_managerがない場合はスキップ）
        if hasattr(self, 'memory_manager'):
            with self.memory_manager.batch_context(n_steps):
                return self._detect_boundaries_impl(structures, window_steps, n_steps)
        else:
            return self._detect_boundaries_impl(structures, window_steps, n_steps)
    
    def _detect_boundaries_impl(self, structures, window_steps, n_steps):
        """境界検出の実装部分"""
        # 各指標をGPUで計算
        fractal_dims = self._compute_fractal_dimensions_gpu(
            structures['Q_cumulative'], window_steps
        )
        
        coherence = self._get_coherence_gpu(structures)
        
        coupling = self._compute_coupling_strength_gpu(
            structures['Q_cumulative'], window_steps
        )
        
        entropy = self._compute_structural_entropy_gpu(
            structures['rho_T'], window_steps
        )
        
        # 境界スコア計算
        boundary_score = self._compute_boundary_score_gpu(
            fractal_dims, coherence, coupling, entropy
        )
        
        # ピーク検出
        peaks, properties = self._detect_peaks_gpu(
            boundary_score, n_steps
        )
        
        # CPU形式で返す
        return {
            'boundary_score': self.to_cpu(boundary_score),
            'boundary_locations': self.to_cpu(peaks),
            'boundary_strengths': self.to_cpu(boundary_score[peaks]) if len(peaks) > 0 else np.array([]),
            'fractal_dimension': self.to_cpu(fractal_dims),
            'structural_coherence': self.to_cpu(coherence),
            'coupling_strength': self.to_cpu(coupling),
            'structural_entropy': self.to_cpu(entropy)
        }
    
    def _compute_fractal_dimensions_gpu(self,
                                      q_cumulative: np.ndarray,
                                      window: int) -> NDArray:
        """局所フラクタル次元の計算（GPU版）"""
        q_cum_gpu = self.to_gpu(q_cumulative)
        n = len(q_cum_gpu)
        dims = cp.ones(n) if self.is_gpu else np.ones(n)
        
        # CUDAカーネルの代わりに簡易実装
        threads = 256
        blocks = (n + threads - 1) // threads
        
        # カスタムカーネルの代わりにCuPyで実装
        for i in range(window, n - window):
            local_data = q_cum_gpu[i-window:i+window]
            # 簡易的なフラクタル次元計算
            if len(local_data) > 2:
                dims[i] = self._compute_local_fractal_dim(local_data)
        
        return dims
    
    def _compute_local_fractal_dim(self, data):
        """簡易的なフラクタル次元計算"""
        if len(data) < 3:
            return 1.0
        
        # ボックスカウント法の簡易版
        xp = cp if self.is_gpu else np
        eps_values = xp.logspace(-2, 0, 10)
        counts = []
        
        for eps in eps_values:
            count = len(xp.unique(xp.floor(data / eps)))
            counts.append(count)
        
        if len(counts) > 2:
            log_eps = xp.log(eps_values)
            log_counts = xp.log(xp.array(counts))
            # 線形回帰で傾きを計算
            slope = -xp.polyfit(log_eps, log_counts, 1)[0]
            return float(slope)
        
        return 1.0
    
    def _get_coherence_gpu(self, structures: Dict) -> NDArray:
        """構造的コヒーレンスを取得"""
        if 'structural_coherence' in structures:
            return self.to_gpu(structures['structural_coherence'])
        else:
            # なければゼロ配列
            xp = cp if self.is_gpu else np
            return xp.zeros(len(structures['rho_T']))
    
    def _compute_coupling_strength_gpu(self,
                                     q_cumulative: np.ndarray,
                                     window: int) -> NDArray:
        """結合強度の計算（GPU版）"""
        q_cum_gpu = self.to_gpu(q_cumulative)
        xp = cp if self.is_gpu else np
        coupling = xp.ones_like(q_cum_gpu)
        
        # 並列で局所分散を計算
        for i in range(window, len(q_cum_gpu) - window):
            local_q = q_cum_gpu[i-window:i+window]
            var = xp.var(local_q)
            if var > 1e-10:
                coupling[i] = 1.0 / (1.0 + var)
        
        return coupling
    
    def _compute_structural_entropy_gpu(self,
                                      rho_t: np.ndarray,
                                      window: int) -> NDArray:
        """構造エントロピーの計算（GPU版）"""
        rho_t_gpu = self.to_gpu(rho_t)
        n = len(rho_t_gpu)
        xp = cp if self.is_gpu else np
        entropy = xp.zeros(n)
        
        # シャノンエントロピー計算（簡易版）
        for i in range(window, n - window):
            local_data = rho_t_gpu[i-window:i+window]
            entropy[i] = self._compute_shannon_entropy(local_data)
        
        return entropy
    
    def _compute_shannon_entropy(self, data):
        """シャノンエントロピー計算"""
        # ヒストグラムを作成
        xp = cp if self.is_gpu else np
        hist, _ = xp.histogram(data, bins=10)
        hist = hist / xp.sum(hist)  # 正規化
        
        # エントロピー計算
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * xp.log(p)
        
        return entropy
    
    def _compute_boundary_score_gpu(self,
                                  fractal_dims: NDArray,
                                  coherence: NDArray,
                                  coupling: NDArray,
                                  entropy: NDArray) -> NDArray:
        """統合境界スコアの計算"""
        xp = cp if self.is_gpu else np
        
        # 長さを揃える
        min_len = min(len(fractal_dims), len(coupling), len(entropy))
        if len(coherence) > 0:
            min_len = min(min_len, len(coherence))
        
        # 各成分の計算
        fractal_gradient = xp.abs(xp.gradient(fractal_dims[:min_len]))
        coherence_drop = 1 - coherence[:min_len] if len(coherence) > 0 else xp.zeros(min_len)
        coupling_weakness = 1 - coupling[:min_len]
        entropy_gradient = xp.abs(xp.gradient(entropy[:min_len]))
        
        # 重み付き統合
        boundary_score = (
            2.0 * fractal_gradient +      # フラクタル次元の変化
            1.5 * coherence_drop +        # 構造的一貫性の低下
            1.0 * coupling_weakness +     # 結合の弱まり
            1.0 * entropy_gradient        # 情報障壁
        ) / 5.5
        
        return boundary_score
    
    def _detect_peaks_gpu(self,
                        boundary_score: NDArray,
                        n_steps: int) -> Tuple[NDArray, Dict]:
        """ピーク検出（GPU版）"""
        xp = cp if self.is_gpu else np
        
        if len(boundary_score) > 10:
            min_distance_steps = max(50, n_steps // 30)
            height_threshold = xp.mean(boundary_score) + xp.std(boundary_score)
            
            # CuPyのfind_peaks使用
            if self.is_gpu:
                peaks, properties = find_peaks_gpu(
                    boundary_score,
                    height=height_threshold,
                    distance=min_distance_steps
                )
            else:
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(
                    boundary_score,
                    height=height_threshold,
                    distance=min_distance_steps
                )
        else:
            peaks = xp.array([])
            properties = {}
        
        print(f"   Found {len(peaks)} structural boundaries")
        
        return peaks, properties
    
    def detect_topological_breaks(self,
                                structures: Dict[str, np.ndarray],
                                window_steps: int) -> Dict[str, np.ndarray]:
        """
        トポロジカル破れの検出（GPU版）
        
        Returns
        -------
        Dict[str, np.ndarray]
            各種破れ検出結果
        """
        print("\n💥 Detecting topological breaks on GPU...")
        
        if hasattr(self, 'memory_manager'):
            with self.memory_manager.batch_context(len(structures['rho_T'])):
                return self._detect_breaks_impl(structures, window_steps)
        else:
            return self._detect_breaks_impl(structures, window_steps)
    
    def _detect_breaks_impl(self, structures, window_steps):
        """破れ検出の実装部分"""
        # 1. ΛF異常
        lambda_f_anomaly = self._detect_lambda_anomalies_gpu(
            structures['lambda_F_mag'], window_steps
        )
        
        # 2. ΛFF異常
        lambda_ff_anomaly = self._detect_lambda_anomalies_gpu(
            structures['lambda_FF_mag'], window_steps // 2
        )
        
        # 3. テンション場ジャンプ
        rho_t_breaks = self._detect_tension_jumps_gpu(
            structures['rho_T'], window_steps
        )
        
        # 4. トポロジカルチャージ異常
        q_breaks = self._detect_phase_breaks_gpu(structures['Q_lambda'])
        
        # 5. 統合異常スコア
        combined_anomaly = self._combine_anomalies_gpu(
            lambda_f_anomaly,
            lambda_ff_anomaly,
            rho_t_breaks,
            q_breaks
        )
        
        return {
            'lambda_F_anomaly': self.to_cpu(lambda_f_anomaly),
            'lambda_FF_anomaly': self.to_cpu(lambda_ff_anomaly),
            'rho_T_breaks': self.to_cpu(rho_t_breaks),
            'Q_breaks': self.to_cpu(q_breaks),
            'combined_anomaly': self.to_cpu(combined_anomaly)
        }
    
    def _detect_lambda_anomalies_gpu(self,
                                   series: np.ndarray,
                                   window: int) -> NDArray:
        """Lambda系列の異常検出"""
        from .anomaly_detection_gpu import AnomalyDetectorGPU
        
        # AnomalyDetectorGPUのインスタンスを使用
        detector = AnomalyDetectorGPU(self.force_cpu)
        return self.to_gpu(detector.detect_local_anomalies(series, window))
    
    def _detect_tension_jumps_gpu(self,
                                 rho_t: np.ndarray,
                                 window_steps: int) -> NDArray:
        """テンション場のジャンプ検出"""
        rho_t_gpu = self.to_gpu(rho_t)
        
        # ガウシアンフィルタでスムージング
        sigma = window_steps / 3
        if self.is_gpu:
            rho_t_smooth = gaussian_filter1d_gpu(rho_t_gpu, sigma=sigma)
        else:
            from scipy.ndimage import gaussian_filter1d
            rho_t_smooth = gaussian_filter1d(rho_t_gpu, sigma=sigma)
        
        # ジャンプ = 元データとスムージングの差
        xp = cp if self.is_gpu else np
        jumps = xp.abs(rho_t_gpu - rho_t_smooth)
        
        return jumps
    
    def _detect_phase_breaks_gpu(self, phase_series: np.ndarray) -> NDArray:
        """位相破れの検出"""
        phase_gpu = self.to_gpu(phase_series)
        xp = cp if self.is_gpu else np
        breaks = xp.zeros_like(phase_gpu)
        
        # 位相差を計算
        phase_diff = xp.abs(xp.diff(phase_gpu))
        
        # 急激な位相ジャンプを検出（0.1 * 2π radians）
        breaks[1:] = xp.where(phase_diff > 0.1, phase_diff, 0)
        
        return breaks
    
    def _combine_anomalies_gpu(self, *anomalies) -> NDArray:
        """異常スコアの統合"""
        xp = cp if self.is_gpu else np
        
        # 全ての長さを揃える
        min_len = min(len(a) for a in anomalies)
        
        # 重み付き統合
        weights = [1.0, 0.8, 0.6, 1.2]  # ΛF, ΛFF, ρT, Q
        combined = xp.zeros(min_len)
        
        for anomaly, weight in zip(anomalies, weights):
            combined += weight * anomaly[:min_len]
        
        combined /= sum(weights)
        
        return combined


# 追加のGPU最適化関数
def compute_structural_boundaries_batch_gpu(
    structures_list: List[Dict[str, np.ndarray]],
    window_steps_list: List[int],
    gpu_backend: GPUBackend = None
) -> List[Dict[str, Any]]:
    """
    複数の構造に対してバッチで境界検出を実行
    
    Parameters
    ----------
    structures_list : List[Dict]
        構造辞書のリスト
    window_steps_list : List[int]
        各構造のウィンドウサイズ
    gpu_backend : GPUBackend, optional
        共有GPUバックエンド
        
    Returns
    -------
    List[Dict]
        境界検出結果のリスト
    """
    if gpu_backend is None:
        detector = BoundaryDetectorGPU()
    else:
        detector = BoundaryDetectorGPU()
        detector.device = gpu_backend.device
        if hasattr(gpu_backend, 'memory_manager'):
            detector.memory_manager = gpu_backend.memory_manager
    
    results = []
    
    # バッチ処理で効率化
    for structures, window_steps in zip(structures_list, window_steps_list):
        result = detector.detect_structural_boundaries(structures, window_steps)
        results.append(result)
    
    return results
