"""
Causality Analysis for Lambda³ (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

残基間の因果関係をGPUで高速解析！
構造的因果性、Transfer Entropy、遅延相関とか全部できるよ〜！💕

by 環ちゃん
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


logger = logging.getLogger('lambda3_gpu.residue.causality')

# ===============================
# Data Classes
# ===============================

@dataclass
class CausalityResult:
    """因果関係解析の結果"""
    pair: Tuple[int, int]           # (from_residue, to_residue)
    causality_strength: float       # 因果強度
    optimal_lag: int               # 最適ラグ
    causality_profile: np.ndarray  # ラグごとの因果強度
    transfer_entropy: Optional[float] = None  # Transfer Entropy
    granger_causality: Optional[float] = None  # Granger因果
    confidence: float = 1.0        # 信頼度
    p_value: Optional[float] = None  # p値

# ===============================
# CUDA Kernels
# ===============================

# 構造的因果性計算カーネル
STRUCTURAL_CAUSALITY_KERNEL = r'''
extern "C" __global__
void compute_structural_causality_kernel(
    const float* __restrict__ anomaly_i,    // (n_frames,)
    const float* __restrict__ anomaly_j,    // (n_frames,)
    float* __restrict__ causality_profile,  // (n_lags,)
    const int n_frames,
    const int n_lags,
    const float event_threshold
) {
    const int lag = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lag >= n_lags || lag == 0) return;
    
    // イベント検出
    int cause_events = 0;
    int effect_given_cause = 0;
    
    for (int t = 0; t < n_frames - lag; t++) {
        bool cause = anomaly_i[t] > event_threshold;
        bool effect = anomaly_j[t + lag] > event_threshold;
        
        if (cause) {
            cause_events++;
            if (effect) {
                effect_given_cause++;
            }
        }
    }
    
    // 条件付き確率 P(effect|cause)
    if (cause_events > 0) {
        causality_profile[lag] = (float)effect_given_cause / cause_events;
    } else {
        causality_profile[lag] = 0.0f;
    }
}
'''

# Transfer Entropy計算カーネル（簡易版）
TRANSFER_ENTROPY_KERNEL = r'''
extern "C" __global__
void compute_transfer_entropy_kernel(
    const float* __restrict__ source,      // (n_frames,)
    const float* __restrict__ target,      // (n_frames,) 
    float* __restrict__ te_values,        // (n_lags,)
    const int n_frames,
    const int n_lags,
    const int history_length,
    const int n_bins
) {
    const int lag = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lag >= n_lags || lag == 0) return;
    
    // 簡易的なTransfer Entropy計算
    // 実際は離散化とエントロピー計算が必要
    
    float te = 0.0f;
    int valid_samples = 0;
    
    for (int t = history_length; t < n_frames - lag; t++) {
        // 簡易版：相関の非対称性を使用
        float past_target = target[t - 1];
        float current_target = target[t];
        float past_source = source[t - lag];
        
        // 条件付き相互情報量の近似
        float prediction_without_source = past_target;
        float prediction_with_source = 0.5f * (past_target + past_source);
        
        float error_without = fabsf(current_target - prediction_without_source);
        float error_with = fabsf(current_target - prediction_with_source);
        
        te += logf((error_without + 1e-10f) / (error_with + 1e-10f));
        valid_samples++;
    }
    
    te_values[lag] = (valid_samples > 0) ? te / valid_samples : 0.0f;
}
'''

# 遅延相関計算カーネル
LAGGED_CORRELATION_KERNEL = r'''
extern "C" __global__
void compute_lagged_correlation_kernel(
    const float* __restrict__ series_i,     // (n_frames,)
    const float* __restrict__ series_j,     // (n_frames,)
    float* __restrict__ correlations,      // (n_lags,)
    const int n_frames,
    const int n_lags
) {
    const int lag = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lag >= n_lags) return;
    
    // 有効なサンプル数
    int n_valid = n_frames - lag;
    if (n_valid <= 1) {
        correlations[lag] = 0.0f;
        return;
    }
    
    // 平均計算
    float mean_i = 0.0f, mean_j = 0.0f;
    for (int t = 0; t < n_valid; t++) {
        mean_i += series_i[t];
        mean_j += series_j[t + lag];
    }
    mean_i /= n_valid;
    mean_j /= n_valid;
    
    // 共分散と分散
    float cov = 0.0f, var_i = 0.0f, var_j = 0.0f;
    for (int t = 0; t < n_valid; t++) {
        float di = series_i[t] - mean_i;
        float dj = series_j[t + lag] - mean_j;
        cov += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }
    
    // 相関係数
    float denominator = sqrtf(var_i * var_j);
    correlations[lag] = (denominator > 1e-10f) ? cov / denominator : 0.0f;
}
'''

# ===============================
# Causality Analyzer GPU Class
# ===============================

class CausalityAnalyzerGPU(GPUBackend):
    """
    因果関係解析のGPU実装
    
    残基間の因果関係を複数の手法で解析！
    構造的因果性、Transfer Entropy、Granger因果とか全部高速！
    """
    
    def __init__(self,
                 event_threshold: float = 1.0,
                 min_lag: int = 1,
                 max_lag: int = 200,
                 history_length: int = 5,
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
            最大ラグ
        history_length : int
            履歴長（Transfer Entropy用）
        """
        super().__init__(**kwargs)
        self.event_threshold = event_threshold
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.history_length = history_length
        self.memory_manager = memory_manager or GPUMemoryManager()
        
        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.structural_causality_kernel = cp.RawKernel(
                STRUCTURAL_CAUSALITY_KERNEL, 'compute_structural_causality_kernel'
            )
            self.transfer_entropy_kernel = cp.RawKernel(
                TRANSFER_ENTROPY_KERNEL, 'compute_transfer_entropy_kernel'
            )
            self.lagged_correlation_kernel = cp.RawKernel(
                LAGGED_CORRELATION_KERNEL, 'compute_lagged_correlation_kernel'
            )
            logger.debug("Causality analysis kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.structural_causality_kernel = None
            self.transfer_entropy_kernel = None
            self.lagged_correlation_kernel = None
    
    @handle_gpu_errors
    @profile_gpu
    def calculate_structural_causality(self,
                                     anomaly_i: np.ndarray,
                                     anomaly_j: np.ndarray,
                                     lag_window: Optional[int] = None) -> CausalityResult:
        """
        構造的因果性を計算
        
        Parameters
        ----------
        anomaly_i : np.ndarray
            原因側の異常スコア
        anomaly_j : np.ndarray
            結果側の異常スコア
        lag_window : int, optional
            ラグウィンドウ（Noneなら最大ラグ使用）
            
        Returns
        -------
        CausalityResult
            因果関係解析結果
        """
        if lag_window is None:
            lag_window = self.max_lag
        
        n_lags = min(lag_window, len(anomaly_i) // 2)
        
        # GPU転送
        anomaly_i_gpu = self.to_gpu(anomaly_i)
        anomaly_j_gpu = self.to_gpu(anomaly_j)
        
        if self.is_gpu and self.structural_causality_kernel is not None:
            # カスタムカーネル使用
            causality_profile = self.zeros(n_lags, dtype=cp.float32)
            
            # カーネル実行
            block_size = 256
            grid_size = (n_lags + block_size - 1) // block_size
            
            self.structural_causality_kernel(
                (grid_size,), (block_size,),
                (anomaly_i_gpu, anomaly_j_gpu, causality_profile,
                 len(anomaly_i), n_lags, self.event_threshold)
            )
        else:
            # 汎用GPU/CPU版
            causality_profile = self._calculate_structural_causality_generic(
                anomaly_i_gpu, anomaly_j_gpu, n_lags
            )
        
        # 最大値と最適ラグ
        max_causality = float(self.xp.max(causality_profile))
        optimal_lag = int(self.xp.argmax(causality_profile))
        
        # Transfer Entropy（オプション）
        transfer_entropy = None
        if optimal_lag > 0:
            with self.timer('transfer_entropy'):
                transfer_entropy = self._calculate_transfer_entropy(
                    anomaly_i_gpu, anomaly_j_gpu, optimal_lag
                )
        
        # 結果作成
        result = CausalityResult(
            pair=(0, 1),  # ダミー（呼び出し側で設定）
            causality_strength=max_causality,
            optimal_lag=optimal_lag,
            causality_profile=self.to_cpu(causality_profile),
            transfer_entropy=transfer_entropy
        )
        
        return result
    
    def _calculate_structural_causality_generic(self,
                                              anomaly_i: Union[np.ndarray, cp.ndarray],
                                              anomaly_j: Union[np.ndarray, cp.ndarray],
                                              n_lags: int) -> Union[np.ndarray, cp.ndarray]:
        """構造的因果性の汎用計算"""
        causality_profile = self.zeros(n_lags)
        
        # イベント化
        events_i = (anomaly_i > self.event_threshold).astype(self.xp.float32)
        events_j = (anomaly_j > self.event_threshold).astype(self.xp.float32)
        
        for lag in range(1, n_lags):
            cause = events_i[:-lag]
            effect = events_j[lag:]
            
            # 条件付き確率 P(effect|cause)
            cause_mask = cause > 0
            if self.xp.sum(cause_mask) > 0:
                causality_profile[lag] = self.xp.mean(effect[cause_mask])
        
        return causality_profile
    
    def _calculate_transfer_entropy(self,
                                  source: Union[np.ndarray, cp.ndarray],
                                  target: Union[np.ndarray, cp.ndarray],
                                  lag: int) -> float:
        """Transfer Entropy計算（簡易版）"""
        if len(source) < self.history_length + lag:
            return 0.0
        
        # 簡易的な実装（実際は離散化が必要）
        te = 0.0
        n_samples = 0
        
        for t in range(self.history_length, len(target) - lag):
            # 過去の情報
            past_target = target[t-self.history_length:t]
            current_target = target[t]
            past_source = source[t-lag-self.history_length:t-lag]
            
            # 予測誤差の比較（簡易版）
            pred_without = self.xp.mean(past_target)
            pred_with = (self.xp.mean(past_target) + self.xp.mean(past_source)) / 2
            
            error_without = abs(current_target - pred_without)
            error_with = abs(current_target - pred_with)
            
            if error_without > 1e-10:
                te += self.xp.log(error_without / (error_with + 1e-10))
                n_samples += 1
        
        return float(te / n_samples) if n_samples > 0 else 0.0
    
    @profile_gpu
    def compute_lagged_correlation(self,
                                 series_i: np.ndarray,
                                 series_j: np.ndarray,
                                 max_lag: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        遅延相関を計算
        
        Parameters
        ----------
        series_i : np.ndarray
            系列1
        series_j : np.ndarray
            系列2
        max_lag : int, optional
            最大ラグ
            
        Returns
        -------
        correlations : np.ndarray
            各ラグでの相関
        optimal_lag : int
            最適ラグ
        max_correlation : float
            最大相関
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        n_lags = min(max_lag, len(series_i) // 2)
        
        # GPU転送
        series_i_gpu = self.to_gpu(series_i)
        series_j_gpu = self.to_gpu(series_j)
        
        if self.is_gpu and self.lagged_correlation_kernel is not None:
            # カスタムカーネル使用
            correlations = self.zeros(n_lags, dtype=cp.float32)
            
            block_size = 256
            grid_size = (n_lags + block_size - 1) // block_size
            
            self.lagged_correlation_kernel(
                (grid_size,), (block_size,),
                (series_i_gpu, series_j_gpu, correlations, len(series_i), n_lags)
            )
        else:
            # 汎用版
            correlations = self.zeros(n_lags)
            
            for lag in range(n_lags):
                if lag == 0:
                    corr = self.xp.corrcoef(series_i_gpu, series_j_gpu)[0, 1]
                else:
                    corr = self.xp.corrcoef(series_i_gpu[:-lag], series_j_gpu[lag:])[0, 1]
                
                correlations[lag] = corr
        
        # 最大相関とラグ
        abs_corr = self.xp.abs(correlations)
        optimal_lag = int(self.xp.argmax(abs_corr))
        max_correlation = float(correlations[optimal_lag])
        
        return self.to_cpu(correlations), optimal_lag, max_correlation
    
    @handle_gpu_errors
    def detect_causal_pairs(self,
                          anomaly_scores: Dict[int, np.ndarray],
                          threshold: float = 0.2,
                          batch_size: int = 100) -> List[CausalityResult]:
        """
        複数残基ペアの因果関係を検出
        
        Parameters
        ----------
        anomaly_scores : dict
            残基ID -> 異常スコア
        threshold : float
            因果強度閾値
        batch_size : int
            バッチサイズ
            
        Returns
        -------
        list of CausalityResult
            検出された因果関係
        """
        residue_ids = sorted(anomaly_scores.keys())
        n_residues = len(residue_ids)
        causal_pairs = []
        
        logger.info(f"🔍 Detecting causal pairs among {n_residues} residues")
        
        # バッチ処理
        pair_count = 0
        for i in range(0, n_residues, batch_size):
            for j in range(0, n_residues, batch_size):
                # バッチ内のペアを処理
                batch_results = self._process_pair_batch(
                    residue_ids[i:i+batch_size],
                    residue_ids[j:j+batch_size],
                    anomaly_scores,
                    threshold
                )
                
                causal_pairs.extend(batch_results)
                pair_count += len(batch_results)
                
                if pair_count % 1000 == 0:
                    logger.debug(f"   Processed {pair_count} pairs...")
        
        # 強度でソート
        causal_pairs.sort(key=lambda x: x.causality_strength, reverse=True)
        
        logger.info(f"   Found {len(causal_pairs)} causal relationships")
        
        return causal_pairs
    
    def _process_pair_batch(self,
                          residues_i: List[int],
                          residues_j: List[int],
                          anomaly_scores: Dict[int, np.ndarray],
                          threshold: float) -> List[CausalityResult]:
        """ペアのバッチ処理"""
        results = []
        
        for res_i in residues_i:
            for res_j in residues_j:
                if res_i == res_j:
                    continue
                
                # 因果性計算
                result = self.calculate_structural_causality(
                    anomaly_scores[res_i],
                    anomaly_scores[res_j]
                )
                
                # 閾値チェック
                if result.causality_strength >= threshold:
                    result.pair = (res_i, res_j)
                    results.append(result)
        
        return results
    
    def compute_granger_causality(self,
                                series_i: np.ndarray,
                                series_j: np.ndarray,
                                order: int = 5) -> float:
        """
        Granger因果性を計算（簡易版）
        
        Parameters
        ----------
        series_i : np.ndarray
            原因候補の系列
        series_j : np.ndarray
            結果候補の系列
        order : int
            自己回帰の次数
            
        Returns
        -------
        float
            Granger因果性スコア
        """
        # GPU転送
        x_gpu = self.to_gpu(series_i)
        y_gpu = self.to_gpu(series_j)
        
        n = len(x_gpu)
        
        # 制限モデル（yの過去のみ）
        y_lagged = self.xp.zeros((n - order, order))
        for i in range(order):
            y_lagged[:, i] = y_gpu[order-i-1:-i-1]
        
        y_target = y_gpu[order:]
        
        # 最小二乗法
        coef_restricted = self.xp.linalg.lstsq(y_lagged, y_target, rcond=None)[0]
        residuals_restricted = y_target - self.xp.dot(y_lagged, coef_restricted)
        rss_restricted = self.xp.sum(residuals_restricted ** 2)
        
        # 非制限モデル（xとyの過去）
        xy_lagged = self.xp.zeros((n - order, 2 * order))
        xy_lagged[:, :order] = y_lagged
        for i in range(order):
            xy_lagged[:, order+i] = x_gpu[order-i-1:-i-1]
        
        coef_unrestricted = self.xp.linalg.lstsq(xy_lagged, y_target, rcond=None)[0]
        residuals_unrestricted = y_target - self.xp.dot(xy_lagged, coef_unrestricted)
        rss_unrestricted = self.xp.sum(residuals_unrestricted ** 2)
        
        # F統計量の近似
        f_stat = ((rss_restricted - rss_unrestricted) / order) / (rss_unrestricted / (n - 2*order))
        
        return float(f_stat)

# ===============================
# Standalone Functions
# ===============================

def calculate_structural_causality_gpu(anomaly_i: np.ndarray,
                                     anomaly_j: np.ndarray,
                                     lag_window: int = 200,
                                     event_threshold: float = 1.0,
                                     **kwargs) -> CausalityResult:
    """構造的因果性計算のスタンドアロン関数"""
    analyzer = CausalityAnalyzerGPU(event_threshold=event_threshold, **kwargs)
    return analyzer.calculate_structural_causality(anomaly_i, anomaly_j, lag_window)

def compute_lagged_correlation_gpu(series_i: np.ndarray,
                                 series_j: np.ndarray,
                                 max_lag: int = 200,
                                 backend: Optional[GPUBackend] = None) -> Tuple[np.ndarray, int, float]:
    """遅延相関計算のスタンドアロン関数"""
    analyzer = CausalityAnalyzerGPU() if backend is None else CausalityAnalyzerGPU(device=backend.device)
    return analyzer.compute_lagged_correlation(series_i, series_j, max_lag)

def detect_causal_pairs_gpu(anomaly_scores: Dict[int, np.ndarray],
                          threshold: float = 0.2,
                          **kwargs) -> List[CausalityResult]:
    """因果ペア検出のスタンドアロン関数"""
    analyzer = CausalityAnalyzerGPU(**kwargs)
    return analyzer.detect_causal_pairs(anomaly_scores, threshold)

def compute_transfer_entropy_gpu(source: np.ndarray,
                               target: np.ndarray,
                               lag: int,
                               history_length: int = 5,
                               backend: Optional[GPUBackend] = None) -> float:
    """Transfer Entropy計算のスタンドアロン関数"""
    analyzer = CausalityAnalyzerGPU(history_length=history_length)
    if backend:
        analyzer.device = backend.device
        analyzer.is_gpu = backend.is_gpu
        analyzer.xp = backend.xp
    
    source_gpu = analyzer.to_gpu(source)
    target_gpu = analyzer.to_gpu(target)
    
    return analyzer._calculate_transfer_entropy(source_gpu, target_gpu, lag)
