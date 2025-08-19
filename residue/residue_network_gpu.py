"""
Residue Network Analysis (GPU Version) - v4.0 Design Unified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

残基間ネットワーク解析のGPU実装 - 3パターン設計統一版！
quantum_validation_v4.pyの設計思想に完全準拠💕

Version: 4.0-unified
by 環ちゃん - Design Unified Edition
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist as cp_cdist
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors

logger = logging.getLogger('lambda3_gpu.residue.network')

# ===============================
# Data Classes
# ===============================

@dataclass
class NetworkLink:
    """ネットワークリンク"""
    from_res: int
    to_res: int
    strength: float
    lag: int = 0
    distance: Optional[float] = None
    sync_rate: Optional[float] = None
    link_type: str = 'causal'  # 'causal', 'sync', 'async', 'instantaneous', 'transition', 'cascade'
    confidence: float = 1.0
    quantum_signature: Optional[str] = None  # 'entanglement', 'tunneling', 'jump', etc.

@dataclass
class NetworkAnalysisResult:
    """ネットワーク解析結果"""
    causal_network: List[NetworkLink]
    sync_network: List[NetworkLink]
    async_strong_bonds: List[NetworkLink]
    spatial_constraints: Dict[Tuple[int, int], float]
    adaptive_windows: Dict[int, int]
    network_stats: Dict[str, Any]
    
    @property
    def n_causal_links(self) -> int:
        return len(self.causal_network)
    
    @property
    def n_sync_links(self) -> int:
        return len(self.sync_network)
    
    @property
    def n_async_bonds(self) -> int:
        return len(self.async_strong_bonds)

# ===============================
# CUDA Kernels（既存のまま）
# ===============================

# 適応的ウィンドウサイズ計算カーネル
ADAPTIVE_WINDOW_KERNEL = r'''
extern "C" __global__
void compute_adaptive_windows_kernel(
    const float* __restrict__ anomaly_scores,  // (n_residues, n_frames)
    int* __restrict__ window_sizes,          // (n_residues,)
    const int n_residues,
    const int n_frames,
    const int min_window,
    const int max_window,
    const int base_window
) {
    const int res_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (res_id >= n_residues) return;
    
    // 該当残基のスコアを解析
    const float* res_scores = &anomaly_scores[res_id * n_frames];
    
    // イベント密度計算
    int n_events = 0;
    float score_sum = 0.0f;
    float score_sq_sum = 0.0f;
    
    for (int i = 0; i < n_frames; i++) {
        float score = res_scores[i];
        if (score > 1.0f) n_events++;
        score_sum += score;
        score_sq_sum += score * score;
    }
    
    float event_density = (float)n_events / n_frames;
    float mean = score_sum / n_frames;
    float variance = (score_sq_sum / n_frames) - (mean * mean);
    float volatility = (mean > 1e-10f) ? sqrtf(variance) / mean : 0.0f;
    
    // スケールファクター計算
    float scale_factor = 1.0f;
    
    if (event_density > 0.1f) {
        scale_factor *= 0.7f;
    } else if (event_density < 0.02f) {
        scale_factor *= 2.0f;
    }
    
    if (volatility > 2.0f) {
        scale_factor *= 0.8f;
    } else if (volatility < 0.5f) {
        scale_factor *= 1.3f;
    }
    
    // ウィンドウサイズ決定
    int adaptive_window = (int)(base_window * scale_factor);
    window_sizes[res_id] = max(min_window, min(max_window, adaptive_window));
}
'''

# 空間制約フィルタリングカーネル
SPATIAL_FILTER_KERNEL = r'''
extern "C" __global__
void filter_by_distance_kernel(
    const float* __restrict__ distances,      // (n_pairs,)
    const int* __restrict__ pair_indices,     // (n_pairs, 2)
    bool* __restrict__ valid_mask,           // (n_pairs,)
    float* __restrict__ weights,             // (n_pairs,)
    const int n_pairs,
    const float max_distance,
    const float contact_distance,
    const float near_distance
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_pairs) return;
    
    float dist = distances[idx];
    
    // 距離ベースの重み付け
    if (dist < contact_distance) {
        valid_mask[idx] = true;
        weights[idx] = 1.0f;
    } else if (dist < near_distance) {
        valid_mask[idx] = true;
        weights[idx] = 0.8f;
    } else if (dist < max_distance) {
        valid_mask[idx] = true;
        weights[idx] = 0.5f;
    } else {
        valid_mask[idx] = false;
        weights[idx] = 0.0f;
    }
}
'''

# ===============================
# Residue Network GPU Class (v4.0 Unified)
# ===============================

class ResidueNetworkGPU(GPUBackend):
    """
    残基ネットワーク解析のGPU実装
    v4.0 Design Unified - 3パターン設計統一版！
    """
    
    def __init__(self,
                 max_interaction_distance: float = 15.0,
                 correlation_threshold: float = 0.15,
                 sync_threshold: float = 0.2,
                 min_causality_strength: float = 0.2,
                 max_causal_links: int = 500,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        max_interaction_distance : float
            最大相互作用距離（Å）
        correlation_threshold : float
            相関閾値
        sync_threshold : float
            同期判定閾値
        min_causality_strength : float
            最小因果強度
        max_causal_links : int
            最大因果リンク数
        """
        super().__init__(**kwargs)
        self.max_interaction_distance = max_interaction_distance
        self.correlation_threshold = correlation_threshold
        self.sync_threshold = sync_threshold
        self.min_causality_strength = min_causality_strength
        self.max_causal_links = max_causal_links
        self.memory_manager = memory_manager or GPUMemoryManager()
        
        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.adaptive_window_kernel = cp.RawKernel(
                ADAPTIVE_WINDOW_KERNEL, 'compute_adaptive_windows_kernel'
            )
            self.spatial_filter_kernel = cp.RawKernel(
                SPATIAL_FILTER_KERNEL, 'filter_by_distance_kernel'
            )
            logger.debug("Network analysis kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.adaptive_window_kernel = None
            self.spatial_filter_kernel = None
    
    @handle_gpu_errors
    def analyze_network(self,
                       residue_anomaly_scores: Dict[int, np.ndarray],
                       residue_coupling: np.ndarray,
                       residue_coms: Optional[np.ndarray] = None,
                       lag_window: int = 200) -> NetworkAnalysisResult:
        """
        残基ネットワークを解析（v4.0 3パターン設計）
        
        Parameters
        ----------
        residue_anomaly_scores : dict
            残基ID -> 異常スコアのマッピング
        residue_coupling : np.ndarray
            残基間カップリング (n_frames, n_residues, n_residues)
        residue_coms : np.ndarray, optional
            残基COM座標 (n_frames, n_residues, 3)
        lag_window : int
            遅延相関の最大ラグ
            
        Returns
        -------
        NetworkAnalysisResult
            ネットワーク解析結果
        """
        with self.timer('analyze_network'):
            logger.info("🎯 Analyzing residue interaction network (v4.0 Unified)")
            
            # フレーム数を確認
            if not residue_anomaly_scores:
                logger.warning("No anomaly scores provided")
                return self._create_empty_result()
            
            first_score = next(iter(residue_anomaly_scores.values()))
            n_frames = len(first_score)
            
            if n_frames <= 0:
                logger.warning("No frames to analyze")
                return self._create_empty_result()
            
            # ========================================
            # v4.0 3パターン判定
            # ========================================
            
            # 1. CASCADE判定（async_bondsの潜在性をチェック）
            has_cascade_potential = self._check_cascade_potential(
                residue_anomaly_scores, residue_coupling, n_frames
            )
            
            if has_cascade_potential and n_frames >= 2:
                logger.info("   🌐 CASCADE pattern detected - Network cascade analysis")
                return self._analyze_cascade_pattern(
                    residue_anomaly_scores, residue_coupling, residue_coms, lag_window
                )
            
            # 2. INSTANTANEOUS判定（単一フレーム）
            elif n_frames == 1:
                logger.info("   ⚡ INSTANTANEOUS pattern detected - Quantum-like analysis")
                return self._analyze_instantaneous_pattern(
                    residue_anomaly_scores, residue_coupling, residue_coms
                )
            
            # 3. TRANSITION判定（複数フレーム）
            else:
                logger.info(f"   📈 TRANSITION pattern detected - {n_frames} frames analysis")
                return self._analyze_transition_pattern(
                    residue_anomaly_scores, residue_coupling, residue_coms, lag_window, n_frames
                )
    
    # ========================================
    # CASCADE潜在性チェック
    # ========================================
    
    def _check_cascade_potential(self,
                                residue_anomaly_scores: Dict[int, np.ndarray],
                                residue_coupling: np.ndarray,
                                n_frames: int) -> bool:
        """CASCADEパターンの潜在性をチェック"""
        if n_frames < 2:
            return False
        
        # 複数残基が同時に高い異常を示すかチェック
        high_anomaly_count = 0
        threshold = 2.0  # 異常スコア閾値
        
        for scores in residue_anomaly_scores.values():
            if np.any(scores > threshold):
                high_anomaly_count += 1
        
        # 3残基以上が異常 → CASCADE可能性
        if high_anomaly_count >= 3:
            # カップリングの変動もチェック
            if residue_coupling.ndim == 3 and residue_coupling.shape[0] >= 2:
                coupling_std = np.std(residue_coupling, axis=0)
                if np.max(coupling_std) > 0.3:  # 高い変動
                    return True
        
        return False
    
    # ========================================
    # INSTANTANEOUS パターン解析
    # ========================================
    
    def _analyze_instantaneous_pattern(self,
                                      residue_anomaly_scores: Dict[int, np.ndarray],
                                      residue_coupling: np.ndarray,
                                      residue_coms: Optional[np.ndarray]) -> NetworkAnalysisResult:
        """INSTANTANEOUSパターン：瞬間的変化の解析"""
        
        residue_ids = sorted(residue_anomaly_scores.keys())
        n_residues = len(residue_ids)
        
        # カップリング行列から強い相互作用を検出
        if residue_coupling.ndim == 3 and residue_coupling.shape[0] > 0:
            coupling = residue_coupling[0]
        elif residue_coupling.ndim == 2:
            coupling = residue_coupling
        else:
            logger.warning("Invalid coupling matrix shape")
            return self._create_empty_result()
        
        # 統計的閾値設定
        mean_coupling = float(np.mean(coupling))
        std_coupling = float(np.std(coupling))
        threshold = mean_coupling + 2 * std_coupling
        
        # ネットワークリンクを構築
        causal_links = []
        sync_links = []
        async_bonds = []
        
        for i, res_i in enumerate(residue_ids):
            for j, res_j in enumerate(residue_ids[i+1:], i+1):
                if res_i < coupling.shape[0] and res_j < coupling.shape[1]:
                    coupling_strength = coupling[res_i, res_j]
                    
                    if coupling_strength > threshold:
                        # 瞬間的な強い結合 = 量子もつれ的
                        link = NetworkLink(
                            from_res=res_i,
                            to_res=res_j,
                            strength=float(coupling_strength),
                            lag=0,
                            sync_rate=1.0,  # 完全同期
                            link_type='instantaneous',
                            confidence=1.0,
                            quantum_signature='entanglement'
                        )
                        sync_links.append(link)
                        
                        # 特に強い結合はasync_bondsにも追加
                        if coupling_strength > threshold * 1.5:
                            async_bonds.append(link)
        
        # 空間制約
        spatial_constraints = {}
        if residue_coms is not None and residue_coms.shape[0] > 0:
            spatial_constraints = self._compute_spatial_constraints(
                residue_ids, residue_coms
            )
        
        return NetworkAnalysisResult(
            causal_network=causal_links,
            sync_network=sync_links,
            async_strong_bonds=async_bonds,
            spatial_constraints=spatial_constraints,
            adaptive_windows={res_id: 1 for res_id in residue_ids},
            network_stats={
                'n_causal': 0,
                'n_sync': len(sync_links),
                'n_async': len(async_bonds),
                'event_type': 'INSTANTANEOUS',
                'pattern': 'instantaneous',
                'quantum_signature': 'entanglement',
                'n_frames': 1
            }
        )
    
    # ========================================
    # TRANSITION パターン解析
    # ========================================
    
    def _analyze_transition_pattern(self,
                                  residue_anomaly_scores: Dict[int, np.ndarray],
                                  residue_coupling: np.ndarray,
                                  residue_coms: Optional[np.ndarray],
                                  lag_window: int,
                                  n_frames: int) -> NetworkAnalysisResult:
        """TRANSITIONパターン：遷移過程の解析"""
        
        residue_ids = sorted(residue_anomaly_scores.keys())
        
        # フレーム数によるサブパターン分類
        if n_frames == 2:
            quantum_signature = 'tunneling'
            analysis_mode = 'quantum_tunneling'
        elif n_frames == 3:
            quantum_signature = 'jump'
            analysis_mode = 'quantum_jump'
        elif n_frames < 10:
            quantum_signature = 'short_transition'
            analysis_mode = 'short_timeseries'
        else:
            quantum_signature = None
            analysis_mode = 'classical'
        
        logger.info(f"      Analysis mode: {analysis_mode}")
        
        # 適応的ウィンドウサイズ計算
        with self.timer('adaptive_windows'):
            adaptive_windows = self._compute_adaptive_windows(residue_anomaly_scores)
        
        # 空間制約計算
        with self.timer('spatial_constraints'):
            if residue_coms is not None:
                spatial_constraints = self._compute_spatial_constraints(
                    residue_ids, residue_coms
                )
            else:
                spatial_constraints = self._create_all_pairs(residue_ids)
        
        # ネットワーク構築
        with self.timer('build_network'):
            if analysis_mode in ['quantum_tunneling', 'quantum_jump']:
                networks = self._build_quantum_transition_network(
                    residue_anomaly_scores,
                    residue_coupling,
                    spatial_constraints,
                    n_frames,
                    quantum_signature
                )
            elif analysis_mode == 'short_timeseries':
                networks = self._build_short_transition_network(
                    residue_anomaly_scores,
                    residue_coupling,
                    spatial_constraints,
                    n_frames
                )
            else:
                networks = self._build_classical_transition_network(
                    residue_anomaly_scores,
                    residue_coupling,
                    spatial_constraints,
                    adaptive_windows,
                    lag_window
                )
        
        # 統計情報
        network_stats = self._compute_network_stats(networks, spatial_constraints)
        network_stats.update({
            'event_type': 'TRANSITION',
            'pattern': 'transition',
            'quantum_signature': quantum_signature,
            'analysis_mode': analysis_mode,
            'n_frames': n_frames
        })
        
        result = NetworkAnalysisResult(
            causal_network=networks['causal'],
            sync_network=networks['sync'],
            async_strong_bonds=networks['async'],
            spatial_constraints=spatial_constraints,
            adaptive_windows=adaptive_windows,
            network_stats=network_stats
        )
        
        self._print_summary(result)
        return result
    
    # ========================================
    # CASCADE パターン解析
    # ========================================
    
    def _analyze_cascade_pattern(self,
                               residue_anomaly_scores: Dict[int, np.ndarray],
                               residue_coupling: np.ndarray,
                               residue_coms: Optional[np.ndarray],
                               lag_window: int) -> NetworkAnalysisResult:
        """CASCADEパターン：ネットワークカスケードの解析"""
        
        residue_ids = sorted(residue_anomaly_scores.keys())
        n_frames = len(next(iter(residue_anomaly_scores.values())))
        
        logger.info("      Detecting cascade propagation paths...")
        
        # 適応的ウィンドウ（カスケード用に短く）
        adaptive_windows = {res_id: min(50, n_frames//2) for res_id in residue_ids}
        
        # 空間制約
        if residue_coms is not None:
            spatial_constraints = self._compute_spatial_constraints(
                residue_ids, residue_coms
            )
        else:
            spatial_constraints = self._create_all_pairs(residue_ids)
        
        # カスケード特有の解析
        causal_links = []
        sync_links = []
        async_bonds = []
        
        # 開始点の検出（最初に異常を示した残基）
        initiators = self._find_cascade_initiators(residue_anomaly_scores)
        
        # 伝播経路の構築
        for initiator in initiators:
            paths = self._trace_cascade_paths(
                initiator,
                residue_anomaly_scores,
                residue_coupling,
                spatial_constraints,
                max_depth=5
            )
            
            for path in paths:
                for i in range(len(path) - 1):
                    from_res = path[i]
                    to_res = path[i + 1]
                    
                    # 伝播強度の計算
                    strength = self._calculate_cascade_strength(
                        from_res, to_res,
                        residue_anomaly_scores,
                        residue_coupling
                    )
                    
                    if strength > self.min_causality_strength:
                        link = NetworkLink(
                            from_res=from_res,
                            to_res=to_res,
                            strength=strength,
                            lag=i + 1,  # カスケードのステップ
                            sync_rate=0.0,  # 非同期
                            link_type='cascade',
                            confidence=0.8,
                            quantum_signature='information_transfer'
                        )
                        causal_links.append(link)
                        async_bonds.append(link)
        
        # 同期的な副次効果も検出
        sync_links = self._detect_cascade_synchrony(
            residue_anomaly_scores,
            residue_coupling,
            spatial_constraints
        )
        
        # 統計情報
        network_stats = {
            'n_causal': len(causal_links),
            'n_sync': len(sync_links),
            'n_async': len(async_bonds),
            'event_type': 'CASCADE',
            'pattern': 'cascade',
            'quantum_signature': 'information_transfer',
            'n_initiators': len(initiators),
            'cascade_depth': max([link.lag for link in causal_links]) if causal_links else 0,
            'n_frames': n_frames
        }
        
        result = NetworkAnalysisResult(
            causal_network=causal_links,
            sync_network=sync_links,
            async_strong_bonds=async_bonds,
            spatial_constraints=spatial_constraints,
            adaptive_windows=adaptive_windows,
            network_stats=network_stats
        )
        
        self._print_summary(result)
        return result
    
    # ========================================
    # TRANSITION サブパターン用メソッド
    # ========================================
    
    def _build_quantum_transition_network(self,
                                        anomaly_scores: Dict[int, np.ndarray],
                                        residue_coupling: np.ndarray,
                                        spatial_constraints: Dict,
                                        n_frames: int,
                                        quantum_signature: str) -> Dict:
        """量子遷移（2-3フレーム）のネットワーク構築"""
        causal_links = []
        sync_links = []
        async_bonds = []
        
        residue_ids = sorted(anomaly_scores.keys())
        
        for i, res_i in enumerate(residue_ids):
            for j, res_j in enumerate(residue_ids[i+1:], i+1):
                if (res_i, res_j) not in spatial_constraints:
                    continue
                
                scores_i = anomaly_scores[res_i]
                scores_j = anomaly_scores[res_j]
                
                # 変化量計算
                if n_frames == 2:
                    delta_i = scores_i[1] - scores_i[0]
                    delta_j = scores_j[1] - scores_j[0]
                    
                    # 同じ方向への変化 = トンネリングペア
                    if delta_i * delta_j > 0 and abs(delta_i) > 0.5 and abs(delta_j) > 0.5:
                        link = NetworkLink(
                            from_res=res_i,
                            to_res=res_j,
                            strength=float(np.sqrt(abs(delta_i * delta_j))),
                            lag=0,
                            sync_rate=0.8,
                            link_type='transition',
                            quantum_signature=quantum_signature,
                            distance=spatial_constraints[(res_i, res_j)]
                        )
                        sync_links.append(link)
                        if link.strength > 0.7:
                            async_bonds.append(link)
                
                elif n_frames == 3:
                    # ジャンプ検出
                    jump_i = abs(scores_i[1] - scores_i[0]) + abs(scores_i[2] - scores_i[1])
                    jump_j = abs(scores_j[1] - scores_j[0]) + abs(scores_j[2] - scores_j[1])
                    
                    if jump_i > 0.3 and jump_j > 0.3:
                        link = NetworkLink(
                            from_res=res_i,
                            to_res=res_j,
                            strength=float(np.sqrt(jump_i * jump_j)),
                            lag=1,
                            sync_rate=0.5,
                            link_type='transition',
                            quantum_signature=quantum_signature,
                            distance=spatial_constraints[(res_i, res_j)]
                        )
                        sync_links.append(link)
                        if link.strength > 0.5:
                            async_bonds.append(link)
        
        return {
            'causal': causal_links,
            'sync': sync_links,
            'async': async_bonds
        }
    
    def _build_short_transition_network(self,
                                       anomaly_scores: Dict[int, np.ndarray],
                                       residue_coupling: np.ndarray,
                                       spatial_constraints: Dict,
                                       n_frames: int) -> Dict:
        """短期遷移（4-9フレーム）のネットワーク構築"""
        causal_links = []
        sync_links = []
        async_bonds = []
        
        residue_ids = sorted(anomaly_scores.keys())
        
        for i, res_i in enumerate(residue_ids):
            for j, res_j in enumerate(residue_ids[i+1:], i+1):
                if (res_i, res_j) not in spatial_constraints:
                    continue
                
                scores_i = anomaly_scores[res_i]
                scores_j = anomaly_scores[res_j]
                
                # 前半と後半の比較
                mid_point = n_frames // 2
                first_half_i = np.mean(scores_i[:mid_point])
                second_half_i = np.mean(scores_i[mid_point:])
                first_half_j = np.mean(scores_j[:mid_point])
                second_half_j = np.mean(scores_j[mid_point:])
                
                change_i = second_half_i - first_half_i
                change_j = second_half_j - first_half_j
                
                threshold = 0.2
                
                if abs(change_i) > threshold and abs(change_j) > threshold:
                    if change_i * change_j > 0:
                        # 同期的変化
                        link = NetworkLink(
                            from_res=res_i,
                            to_res=res_j,
                            strength=float(np.sqrt(abs(change_i * change_j))),
                            lag=0,
                            sync_rate=0.6,
                            link_type='transition',
                            distance=spatial_constraints[(res_i, res_j)]
                        )
                        sync_links.append(link)
                    else:
                        # 因果的変化
                        if abs(change_i) > abs(change_j):
                            from_res, to_res = res_i, res_j
                        else:
                            from_res, to_res = res_j, res_i
                        
                        link = NetworkLink(
                            from_res=from_res,
                            to_res=to_res,
                            strength=float(max(abs(change_i), abs(change_j))),
                            lag=mid_point,
                            sync_rate=0.2,
                            link_type='transition',
                            distance=spatial_constraints.get((from_res, to_res), 0.0)
                        )
                        causal_links.append(link)
                        
                        if link.strength > 0.5:
                            async_bonds.append(link)
        
        return {
            'causal': causal_links,
            'sync': sync_links,
            'async': async_bonds
        }
    
    def _build_classical_transition_network(self,
                                          anomaly_scores: Dict[int, np.ndarray],
                                          residue_coupling: np.ndarray,
                                          spatial_constraints: Dict,
                                          adaptive_windows: Dict[int, int],
                                          lag_window: int) -> Dict:
        """古典的遷移（10+フレーム）のネットワーク構築"""
        # 既存の_build_networksメソッドの内容を使用
        return self._build_networks(
            anomaly_scores,
            residue_coupling,
            spatial_constraints,
            adaptive_windows,
            lag_window
        )
    
    # ========================================
    # CASCADE用ヘルパーメソッド
    # ========================================
    
    def _find_cascade_initiators(self,
                                anomaly_scores: Dict[int, np.ndarray]) -> List[int]:
        """カスケードの開始点を検出"""
        initiators = []
        
        # 最初に異常を示した残基を探す
        for res_id, scores in anomaly_scores.items():
            # 最初の高異常フレームを探す
            first_anomaly = np.where(scores > 2.0)[0]
            if len(first_anomaly) > 0:
                initiators.append((res_id, first_anomaly[0]))
        
        # 時間順にソート
        initiators.sort(key=lambda x: x[1])
        
        # 上位5個の開始点
        return [res_id for res_id, _ in initiators[:5]]
    
    def _trace_cascade_paths(self,
                           initiator: int,
                           anomaly_scores: Dict[int, np.ndarray],
                           residue_coupling: np.ndarray,
                           spatial_constraints: Dict,
                           max_depth: int = 5) -> List[List[int]]:
        """カスケード伝播経路を追跡"""
        paths = []
        
        def dfs(current: int, path: List[int], depth: int):
            if depth >= max_depth:
                paths.append(path.copy())
                return
            
            # 隣接残基を探す
            neighbors = []
            for (res_i, res_j) in spatial_constraints.keys():
                if res_i == current:
                    neighbors.append(res_j)
                elif res_j == current:
                    neighbors.append(res_i)
            
            # スコアでソート
            neighbors = sorted(neighbors, 
                             key=lambda x: np.max(anomaly_scores.get(x, [0])),
                             reverse=True)
            
            for neighbor in neighbors[:3]:  # 上位3経路
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
            
            if len(neighbors) == 0:
                paths.append(path.copy())
        
        dfs(initiator, [initiator], 0)
        return paths
    
    def _calculate_cascade_strength(self,
                                  from_res: int,
                                  to_res: int,
                                  anomaly_scores: Dict[int, np.ndarray],
                                  residue_coupling: np.ndarray) -> float:
        """カスケード伝播の強度を計算"""
        # 簡易実装
        scores_from = anomaly_scores.get(from_res, np.array([0]))
        scores_to = anomaly_scores.get(to_res, np.array([0]))
        
        # 相関計算
        if len(scores_from) > 1 and len(scores_to) > 1:
            try:
                corr = np.corrcoef(scores_from, scores_to)[0, 1]
                return abs(corr)
            except:
                return 0.0
        return 0.0
    
    def _detect_cascade_synchrony(self,
                                 anomaly_scores: Dict[int, np.ndarray],
                                 residue_coupling: np.ndarray,
                                 spatial_constraints: Dict) -> List[NetworkLink]:
        """カスケードに伴う同期的変化を検出"""
        sync_links = []
        
        # 簡易実装：高異常を同時に示すペアを探す
        residue_ids = sorted(anomaly_scores.keys())
        
        for i, res_i in enumerate(residue_ids):
            for j, res_j in enumerate(residue_ids[i+1:], i+1):
                if (res_i, res_j) not in spatial_constraints:
                    continue
                
                scores_i = anomaly_scores[res_i]
                scores_j = anomaly_scores[res_j]
                
                # 同時に高い異常
                high_i = scores_i > 2.0
                high_j = scores_j > 2.0
                
                overlap = np.sum(high_i & high_j)
                
                if overlap > 0:
                    link = NetworkLink(
                        from_res=res_i,
                        to_res=res_j,
                        strength=float(overlap) / len(scores_i),
                        lag=0,
                        sync_rate=1.0,
                        link_type='cascade',
                        quantum_signature='synchronous_cascade',
                        distance=spatial_constraints[(res_i, res_j)]
                    )
                    sync_links.append(link)
        
        return sync_links
    
    # ========================================
    # 既存のヘルパーメソッド（変更最小限）
    # ========================================
    
    def _compute_adaptive_windows(self,
                                anomaly_scores: Dict[int, np.ndarray]) -> Dict[int, int]:
        """適応的ウィンドウサイズ計算（既存のまま）"""
        n_residues = len(anomaly_scores)
        residue_ids = sorted(anomaly_scores.keys())
        
        # スコアを配列に整理
        n_frames = len(next(iter(anomaly_scores.values())))
        scores_array = np.zeros((n_residues, n_frames), dtype=np.float32)
        
        for i, res_id in enumerate(residue_ids):
            scores_array[i] = anomaly_scores[res_id]
        
        if self.is_gpu and self.adaptive_window_kernel is not None:
            # GPU版
            scores_gpu = self.to_gpu(scores_array)
            window_sizes_gpu = self.zeros(n_residues, dtype=cp.int32)
            
            # カーネル実行
            block_size = 256
            grid_size = (n_residues + block_size - 1) // block_size
            
            self.adaptive_window_kernel(
                (grid_size,), (block_size,),
                (scores_gpu.ravel(), window_sizes_gpu, n_residues, n_frames,
                 30, 300, 100)  # min, max, base window
            )
            
            window_sizes = self.to_cpu(window_sizes_gpu)
        else:
            # CPU版
            window_sizes = np.zeros(n_residues, dtype=np.int32)
            
            for i, scores in enumerate(scores_array):
                # イベント密度
                n_events = np.sum(scores > 1.0)
                event_density = n_events / n_frames
                
                # ボラティリティ
                if np.mean(scores) > 1e-10:
                    volatility = np.std(scores) / np.mean(scores)
                else:
                    volatility = 0.0
                
                # スケールファクター
                scale_factor = 1.0
                if event_density > 0.1:
                    scale_factor *= 0.7
                elif event_density < 0.02:
                    scale_factor *= 2.0
                
                if volatility > 2.0:
                    scale_factor *= 0.8
                elif volatility < 0.5:
                    scale_factor *= 1.3
                
                window_sizes[i] = int(np.clip(100 * scale_factor, 30, 300))
        
        # 辞書に変換
        return {res_id: int(window_sizes[i]) for i, res_id in enumerate(residue_ids)}
    
    def _compute_spatial_constraints(self,
                                   residue_ids: List[int],
                                   residue_coms: np.ndarray) -> Dict[Tuple[int, int], float]:
        """空間制約計算（既存のまま）"""
        n_frames, n_all_residues, _ = residue_coms.shape
        
        # サンプルフレームで平均距離計算
        sample_frames = np.linspace(0, n_frames-1, min(10, n_frames), dtype=int)
        
        if self.is_gpu:
            coms_gpu = self.to_gpu(residue_coms[sample_frames])
            
            # 各フレームで距離計算
            avg_distances = self.zeros((n_all_residues, n_all_residues))
            
            for frame_coms in coms_gpu:
                distances = cp_cdist(frame_coms, frame_coms)
                avg_distances += distances / len(sample_frames)
        else:
            # CPU版
            from scipy.spatial.distance import cdist
            avg_distances = np.zeros((n_all_residues, n_all_residues))
            
            for frame_idx in sample_frames:
                distances = cdist(residue_coms[frame_idx], residue_coms[frame_idx])
                avg_distances += distances / len(sample_frames)
        
        # 有効なペアを抽出
        spatial_constraints = {}
        
        for i, res_i in enumerate(residue_ids):
            for j, res_j in enumerate(residue_ids):
                if i < j and res_i < n_all_residues and res_j < n_all_residues:
                    dist = float(avg_distances[res_i, res_j])
                    
                    if dist < self.max_interaction_distance:
                        spatial_constraints[(res_i, res_j)] = dist
        
        logger.info(f"   Found {len(spatial_constraints)} spatially valid pairs "
                   f"(< {self.max_interaction_distance} Å)")
        
        return spatial_constraints
    
    def _create_all_pairs(self, residue_ids: List[int]) -> Dict[Tuple[int, int], float]:
        """全ペアを作成（既存のまま）"""
        pairs = {}
        for i, res_i in enumerate(residue_ids):
            for j, res_j in enumerate(residue_ids[i+1:], i+1):
                pairs[(res_i, res_j)] = 0.0  # 距離不明
        return pairs
    
    def _build_networks(self,
                       anomaly_scores: Dict[int, np.ndarray],
                       residue_coupling: np.ndarray,
                       spatial_constraints: Dict[Tuple[int, int], float],
                       adaptive_windows: Dict[int, int],
                       lag_window: int) -> Dict[str, List[NetworkLink]]:
        """ネットワーク構築（既存の実装を活用）"""
        causal_links = []
        sync_links = []
        async_bonds = []
        
        # GPUで並列処理できるようにペアをバッチ化
        pairs_list = list(spatial_constraints.keys())
        n_pairs = len(pairs_list)
        batch_size = min(1000, n_pairs)
        
        logger.info(f"   Analyzing {n_pairs} residue pairs in batches of {batch_size}")
        
        for batch_start in range(0, n_pairs, batch_size):
            batch_end = min(batch_start + batch_size, n_pairs)
            batch_pairs = pairs_list[batch_start:batch_end]
            
            # バッチ処理
            batch_results = self._analyze_pair_batch(
                batch_pairs,
                anomaly_scores,
                residue_coupling,
                spatial_constraints,
                adaptive_windows,
                lag_window
            )
            
            # 結果を分類
            for result in batch_results:
                if result['type'] == 'causal':
                    causal_links.append(result['link'])
                elif result['type'] == 'sync':
                    sync_links.append(result['link'])
                
                # 非同期強結合チェック
                if (result.get('has_causality', False) and 
                    abs(result.get('sync_rate', 0)) <= self.sync_threshold):
                    async_bonds.append(result['link'])
        
        # 因果ネットワークをフィルタリング
        causal_links = self._filter_causal_network(causal_links)
        
        return {
            'causal': causal_links,
            'sync': sync_links,
            'async': async_bonds
        }
    
    def _analyze_pair_batch(self,
                          pairs: List[Tuple[int, int]],
                          anomaly_scores: Dict[int, np.ndarray],
                          residue_coupling: np.ndarray,
                          spatial_constraints: Dict[Tuple[int, int], float],
                          adaptive_windows: Dict[int, int],
                          lag_window: int) -> List[Dict]:
        """ペアのバッチ解析（link_typeを'transition'に統一）"""
        results = []
        
        for res_i, res_j in pairs:
            if res_i not in anomaly_scores or res_j not in anomaly_scores:
                continue
            
            scores_i = anomaly_scores[res_i]
            scores_j = anomaly_scores[res_j]
            
            n_frames = len(scores_i)
            if n_frames < 10:
                continue
            
            # 最適ウィンドウ
            window = (adaptive_windows.get(res_i, 100) + 
                     adaptive_windows.get(res_j, 100)) // 2
            
            # 因果性解析
            max_correlation = 0.0
            optimal_lag = 0
            
            # GPU上で相関計算
            if self.is_gpu:
                scores_i_gpu = self.to_gpu(scores_i)
                scores_j_gpu = self.to_gpu(scores_j)
                
                for lag in range(0, min(lag_window, len(scores_i)//2), 10):
                    if lag < len(scores_i):
                        try:
                            # 前方向
                            corr = float(self.xp.corrcoef(
                                scores_i_gpu[:-lag] if lag > 0 else scores_i_gpu,
                                scores_j_gpu[lag:] if lag > 0 else scores_j_gpu
                            )[0, 1])
                            
                            if abs(corr) > abs(max_correlation):
                                max_correlation = corr
                                optimal_lag = lag
                            
                            # 後方向
                            if lag > 0:
                                corr = float(self.xp.corrcoef(
                                    scores_i_gpu[lag:],
                                    scores_j_gpu[:-lag]
                                )[0, 1])
                                
                                if abs(corr) > abs(max_correlation):
                                    max_correlation = corr
                                    optimal_lag = -lag
                        except:
                            continue
            else:
                # CPU版
                for lag in range(0, min(lag_window, len(scores_i)//2), 10):
                    if lag < len(scores_i):
                        try:
                            corr = np.corrcoef(
                                scores_i[:-lag] if lag > 0 else scores_i,
                                scores_j[lag:] if lag > 0 else scores_j
                            )[0, 1]
                            
                            if abs(corr) > abs(max_correlation):
                                max_correlation = corr
                                optimal_lag = lag
                        except:
                            continue
            
            # 同期率
            try:
                sync_rate = float(np.corrcoef(scores_i, scores_j)[0, 1])
            except:
                sync_rate = 0.0
            
            # リンク作成
            distance = spatial_constraints.get((res_i, res_j), 0.0)
            
            # 閾値チェック
            if abs(max_correlation) > self.correlation_threshold:
                # 因果方向決定
                if optimal_lag >= 0:
                    from_res, to_res = res_i, res_j
                else:
                    from_res, to_res = res_j, res_i
                    optimal_lag = -optimal_lag
                
                link = NetworkLink(
                    from_res=from_res,
                    to_res=to_res,
                    strength=abs(max_correlation),
                    lag=optimal_lag,
                    distance=distance,
                    sync_rate=sync_rate,
                    link_type='transition'  # v4.0: 'causal'から変更
                )
                
                results.append({
                    'type': 'causal',
                    'link': link,
                    'has_causality': True,
                    'sync_rate': sync_rate
                })
            
            # 同期チェック
            if abs(sync_rate) > self.sync_threshold:
                link = NetworkLink(
                    from_res=res_i,
                    to_res=res_j,
                    strength=abs(sync_rate),
                    lag=0,
                    distance=distance,
                    sync_rate=sync_rate,
                    link_type='transition'  # v4.0: 'sync'から変更
                )
                
                results.append({
                    'type': 'sync',
                    'link': link
                })
        
        return results
    
    def _filter_causal_network(self,
                             causal_links: List[NetworkLink]) -> List[NetworkLink]:
        """因果ネットワークのフィルタリング（既存のまま）"""
        # 強度でソート
        causal_links.sort(key=lambda x: x.strength, reverse=True)
        
        # 上位N個を選択
        if len(causal_links) > self.max_causal_links:
            logger.info(f"   Filtering causal network: {len(causal_links)} → "
                       f"{self.max_causal_links} links")
            causal_links = causal_links[:self.max_causal_links]
        
        # 最小強度でフィルタ
        causal_links = [
            link for link in causal_links 
            if link.strength >= self.min_causality_strength
        ]
        
        return causal_links
    
    def _compute_network_stats(self,
                             networks: Dict[str, List[NetworkLink]],
                             spatial_constraints: Dict) -> Dict[str, Any]:
        """ネットワーク統計計算（既存のまま）"""
        # 次数分布
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for link in networks['causal']:
            out_degree[link.from_res] += 1
            in_degree[link.to_res] += 1
        
        # ハブ残基
        hubs = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        sinks = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 平均距離
        distances = [link.distance for link in networks['causal'] if link.distance]
        avg_distance = np.mean(distances) if distances else 0.0
        
        # ラグ分布
        lags = [link.lag for link in networks['causal']]
        avg_lag = np.mean(lags) if lags else 0.0
        
        return {
            'n_causal': len(networks['causal']),
            'n_sync': len(networks['sync']),
            'n_async': len(networks['async']),
            'hub_residues': hubs,
            'sink_residues': sinks,
            'avg_interaction_distance': avg_distance,
            'avg_causal_lag': avg_lag,
            'n_spatial_pairs': len(spatial_constraints)
        }
    
    def _print_summary(self, result: NetworkAnalysisResult):
        """結果サマリー出力（v4.0対応）"""
        logger.info("\n🌐 Network Analysis Summary (v4.0):")
        logger.info(f"   Pattern: {result.network_stats.get('pattern', 'unknown')}")
        logger.info(f"   Event type: {result.network_stats.get('event_type', 'unknown')}")
        
        if result.network_stats.get('quantum_signature'):
            logger.info(f"   Quantum signature: {result.network_stats['quantum_signature']}")
        
        logger.info(f"   Causal links: {result.n_causal_links}")
        logger.info(f"   Synchronous links: {result.n_sync_links}")
        logger.info(f"   Async strong bonds: {result.n_async_bonds}")
        
        stats = result.network_stats
        if stats.get('hub_residues'):
            logger.info(f"\n   Top hub residues:")
            for res_id, degree in stats['hub_residues']:
                logger.info(f"     Residue {res_id}: {degree} outgoing links")
    
    def _create_empty_result(self) -> NetworkAnalysisResult:
        """空の結果を生成"""
        return NetworkAnalysisResult(
            causal_network=[],
            sync_network=[],
            async_strong_bonds=[],
            spatial_constraints={},
            adaptive_windows={},
            network_stats={
                'error': 'No data to analyze',
                'event_type': 'NONE',
                'pattern': 'none'
            }
        )

# ===============================
# Standalone Functions（v4.0対応）
# ===============================

def analyze_residue_network_gpu(residue_anomaly_scores: Dict[int, np.ndarray],
                              residue_coupling: np.ndarray,
                              residue_coms: Optional[np.ndarray] = None,
                              max_interaction_distance: float = 15.0,
                              **kwargs) -> NetworkAnalysisResult:
    """残基ネットワーク解析のスタンドアロン関数（v4.0）"""
    analyzer = ResidueNetworkGPU(
        max_interaction_distance=max_interaction_distance,
        **kwargs
    )
    return analyzer.analyze_network(
        residue_anomaly_scores, residue_coupling, residue_coms
    )

def compute_spatial_constraints_gpu(residue_ids: List[int],
                                  residue_coms: np.ndarray,
                                  max_distance: float = 15.0,
                                  backend: Optional[GPUBackend] = None) -> Dict[Tuple[int, int], float]:
    """空間制約計算のスタンドアロン関数"""
    analyzer = ResidueNetworkGPU(max_interaction_distance=max_distance)
    if backend:
        analyzer.device = backend.device
        analyzer.is_gpu = backend.is_gpu
        analyzer.xp = backend.xp
    
    return analyzer._compute_spatial_constraints(residue_ids, residue_coms)

def filter_causal_network_gpu(causal_links: List[NetworkLink],
                            max_links: int = 500,
                            min_strength: float = 0.2) -> List[NetworkLink]:
    """因果ネットワークフィルタリングのスタンドアロン関数"""
    # 強度でソート
    causal_links.sort(key=lambda x: x.strength, reverse=True)
    
    # フィルタリング
    filtered = []
    for link in causal_links[:max_links]:
        if link.strength >= min_strength:
            filtered.append(link)
    
    return filtered

def build_propagation_paths_gpu(initiators: List[int],
                              causal_links: List[NetworkLink],
                              max_depth: int = 5) -> List[List[int]]:
    """伝播経路構築のスタンドアロン関数"""
    # グラフ構築
    graph = defaultdict(list)
    for link in causal_links:
        graph[link.from_res].append((link.to_res, link.strength))
    
    paths = []
    
    def dfs(current: int, path: List[int], depth: int):
        if depth >= max_depth:
            paths.append(path.copy())
            return
        
        if current in graph:
            # 強度でソートして探索
            neighbors = sorted(graph[current], key=lambda x: x[1], reverse=True)
            
            for neighbor, weight in neighbors[:3]:  # 上位3経路
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
        else:
            paths.append(path.copy())
    
    # 各開始点から探索
    for initiator in initiators:
        dfs(initiator, [initiator], 0)
    
    # 重複除去と長さでソート
    unique_paths = []
    seen = set()
    
    for path in sorted(paths, key=len, reverse=True):
        path_tuple = tuple(path)
        if path_tuple not in seen and len(path) > 1:
            seen.add(path_tuple)
            unique_paths.append(path)
    
    return unique_paths
