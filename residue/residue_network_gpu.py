"""
Residue Network Analysis (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

残基間ネットワーク解析のGPU実装！
空間的制約とか、因果ネットワークとか、全部高速に解析するよ〜！💕

by 環ちゃん
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
    link_type: str = 'causal'  # 'causal', 'sync', 'async'
    confidence: float = 1.0

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
# CUDA Kernels
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
# Residue Network GPU Class
# ===============================

class ResidueNetworkGPU(GPUBackend):
    """
    残基ネットワーク解析のGPU実装
    
    残基間の相互作用ネットワークを高速に解析！
    空間的制約も考慮した現実的なネットワークを構築するよ〜！
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
        残基ネットワークを解析
        
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
            logger.info("🎯 Analyzing residue interaction network on GPU")
            
            residue_ids = sorted(residue_anomaly_scores.keys())
            n_residues = len(residue_ids)
            
            # 1. 適応的ウィンドウサイズ計算
            with self.timer('adaptive_windows'):
                adaptive_windows = self._compute_adaptive_windows(residue_anomaly_scores)
            
            # 2. 空間制約計算
            with self.timer('spatial_constraints'):
                if residue_coms is not None:
                    spatial_constraints = self._compute_spatial_constraints(
                        residue_ids, residue_coms
                    )
                else:
                    logger.warning("No spatial information provided, analyzing all pairs")
                    spatial_constraints = self._create_all_pairs(residue_ids)
            
            # 3. ネットワーク構築
            with self.timer('build_network'):
                networks = self._build_networks(
                    residue_anomaly_scores,
                    residue_coupling,
                    spatial_constraints,
                    adaptive_windows,
                    lag_window
                )
            
            # 4. 統計情報計算
            network_stats = self._compute_network_stats(networks, spatial_constraints)
            
            # 結果をまとめる
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
    
    def _compute_adaptive_windows(self,
                                anomaly_scores: Dict[int, np.ndarray]) -> Dict[int, int]:
        """適応的ウィンドウサイズ計算（GPU版）"""
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
        """空間制約計算（GPU版）"""
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
        """全ペアを作成（空間情報なし）"""
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
        """ネットワーク構築"""
        causal_links = []
        sync_links = []
        async_bonds = []
        
        # GPUで並列処理できるようにペアをバッチ化
        pairs_list = list(spatial_constraints.keys())
        n_pairs = len(pairs_list)
        batch_size = min(1000, n_pairs)  # メモリ考慮
        
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
        """ペアのバッチ解析"""
        results = []
        
        for res_i, res_j in pairs:
            if res_i not in anomaly_scores or res_j not in anomaly_scores:
                continue
            
            scores_i = anomaly_scores[res_i]
            scores_j = anomaly_scores[res_j]
            
            # 最適ウィンドウ
            window = (adaptive_windows.get(res_i, 100) + 
                     adaptive_windows.get(res_j, 100)) // 2
            
            # 因果性解析（簡略版）
            max_correlation = 0.0
            optimal_lag = 0
            
            # GPU上で相関計算
            if self.is_gpu:
                scores_i_gpu = self.to_gpu(scores_i)
                scores_j_gpu = self.to_gpu(scores_j)
                
                for lag in range(0, min(lag_window, len(scores_i)//2), 10):
                    if lag < len(scores_i):
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
            else:
                # CPU版（同じロジック）
                for lag in range(0, min(lag_window, len(scores_i)//2), 10):
                    if lag < len(scores_i):
                        corr = np.corrcoef(
                            scores_i[:-lag] if lag > 0 else scores_i,
                            scores_j[lag:] if lag > 0 else scores_j
                        )[0, 1]
                        
                        if abs(corr) > abs(max_correlation):
                            max_correlation = corr
                            optimal_lag = lag
            
            # 同期率
            sync_rate = float(np.corrcoef(scores_i, scores_j)[0, 1])
            
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
                    link_type='causal'
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
                    link_type='sync'
                )
                
                results.append({
                    'type': 'sync',
                    'link': link
                })
        
        return results
    
    def _filter_causal_network(self,
                             causal_links: List[NetworkLink]) -> List[NetworkLink]:
        """因果ネットワークのフィルタリング"""
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
        """ネットワーク統計計算"""
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
        """結果サマリー出力"""
        logger.info("\n🌐 Network Analysis Summary:")
        logger.info(f"   Causal links: {result.n_causal_links}")
        logger.info(f"   Synchronous links: {result.n_sync_links}")
        logger.info(f"   Async strong bonds: {result.n_async_bonds}")
        
        stats = result.network_stats
        if stats['hub_residues']:
            logger.info(f"\n   Top hub residues:")
            for res_id, degree in stats['hub_residues']:
                logger.info(f"     Residue {res_id}: {degree} outgoing links")

# ===============================
# Standalone Functions
# ===============================

def analyze_residue_network_gpu(residue_anomaly_scores: Dict[int, np.ndarray],
                              residue_coupling: np.ndarray,
                              residue_coms: Optional[np.ndarray] = None,
                              max_interaction_distance: float = 15.0,
                              **kwargs) -> NetworkAnalysisResult:
    """残基ネットワーク解析のスタンドアロン関数"""
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
