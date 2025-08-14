"""
Lambda³ GPU版2段階解析モジュール
マクロレベル→残基レベルの階層的解析をGPUで高速化
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from concurrent.futures import ThreadPoolExecutor
import warnings

from ..core.gpu_utils import GPUBackend
from ..residue.residue_structures_gpu import ResidueStructuresGPU
from ..residue.residue_network_gpu import ResidueNetworkGPU
from ..residue.causality_analysis_gpu import CausalityAnalyzerGPU
from ..residue.confidence_analysis_gpu import ConfidenceAnalyzerGPU
from .md_lambda3_detector_gpu import MDLambda3Result
from ..types import ArrayType, NDArray

warnings.filterwarnings('ignore')


@dataclass
class ResidueAnalysisConfig:
    """残基レベル解析の設定"""
    # 解析パラメータ
    sensitivity: float = 1.0
    correlation_threshold: float = 0.15
    sync_threshold: float = 0.2
    
    # ウィンドウパラメータ
    min_window: int = 30
    max_window: int = 300
    base_window: int = 50
    base_lag_window: int = 100
    
    # ネットワーク制約
    max_causal_links: int = 500
    min_causality_strength: float = 0.2
    
    # Bootstrap パラメータ
    use_confidence: bool = True
    n_bootstrap: int = 50
    confidence_level: float = 0.95
    
    # GPU設定
    gpu_batch_residues: int = 50
    parallel_events: bool = True
    
    # イベント固有設定
    event_sensitivities: Dict[str, float] = field(default_factory=lambda: {
        'ligand_binding_effect': 1.5,
        'slow_helix_destabilization': 1.0,
        'rapid_partial_unfold': 0.8,
        'transient_refolding_attempt': 1.2,
        'aggregation_onset': 1.0
    })
    
    event_windows: Dict[str, int] = field(default_factory=lambda: {
        'ligand_binding_effect': 100,
        'slow_helix_destabilization': 500,
        'rapid_partial_unfold': 50,
        'transient_refolding_attempt': 200,
        'aggregation_onset': 300
    })


@dataclass
class ResidueEvent:
    """残基レベルイベント"""
    residue_id: int
    residue_name: str
    start_frame: int
    end_frame: int
    peak_lambda_f: float
    propagation_delay: int
    role: str
    adaptive_window: int = 100


@dataclass
class ResidueLevelAnalysis:
    """残基レベル解析結果"""
    event_name: str
    macro_start: int
    macro_end: int
    residue_events: List[ResidueEvent] = field(default_factory=list)
    causality_chain: List[Tuple[int, int, float]] = field(default_factory=list)
    initiator_residues: List[int] = field(default_factory=list)
    key_propagation_paths: List[List[int]] = field(default_factory=list)
    async_strong_bonds: List[Dict] = field(default_factory=list)
    sync_network: List[Dict] = field(default_factory=list)
    network_stats: Dict = field(default_factory=dict)
    confidence_results: List[Any] = field(default_factory=list)
    gpu_time: float = 0.0


@dataclass
class TwoStageLambda3Result:
    """2段階解析の統合結果"""
    macro_result: MDLambda3Result
    residue_analyses: Dict[str, ResidueLevelAnalysis]
    global_residue_importance: Dict[int, float]
    suggested_intervention_points: List[int]
    global_network_stats: Dict
    total_gpu_time: float = 0.0


class TwoStageAnalyzerGPU(GPUBackend):
    """GPU版2段階解析器"""
    
    def __init__(self, config: ResidueAnalysisConfig = None):
        super().__init__()
        self.config = config or ResidueAnalysisConfig()
        
        # GPU版コンポーネント初期化
        self.residue_structures = ResidueStructuresGPU()
        self.residue_network = ResidueNetworkGPU()
        self.causality_analyzer = CausalityAnalyzerGPU()
        self.confidence_analyzer = ConfidenceAnalyzerGPU()
        
        # メモリマネージャ共有
        for component in [self.residue_structures, self.residue_network,
                         self.causality_analyzer, self.confidence_analyzer]:
            component.memory_manager = self.memory_manager
            component.device = self.device
    
    def analyze_trajectory(self,
                          trajectory: np.ndarray,
                          macro_result: MDLambda3Result,
                          detected_events: List[Tuple[int, int, str]],
                          n_residues: int = 129) -> TwoStageLambda3Result:
        """
        2段階Lambda³解析の実行（GPU高速化版）
        
        Parameters
        ----------
        trajectory : np.ndarray
            MD軌道 (n_frames, n_atoms, 3)
        macro_result : MDLambda3Result
            マクロレベル解析結果
        detected_events : List[Tuple[int, int, str]]
            検出イベントリスト
        n_residues : int, default=129
            残基数
            
        Returns
        -------
        TwoStageLambda3Result
            統合解析結果
        """
        start_time = time.time()
        
        print("\n" + "="*60)
        print("=== Two-Stage Lambda³ Analysis (GPU) ===")
        print("="*60)
        print(f"Events to analyze: {len(detected_events)}")
        print(f"Number of residues: {n_residues}")
        print(f"GPU Device: {self.device}")
        
        # 入力検証
        if len(trajectory.shape) != 3:
            raise ValueError(f"Expected 3D trajectory, got shape {trajectory.shape}")
        
        # セットアップ
        residue_atoms = self._create_residue_mapping(trajectory.shape[1], n_residues)
        residue_names = self._get_residue_names(n_residues)
        
        # 各イベントを解析
        residue_analyses = {}
        all_important_residues = {}
        
        # 並列処理の決定
        if self.config.parallel_events and len(detected_events) > 1:
            residue_analyses = self._analyze_events_parallel(
                trajectory, detected_events, residue_atoms, residue_names
            )
        else:
            residue_analyses = self._analyze_events_sequential(
                trajectory, detected_events, residue_atoms, residue_names
            )
        
        # グローバル重要度の計算
        for event_name, analysis in residue_analyses.items():
            for event in analysis.residue_events:
                res_id = event.residue_id
                if res_id not in all_important_residues:
                    all_important_residues[res_id] = 0.0
                
                # 重要度スコア
                importance = event.peak_lambda_f * (1 + 0.1 * (100 / event.adaptive_window))
                all_important_residues[res_id] += importance
        
        # 介入ポイントの特定
        intervention_points = self._identify_intervention_points_gpu(
            all_important_residues
        )
        
        # グローバル統計
        global_stats = self._compute_global_stats(residue_analyses)
        
        # 結果サマリー
        self._print_summary(all_important_residues, global_stats, intervention_points)
        
        total_time = time.time() - start_time
        
        return TwoStageLambda3Result(
            macro_result=macro_result,
            residue_analyses=residue_analyses,
            global_residue_importance=all_important_residues,
            suggested_intervention_points=intervention_points,
            global_network_stats=global_stats,
            total_gpu_time=total_time
        )
    
    def _analyze_events_parallel(self,
                               trajectory: np.ndarray,
                               detected_events: List[Tuple[int, int, str]],
                               residue_atoms: Dict[int, List[int]],
                               residue_names: Dict[int, str]) -> Dict[str, ResidueLevelAnalysis]:
        """イベントの並列解析"""
        print("\n📍 Processing events in parallel on GPU...")
        
        residue_analyses = {}
        
        # ThreadPoolExecutorで並列実行
        with ThreadPoolExecutor(max_workers=min(4, len(detected_events))) as executor:
            futures = []
            
            for start, end, event_name in detected_events:
                future = executor.submit(
                    self._analyze_single_event_gpu,
                    trajectory, event_name, start, end,
                    residue_atoms, residue_names
                )
                futures.append((event_name, future))
            
            # 結果収集
            for event_name, future in futures:
                try:
                    analysis = future.result()
                    residue_analyses[event_name] = analysis
                    print(f"  ✓ {event_name} complete (GPU time: {analysis.gpu_time:.2f}s)")
                except Exception as e:
                    print(f"  ✗ {event_name} failed: {str(e)}")
        
        return residue_analyses
    
    def _analyze_events_sequential(self,
                                 trajectory: np.ndarray,
                                 detected_events: List[Tuple[int, int, str]],
                                 residue_atoms: Dict[int, List[int]],
                                 residue_names: Dict[int, str]) -> Dict[str, ResidueLevelAnalysis]:
        """イベントの逐次解析"""
        print("\n📍 Processing events sequentially on GPU...")
        
        residue_analyses = {}
        
        for start, end, event_name in detected_events:
            print(f"\n  → Analyzing {event_name}...")
            analysis = self._analyze_single_event_gpu(
                trajectory, event_name, start, end,
                residue_atoms, residue_names
            )
            residue_analyses[event_name] = analysis
            print(f"    GPU time: {analysis.gpu_time:.2f}s")
        
        return residue_analyses
    
    def _analyze_single_event_gpu(self,
                              trajectory: np.ndarray,
                              event_name: str,
                              start_frame: int,
                              end_frame: int,
                              residue_atoms: Dict[int, List[int]],
                              residue_names: Dict[int, str]) -> ResidueLevelAnalysis:
        """単一イベントのGPU解析"""
        event_start_time = time.time()
        
        # GPUメモリコンテキスト
        event_frames = end_frame - start_frame
        with self.memory_manager.batch_context(event_frames * len(residue_atoms) * 3 * 4):
            
            # 1. 残基構造計算
            structures = self.residue_structures.compute_residue_structures(
                trajectory, start_frame, end_frame, residue_atoms
            )
            
            # 2. 異常検出
            anomaly_scores = self._detect_residue_anomalies_gpu(
                structures, event_name
            )
            
            # 3. ネットワーク解析
            network_results = self.residue_network.analyze_network(
                anomaly_scores,
                structures.residue_coupling,
                structures.residue_coms 
            )
            
            # 4. イベント構築
            residue_events = self._build_residue_events_gpu(
                anomaly_scores, residue_names, start_frame, network_results
            )
            
            # 5. イニシエータ特定
            initiators = self._find_initiators_gpu(
                residue_events, network_results['causal_network']
            )
            
            # 6. 因果連鎖
            causality_chains = [
                (link.from_res, link.to_res, link.strength)
                for link in network_results.causal_network
            ]

            # 7. 伝播経路
            propagation_paths = self._build_propagation_paths_gpu(
                initiators, causality_chains
            )
            
            # 8. 信頼区間解析（オプション）
            confidence_results = []
            if self.config.use_confidence and causality_chains:
                confidence_results = self.confidence_analyzer.analyze(
                    causality_chains[:10], anomaly_scores
                )
        
        gpu_time = time.time() - event_start_time
        
        # 結果を返す
        network_stats = {
            'n_causal': network_results.n_causal_links,
            'n_sync': network_results.n_sync_links,
            'n_async': network_results.n_async_bonds,
            'mean_adaptive_window': np.mean(list(network_results.adaptive_windows.values())) if network_results.adaptive_windows else 100
        }
        
        # 結果を返す（修正）
        return ResidueLevelAnalysis(
            event_name=event_name,
            macro_start=start_frame,
            macro_end=end_frame,
            residue_events=residue_events,
            causality_chain=causality_chains,
            initiator_residues=initiators,
            key_propagation_paths=propagation_paths[:5],
            async_strong_bonds=network_results.async_strong_bonds,  # 直接アクセス
            sync_network=network_results.sync_network,  # 直接アクセス
            network_stats=network_stats,
            confidence_results=confidence_results,
            gpu_time=gpu_time
        )
    
    def _detect_residue_anomalies_gpu(self,
                                    structures,  # ResidueStructureResultまたはDict
                                    event_type: str) -> Dict[int, np.ndarray]:
        """
        残基異常検出（GPU最適化）
        
        Parameters
        ----------
        structures : ResidueStructureResult or Dict
            残基構造解析結果（データクラスまたは辞書）
        event_type : str
            イベントタイプ名
            
        Returns
        -------
        Dict[int, np.ndarray]
            残基ID -> 異常スコア時系列
        """
        # structuresがデータクラスか辞書か判定
        if hasattr(structures, 'residue_rho_t'):
            # データクラスの場合（ResidueStructureResult）
            residue_rho_t = structures.residue_rho_t
            residue_lambda_f_mag = structures.residue_lambda_f_mag
        else:
            # 辞書の場合（後方互換性）
            residue_rho_t = structures['residue_rho_t']
            residue_lambda_f_mag = structures['residue_lambda_f_mag']
        
        n_frames, n_residues = residue_rho_t.shape
        
        # イベント固有の感度
        sensitivity = self.config.event_sensitivities.get(
            event_type, self.config.sensitivity
        )
        
        # GPU上で異常検出
        residue_anomaly_scores = {}
        
        # バッチ処理で残基を解析
        batch_size = self.config.gpu_batch_residues
        
        for batch_start in range(0, n_residues, batch_size):
            batch_end = min(batch_start + batch_size, n_residues)
            
            # バッチデータをGPUに転送
            batch_lambda_f = self.to_gpu(
                residue_lambda_f_mag[:, batch_start:batch_end]
            )
            batch_rho_t = self.to_gpu(
                residue_rho_t[:, batch_start:batch_end]
            )
            
            # GPU上で異常スコア計算
            for i, res_id in enumerate(range(batch_start, batch_end)):
                lambda_anomaly = self._compute_anomaly_gpu(batch_lambda_f[:, i])
                rho_anomaly = self._compute_anomaly_gpu(batch_rho_t[:, i])
                
                # 統合スコア
                combined = (lambda_anomaly + rho_anomaly) / 2
                
                # 有意な異常のみ保存
                if cp.max(combined) > sensitivity:
                    residue_anomaly_scores[res_id] = self.to_cpu(combined)
        
        return residue_anomaly_scores
    
    def _compute_anomaly_gpu(self, series: cp.ndarray, window: int = 50) -> cp.ndarray:
        """GPU上で異常スコア計算"""
        anomaly = cp.zeros_like(series)
        
        for i in range(len(series)):
            start = max(0, i - window)
            end = min(len(series), i + window + 1)
            
            local_mean = cp.mean(series[start:end])
            local_std = cp.std(series[start:end])
            
            if local_std > 1e-10:
                anomaly[i] = cp.abs(series[i] - local_mean) / local_std
        
        return anomaly
    
    def _build_residue_events_gpu(self,
                            anomaly_scores: Dict[int, np.ndarray],
                            residue_names: Dict[int, str],
                            start_frame: int,
                            network_results) -> List[ResidueEvent]:
        """修正版：NetworkAnalysisResultを正しく扱う"""
        import numpy as np  # ローカルインポート
        events = []
        
        # find_peaksのインポート（既存のコード）
        find_peaks_func = None
        use_gpu_peaks = False
        
        try:
            if self.is_gpu and HAS_CUPY:
                from cupyx.scipy.signal import find_peaks as find_peaks_func
                use_gpu_peaks = True
            else:
                from scipy.signal import find_peaks as find_peaks_func
                use_gpu_peaks = False
        except ImportError:
            print("  ⚠️ find_peaks not available, using simple peak detection")
            find_peaks_func = None
        
        for res_id, scores in anomaly_scores.items():
            peaks = []
            peak_heights = []
            
            # ピーク検出（既存のコード）
            if find_peaks_func is not None:
                try:
                    if use_gpu_peaks:
                        scores_gpu = self.to_gpu(scores)
                        peaks, properties = find_peaks_func(
                            scores_gpu,
                            height=self.config.sensitivity,
                            distance=50
                        )
                        peaks = self.to_cpu(peaks)
                        peak_heights = self.to_cpu(properties['peak_heights'])
                    else:
                        peaks, properties = find_peaks_func(
                            scores,
                            height=self.config.sensitivity,
                            distance=50
                        )
                        peak_heights = properties['peak_heights']
                except Exception as e:
                    print(f"  ⚠️ find_peaks failed for residue {res_id}: {e}")
                    find_peaks_func = None
            
            # フォールバック処理（numpy修正）
            if find_peaks_func is None or len(peaks) == 0:
                for i in range(1, len(scores)-1):
                    if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                        if scores[i] > self.config.sensitivity:
                            peaks.append(i)
                            peak_heights.append(scores[i])
                peaks = np.array(peaks) if peaks else np.array([])  # np定義済み！
                peak_heights = np.array(peak_heights) if peak_heights else np.array([])
            
            # イベント構築（NetworkAnalysisResult対応）
            if len(peaks) > 0:
                first_peak = peaks[0]
                peak_height = peak_heights[0]
                
                # NetworkAnalysisResultの属性アクセス（修正）
                adaptive_window = 100  # デフォルト値
                if hasattr(network_results, 'adaptive_windows'):
                    adaptive_window = network_results.adaptive_windows.get(res_id, 100)
                
                event = ResidueEvent(
                    residue_id=res_id,
                    residue_name=residue_names.get(res_id, f"RES{res_id}"),
                    start_frame=start_frame + first_peak,
                    end_frame=start_frame + min(first_peak + 100, len(scores)),
                    peak_lambda_f=float(peak_height),
                    propagation_delay=first_peak,
                    role='initiator' if first_peak < 50 else 'propagator',
                    adaptive_window=adaptive_window
                )
                events.append(event)
        
        return events
    
    def _find_initiators_gpu(self,
                           residue_events: List[ResidueEvent],
                           causal_network: List[Dict]) -> List[int]:
        """イニシエータ残基の特定"""
        initiators = []
        
        # 早期応答残基
        for event in residue_events:
            if event.propagation_delay < 50:
                initiators.append(event.residue_id)
        
        # ネットワークのハブ
        if causal_network:
            out_degree = {}
            for link in causal_network:
                from_res = link.from_res
                if from_res not in out_degree:
                    out_degree[from_res] = 0
                out_degree[from_res] += 1
            
            # 上位ハブを追加
            sorted_hubs = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
            for res_id, degree in sorted_hubs[:3]:
                if res_id not in initiators and degree >= 3:
                    initiators.append(res_id)
        
        return initiators
    
    def _build_propagation_paths_gpu(self,
                                   initiators: List[int],
                                   causality_chains: List[Tuple[int, int, float]],
                                   max_depth: int = 5) -> List[List[int]]:
        """伝播経路の構築"""
        # グラフ構築
        graph = {}
        for res1, res2, weight in causality_chains:
            if res1 not in graph:
                graph[res1] = []
            graph[res1].append((res2, weight))
        
        paths = []
        
        def dfs(current: int, path: List[int], depth: int):
            if depth >= max_depth:
                paths.append(path.copy())
                return
            
            if current in graph:
                for neighbor, weight in graph[current]:
                    if neighbor not in path:
                        path.append(neighbor)
                        dfs(neighbor, path, depth + 1)
                        path.pop()
            else:
                paths.append(path.copy())
        
        # 各イニシエータから探索
        for initiator in initiators:
            dfs(initiator, [initiator], 0)
        
        # 重複除去と並べ替え
        unique_paths = []
        seen = set()
        
        for path in sorted(paths, key=len, reverse=True):
            path_tuple = tuple(path)
            if path_tuple not in seen and len(path) > 1:
                seen.add(path_tuple)
                unique_paths.append(path)
        
        return unique_paths
    
    def _identify_intervention_points_gpu(self,
                                        importance_scores: Dict[int, float],
                                        top_n: int = 10) -> List[int]:
        """介入ポイントの特定（GPU使用）"""
        # GPU上でソート
        if importance_scores:
            residues = cp.array(list(importance_scores.keys()))
            scores = cp.array(list(importance_scores.values()))
            
            # 降順でソート
            sorted_indices = cp.argsort(scores)[::-1]
            
            # 上位を取得
            top_residues = residues[sorted_indices[:top_n]]
            
            return self.to_cpu(top_residues).tolist()
        
        return []
    
    def _create_residue_mapping(self, n_atoms: int, n_residues: int) -> Dict[int, List[int]]:
        """残基マッピングの作成"""
        atoms_per_residue = n_atoms // n_residues
        residue_atoms = {}
        
        for res_id in range(n_residues):
            start_atom = res_id * atoms_per_residue
            end_atom = min(start_atom + atoms_per_residue, n_atoms)
            residue_atoms[res_id] = list(range(start_atom, end_atom))
        
        # 余った原子を最後の残基に
        if n_atoms % n_residues != 0:
            remaining_start = n_residues * atoms_per_residue
            residue_atoms[n_residues-1].extend(range(remaining_start, n_atoms))
        
        return residue_atoms
    
    def _get_residue_names(self, n_residues: int) -> Dict[int, str]:
        """残基名の取得"""
        return {i: f"RES{i+1}" for i in range(n_residues)}
    
    def _compute_global_stats(self,
                            residue_analyses: Dict[str, ResidueLevelAnalysis]) -> Dict:
        """グローバル統計の計算"""
        total_causal = sum(a.network_stats['n_causal'] for a in residue_analyses.values())
        total_sync = sum(a.network_stats['n_sync'] for a in residue_analyses.values())
        total_async = sum(a.network_stats['n_async'] for a in residue_analyses.values())
        
        total_gpu_time = sum(a.gpu_time for a in residue_analyses.values())
        
        return {
            'total_causal_links': total_causal,
            'total_sync_links': total_sync,
            'total_async_bonds': total_async,
            'async_to_causal_ratio': total_async / (total_causal + 1e-10),
            'mean_adaptive_window': np.mean([
                a.network_stats['mean_adaptive_window'] 
                for a in residue_analyses.values()
            ]),
            'total_gpu_time': total_gpu_time,
            'events_analyzed': len(residue_analyses)
        }
    
    def _print_summary(self,
                     importance_scores: Dict,
                     global_stats: Dict,
                     intervention_points: List[int]):
        """解析サマリーの表示"""
        print("\n🎯 Global Analysis Complete!")
        print(f"   Key residues identified: {len(importance_scores)}")
        print(f"   Total causal links: {global_stats['total_causal_links']}")
        print(f"   Total async strong bonds: {global_stats['total_async_bonds']} "
              f"({global_stats['async_to_causal_ratio']:.1%})")
        print(f"   Mean adaptive window: {global_stats['mean_adaptive_window']:.1f} frames")
        print(f"   Total GPU time: {global_stats['total_gpu_time']:.2f} seconds")
        print(f"   Suggested intervention points: {intervention_points[:5]}")


# ユーティリティ関数
def perform_two_stage_analysis_gpu(trajectory: np.ndarray,
                                 macro_result: MDLambda3Result,
                                 detected_events: List[Tuple[int, int, str]],
                                 n_residues: int = 129,
                                 config: ResidueAnalysisConfig = None) -> TwoStageLambda3Result:
    """
    2段階解析の便利なラッパー関数（後方互換性）
    """
    analyzer = TwoStageAnalyzerGPU(config)
    return analyzer.analyze_trajectory(trajectory, macro_result, detected_events, n_residues)
