"""
MD Lambda³ Detector GPU - リファクタリング版
バッチ処理後の解析ステップを適切に実装
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

from ..core.gpu_utils import GPUBackend
from ..structures.lambda_structures_gpu import LambdaStructuresGPU
from ..structures.md_features_gpu import MDFeaturesGPU
from ..detection.anomaly_detection_gpu import AnomalyDetectorGPU
from ..detection.boundary_detection_gpu import BoundaryDetectorGPU  # detectionから！
from ..detection.topology_breaks_gpu import TopologyBreaksDetectorGPU
from ..detection.phase_space_gpu import PhaseSpaceAnalyzerGPU  
from ..detection.extended_detection_gpu import ExtendedDetectorGPU

@dataclass
class MDConfig:
    """MD解析設定"""
    # Lambda³パラメータ
    adaptive_window: bool = True
    extended_detection: bool = True
    use_extended_detection: bool = True
    use_phase_space: bool = False
    sensitivity: float = 2.0
    min_boundary_gap: int = 100
    
    # MD特有の設定
    use_rmsd: bool = True
    use_rg: bool = True
    use_dihedrals: bool = True
    use_contacts: bool = True
    
    # GPU設定
    gpu_batch_size: int = 10000
    mixed_precision: bool = False
    benchmark_mode: bool = False


@dataclass
class MDLambda3Result:
    """MD Lambda³解析結果のデータクラス"""
    # Core Lambda³構造
    lambda_structures: Dict[str, np.ndarray]
    structural_boundaries: Dict[str, Any]
    topological_breaks: Dict[str, np.ndarray]
    
    # MD特有の特徴
    md_features: Dict[str, np.ndarray]
    
    # 解析結果
    anomaly_scores: Dict[str, np.ndarray]
    detected_structures: List[Dict]
    
    # 位相空間解析（オプション）
    phase_space_analysis: Optional[Dict] = None
    
    # メタデータ
    n_frames: int = 0
    n_atoms: int = 0
    window_steps: int = 0
    computation_time: float = 0.0
    gpu_info: Dict = None
    critical_events: List = field(default_factory=list)


class MDLambda3DetectorGPU(GPUBackend):
    """GPU版Lambda³ MD検出器（リファクタリング版）"""
    
    def __init__(self, config: MDConfig = None, device: str = 'auto'):
        """
        Parameters
        ----------
        config : MDConfig, optional
            設定パラメータ
        device : str, default='auto'
            'auto', 'gpu', 'cpu'のいずれか
        """
        # GPUBackendの初期化
        if device == 'cpu':
            super().__init__(device='cpu', force_cpu=True)
        else:
            super().__init__(device=device, force_cpu=False)
        
        self.config = config or MDConfig()
        self.verbose = True
        
        # force_cpuフラグを決定
        force_cpu_flag = not self.is_gpu
        
        # GPU版コンポーネントの初期化
        self.structure_computer = LambdaStructuresGPU(force_cpu_flag)
        self.feature_extractor = MDFeaturesGPU(force_cpu_flag)
        self.anomaly_detector = AnomalyDetectorGPU(force_cpu_flag)
        self.boundary_detector = BoundaryDetectorGPU(force_cpu_flag)
        self.topology_detector = TopologyBreaksDetectorGPU(force_cpu_flag)
        self.extended_detector = ExtendedDetectorGPU(force_cpu_flag)
        self.phase_space_analyzer = PhaseSpaceAnalyzerGPU(force_cpu_flag)
        
        # メモリマネージャとデバイスを共有
        for component in [self.structure_computer, self.feature_extractor,
                         self.anomaly_detector, self.boundary_detector,
                         self.topology_detector, self.extended_detector,
                         self.phase_space_analyzer]:
            if hasattr(component, 'memory_manager'):
                component.memory_manager = self.memory_manager
            if hasattr(component, 'device'):
                component.device = self.device
        
        self._print_initialization_info()
    
    def analyze(self, 
                trajectory: np.ndarray,
                backbone_indices: Optional[np.ndarray] = None) -> MDLambda3Result:
        """
        MD軌道のLambda³解析（完全GPU化）
        
        Parameters
        ----------
        trajectory : np.ndarray
            MD軌道 (n_frames, n_atoms, 3)
        backbone_indices : np.ndarray, optional
            バックボーン原子のインデックス
            
        Returns
        -------
        MDLambda3Result
            解析結果
        """
        start_time = time.time()
        n_frames, n_atoms, _ = trajectory.shape
        
        print(f"\n{'='*60}")
        print(f"=== Lambda³ MD Analysis (GPU) ===")
        print(f"{'='*60}")
        print(f"Trajectory: {n_frames} frames, {n_atoms} atoms")
        print(f"GPU Device: {self.device}")
        
        # メモリ情報の安全な取得
        try:
            mem_info = self.memory_manager.get_memory_info()
            print(f"Available GPU Memory: {mem_info.free / 1e9:.2f} GB")
        except Exception as e:
            print(f"Memory info unavailable: {e}")
        
        # バッチ処理の判定
        batch_size = min(self.config.gpu_batch_size, n_frames)
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        if n_batches > 1:
            print(f"Processing in {n_batches} batches of {batch_size} frames")
            result = self._analyze_batched(trajectory, backbone_indices, batch_size)
        else:
            # 単一バッチ処理
            result = self._analyze_single_trajectory(trajectory, backbone_indices)
        
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        
        self._print_summary(result)
        
        return result
    
    def _analyze_single_trajectory(self,
                                 trajectory: np.ndarray,
                                 backbone_indices: Optional[np.ndarray]) -> MDLambda3Result:
        """単一軌道の解析（メモリに収まる場合）"""
        n_frames, n_atoms, _ = trajectory.shape
        
        # 1. MD特徴抽出
        print("\n1. Extracting MD features on GPU...")
        md_features = self.feature_extractor.extract_md_features(
            trajectory, backbone_indices
        )
        
        # 2. 初期ウィンドウサイズ
        initial_window = self._compute_initial_window(n_frames)
        
        # 3. Lambda構造計算（第1回）
        print("\n2. Computing Lambda³ structures (first pass)...")
        lambda_structures = self.structure_computer.compute_lambda_structures(
            trajectory, md_features, initial_window
        )
        
        # 4. 適応的ウィンドウサイズ決定
        adaptive_windows = self._determine_adaptive_windows(
            lambda_structures, initial_window
        )
        primary_window = adaptive_windows.get('primary', initial_window)
        
        # 5. 構造境界検出
        print("\n3. Detecting structural boundaries...")
        boundary_window = adaptive_windows.get('boundary', primary_window // 3)
        structural_boundaries = self.boundary_detector.detect_structural_boundaries(
            lambda_structures, boundary_window
        )
        
        # 6. トポロジカル破れ検出
        print("\n4. Detecting topological breaks...")
        fast_window = adaptive_windows.get('fast', primary_window // 2)
        topological_breaks = self.topology_detector.detect_topological_breaks(
            lambda_structures, fast_window
        )
        
        # 7. マルチスケール異常検出
        print("\n5. Computing multi-scale anomaly scores...")
        anomaly_scores = self.anomaly_detector.compute_multiscale_anomalies(
            lambda_structures,
            structural_boundaries,
            topological_breaks,
            md_features,
            self.config
        )
        
        # 8. 構造パターン検出
        print("\n6. Detecting structural patterns...")
        slow_window = adaptive_windows.get('slow', primary_window * 2)
        detected_structures = self._detect_structural_patterns(
            lambda_structures, structural_boundaries, slow_window
        )
        
        # 9. 位相空間解析（オプション）
        phase_space_analysis = None
        if self.config.use_phase_space:
            print("\n7. Performing phase space analysis...")
            try:
                phase_space_analysis = self.phase_space_analyzer.analyze_phase_space(
                    lambda_structures
                )
            except Exception as e:
                print(f"Phase space analysis failed: {e}")
        
        # 臨界イベントの検出
        critical_events = self._detect_critical_events(anomaly_scores)
        
        # GPU情報を収集
        gpu_info = self._get_gpu_info()
        
        # 結果を構築
        result = MDLambda3Result(
            lambda_structures=self._to_cpu_dict(lambda_structures),
            structural_boundaries=structural_boundaries,
            topological_breaks=self._to_cpu_dict(topological_breaks),
            md_features=self._to_cpu_dict(md_features),
            anomaly_scores=self._to_cpu_dict(anomaly_scores),
            detected_structures=detected_structures,
            phase_space_analysis=phase_space_analysis,
            critical_events=critical_events,
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=primary_window,
            computation_time=0.0,
            gpu_info=gpu_info
        )
        
        return result
    
    def _analyze_batched(self,
                        trajectory: np.ndarray,
                        backbone_indices: Optional[np.ndarray],
                        batch_size: int) -> MDLambda3Result:
        """バッチ処理による解析（大規模データ用）- リファクタリング版"""
        print("\n⚡ Running batched GPU analysis...")
        
        n_frames = trajectory.shape[0]
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        # バッチごとの結果を蓄積
        batch_results = []
        
        # Step 1: 重い処理をバッチで実行（MD特徴とLambda構造）
        print("\n[Step 1] Processing batches for feature extraction...")
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_frames)
            
            print(f"  Batch {batch_idx + 1}/{n_batches}: frames {start_idx}-{end_idx}")
            
            batch_trajectory = trajectory[start_idx:end_idx]
            
            # バッチ解析（特徴抽出とLambda構造計算のみ）
            batch_result = self._analyze_single_batch(
                batch_trajectory, backbone_indices, start_idx
            )
            
            batch_results.append(batch_result)
            
            # メモリクリア
            self.memory_manager.clear_cache()
        
        # Step 2: 結果をマージ（データ結合のみ）
        print("\n[Step 2] Merging batch results...")
        merged_result = self._merge_batch_results(batch_results, trajectory.shape)
        
        # Step 3: マージされたデータで解析を完了（軽い処理）
        print("\n[Step 3] Completing analysis on merged data...")
        final_result = self._complete_analysis(merged_result)
        
        return final_result
    
    def _analyze_single_batch(self,
                            batch_trajectory: np.ndarray,
                            backbone_indices: Optional[np.ndarray],
                            offset: int) -> Dict:
        """単一バッチの解析（特徴抽出とLambda構造計算のみ）"""
        # MD特徴抽出（重い処理）
        md_features = self.feature_extractor.extract_md_features(
            batch_trajectory, backbone_indices
        )
        
        window = self._compute_initial_window(len(batch_trajectory))
        
        # Lambda構造計算（重い処理）
        lambda_structures = self.structure_computer.compute_lambda_structures(
            batch_trajectory, md_features, window
        )
        
        return {
            'offset': offset,
            'n_frames': len(batch_trajectory),
            'lambda_structures': lambda_structures,
            'md_features': md_features,
            'window': window
        }
    
    def _merge_batch_results(self,
                           batch_results: List[Dict],
                           original_shape: Tuple) -> MDLambda3Result:
        """バッチ結果の統合（データ結合のみ）"""
        print("  Merging data from all batches...")
        
        n_frames, n_atoms, _ = original_shape
        
        if not batch_results:
            return self._create_empty_result(n_frames, n_atoms)
        
        # 結果を保存する辞書を初期化
        merged_lambda_structures = {}
        merged_md_features = {}
        
        # 最初のバッチからキーと形状を取得
        first_batch = batch_results[0]
        lambda_keys = first_batch.get('lambda_structures', {}).keys()
        feature_keys = first_batch.get('md_features', {}).keys()
        
        print(f"    Lambda structure keys: {list(lambda_keys)}")
        print(f"    MD feature keys: {list(feature_keys)}")
        
        # Lambda構造の配列を初期化
        for key in lambda_keys:
            sample = first_batch['lambda_structures'][key]
            if isinstance(sample, (np.ndarray, self.xp.ndarray)):
                rest_shape = sample.shape[1:] if len(sample.shape) > 1 else ()
                full_shape = (n_frames,) + rest_shape
                dtype = sample.dtype
                merged_lambda_structures[key] = np.full(full_shape, np.nan, dtype=dtype)
        
        # MD特徴の配列を初期化
        for key in feature_keys:
            sample = first_batch['md_features'][key]
            if isinstance(sample, (np.ndarray, self.xp.ndarray)):
                rest_shape = sample.shape[1:] if len(sample.shape) > 1 else ()
                full_shape = (n_frames,) + rest_shape
                dtype = sample.dtype
                merged_md_features[key] = np.full(full_shape, np.nan, dtype=dtype)
        
        # 各バッチの結果を正しい位置に配置
        for batch_idx, batch_result in enumerate(batch_results):
            offset = batch_result['offset']
            batch_n_frames = batch_result['n_frames']
            end_idx = offset + batch_n_frames
            
            # 範囲チェック
            if end_idx > n_frames:
                end_idx = n_frames
                batch_n_frames = end_idx - offset
            
            # Lambda構造をマージ
            for key, value in batch_result.get('lambda_structures', {}).items():
                if key in merged_lambda_structures:
                    if hasattr(value, 'get'):  # CuPy配列の場合
                        value = self.to_cpu(value)
                    actual_frames = min(len(value), batch_n_frames)
                    merged_lambda_structures[key][offset:offset + actual_frames] = value[:actual_frames]
            
            # MD特徴をマージ
            for key, value in batch_result.get('md_features', {}).items():
                if key in merged_md_features:
                    if hasattr(value, 'get'):
                        value = self.to_cpu(value)
                    actual_frames = min(len(value), batch_n_frames)
                    merged_md_features[key][offset:offset + actual_frames] = value[:actual_frames]
        
        # NaNチェック
        for key, arr in merged_lambda_structures.items():
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                print(f"    ⚠️ Warning: {key} has {nan_count} unprocessed frames")
        
        # ウィンドウステップの計算
        window_steps = 100  # デフォルト値
        if 'window' in first_batch:
            windows = [b.get('window', 100) for b in batch_results if 'window' in b]
            if windows:
                window_steps = int(np.mean(windows))
        
        # GPU情報の構築
        gpu_info = {
            'computation_mode': 'batched',
            'n_batches': len(batch_results),
            'device_name': str(self.device),
            'batch_sizes': [b['n_frames'] for b in batch_results]
        }
        
        print(f"  ✅ Merged {n_frames} frames successfully")
        
        # マージ結果を返す（解析は未完了）
        return MDLambda3Result(
            lambda_structures=merged_lambda_structures,
            structural_boundaries={},  # 後で計算
            topological_breaks={},      # 後で計算
            md_features=merged_md_features,
            anomaly_scores={},          # 後で計算
            detected_structures=[],     # 後で計算
            critical_events=[],         # 後で計算
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=window_steps,
            computation_time=0.0,
            gpu_info=gpu_info
        )
    
    def _complete_analysis(self, merged_result: MDLambda3Result) -> MDLambda3Result:
        """マージ後のデータで解析を完了（境界検出、異常スコア計算など）"""
        
        lambda_structures = merged_result.lambda_structures
        md_features = merged_result.md_features
        n_frames = merged_result.n_frames
        
        # 適応的ウィンドウサイズ決定
        initial_window = merged_result.window_steps
        adaptive_windows = self._determine_adaptive_windows(
            lambda_structures, initial_window
        )
        primary_window = adaptive_windows.get('primary', initial_window)
        
        # 5. 構造境界検出（全フレームで実行 - 軽い処理）
        print("  - Detecting structural boundaries...")
        boundary_window = adaptive_windows.get('boundary', primary_window // 3)
        structural_boundaries = self.boundary_detector.detect_structural_boundaries(
            lambda_structures, boundary_window
        )
        
        # 6. トポロジカル破れ検出（全フレームで実行 - 軽い処理）
        print("  - Detecting topological breaks...")
        fast_window = adaptive_windows.get('fast', primary_window // 2)
        topological_breaks = self.topology_detector.detect_topological_breaks(
            lambda_structures, fast_window
        )
        
        # 7. マルチスケール異常検出（全フレームで実行 - 軽い処理）
        print("  - Computing anomaly scores...")
        anomaly_scores = self.anomaly_detector.compute_multiscale_anomalies(
            lambda_structures,
            structural_boundaries,
            topological_breaks,
            md_features,
            self.config
        )
        
        # 8. 構造パターン検出
        print("  - Detecting structural patterns...")
        slow_window = adaptive_windows.get('slow', primary_window * 2)
        detected_structures = self._detect_structural_patterns(
            lambda_structures, structural_boundaries, slow_window
        )
        
        # 9. 位相空間解析（オプション）
        phase_space_analysis = None
        if self.config.use_phase_space:
            print("  - Performing phase space analysis...")
            try:
                phase_space_analysis = self.phase_space_analyzer.analyze_phase_space(
                    lambda_structures
                )
            except Exception as e:
                print(f"    Phase space analysis failed: {e}")
        
        # 臨界イベントの検出
        critical_events = self._detect_critical_events(anomaly_scores)
        
        print("  ✅ Analysis completed!")
        
        # 結果を更新
        merged_result.structural_boundaries = structural_boundaries
        merged_result.topological_breaks = topological_breaks
        merged_result.anomaly_scores = anomaly_scores
        merged_result.detected_structures = detected_structures
        merged_result.phase_space_analysis = phase_space_analysis
        merged_result.critical_events = critical_events
        merged_result.window_steps = primary_window
        
        return merged_result
    
    # === ヘルパーメソッド ===
    
    def _compute_initial_window(self, n_frames: int) -> int:
        """初期ウィンドウサイズの計算"""
        return min(100, n_frames // 10)
    
    def _determine_adaptive_windows(self, lambda_structures: Dict, 
                                   initial_window: int) -> Dict[str, int]:
        """適応的ウィンドウサイズの決定"""
        if not self.config.adaptive_window:
            return {'primary': initial_window}
        
        # Lambda構造の変動から最適なウィンドウサイズを推定
        windows = {
            'primary': initial_window,
            'fast': max(10, initial_window // 2),
            'slow': min(500, initial_window * 2),
            'boundary': max(20, initial_window // 3)
        }
        
        return windows
    
    def _detect_structural_patterns(self, lambda_structures: Dict,
                                   boundaries: Dict, window: int) -> List[Dict]:
        """構造パターンの検出"""
        patterns = []
        
        if isinstance(boundaries, dict) and 'boundary_locations' in boundaries:
            boundary_locs = boundaries['boundary_locations']
            
            # 境界間のセグメントを構造として認識
            if len(boundary_locs) > 0:
                # 最初のセグメント
                if boundary_locs[0] > 50:
                    patterns.append({
                        'type': 'initial_structure',
                        'start': 0,
                        'end': boundary_locs[0],
                        'duration': boundary_locs[0]
                    })
                
                # 中間セグメント
                for i in range(len(boundary_locs) - 1):
                    duration = boundary_locs[i+1] - boundary_locs[i]
                    if duration > 30:
                        patterns.append({
                            'type': 'intermediate_structure',
                            'start': boundary_locs[i],
                            'end': boundary_locs[i+1],
                            'duration': duration
                        })
        
        return patterns
    
    def _detect_critical_events(self, anomaly_scores: Dict) -> List:
        """臨界イベントの検出"""
        events = []
        
        if 'combined' in anomaly_scores:
            scores = anomaly_scores['combined']
            threshold = np.mean(scores) + 2 * np.std(scores)
            
            # 閾値を超えるフレームを検出
            critical_frames = np.where(scores > threshold)[0]
            
            # 連続したフレームをイベントとしてグループ化
            if len(critical_frames) > 0:
                current_event_start = critical_frames[0]
                current_event_end = critical_frames[0]
                
                for frame in critical_frames[1:]:
                    if frame == current_event_end + 1:
                        current_event_end = frame
                    else:
                        # イベントを記録
                        events.append((current_event_start, current_event_end))
                        current_event_start = frame
                        current_event_end = frame
                
                # 最後のイベントを記録
                events.append((current_event_start, current_event_end))
        
        return events
    
    def _to_cpu_dict(self, data_dict: Dict) -> Dict:
        """辞書内のGPU配列をCPUに転送"""
        cpu_dict = {}
        for key, value in data_dict.items():
            if hasattr(value, 'get'):  # CuPy配列の場合
                cpu_dict[key] = self.to_cpu(value)
            elif isinstance(value, dict):
                cpu_dict[key] = self._to_cpu_dict(value)
            else:
                cpu_dict[key] = value
        return cpu_dict
    
    def _get_gpu_info(self) -> Dict:
        """GPU情報を安全に取得"""
        gpu_info = {
            'device_name': str(self.device),
            'computation_mode': 'single_batch'
        }
        
        try:
            mem_info = self.memory_manager.get_memory_info()
            gpu_info['memory_used'] = mem_info.used / 1e9
        except:
            gpu_info['memory_used'] = 0.0
        
        return gpu_info
    
    def _create_empty_result(self, n_frames: int, n_atoms: int) -> MDLambda3Result:
        """空の結果を作成"""
        return MDLambda3Result(
            lambda_structures={},
            structural_boundaries={},
            topological_breaks={},
            md_features={},
            anomaly_scores={},
            detected_structures=[],
            critical_events=[],
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=100,
            computation_time=0.0,
            gpu_info={'computation_mode': 'batched', 'n_batches': 0}
        )
    
    def _print_initialization_info(self):
        """初期化情報の表示"""
        if self.verbose:
            print(f"\n🚀 Lambda³ GPU Detector Initialized")
            print(f"   Device: {self.device}")
            print(f"   GPU Mode: {self.is_gpu}")
            print(f"   Memory Limit: {self.memory_manager.max_memory / 1e9:.2f} GB")
            
            try:
                mem_info = self.memory_manager.get_memory_info()
                print(f"   Available: {mem_info.free / 1e9:.2f} GB")
            except:
                pass
            
            print(f"   Batch Size: {self.config.gpu_batch_size} frames")
            print(f"   Extended Detection: {'ON' if self.config.use_extended_detection else 'OFF'}")
            print(f"   Phase Space Analysis: {'ON' if self.config.use_phase_space else 'OFF'}")
    
    def _print_summary(self, result: MDLambda3Result):
        """結果サマリーの表示"""
        print("\n" + "="*60)
        print("=== Analysis Complete ===")
        print("="*60)
        print(f"Total frames: {result.n_frames}")
        print(f"Computation time: {result.computation_time:.2f} seconds")
        
        if result.computation_time > 0:
            print(f"Speed: {result.n_frames / result.computation_time:.1f} frames/second")
        
        if result.gpu_info:
            print(f"\nGPU Performance:")
            print(f"  Memory used: {result.gpu_info.get('memory_used', 0):.2f} GB")
            print(f"  Computation mode: {result.gpu_info.get('computation_mode', 'unknown')}")
        
        print(f"\nDetected features:")
        if isinstance(result.structural_boundaries, dict):
            n_boundaries = len(result.structural_boundaries.get('boundary_locations', []))
        else:
            n_boundaries = 0
        print(f"  Structural boundaries: {n_boundaries}")
        print(f"  Detected patterns: {len(result.detected_structures)}")
        print(f"  Critical events: {len(result.critical_events)}")
        
        if 'combined' in result.anomaly_scores:
            scores = result.anomaly_scores['combined']
            print(f"\nAnomaly statistics:")
            print(f"  Mean score: {np.mean(scores):.3f}")
            print(f"  Max score: {np.max(scores):.3f}")
            print(f"  Frames > 2σ: {np.sum(scores > 2.0)}")
    
    # === 追加のユーティリティメソッド ===
    
    def enable_mixed_precision(self):
        """混合精度モードを有効化"""
        self.config.mixed_precision = True
        print("✓ Mixed precision mode enabled")
    
    def benchmark_mode(self, enable: bool = True):
        """ベンチマークモードの切り替え"""
        self.config.benchmark_mode = enable
        if enable:
            print("⏱️ Benchmark mode enabled - timing all operations")
    
    def set_batch_size(self, batch_size: int):
        """バッチサイズの設定"""
        self.config.gpu_batch_size = batch_size
        print(f"✓ Batch size set to {batch_size} frames")
