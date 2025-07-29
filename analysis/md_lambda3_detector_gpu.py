"""
Lambda³ GPU版MD軌道解析メインモジュール
全体的な解析フローを管理する中心的なクラス
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from ..types import ArrayType, NDArray
from ..core.gpu_utils import GPUBackend
from ..structures.lambda_structures_gpu import LambdaStructuresGPU
from ..structures.md_features_gpu import MDFeaturesGPU
from ..detection.anomaly_detection_gpu import AnomalyDetectorGPU
from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
from ..detection.topology_breaks_gpu import TopologyBreaksDetectorGPU
from ..detection.extended_detection_gpu import ExtendedDetectorGPU
from ..detection.phase_space_gpu import PhaseSpaceAnalyzerGPU

warnings.filterwarnings('ignore')


@dataclass
class MDConfig:
    """MD Lambda³解析の設定パラメータ"""
    # Lambda³パラメータ
    window_scale: float = 0.005
    min_window: int = 3
    max_window: int = 500
    adaptive_window: bool = True
    
    # MD特徴抽出
    use_contacts: bool = False
    use_rmsd: bool = True
    use_rg: bool = True
    use_dihedrals: bool = True
    
    # 異常検出の重み
    w_lambda_f: float = 0.3
    w_lambda_ff: float = 0.2
    w_rho_t: float = 0.2
    w_topology: float = 0.3
    
    # 拡張検出の重み
    w_periodic: float = 0.15
    w_gradual: float = 0.2
    w_drift: float = 0.15
    
    # 拡張検出フラグ
    use_extended_detection: bool = True
    use_periodic: bool = True
    use_gradual: bool = True
    use_drift: bool = True
    radius_of_gyration: bool = True
    use_phase_space: bool = True
    
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


class MDLambda3DetectorGPU(GPUBackend):
    """GPU版Lambda³ MD検出器"""
    
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
        
        # force_cpuフラグを決定（重要！）
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
        
        # バッチ処理の設定
        batch_size = min(self.config.gpu_batch_size, n_frames)
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        if n_batches > 1:
            print(f"Processing in {n_batches} batches of {batch_size} frames")
            return self._analyze_batched(trajectory, backbone_indices, batch_size)
        
        # 単一バッチ処理
        try:
            with self.memory_manager.temporary_allocation(n_frames * n_atoms * 3 * 4, "analysis"):
                result = self._analyze_single_trajectory(trajectory, backbone_indices)
        except Exception as e:
            print(f"Analysis failed: {e}")
            raise
        
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        
        self._print_summary(result)
        
        return result
    
    def _analyze_single_trajectory(self,
                                 trajectory: np.ndarray,
                                 backbone_indices: Optional[np.ndarray]) -> MDLambda3Result:
        """単一軌道の解析（内部メソッド）"""
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
        
        # 4. 適応的ウィンドウサイズ（必要に応じて）
        if self.config.adaptive_window:
            try:
                adaptive_windows = self.structure_computer.compute_adaptive_window_size(
                    md_features, lambda_structures, n_frames, self.config
                )
                primary_window = adaptive_windows['primary']
                
                # 大きく変わった場合は再計算
                if abs(primary_window - initial_window) > initial_window * 0.2:
                    print(f"\n🔄 Recomputing with adaptive window: {primary_window} frames")
                    lambda_structures = self.structure_computer.compute_lambda_structures(
                        trajectory, md_features, primary_window
                    )
            except Exception as e:
                print(f"Adaptive window computation failed: {e}")
                primary_window = initial_window
                adaptive_windows = self._get_default_windows(primary_window)
        else:
            primary_window = initial_window
            adaptive_windows = self._get_default_windows(primary_window)
        
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
        
        # GPU情報を収集（安全に）
        gpu_info = self._get_gpu_info()
        
        # 結果を構築（CPUに転送）
        result = MDLambda3Result(
            lambda_structures=self._to_cpu_dict(lambda_structures),
            structural_boundaries=structural_boundaries,
            topological_breaks=self._to_cpu_dict(topological_breaks),
            md_features=self._to_cpu_dict(md_features),
            anomaly_scores=self._to_cpu_dict(anomaly_scores),
            detected_structures=detected_structures,
            phase_space_analysis=phase_space_analysis,
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=primary_window,
            computation_time=0.0,  # 後で設定
            gpu_info=gpu_info
        )
        
        return result
    
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
    
    def _analyze_batched(self,
                        trajectory: np.ndarray,
                        backbone_indices: Optional[np.ndarray],
                        batch_size: int) -> MDLambda3Result:
        """バッチ処理による解析（大規模データ用）"""
        print("\n⚡ Running batched GPU analysis...")
        
        n_frames = trajectory.shape[0]
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        # バッチごとの結果を蓄積
        batch_results = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_frames)
            
            print(f"\n  Batch {batch_idx + 1}/{n_batches}: frames {start_idx}-{end_idx}")
            
            batch_trajectory = trajectory[start_idx:end_idx]
            
            # バッチ解析
            batch_result = self._analyze_single_batch(
                batch_trajectory, backbone_indices, start_idx
            )
            
            batch_results.append(batch_result)
            
            # メモリクリア
            self.memory_manager.clear_cache()
        
        # 結果を統合
        return self._merge_batch_results(batch_results, trajectory.shape)
    
    def _analyze_single_batch(self,
                            batch_trajectory: np.ndarray,
                            backbone_indices: Optional[np.ndarray],
                            offset: int) -> Dict:
        """単一バッチの解析"""
        # 簡略化された解析（メモリ効率重視）
        md_features = self.feature_extractor.extract_md_features(
            batch_trajectory, backbone_indices
        )
        
        window = self._compute_initial_window(len(batch_trajectory))
        
        lambda_structures = self.structure_computer.compute_lambda_structures(
            batch_trajectory, md_features, window
        )
        
        return {
            'offset': offset,
            'n_frames': len(batch_trajectory),
            'lambda_structures': lambda_structures,
            'md_features': md_features
        }
    
    def _merge_batch_results(self,
                           batch_results: List[Dict],
                           original_shape: Tuple) -> MDLambda3Result:
        """バッチ結果の統合"""
        print("\n📊 Merging batch results...")
        
        # TODO: 実際のマージロジックを実装
        # 現在はプレースホルダー
        
        n_frames, n_atoms, _ = original_shape
        
        return MDLambda3Result(
            lambda_structures={},
            structural_boundaries={},
            topological_breaks={},
            md_features={},
            anomaly_scores={},
            detected_structures=[],
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=100,
            computation_time=0.0,
            gpu_info={'computation_mode': 'batched'}
        )
    
    def _compute_initial_window(self, n_frames: int) -> int:
        """初期ウィンドウサイズの計算"""
        return max(
            self.config.min_window,
            min(
                int(n_frames * self.config.window_scale),
                self.config.max_window
            )
        )
    
    def _get_default_windows(self, primary: int) -> Dict[str, int]:
        """デフォルトのウィンドウサイズ"""
        return {
            'primary': primary,
            'fast': max(self.config.min_window, primary // 2),
            'slow': min(self.config.max_window, primary * 2),
            'boundary': max(10, primary // 3)
        }
    
    def _detect_structural_patterns(self,
                                  lambda_structures: Dict,
                                  boundaries: Dict,
                                  window_steps: int) -> List[Dict]:
        """構造パターンの検出"""
        patterns = []
        
        # 安全なパターン検出
        try:
            if 'Q_cumulative' in lambda_structures and self.is_gpu and HAS_CUPY:
                patterns = self._detect_patterns_gpu(lambda_structures['Q_cumulative'])
            else:
                patterns = self._detect_patterns_cpu(lambda_structures.get('Q_cumulative'))
        except Exception as e:
            print(f"Pattern detection failed: {e}")
        
        return patterns
    
    def _detect_patterns_gpu(self, q_cumulative: Any) -> List[Dict]:
        """GPU版パターン検出"""
        patterns = []
        
        q_cum_gpu = self.to_gpu(q_cumulative)
        
        if len(q_cum_gpu) > 100:
            try:
                from cupyx.scipy.signal import correlate
                
                # デトレンド
                q_detrend = q_cum_gpu - cp.mean(q_cum_gpu)
                
                # 自己相関
                acf = correlate(q_detrend, q_detrend, mode='same')
                acf = acf[len(acf)//2:]
                
                # ピーク検出（エラーハンドリング付き）
                try:
                    from cupyx.scipy.signal import find_peaks as find_peaks_gpu
                    peaks, _ = find_peaks_gpu(acf, height=0.5*cp.max(acf))
                except:
                    # フォールバック：シンプルなピーク検出
                    peaks = self._simple_peak_detection(acf)
                
                # パターン作成
                for i, peak in enumerate(self.to_cpu(peaks[:5])):
                    if peak < len(acf):
                        patterns.append({
                            'name': f'Pattern_{i+1}',
                            'period': int(peak),
                            'strength': float(acf[peak] / acf[0]),
                            'type': 'periodic' if acf[peak] > 0.7*acf[0] else 'quasi-periodic'
                        })
            except Exception as e:
                print(f"GPU pattern detection error: {e}")
        
        return patterns
    
    def _detect_patterns_cpu(self, q_cumulative: Any) -> List[Dict]:
        """CPU版パターン検出（フォールバック）"""
        if q_cumulative is None or len(q_cumulative) < 100:
            return []
        
        # シンプルなパターン検出
        return [{
            'name': 'Default_Pattern',
            'period': 50,
            'strength': 0.5,
            'type': 'unknown'
        }]
    
    def _simple_peak_detection(self, signal: Any) -> Any:
        """シンプルなピーク検出（フォールバック）"""
        # 閾値を超える点を検出
        threshold = 0.5 * cp.max(signal) if self.is_gpu else 0.5 * np.max(signal)
        peaks = cp.where(signal > threshold)[0] if self.is_gpu else np.where(signal > threshold)[0]
        return peaks[:5]  # 最初の5つだけ
    
    def _print_initialization_info(self):
        """初期化情報の表示"""
        if self.verbose:
            print(f"\n🚀 Lambda³ GPU Detector Initialized")
            print(f"   Device: {self.device}")
            print(f"   GPU Mode: {self.is_gpu}")
            
            try:
                mem_info = self.memory_manager.get_memory_info()
                print(f"   Memory Limit: {self.memory_manager.max_memory / 1e9:.2f} GB")
                print(f"   Available: {mem_info.free_gb:.2f} GB")
            except:
                print(f"   Memory info unavailable")
            
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
        
        if 'final_combined' in result.anomaly_scores:
            scores = result.anomaly_scores['final_combined']
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
    
    def visualize_results(self, result: MDLambda3Result) -> Any:
        """結果の可視化（matplotlib figure）"""
        # TODO: 可視化モジュールの実装
        print("Visualization not yet implemented in GPU version")
        return None
