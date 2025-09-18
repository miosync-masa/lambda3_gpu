#!/usr/bin/env python3
"""
Material Lambda³ Detector GPU - Refactored Pipeline Edition
==========================================================

材料解析用Lambda³検出器のGPU実装（リファクタリング版）

主な改善点：
- 前処理（バッチごと）と後処理（マージ後）の明確な分離
- MaterialAnalyticsGPUの欠陥解析を前処理に移動
- メモリ効率的なバッチ処理

Version: 2.1.0
Author: 環ちゃん
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import logging

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

# Lambda³ GPU imports
from ..core.gpu_utils import GPUBackend
from ..structures.lambda_structures_gpu import LambdaStructuresGPU
from ..structures.md_features import MDFeaturesGPU  # 基本MD特徴抽出
from ..material.material_md_features_gpu import MaterialMDFeaturesGPU  # 材料特徴抽出
from ..detection.anomaly_detection_gpu import AnomalyDetectorGPU
from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
from ..detection.topology_breaks_gpu import TopologyBreaksDetectorGPU
from ..detection.extended_detection_gpu import ExtendedDetectorGPU
from ..detection.phase_space_gpu import PhaseSpaceAnalyzerGPU

# Material specific imports
from ..material.material_analytics_gpu import MaterialAnalyticsGPU

# Logger設定
logger = logging.getLogger(__name__)

# ===============================
# Configuration
# ===============================

@dataclass
class MaterialConfig:
    """材料解析設定"""
    # Lambda³パラメータ
    adaptive_window: bool = True
    extended_detection: bool = True
    use_extended_detection: bool = True
    use_phase_space: bool = True
    sensitivity: float = 1.5
    min_boundary_gap: int = 50
    
    # 材料特有の設定
    material_type: str = 'SUJ2'
    use_material_analytics: bool = True
    
    # GPU設定
    gpu_batch_size: int = 10000
    mixed_precision: bool = False
    benchmark_mode: bool = False
    
    # 異常検出の重み
    w_lambda_f: float = 0.3
    w_lambda_ff: float = 0.2
    w_rho_t: float = 0.2
    w_topology: float = 0.3
    
    # 材料特有の重み
    w_defect: float = 0.2
    w_coherence: float = 0.1
    
    # 適応的ウィンドウのスケール設定
    window_scale: float = 0.005

@dataclass
class MaterialLambda3Result:
    """材料Lambda³解析結果"""
    # Core Lambda³構造
    lambda_structures: Dict[str, np.ndarray]
    structural_boundaries: Dict[str, Any]
    topological_breaks: Dict[str, np.ndarray]
    
    # MD/材料特徴
    md_features: Dict[str, np.ndarray]
    
    # 解析結果
    anomaly_scores: Dict[str, np.ndarray]
    detected_structures: List[Dict]
    
    # オプション
    material_features: Optional[Dict[str, np.ndarray]] = None
    phase_space_analysis: Optional[Dict] = None
    defect_analysis: Optional[Dict] = None
    structural_coherence: Optional[np.ndarray] = None
    failure_prediction: Optional[Dict] = None
    material_events: Optional[List[Tuple[int, int, str]]] = None
    
    # メタデータ
    n_frames: int = 0
    n_atoms: int = 0
    window_steps: int = 0
    computation_time: float = 0.0
    gpu_info: Dict = field(default_factory=dict)
    critical_events: List = field(default_factory=list)

# ===============================
# Main Detector Class
# ===============================

class MaterialLambda3DetectorGPU(GPUBackend):
    """
    GPU版Lambda³材料検出器（リファクタリング版）
    
    処理フロー：
    1. 前処理（バッチごと）: MD特徴抽出 → 欠陥解析 → Lambda構造計算
    2. 後処理（マージ後）: 境界検出 → 異常検出 → 材料解析（MD特徴ベース）
    """
    
    def __init__(self, config: MaterialConfig = None, device: str = 'auto'):
        # GPUBackendの初期化
        if device == 'cpu':
            super().__init__(device='cpu', force_cpu=True)
        else:
            super().__init__(device=device, force_cpu=False)
        
        self.config = config or MaterialConfig()
        self.verbose = True
        
        force_cpu_flag = not self.is_gpu
        
        # コンポーネント構成
        self.structure_computer = LambdaStructuresGPU(force_cpu_flag)
        
        # === 重要: 2つの独立した特徴抽出器 ===
        # Lambda構造計算用（全原子）
        self.feature_extractor = MDFeaturesGPU(force_cpu_flag)
        # 材料解析用（欠陥領域）
        self.material_feature_extractor = MaterialMDFeaturesGPU(force_cpu_flag)
        
        self.anomaly_detector = AnomalyDetectorGPU(force_cpu_flag)
        self.boundary_detector = BoundaryDetectorGPU(force_cpu_flag)
        self.topology_detector = TopologyBreaksDetectorGPU(force_cpu_flag)
        self.extended_detector = ExtendedDetectorGPU(force_cpu_flag)
        self.phase_space_analyzer = PhaseSpaceAnalyzerGPU(force_cpu_flag)
        
        # 材料特有のコンポーネント
        if self.config.use_material_analytics:
            self.material_analytics = MaterialAnalyticsGPU(
                material_type=self.config.material_type,
                force_cpu=force_cpu_flag
            )
        else:
            self.material_analytics = None
        
        # メモリマネージャとデバイスを共有
        self._share_resources()
        self._print_initialization_info()
    
    def _share_resources(self):
        """メモリマネージャとデバイスをコンポーネント間で共有"""
        components = [
            self.structure_computer, 
            self.feature_extractor,  # 基本MD特徴抽出器
            self.material_feature_extractor,  # 材料特徴抽出器
            self.anomaly_detector, 
            self.boundary_detector,
            self.topology_detector, 
            self.extended_detector,
            self.phase_space_analyzer
        ]
        
        if self.material_analytics:
            components.append(self.material_analytics)
        
        for component in components:
            if hasattr(component, 'memory_manager'):
                component.memory_manager = self.memory_manager
            if hasattr(component, 'device'):
                component.device = self.device

    def analyze(self,
                trajectory: np.ndarray,
                backbone_indices: Optional[np.ndarray] = None,
                atom_types: Optional[np.ndarray] = None,
                cluster_definition_path: Optional[str] = None,
                cluster_atoms: Optional[Dict[int, List[int]]] = None,
                **kwargs) -> MaterialLambda3Result:
        """
        材料軌道のLambda³解析
        """
        start_time = time.time()
        
        # 前処理
        trajectory, atom_types = self._preprocess_inputs(
            trajectory, atom_types, backbone_indices
        )
        
        n_frames, n_atoms, _ = trajectory.shape
        self._print_analysis_header(n_frames, n_atoms)
        
        # バッチ処理の判定
        batch_size = min(self.config.gpu_batch_size, n_frames)
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        if n_batches > 1:
            print(f"Processing in {n_batches} batches of {batch_size} frames")
            result = self._analyze_batched(
                trajectory, backbone_indices, atom_types, 
                batch_size, cluster_definition_path, cluster_atoms
            )
        else:
            result = self._analyze_single_trajectory(
                trajectory, backbone_indices, atom_types,
                cluster_definition_path, cluster_atoms
            )
        
        result.computation_time = time.time() - start_time
        self._print_summary(result)
        
        return result
    
    def _preprocess_inputs(self, trajectory, atom_types, backbone_indices):
        """入力データの前処理とGPU転送"""
        # 原子タイプの文字列→数値変換
        if atom_types is not None and atom_types.dtype.kind == 'U':
            atom_type_names = np.unique(atom_types)
            type_map = {t: i for i, t in enumerate(atom_type_names)}
            atom_types = np.array([type_map[t] for t in atom_types], dtype=np.int32)
        
        # GPU転送
        if self.is_gpu and cp is not None:
            print("📊 Converting arrays to GPU...")
            trajectory = cp.asarray(trajectory)
            if backbone_indices is not None:
                backbone_indices = cp.asarray(backbone_indices)
            if atom_types is not None:
                atom_types = cp.asarray(atom_types)
        
        return trajectory, atom_types
    
    def _analyze_single_trajectory(self,
                                  trajectory: np.ndarray,
                                  backbone_indices: Optional[np.ndarray],
                                  atom_types: Optional[np.ndarray],
                                  cluster_definition_path: Optional[str] = None,
                                  cluster_atoms: Optional[Dict[int, List[int]]] = None
                                  ) -> MaterialLambda3Result:
        """単一軌道の完全解析"""
        n_frames, n_atoms, _ = trajectory.shape
        
        # 欠陥領域の特定
        backbone_indices = self._identify_defect_regions(
            cluster_atoms, backbone_indices
        )
        
        # === 前処理 ===
        print("\n[PREPROCESSING]")
        
        # 1. 基本MD特徴抽出（全原子 - Lambda構造用）
        print("  1. Extracting MD features (full atoms)...")
        md_features = self.feature_extractor.extract_md_features(
            trajectory, 
            None,  # 全原子で計算
            cluster_definition_path=cluster_definition_path,
            atom_types=atom_types
        )
        
        # 2. 材料特有の特徴抽出（欠陥領域 - 材料解析用）
        material_features = None
        if self.config.use_material_analytics and backbone_indices is not None:
            print("  2. Extracting material features (defect region)...")
            material_features = self.material_feature_extractor.extract_md_features(
                trajectory, 
                backbone_indices,  # 欠陥領域のみ
                cluster_definition_path=cluster_definition_path,
                atom_types=atom_types
            )
            
            # 欠陥解析結果をMD特徴に追加（参照用）
            print("  3. Computing crystal defect charges...")
            self._add_defect_analysis_to_features(
                trajectory, md_features, cluster_atoms, n_atoms
            )
        
        # 4. Lambda構造計算（基本MD特徴を使用）
        initial_window = self._compute_initial_window(n_frames)
        print("  4. Computing Lambda³ structures...")
        lambda_structures = self.structure_computer.compute_lambda_structures(
            trajectory, md_features, initial_window  # md_features（全原子）を使用
        )
        
        # === 後処理 ===
        print("\n[POSTPROCESSING]")
        
        # 解析結果を統合
        result = self._perform_postprocessing(
            lambda_structures, md_features, material_features,
            n_frames, n_atoms, initial_window
        )
        
        return result
    
    def _analyze_batched(self,
                        trajectory: np.ndarray,
                        backbone_indices: Optional[np.ndarray],
                        atom_types: Optional[np.ndarray],
                        batch_size: int,
                        cluster_definition_path: Optional[str] = None,
                        cluster_atoms: Optional[Dict[int, List[int]]] = None
                        ) -> MaterialLambda3Result:
        """バッチ処理による解析"""
        print("\n⚡ Running batched GPU analysis...")
        
        n_frames = trajectory.shape[0]
        batches = self._optimize_batch_plan(n_frames, batch_size)
        
        # Step 1: 前処理（バッチごと）
        batch_results = self._process_batches(
            trajectory, backbone_indices, atom_types,
            batches, cluster_definition_path
        )
        
        if not batch_results:
            return self._create_empty_result(n_frames, trajectory.shape[1])
        
        # Step 2: 結果をマージ
        print("\n[MERGING]")
        merged_result = self._merge_batch_results(
            batch_results, trajectory.shape, atom_types
        )
        
        # Step 3: 後処理
        print("\n[POSTPROCESSING]")
        final_result = self._complete_analysis(merged_result, atom_types)
        
        return final_result
    
    def _analyze_single_batch(self,
                             batch_trajectory: np.ndarray,
                             backbone_indices: Optional[np.ndarray],
                             atom_types: Optional[np.ndarray],
                             offset: int,
                             cluster_definition_path: Optional[str] = None
                             ) -> Dict:
        """
        単一バッチの前処理
        
        実行内容：
        1. 基本MD特徴抽出（MDFeaturesGPU - 全原子）
        2. 材料特徴抽出（MaterialMDFeaturesGPU - 欠陥領域）
        3. 欠陥解析（MaterialAnalyticsGPU - トラジェクトリ必要）
        4. Lambda構造計算
        """
        n_atoms = batch_trajectory.shape[1]
        
        # 1. 基本MD特徴抽出（全原子 - Lambda構造用）
        md_features = self.feature_extractor.extract_md_features(
            batch_trajectory, 
            None,  # 全原子で計算
            cluster_definition_path=cluster_definition_path,
            atom_types=atom_types
        )
        
        # 2. 材料特有の特徴抽出（欠陥領域 - 材料解析用）
        material_features = None
        if self.config.use_material_analytics and backbone_indices is not None:
            material_features = self.material_feature_extractor.extract_md_features(
                batch_trajectory, 
                backbone_indices,  # 欠陥領域のみ
                cluster_definition_path=cluster_definition_path,
                atom_types=atom_types
            )
            
            # 3. 欠陥解析（前処理で実行）
            self._add_defect_analysis_to_features(
                batch_trajectory, md_features, None, n_atoms
            )
        
        # 4. Lambda構造計算（基本MD特徴を使用）
        window = self._compute_initial_window(len(batch_trajectory))
        lambda_structures = self.structure_computer.compute_lambda_structures(
            batch_trajectory, md_features, window  # md_features（全原子）を使用
        )
        
        return {
            'offset': offset,
            'n_frames': len(batch_trajectory),
            'lambda_structures': lambda_structures,
            'md_features': md_features,
            'material_features': material_features,  # 別途保持
            'window': window
        }
    
    def _complete_analysis(self,
                          merged_result: MaterialLambda3Result,
                          atom_types: Optional[np.ndarray]
                          ) -> MaterialLambda3Result:
        """
        マージ後の後処理
        
        実行内容：
        1. 境界検出・トポロジー解析
        2. 異常スコア計算
        3. 材料特有の解析（MD特徴ベース）
        """
        lambda_structures = merged_result.lambda_structures
        md_features = merged_result.md_features
        n_frames = merged_result.n_frames
        n_atoms = merged_result.n_atoms
        
        # ウィンドウサイズ決定
        initial_window = merged_result.window_steps
        adaptive_windows = self._determine_adaptive_windows(
            lambda_structures, initial_window
        )
        primary_window = adaptive_windows.get('primary', initial_window)
        
        # 境界・トポロジー検出
        print("  - Detecting boundaries and topology...")
        structural_boundaries = self.boundary_detector.detect_structural_boundaries(
            lambda_structures, adaptive_windows.get('boundary', primary_window // 3)
        )
        topological_breaks = self.topology_detector.detect_topological_breaks(
            lambda_structures, adaptive_windows.get('fast', primary_window // 2)
        )
        
        # 異常スコア計算
        print("  - Computing anomaly scores...")
        anomaly_scores = self.anomaly_detector.compute_multiscale_anomalies(
            lambda_structures, structural_boundaries,
            topological_breaks, md_features, self.config
        )
        
        # 材料特有の解析（MD特徴ベース）
        defect_analysis = None
        structural_coherence = None
        failure_prediction = None
        material_events = None
        
        if self.config.use_material_analytics and self.material_analytics:
            print("  - Material-specific analysis...")
            
            # 欠陥情報の取得
            if 'defect_charge' in md_features:
                defect_analysis = {
                    'defect_charge': md_features.get('defect_charge'),
                    'cumulative_charge': md_features.get('cumulative_charge')
                }
            
            # 構造一貫性
            if 'coordination' in md_features:
                structural_coherence = self.material_analytics.compute_structural_coherence(
                    md_features['coordination'],
                    md_features.get('strain', np.zeros((n_frames, 1))),
                    window=primary_window // 2
                )
            
            # 破壊予測
            if 'strain' in md_features:
                failure_prediction = self.material_analytics.predict_failure(
                    md_features['strain'],
                    damage_history=md_features.get('damage'),
                    defect_charge=md_features.get('cumulative_charge')
                )
                failure_prediction = {
                    'failure_probability': failure_prediction.failure_probability,
                    'reliability_index': failure_prediction.reliability_index,
                    'failure_mode': failure_prediction.failure_mode
                }
            
            # イベント分類
            critical_events = self._detect_critical_events(anomaly_scores)
            if critical_events and structural_coherence is not None:
                material_events = self.material_analytics.classify_material_events(
                    critical_events, anomaly_scores,
                    structural_coherence,
                    defect_charge=md_features.get('defect_charge')
                )
        
        # 構造パターン検出
        print("  - Detecting structural patterns...")
        detected_structures = self._detect_structural_patterns(
            lambda_structures, structural_boundaries,
            adaptive_windows.get('slow', primary_window * 2)
        )
        
        # 位相空間解析
        phase_space_analysis = None
        if self.config.use_phase_space:
            try:
                phase_space_analysis = self.phase_space_analyzer.analyze_phase_space(
                    lambda_structures
                )
            except Exception as e:
                logger.warning(f"Phase space analysis failed: {e}")
        
        print("  ✅ Analysis completed!")
        
        # 結果を更新
        merged_result.structural_boundaries = structural_boundaries
        merged_result.topological_breaks = topological_breaks
        merged_result.anomaly_scores = anomaly_scores
        merged_result.detected_structures = detected_structures
        merged_result.phase_space_analysis = phase_space_analysis
        merged_result.defect_analysis = defect_analysis
        merged_result.structural_coherence = structural_coherence
        merged_result.failure_prediction = failure_prediction
        merged_result.material_events = material_events
        merged_result.critical_events = self._detect_critical_events(anomaly_scores)
        merged_result.window_steps = primary_window
        
        return merged_result
    
    # === ヘルパーメソッド ===
    
    def _identify_defect_regions(self,
                                cluster_atoms: Optional[Dict[int, List[int]]],
                                backbone_indices: Optional[np.ndarray]
                                ) -> Optional[np.ndarray]:
        """欠陥領域の特定"""
        if cluster_atoms is not None:
            defect_indices = []
            for cid, atoms in cluster_atoms.items():
                if str(cid) != "0":  # Cluster 0（健全領域）以外
                    defect_indices.extend(atoms)
            if defect_indices:
                backbone_indices = np.array(sorted(defect_indices))
                print(f"  Using {len(backbone_indices)} defect atoms from clusters")
        return backbone_indices
    
    def _add_defect_analysis_to_features(self,
                                        trajectory: np.ndarray,
                                        md_features: Dict,
                                        cluster_atoms: Optional[Dict],
                                        n_atoms: int):
        """MD特徴に欠陥解析結果を追加"""
        try:
            trajectory_cpu = self.to_cpu(trajectory) if self.is_gpu else trajectory
            clusters = cluster_atoms if cluster_atoms else {0: list(range(n_atoms))}
            
            defect_result = self.material_analytics.compute_crystal_defect_charge(
                trajectory_cpu, clusters, cutoff=3.5
            )
            
            md_features['defect_charge'] = defect_result.defect_charge
            md_features['cumulative_charge'] = defect_result.cumulative_charge
        except Exception as e:
            logger.warning(f"Defect analysis failed: {e}")
    
    def _optimize_batch_plan(self, n_frames: int, batch_size: int) -> List[Tuple[int, int]]:
        """バッチ計画の最適化"""
        MIN_BATCH_SIZE = 1000
        batches = []
        current_pos = 0
        
        while current_pos < n_frames:
            batch_end = min(current_pos + batch_size, n_frames)
            remaining = batch_end - current_pos
            
            # 最後の小さいバッチを前のバッチと結合
            if batch_end == n_frames and remaining < MIN_BATCH_SIZE and batches:
                print(f"  Merging last batch ({remaining} frames) with previous")
                prev_start, _ = batches[-1]
                batches[-1] = (prev_start, n_frames)
                break
            
            batches.append((current_pos, batch_end))
            current_pos = batch_end
        
        return batches
    
    def _process_batches(self,
                        trajectory: np.ndarray,
                        backbone_indices: Optional[np.ndarray],
                        atom_types: Optional[np.ndarray],
                        batches: List[Tuple[int, int]],
                        cluster_definition_path: Optional[str]
                        ) -> List[Dict]:
        """バッチ処理の実行"""
        batch_results = []
        
        print("\n[PREPROCESSING - Batch Processing]")
        for batch_idx, (start_idx, end_idx) in enumerate(batches):
            frames_count = end_idx - start_idx
            print(f"  Batch {batch_idx + 1}/{len(batches)}: "
                  f"frames {start_idx}-{end_idx} ({frames_count} frames)")
            
            try:
                batch_result = self._analyze_single_batch(
                    trajectory[start_idx:end_idx],
                    backbone_indices, atom_types,
                    start_idx, cluster_definition_path
                )
                batch_results.append(batch_result)
                
            except Exception as e:
                print(f"    ⚠️ Batch failed: {e}")
                continue
            
            finally:
                self.memory_manager.clear_cache()
        
        return batch_results
    
    def _merge_batch_results(self,
                           batch_results: List[Dict],
                           original_shape: Tuple,
                           atom_types: Optional[np.ndarray]
                           ) -> MaterialLambda3Result:
        """バッチ結果の統合"""
        print("  Merging batch results...")
        n_frames, n_atoms, _ = original_shape
        
        if not batch_results:
            return self._create_empty_result(n_frames, n_atoms)
        
        # 結果配列の初期化
        merged_lambda = {}
        merged_features = {}
        
        # 最初のバッチから構造を取得
        first_batch = batch_results[0]
        
        # Lambda構造の初期化
        for key in first_batch.get('lambda_structures', {}).keys():
            sample = first_batch['lambda_structures'][key]
            if isinstance(sample, (np.ndarray, self.xp.ndarray)):
                shape = (n_frames,) + sample.shape[1:] if sample.ndim > 1 else (n_frames,)
                merged_lambda[key] = np.full(shape, np.nan, dtype=sample.dtype)
        
        # MD特徴の初期化
        for key in first_batch.get('md_features', {}).keys():
            sample = first_batch['md_features'][key]
            if isinstance(sample, (np.ndarray, self.xp.ndarray)):
                shape = (n_frames,) + sample.shape[1:] if sample.ndim > 1 else (n_frames,)
                merged_features[key] = np.full(shape, np.nan, dtype=sample.dtype)
        
        # データのマージ
        for batch in batch_results:
            offset = batch['offset']
            n_batch_frames = batch['n_frames']
            end_idx = min(offset + n_batch_frames, n_frames)
            
            # Lambda構造
            for key, value in batch.get('lambda_structures', {}).items():
                if key in merged_lambda:
                    value = self.to_cpu(value) if hasattr(value, 'get') else value
                    actual_frames = min(len(value), end_idx - offset)
                    merged_lambda[key][offset:offset + actual_frames] = value[:actual_frames]
            
            # MD特徴
            for key, value in batch.get('md_features', {}).items():
                if key in merged_features:
                    value = self.to_cpu(value) if hasattr(value, 'get') else value
                    actual_frames = min(len(value), end_idx - offset)
                    merged_features[key][offset:offset + actual_frames] = value[:actual_frames]
        
        # ウィンドウサイズの平均
        window_steps = int(np.mean([b.get('window', 100) for b in batch_results]))
        
        print(f"  ✅ Merged {n_frames} frames successfully")
        
        return MaterialLambda3Result(
            lambda_structures=merged_lambda,
            md_features=merged_features,
            material_features=merged_features,
            structural_boundaries={},
            topological_breaks={},
            anomaly_scores={},
            detected_structures=[],
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=window_steps,
            gpu_info={'computation_mode': 'batched', 'n_batches': len(batch_results)}
        )
    
    def _perform_postprocessing(self,
                              lambda_structures: Dict,
                              md_features: Dict,
                              material_features: Optional[Dict],
                              n_frames: int,
                              n_atoms: int,
                              initial_window: int
                              ) -> MaterialLambda3Result:
        """単一軌道用の後処理"""
        # MaterialLambda3Resultオブジェクトを作成
        partial_result = MaterialLambda3Result(
            lambda_structures=self._to_cpu_dict(lambda_structures),
            md_features=self._to_cpu_dict(md_features),
            material_features=self._to_cpu_dict(material_features) if material_features else md_features,
            structural_boundaries={},
            topological_breaks={},
            anomaly_scores={},
            detected_structures=[],
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=initial_window,
            gpu_info=self._get_gpu_info()
        )
        
        # 後処理を実行
        return self._complete_analysis(partial_result, None)
    
    def _compute_initial_window(self, n_frames: int) -> int:
        """初期ウィンドウサイズの計算"""
        return min(50, n_frames // 10)
    
    def _determine_adaptive_windows(self,
                                   lambda_structures: Dict,
                                   initial_window: int
                                   ) -> Dict[str, int]:
        """適応的ウィンドウサイズの決定"""
        if not self.config.adaptive_window:
            return {'primary': initial_window}
        
        return {
            'primary': initial_window,
            'fast': max(10, initial_window // 2),
            'slow': min(500, initial_window * 2),
            'boundary': max(20, initial_window // 3)
        }
    
    def _detect_structural_patterns(self,
                                   lambda_structures: Dict,
                                   boundaries: Dict,
                                   window: int
                                   ) -> List[Dict]:
        """構造パターンの検出"""
        patterns = []
        
        if isinstance(boundaries, dict) and 'boundary_locations' in boundaries:
            boundary_locs = boundaries['boundary_locations']
            
            if len(boundary_locs) > 0:
                # 最初のセグメント
                if boundary_locs[0] > 30:
                    patterns.append({
                        'type': 'initial_structure',
                        'start': 0,
                        'end': boundary_locs[0],
                        'duration': boundary_locs[0]
                    })
                
                # 中間セグメント
                for i in range(len(boundary_locs) - 1):
                    duration = boundary_locs[i+1] - boundary_locs[i]
                    if duration > 20:
                        pattern_type = 'elastic' if duration < 100 else 'plastic'
                        patterns.append({
                            'type': pattern_type,
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
            threshold = np.mean(scores) + 1.5 * np.std(scores)
            
            critical_frames = np.where(scores > threshold)[0]
            
            if len(critical_frames) > 0:
                current_start = critical_frames[0]
                current_end = critical_frames[0]
                
                for frame in critical_frames[1:]:
                    if frame == current_end + 1:
                        current_end = frame
                    else:
                        events.append((current_start, current_end))
                        current_start = frame
                        current_end = frame
                
                events.append((current_start, current_end))
        
        return events
    
    def _to_cpu_dict(self, data_dict: Optional[Dict]) -> Dict:
        """GPU配列をCPUに転送"""
        if data_dict is None:
            return {}
        
        cpu_dict = {}
        for key, value in data_dict.items():
            if hasattr(value, 'get'):  # CuPy配列
                cpu_dict[key] = self.to_cpu(value)
            elif isinstance(value, dict):
                cpu_dict[key] = self._to_cpu_dict(value)
            else:
                cpu_dict[key] = value
        return cpu_dict
    
    def _get_gpu_info(self) -> Dict:
        """GPU情報の取得"""
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
    
    def _create_empty_result(self, n_frames: int, n_atoms: int) -> MaterialLambda3Result:
        """空の結果を作成"""
        return MaterialLambda3Result(
            lambda_structures={},
            structural_boundaries={},
            topological_breaks={},
            md_features={},
            anomaly_scores={},
            detected_structures=[],
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=100,
            gpu_info={'computation_mode': 'batched', 'n_batches': 0}
        )
    
    def _print_analysis_header(self, n_frames: int, n_atoms: int):
        """解析ヘッダーの表示"""
        print(f"\n{'='*60}")
        print(f"=== Lambda³ Material Analysis (GPU) ===")
        print(f"{'='*60}")
        print(f"Trajectory: {n_frames} frames, {n_atoms} atoms")
        print(f"Material: {self.config.material_type}")
        print(f"GPU Device: {self.device}")
        
        try:
            mem_info = self.memory_manager.get_memory_info()
            print(f"Available GPU Memory: {mem_info.free / 1e9:.2f} GB")
        except:
            pass
    
    def _print_initialization_info(self):
        """初期化情報の表示"""
        if self.verbose:
            print(f"\n💎 Material Lambda³ GPU Detector Initialized")
            print(f"   Material: {self.config.material_type}")
            print(f"   Device: {self.device}")
            print(f"   GPU Mode: {self.is_gpu}")
            
            try:
                mem_info = self.memory_manager.get_memory_info()
                print(f"   Available Memory: {mem_info.free / 1e9:.2f} GB")
            except:
                pass
            
            print(f"   Batch Size: {self.config.gpu_batch_size} frames")
            print(f"   Material Analytics: {'ON' if self.config.use_material_analytics else 'OFF'}")
    
    def _print_summary(self, result: MaterialLambda3Result):
        """結果サマリーの表示"""
        print("\n" + "="*60)
        print("=== Analysis Complete ===")
        print("="*60)
        print(f"Total frames: {result.n_frames}")
        print(f"Computation time: {result.computation_time:.2f} seconds")
        
        if result.computation_time > 0:
            print(f"Speed: {result.n_frames / result.computation_time:.1f} frames/second")
        
        if result.failure_prediction:
            print(f"\nMaterial Analysis:")
            print(f"  Failure probability: {result.failure_prediction['failure_probability']:.1%}")
            print(f"  Reliability index: {result.failure_prediction['reliability_index']:.2f}")
            print(f"  Failure mode: {result.failure_prediction.get('failure_mode', 'unknown')}")
        
        if result.material_events:
            print(f"  Material events: {len(result.material_events)}")
            event_types = {}
            for event in result.material_events:
                if len(event) >= 3:
                    event_types[event[2]] = event_types.get(event[2], 0) + 1
            
            if event_types:
                print("  Event types:")
                for etype, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {etype}: {count}")

# ===============================
# Module Functions
# ===============================

def detect_material_events(trajectory: np.ndarray,
                          atom_types: np.ndarray,
                          config: Optional[MaterialConfig] = None,
                          **kwargs) -> List[Tuple[int, int, str]]:
    """材料イベントを検出する便利関数"""
    config = config or MaterialConfig()
    detector = MaterialLambda3DetectorGPU(config)
    
    result = detector.analyze(
        trajectory=trajectory,
        atom_types=atom_types,
        **kwargs
    )
    
    if hasattr(result, 'material_events') and result.material_events:
        return result.material_events
    
    events = []
    if hasattr(result, 'critical_events'):
        for event in result.critical_events:
            if isinstance(event, tuple) and len(event) >= 2:
                events.append((event[0], event[1], 'material_event'))
    
    return events
