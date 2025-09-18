#!/usr/bin/env python3
"""
Material Lambda³ Detector GPU - Pipeline Edition
================================================

材料解析用Lambda³検出器のGPU実装（パイプライン版）
MD版の設計思想を踏襲し、材料解析に特化

MD版と同じ3段階バッチ処理と適応的ウィンドウを実装
クラスター解析はTwo-Stageアナライザーに委譲

Version: 2.0.0
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
from ..material.material_md_features_gpu import MaterialMDFeaturesGPU  # 材料版MD特徴を使用
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
    """材料解析設定（MD版を踏襲）"""
    # Lambda³パラメータ（MD版と同じ）
    adaptive_window: bool = True
    extended_detection: bool = True
    use_extended_detection: bool = True
    use_phase_space: bool = True
    sensitivity: float = 1.5  # 材料は低めに
    min_boundary_gap: int = 50  # 材料は短い時間スケール
    
    # 材料特有の設定
    material_type: str = 'SUJ2'  # 'SUJ2', 'AL7075', 'TI6AL4V'
    use_material_analytics: bool = True  # 材料特有の解析を使用
    
    # GPU設定（MD版と同じ）
    gpu_batch_size: int = 10000
    mixed_precision: bool = False
    benchmark_mode: bool = False
    
    # 異常検出の重み（MD版と同じ構造）
    w_lambda_f: float = 0.3
    w_lambda_ff: float = 0.2
    w_rho_t: float = 0.2
    w_topology: float = 0.3
    
    # 材料特有の重み
    w_defect: float = 0.2  # 欠陥チャージの重み
    w_coherence: float = 0.1  # 構造一貫性の重み
    
    # 適応的ウィンドウのスケール設定
    window_scale: float = 0.005

@dataclass
class MaterialLambda3Result:
    """材料Lambda³解析結果（MD版と同じ構造）"""
    # Core Lambda³構造（MD版と同じ）- デフォルト値なし
    lambda_structures: Dict[str, np.ndarray]
    structural_boundaries: Dict[str, Any]
    topological_breaks: Dict[str, np.ndarray]
    
    # MD/材料特徴 - デフォルト値なし
    md_features: Dict[str, np.ndarray]
    
    # 解析結果 - デフォルト値なし
    anomaly_scores: Dict[str, np.ndarray]
    detected_structures: List[Dict]
    
    # ここから下はデフォルト値あり
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
    GPU版Lambda³材料検出器（MD版の設計を踏襲）
    
    MD版と同じ3段階バッチ処理を実装し、
    材料特有の解析はMaterialAnalyticsGPUに委譲
    """
    
    def __init__(self, config: MaterialConfig = None, device: str = 'auto'):
        """
        Parameters
        ----------
        config : MaterialConfig, optional
            設定パラメータ
        device : str, default='auto'
            'auto', 'gpu', 'cpu'のいずれか
        """
        # GPUBackendの初期化（MD版と同じ）
        if device == 'cpu':
            super().__init__(device='cpu', force_cpu=True)
        else:
            super().__init__(device=device, force_cpu=False)
        
        self.config = config or MaterialConfig()
        self.verbose = True
        
        # force_cpuフラグを決定
        force_cpu_flag = not self.is_gpu
        
        # MD版と同じコンポーネント構成（材料版に最適化）
        self.structure_computer = LambdaStructuresGPU(force_cpu_flag)
        self.feature_extractor = MaterialMDFeaturesGPU(force_cpu_flag)  # 材料版MD特徴抽出
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
            # ===== 修正: material_feature_extractorを適切に設定 =====
            # すでにself.feature_extractorがMaterialMDFeaturesGPUなので、それを使う
            self.material_feature_extractor = self.feature_extractor
        else:
            self.material_analytics = None
            self.material_feature_extractor = None
        
        # メモリマネージャとデバイスを共有（MD版と同じ）
        for component in [self.structure_computer, self.feature_extractor,
                         self.anomaly_detector, self.boundary_detector,
                         self.topology_detector, self.extended_detector,
                         self.phase_space_analyzer]:
            if hasattr(component, 'memory_manager'):
                component.memory_manager = self.memory_manager
            if hasattr(component, 'device'):
                component.device = self.device
        
        # material_analyticsのメモリマネージャー共有
        if self.material_analytics:
            self.material_analytics.memory_manager = self.memory_manager
            self.material_analytics.device = self.device
            # material_feature_extractorはself.feature_extractorと同じなので、すでに共有済み
        
        self._print_initialization_info()    

    def analyze(self,
            trajectory: np.ndarray,
            backbone_indices: Optional[np.ndarray] = None,
            atom_types: Optional[np.ndarray] = None,
            cluster_definition_path: Optional[str] = None,
            cluster_atoms: Optional[Dict[int, List[int]]] = None,  # ← 追加！
            **kwargs) -> MaterialLambda3Result:
        """
        材料軌道のLambda³解析（MD版と同じインターフェース）
        
        Parameters
        ----------
        trajectory : np.ndarray
            材料軌道 (n_frames, n_atoms, 3)
        backbone_indices : np.ndarray, optional
            重要原子のインデックス（材料では欠陥領域など）
        atom_types : np.ndarray, optional
            原子タイプ配列
        cluster_definition_path : str, optional
            クラスター定義ファイルパス（欠陥領域の自動検出に使用）
            
        Returns
        -------
        MaterialLambda3Result
            解析結果
        """
        start_time = time.time()
        
        # 原子タイプの前処理（文字列→数値変換）
        atom_type_names = None
        if atom_types is not None and atom_types.dtype.kind == 'U':  # 文字列の場合
            atom_type_names = np.unique(atom_types)  # 元の名前を保存
            type_map = {t: i for i, t in enumerate(atom_type_names)}
            atom_types = np.array([type_map[t] for t in atom_types], dtype=np.int32)
        
        # NumPy配列をGPU（CuPy配列）に変換
        if self.is_gpu and cp is not None:
            print("📊 Converting arrays to GPU...")
            trajectory = cp.asarray(trajectory)
            if backbone_indices is not None:
                backbone_indices = cp.asarray(backbone_indices)
            if atom_types is not None:
                atom_types = cp.asarray(atom_types) 
        
        n_frames, n_atoms, _ = trajectory.shape
        
        print(f"\n{'='*60}")
        print(f"=== Lambda³ Material Analysis (GPU) ===")
        print(f"{'='*60}")
        print(f"Trajectory: {n_frames} frames, {n_atoms} atoms")
        print(f"Material: {self.config.material_type}")
        print(f"GPU Device: {self.device}")
        
        # メモリ情報の安全な取得（MD版と同じ）
        try:
            mem_info = self.memory_manager.get_memory_info()
            print(f"Available GPU Memory: {mem_info.free / 1e9:.2f} GB")
        except Exception as e:
            print(f"Memory info unavailable: {e}")
        
        # バッチ処理の判定（MD版と同じロジック）
        batch_size = min(self.config.gpu_batch_size, n_frames)
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        if n_batches > 1:
            print(f"Processing in {n_batches} batches of {batch_size} frames")
            result = self._analyze_batched(trajectory, backbone_indices, 
                                          atom_types, batch_size, 
                                          cluster_definition_path)
        else:
            # 単一バッチ処理
            result = self._analyze_single_trajectory(trajectory, backbone_indices, 
                                                    atom_types, cluster_definition_path)
        
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        
        self._print_summary(result)
        
        return result

    def _analyze_single_trajectory(self,
                             trajectory: np.ndarray,
                             backbone_indices: Optional[np.ndarray],
                             atom_types: Optional[np.ndarray],
                             cluster_definition_path: Optional[str] = None,
                             cluster_atoms: Optional[Dict[int, List[int]]] = None) -> MaterialLambda3Result:  # ← 追加
        """単一軌道の解析（MD版と同じ構造）"""
        n_frames, n_atoms, _ = trajectory.shape

        if cluster_atoms is not None:
            # Cluster 0以外を欠陥として扱う
            defect_indices = []
            for cid, atoms in cluster_atoms.items():
                if str(cid) != "0":  # Cluster 0（健全領域）以外
                    defect_indices.extend(atoms)
            backbone_indices = np.array(sorted(defect_indices))
            print(f"   Using {len(backbone_indices)} defect atoms from clusters")

        # 1. MD特徴抽出（材料版MD特徴抽出を使用）
        print("\n1. Extracting MD features on GPU...")
        md_features = self.feature_extractor.extract_md_features(
            trajectory, backbone_indices,
            cluster_definition_path=cluster_definition_path,
            atom_types=atom_types
        )
        
        # 1.5. 材料特徴抽出（材料特有）
        material_features = None
        if self.config.use_material_analytics and self.material_feature_extractor:
            print("\n1.5. Extracting material-specific features...")
            
            # クラスター定義が渡されていればそれを使う、なければ簡易定義
            if cluster_atoms is not None:
                clusters_for_features = cluster_atoms
                print(f"    Using provided clusters: {len(clusters_for_features)} clusters")
            else:
                # 簡易クラスター定義（Two-Stageで詳細化）
                clusters_for_features = {0: list(range(n_atoms))}
                print("    Using simple cluster definition (all atoms)")
            
            material_features = self.material_feature_extractor.extract_md_features(
                trajectory=self.to_cpu(trajectory) if self.is_gpu else trajectory,
                backbone_indices=backbone_indices,  # 欠陥領域のインデックス
                cluster_definition_path=cluster_definition_path,
                atom_types=self.to_cpu(atom_types) if atom_types is not None and self.is_gpu else atom_types
            ) 
        
        # 2. 初期ウィンドウサイズ（MD版と同じ）
        initial_window = self._compute_initial_window(n_frames)
        
        # 3. Lambda構造計算（MD版と同じ）
        print("\n2. Computing Lambda³ structures (first pass)...")
        lambda_structures = self.structure_computer.compute_lambda_structures(
            trajectory, md_features, initial_window
        )
        
        # 4. 適応的ウィンドウサイズ決定（MD版と同じ）
        adaptive_windows = self._determine_adaptive_windows(
            lambda_structures, initial_window
        )
        primary_window = adaptive_windows.get('primary', initial_window)
        
        # 5. 構造境界検出（MD版と同じ）
        print("\n3. Detecting structural boundaries...")
        boundary_window = adaptive_windows.get('boundary', primary_window // 3)
        structural_boundaries = self.boundary_detector.detect_structural_boundaries(
            lambda_structures, boundary_window
        )
        
        # 6. トポロジカル破れ検出（MD版と同じ）
        print("\n4. Detecting topological breaks...")
        fast_window = adaptive_windows.get('fast', primary_window // 2)
        topological_breaks = self.topology_detector.detect_topological_breaks(
            lambda_structures, fast_window
        )
        
        # 7. マルチスケール異常検出（MD版と同じ）
        print("\n5. Computing multi-scale anomaly scores...")
        anomaly_scores = self.anomaly_detector.compute_multiscale_anomalies(
            lambda_structures,
            structural_boundaries,
            topological_breaks,
            md_features,
            self.config
        )
        
        # 7.5. 材料特有の解析（材料追加）
        defect_analysis = None
        structural_coherence = None
        failure_prediction = None
        material_events = None
        
        if self.config.use_material_analytics and self.material_analytics:
            print("\n5.5. Performing material-specific analysis...")
            
            # 欠陥解析
            if atom_types is not None:
                trajectory_cpu = self.to_cpu(trajectory) if self.is_gpu else trajectory
                defect_result = self.material_analytics.compute_crystal_defect_charge(
                    trajectory_cpu,
                    {0: list(range(n_atoms))},  # 簡易クラスター
                    cutoff=3.5
                )
                defect_analysis = {
                    'defect_charge': defect_result.defect_charge,
                    'cumulative_charge': defect_result.cumulative_charge
                }
            
            # 構造一貫性
            if material_features and 'coordination' in material_features:
                structural_coherence = self.material_analytics.compute_structural_coherence(
                    material_features['coordination'],
                    material_features.get('strain', np.zeros((n_frames, 1))),
                    window=primary_window // 2
                )
            
            # 破壊予測
            if material_features and 'strain' in material_features:
                failure_prediction = self.material_analytics.predict_failure(
                    material_features['strain'],
                    damage_history=material_features.get('damage'),
                    defect_charge=defect_analysis['cumulative_charge'] if defect_analysis else None
                )
                failure_prediction = {
                    'failure_probability': failure_prediction.failure_probability,
                    'reliability_index': failure_prediction.reliability_index,
                    'failure_mode': failure_prediction.failure_mode
                }
        
        # 8. 構造パターン検出（MD版と同じ）
        print("\n6. Detecting structural patterns...")
        slow_window = adaptive_windows.get('slow', primary_window * 2)
        detected_structures = self._detect_structural_patterns(
            lambda_structures, structural_boundaries, slow_window
        )
        
        # 9. 位相空間解析（MD版と同じ、オプション）
        phase_space_analysis = None
        if self.config.use_phase_space:
            print("\n7. Performing phase space analysis...")
            try:
                phase_space_analysis = self.phase_space_analyzer.analyze_phase_space(
                    lambda_structures
                )
            except Exception as e:
                print(f"Phase space analysis failed: {e}")
        
        # 臨界イベントの検出（MD版と同じ）
        critical_events = self._detect_critical_events(anomaly_scores)
        
        # 材料イベントの分類
        if self.config.use_material_analytics and self.material_analytics:
            if critical_events and structural_coherence is not None:
                material_events = self.material_analytics.classify_material_events(
                    critical_events,
                    anomaly_scores,
                    structural_coherence,
                    defect_charge=defect_analysis['defect_charge'] if defect_analysis else None
                )
        
        # GPU情報を収集（MD版と同じ）
        gpu_info = self._get_gpu_info()
        
        # 結果を構築
        result = MaterialLambda3Result(
            lambda_structures=self._to_cpu_dict(lambda_structures),
            structural_boundaries=structural_boundaries,
            topological_breaks=self._to_cpu_dict(topological_breaks),
            md_features=self._to_cpu_dict(md_features),
            material_features=self._to_cpu_dict(material_features) if material_features else None,
            anomaly_scores=self._to_cpu_dict(anomaly_scores),
            detected_structures=detected_structures,
            phase_space_analysis=phase_space_analysis,
            defect_analysis=defect_analysis,
            structural_coherence=structural_coherence,
            failure_prediction=failure_prediction,
            material_events=material_events,
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
                    atom_types: Optional[np.ndarray],
                    batch_size: int,
                    cluster_definition_path: Optional[str] = None,
                    cluster_atoms: Optional[Dict[int, List[int]]] = None) -> MaterialLambda3Result:
        """バッチ処理による解析（Colab対応版）"""
        print("\n⚡ Running batched GPU analysis...")
        
        n_frames = trajectory.shape[0]
        
        # ===== Colab対応：最小バッチ処理 =====
        MIN_BATCH_SIZE = 1000  # 最小1000フレーム
        
        # バッチ計画を最適化
        batches = []
        current_pos = 0
        
        while current_pos < n_frames:
            batch_end = min(current_pos + batch_size, n_frames)
            remaining_frames = batch_end - current_pos
            
            # 最後の小さいバッチを処理
            if batch_end == n_frames and remaining_frames < MIN_BATCH_SIZE and len(batches) > 0:
                # 前のバッチと結合
                print(f"  ⚠️ Last batch too small ({remaining_frames} frames), merging with previous")
                prev_start, _ = batches[-1]
                batches[-1] = (prev_start, n_frames)
                break
            else:
                batches.append((current_pos, batch_end))
                current_pos = batch_end
        
        print(f"  Optimized to {len(batches)} batches")
        
        # バッチごとの結果を蓄積
        batch_results = []
        
        # Step 1: バッチ処理
        print("\n[Step 1] Processing batches for feature extraction...")
        for batch_idx, (start_idx, end_idx) in enumerate(batches):
            print(f"  Batch {batch_idx + 1}/{len(batches)}: frames {start_idx}-{end_idx} ({end_idx-start_idx} frames)")
            
            try:
                batch_trajectory = trajectory[start_idx:end_idx]
                
                # バッチ解析
                batch_result = self._analyze_single_batch(
                    batch_trajectory, backbone_indices, atom_types, start_idx,
                    cluster_definition_path
                )
                
                batch_results.append(batch_result)
                
            except Exception as e:
                print(f"  ⚠️ Batch {batch_idx + 1} failed: {e}")
                # 失敗したバッチはスキップ
                continue
            
            finally:
                # メモリクリア
                self.memory_manager.clear_cache()
        
        if not batch_results:
            print("  ❌ All batches failed!")
            return self._create_empty_result(n_frames, trajectory.shape[1])
        
        # Step 2: 結果をマージ（MaterialLambda3Resultを返す）
        print("\n[Step 2] Merging batch results...")
        merged_result = self._merge_batch_results(batch_results, trajectory.shape, atom_types)
        
        # Step 3: 完了処理
        print("\n[Step 3] Completing analysis on merged data...")
        
        # ===== 重要：MaterialLambda3Resultオブジェクトをそのまま渡す =====
        # merged_resultはすでにMaterialLambda3Resultなので、変換不要！
        
        try:
            final_result = self._complete_analysis(merged_result, atom_types)
        except AttributeError as e:
            if "'NoneType'" in str(e):
                print(f"  ⚠️ NoneType error in _complete_analysis: {e}")
                print("  Using merged result without additional analysis")
                # エラー時は最低限の解析で返す
                merged_result.structural_boundaries = {'boundary_locations': np.array([])}
                merged_result.topological_breaks = {'break_points': np.array([])}
                merged_result.anomaly_scores = {'combined': np.zeros(n_frames)}
                merged_result.critical_events = []
                merged_result.material_events = []
                return merged_result
            else:
                raise
        
        return final_result
    
    def _analyze_single_batch(self,
                            batch_trajectory: np.ndarray,
                            backbone_indices: Optional[np.ndarray],
                            atom_types: Optional[np.ndarray],
                            offset: int,
                            cluster_definition_path: Optional[str] = None) -> Dict:
        """単一バッチの解析（MD版と同じ - 特徴抽出とLambda構造計算のみ）"""
        # MD特徴抽出（重い処理）
        md_features = self.feature_extractor.extract_md_features(
            batch_trajectory, backbone_indices,
            cluster_definition_path=cluster_definition_path,
            atom_types=atom_types
        )
        
        # 材料特徴抽出（重い処理）
        material_features = None
        if self.config.use_material_analytics and self.material_feature_extractor:
            n_atoms = batch_trajectory.shape[1]
            simple_clusters = {0: list(range(n_atoms))}
            
            batch_traj_cpu = self.to_cpu(batch_trajectory) if self.is_gpu else batch_trajectory
            atom_types_cpu = self.to_cpu(atom_types) if atom_types is not None and self.is_gpu else atom_types
            
            material_features = self.material_feature_extractor.extract_md_features(
                trajectory=batch_traj_cpu,
                backbone_indices=None,  # バッチ処理では省略
                cluster_definition_path=cluster_definition_path,
                atom_types=atom_types_cpu
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
            'material_features': material_features,
            'window': window
        }

    def _merge_batch_results(self,
                       batch_results: List[Dict],
                       original_shape: Tuple,
                       atom_types: Optional[np.ndarray]) -> MaterialLambda3Result:
        """
        バッチ結果の統合（タンパク質版と同じ方式）
        """
        print("  Merging data from all batches...")
        
        n_frames, n_atoms, _ = original_shape
        
        if not batch_results:
            return self._create_empty_result(n_frames, n_atoms)
        
        # 結果を保存する辞書を初期化
        merged_lambda_structures = {}
        merged_md_features = {}
        merged_material_features = {} if self.config.use_material_analytics else None
        
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
        
        # 材料特徴の配列を初期化
        if merged_material_features is not None and first_batch.get('material_features'):
            material_keys = first_batch['material_features'].keys()
            print(f"    Material feature keys: {list(material_keys)}")
            
            for key in material_keys:
                sample = first_batch['material_features'][key]
                if isinstance(sample, (np.ndarray, self.xp.ndarray)):
                    rest_shape = sample.shape[1:] if len(sample.shape) > 1 else ()
                    full_shape = (n_frames,) + rest_shape
                    dtype = sample.dtype
                    merged_material_features[key] = np.full(full_shape, np.nan, dtype=dtype)
        
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
            
            # 材料特徴をマージ
            if merged_material_features is not None and batch_result.get('material_features'):
                for key, value in batch_result['material_features'].items():
                    if key in merged_material_features:
                        if hasattr(value, 'get'):
                            value = self.to_cpu(value)
                            actual_frames = min(len(value), batch_n_frames)
                            merged_material_features[key][offset:offset + actual_frames] = value[:actual_frames]
        
        # ===== NaN処理を削除！タンパク質版と同じ方式 =====
        print("    Checking NaN values...")
        
        # NaNチェック（警告のみ、埋めない）
        for key, arr in merged_lambda_structures.items():
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                print(f"    ⚠️ {key} has {nan_count} unprocessed frames")
                # NaN埋めは削除！そのまま渡す
        
        # MD特徴のNaNチェック（警告のみ）
        for key, arr in merged_md_features.items():
            if isinstance(arr, np.ndarray) and arr.dtype != object:
                nan_count = np.isnan(arr).sum() if arr.dtype in [np.float32, np.float64] else 0
                if nan_count > 0:
                    print(f"    ⚠️ MD feature {key} has {nan_count} NaN values")
        
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
            'batch_sizes': [b['n_frames'] for b in batch_results],
            'nan_handling': 'preserved'  # NaN保持フラグ
        }
        
        print(f"  ✅ Merged {n_frames} frames successfully")
        
        # マージ結果を返す（解析は未完了）
        return MaterialLambda3Result(
            lambda_structures=merged_lambda_structures,
            structural_boundaries={},  # 後で計算
            topological_breaks={},      # 後で計算
            md_features=merged_md_features,
            material_features=merged_material_features,
            anomaly_scores={},          # 後で計算
            detected_structures=[],     # 後で計算
            critical_events=[],         # 後で計算
            n_frames=n_frames,
            n_atoms=n_atoms,
            window_steps=window_steps,
            computation_time=0.0,
            gpu_info=gpu_info
        )
     
    def _complete_analysis(self, 
                          merged_result: MaterialLambda3Result,
                          atom_types: Optional[np.ndarray]) -> MaterialLambda3Result:
        """マージ後のデータで解析を完了（MD版と同じ - 境界検出、異常スコア計算など）"""
        
        lambda_structures = merged_result.lambda_structures
        md_features = merged_result.md_features
        material_features = merged_result.material_features
        n_frames = merged_result.n_frames
        n_atoms = merged_result.n_atoms
        
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
        
        # 7.5. 材料特有の解析（材料追加）
        defect_analysis = None
        structural_coherence = None
        failure_prediction = None
        material_events = None
        
        if self.config.use_material_analytics and self.material_analytics:
            print("  - Performing material-specific analysis...")
            
            # 欠陥解析
            if atom_types is not None:
                # ダミートラジェクトリ作成（Lambda構造から推定）
                # 本来はトラジェクトリ全体を保持すべきだが、メモリ効率のため省略
                # Two-Stageで詳細解析
                pass
            
            # 構造一貫性
            if material_features and 'coordination' in material_features:
                structural_coherence = self.material_analytics.compute_structural_coherence(
                    material_features['coordination'],
                    material_features.get('strain', np.zeros((n_frames, 1))),
                    window=primary_window // 2
                )
            
            # 破壊予測
            if material_features and 'strain' in material_features:
                failure_prediction = self.material_analytics.predict_failure(
                    material_features['strain'],
                    damage_history=material_features.get('damage')
                )
                failure_prediction = {
                    'failure_probability': failure_prediction.failure_probability,
                    'reliability_index': failure_prediction.reliability_index,
                    'failure_mode': failure_prediction.failure_mode
                }
        
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
        
        # 材料イベントの分類
        if self.config.use_material_analytics and self.material_analytics:
            if critical_events and structural_coherence is not None:
                material_events = self.material_analytics.classify_material_events(
                    critical_events,
                    anomaly_scores,
                    structural_coherence
                )
        
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
        merged_result.critical_events = critical_events
        merged_result.window_steps = primary_window
        
        return merged_result
    
    # === ヘルパーメソッド（MD版と同じ） ===
    
    def _compute_initial_window(self, n_frames: int) -> int:
        """初期ウィンドウサイズの計算"""
        return min(50, n_frames // 10)  # 材料は短めに
    
    def _determine_adaptive_windows(self, lambda_structures: Dict, 
                                   initial_window: int) -> Dict[str, int]:
        """適応的ウィンドウサイズの決定（MD版と同じ）"""
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
        """構造パターンの検出（MD版と同じ）"""
        patterns = []
        
        if isinstance(boundaries, dict) and 'boundary_locations' in boundaries:
            boundary_locs = boundaries['boundary_locations']
            
            # 境界間のセグメントを構造として認識
            if len(boundary_locs) > 0:
                # 最初のセグメント
                if boundary_locs[0] > 30:  # 材料は短めの閾値
                    patterns.append({
                        'type': 'initial_structure',
                        'start': 0,
                        'end': boundary_locs[0],
                        'duration': boundary_locs[0]
                    })
                
                # 中間セグメント
                for i in range(len(boundary_locs) - 1):
                    duration = boundary_locs[i+1] - boundary_locs[i]
                    if duration > 20:  # 材料は短めの閾値
                        pattern_type = 'elastic' if duration < 100 else 'plastic'
                        patterns.append({
                            'type': pattern_type,
                            'start': boundary_locs[i],
                            'end': boundary_locs[i+1],
                            'duration': duration
                        })
        
        return patterns
    
    def _detect_critical_events(self, anomaly_scores: Dict) -> List:
        """臨界イベントの検出（MD版と同じ）"""
        events = []
        
        if 'combined' in anomaly_scores:
            scores = anomaly_scores['combined']
            threshold = np.mean(scores) + 1.5 * np.std(scores)  # 材料は低めの閾値
            
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
    
    def _to_cpu_dict(self, data_dict: Optional[Dict]) -> Dict:
        """辞書内のGPU配列をCPUに転送（MD版と同じ）"""
        if data_dict is None:
            return {}
        
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
        """GPU情報を安全に取得（MD版と同じ）"""
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
        """空の結果を作成（MD版と同じ）"""
        return MaterialLambda3Result(
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
            print(f"\n💎 Material Lambda³ GPU Detector Initialized")
            print(f"   Material: {self.config.material_type}")
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
        
        # 材料特有の結果
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
                    event_type = event[2]
                    event_types[event_type] = event_types.get(event_type, 0) + 1
            if event_types:
                print("  Event types:")
                for etype, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {etype}: {count}")
        
        if 'combined' in result.anomaly_scores:
            scores = result.anomaly_scores['combined']
            print(f"\nAnomaly statistics:")
            print(f"  Mean score: {np.mean(scores):.3f}")
            print(f"  Max score: {np.max(scores):.3f}")
            print(f"  Frames > 1.5σ: {np.sum(scores > 1.5)}")
    
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

# ===============================
# Module Level Functions
# ===============================

def detect_material_events(
    trajectory: np.ndarray,
    atom_types: np.ndarray,
    config: Optional[MaterialConfig] = None,
    **kwargs
) -> List[Tuple[int, int, str]]:
    """
    材料イベントを検出する便利関数
    
    MaterialLambda3DetectorGPUのラッパー関数
    """
    config = config or MaterialConfig()
    detector = MaterialLambda3DetectorGPU(config)
    
    # 解析実行
    result = detector.analyze(
        trajectory=trajectory,
        atom_types=atom_types,
        **kwargs
    )
    
    # material_eventsを返す
    if hasattr(result, 'material_events') and result.material_events:
        return result.material_events
    
    # なければcritical_eventsから生成
    events = []
    if hasattr(result, 'critical_events'):
        for event in result.critical_events:
            if isinstance(event, tuple) and len(event) >= 2:
                # (start, end, type)形式に変換
                events.append((event[0], event[1], 'material_event'))
    
    return events
