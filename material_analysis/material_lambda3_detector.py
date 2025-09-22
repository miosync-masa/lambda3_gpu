#!/usr/bin/env python3
"""
Material Lambda³ Detector GPU - Physics-Integrated Edition v3.0
================================================================

材料解析用Lambda³検出器のGPU実装（物理ダメージ計算統合版）

主な改良点（v3.0）：
- PhysicalDamageCalculator統合によるK/V比ベースの物理的ダメージ計算
- Miner則による疲労累積評価
- 温度依存性を考慮した破壊予測
- パーコレーション理論による臨界クラスター特定
- MD特徴からのK/V比自動計算

Version: 3.0.0 - Physics Integration Edition
Author: 環ちゃん - Material Master Edition
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
from ..structures.md_features_gpu import MDFeaturesGPU
from ..material.material_md_features_gpu import MaterialMDFeaturesGPU
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
    """材料解析設定（物理ダメージ解析拡張版）"""
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
    
    # 物理ダメージ解析設定（NEW!）
    use_physical_damage: bool = True
    damage_temperature: float = 300.0  # デフォルト温度 [K]
    kv_critical_override: Optional[float] = None  # 臨界K/V比の上書き
    
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
    w_physical_damage: float = 0.25  # 物理ダメージの重み（NEW!）
    
    # 適応的ウィンドウのスケール設定
    window_scale: float = 0.005

@dataclass
class MaterialLambda3Result:
    """材料Lambda³解析結果（物理ダメージ統合版）"""
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
    
    # 物理ダメージ解析（NEW!）
    physical_damage: Optional[Dict[str, Any]] = None
    kv_ratios: Optional[np.ndarray] = None
    temperature_history: Optional[np.ndarray] = None
    
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
    GPU版Lambda³材料検出器（物理ダメージ統合版 v3.0）
    
    処理フロー：
    1. 前処理（バッチごと）: MD特徴抽出 → 欠陥解析 → Lambda構造計算 → K/V比計算
    2. 後処理（マージ後）: 境界検出 → 異常検出 → 材料解析 → 物理ダメージ評価
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
        
        # 特徴抽出器（2つの独立した抽出器）
        self.feature_extractor = MDFeaturesGPU(force_cpu_flag)  # Lambda構造用（全原子）
        self.material_feature_extractor = MaterialMDFeaturesGPU(force_cpu_flag)  # 材料解析用（欠陥領域）
        
        self.anomaly_detector = AnomalyDetectorGPU(force_cpu_flag)
        self.boundary_detector = BoundaryDetectorGPU(force_cpu_flag)
        self.topology_detector = TopologyBreaksDetectorGPU(force_cpu_flag)
        self.extended_detector = ExtendedDetectorGPU(force_cpu_flag)
        self.phase_space_analyzer = PhaseSpaceAnalyzerGPU(force_cpu_flag)
        
        # 材料特有のコンポーネント（物理ダメージ対応版）
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
            self.feature_extractor,
            self.material_feature_extractor,
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
        材料軌道のLambda³解析（物理ダメージ統合版）
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
    
    def _analyze_single_trajectory(self,
                                  trajectory: np.ndarray,
                                  backbone_indices: Optional[np.ndarray],
                                  atom_types: Optional[np.ndarray],
                                  cluster_definition_path: Optional[str] = None,
                                  cluster_atoms: Optional[Dict[int, List[int]]] = None
                                  ) -> MaterialLambda3Result:
        """単一軌道の完全解析（物理ダメージ計算統合）"""
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
            None  # 全原子で計算
        )
        
        # 2. 材料特有の特徴抽出（欠陥領域 - 材料解析用）
        material_features = None
        if self.config.use_material_analytics and backbone_indices is not None:
            print("  2. Extracting material features (defect region)...")
            material_features = self.material_feature_extractor.extract_md_features(
                trajectory, 
                backbone_indices,
                cluster_definition_path=cluster_definition_path,
                atom_types=atom_types
            )
            
            # 3. 欠陥解析
            print("  3. Computing crystal defect charges...")
            self._add_defect_analysis_to_features(
                trajectory, md_features, cluster_atoms, n_atoms
            )
        
        # 4. K/V比計算（NEW!）
        kv_ratios = None
        if self.config.use_physical_damage:
            print("  4. Computing K/V ratios...")
            kv_ratios = self._extract_kv_ratios(md_features)
            if kv_ratios is not None:
                print(f"     K/V ratio range: [{np.min(kv_ratios):.2f}, {np.max(kv_ratios):.2f}]")
        
        # 5. Lambda構造計算
        initial_window = self._compute_initial_window(n_frames)
        print("  5. Computing Lambda³ structures...")
        lambda_structures = self.structure_computer.compute_lambda_structures(
            trajectory, md_features, initial_window
        )
        
        # === 後処理 ===
        print("\n[POSTPROCESSING]")
        
        # 解析結果を統合
        result = self._perform_postprocessing(
            lambda_structures, md_features, material_features,
            n_frames, n_atoms, initial_window,
            kv_ratios=kv_ratios  # K/V比を渡す
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
        """バッチ処理による解析（物理ダメージ対応）"""
        print("\n⚡ Running batched GPU analysis...")
        
        n_frames = trajectory.shape[0]
        
        # 欠陥領域の特定
        backbone_indices = self._identify_defect_regions(
            cluster_atoms, backbone_indices
        )
        
        batches = self._optimize_batch_plan(n_frames, batch_size)
        
        # Step 1: 前処理（バッチごと）
        batch_results = self._process_batches(
            trajectory, backbone_indices, atom_types,
            batches, cluster_definition_path, cluster_atoms
        )
        
        if not batch_results:
            return self._create_empty_result(n_frames, trajectory.shape[1])
        
        # Step 2: 結果をマージ
        print("\n[MERGING]")
        merged_result = self._merge_batch_results(
            batch_results, trajectory.shape, atom_types
        )
        
        # Step 3: 後処理（物理ダメージ解析を含む）
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
        単一バッチの前処理（物理ダメージ計算対応）
        """
        n_atoms = batch_trajectory.shape[1]
        
        # 1. 基本MD特徴抽出
        md_features = self.feature_extractor.extract_md_features(
            batch_trajectory, 
            None  # 全原子
        )
        
        # 2. 材料特有の特徴抽出
        material_features = None
        if self.config.use_material_analytics and backbone_indices is not None:
            material_features = self.material_feature_extractor.extract_md_features(
                batch_trajectory, 
                backbone_indices,
                cluster_definition_path=cluster_definition_path,
                atom_types=atom_types
            )
            
            # 3. 欠陥解析
            self._add_defect_analysis_to_features(
                batch_trajectory, md_features, None, n_atoms
            )
        
        # 4. K/V比計算（NEW!）
        kv_ratios = None
        if self.config.use_physical_damage:
            kv_ratios = self._extract_kv_ratios(md_features)
        
        # 5. Lambda構造計算
        window = self._compute_initial_window(len(batch_trajectory))
        lambda_structures = self.structure_computer.compute_lambda_structures(
            batch_trajectory, md_features, window
        )
        
        return {
            'offset': offset,
            'n_frames': len(batch_trajectory),
            'lambda_structures': lambda_structures,
            'md_features': md_features,
            'material_features': material_features,
            'kv_ratios': kv_ratios,  # K/V比を保存
            'window': window
        }
    
    def _complete_analysis(self,
                          merged_result: MaterialLambda3Result,
                          atom_types: Optional[np.ndarray]
                          ) -> MaterialLambda3Result:
        """
        マージ後の後処理（物理ダメージ解析統合版）
        """
        lambda_structures = merged_result.lambda_structures
        md_features = merged_result.md_features
        material_features = merged_result.material_features
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
        anomaly_scores = self._compute_enhanced_anomaly_scores(
            lambda_structures, structural_boundaries,
            topological_breaks, md_features, merged_result.physical_damage
        )
        
        # 材料特有の解析
        defect_analysis = None
        structural_coherence = None
        failure_prediction = None
        material_events = None
        physical_damage = None
        temperature_history = None
        
        if self.config.use_material_analytics and self.material_analytics:
            print("  - Material-specific analysis...")
            
            features_for_material = material_features if material_features else md_features
            
            # 欠陥情報
            if 'defect_charge' in md_features:
                defect_analysis = {
                    'defect_charge': md_features.get('defect_charge'),
                    'cumulative_charge': md_features.get('cumulative_charge')
                }
            
            # 構造一貫性
            if 'coordination' in features_for_material:
                structural_coherence = self.material_analytics.compute_structural_coherence(
                    features_for_material['coordination'],
                    features_for_material.get('strain', np.zeros((n_frames, 1))),
                    window=primary_window // 2
                )
            
            # 温度履歴の抽出/推定
            temperature_history = self._extract_temperature_history(features_for_material)
            
            # 物理ダメージ解析（NEW!）
            if self.config.use_physical_damage and merged_result.kv_ratios is not None:
                print("  - Computing physics-based damage assessment...")
                physical_damage = self._compute_physical_damage_analysis(
                    merged_result.kv_ratios,
                    temperature_history,
                    defect_analysis
                )
            
            # 破壊予測（物理ダメージ統合版）
            if 'strain' in features_for_material:
                failure_prediction = self._compute_integrated_failure_prediction(
                    features_for_material,
                    physical_damage,
                    defect_analysis
                )
            
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
        merged_result.physical_damage = physical_damage
        merged_result.temperature_history = temperature_history
        
        return merged_result
    
    # === 物理ダメージ計算関連メソッド（NEW!） ===
    
    def _extract_kv_ratios(self, md_features: Dict) -> Optional[np.ndarray]:
        """MD特徴からK/V比を抽出または計算"""
        try:
            if 'kv_ratio' in md_features:
                # 直接K/V比が計算されている場合
                return md_features['kv_ratio']
            
            elif 'kinetic_energy' in md_features:
                K = md_features['kinetic_energy']  # 運動エネルギー
                
                # ビリアルテンソルまたは圧力から体積を推定
                if 'virial_tensor' in md_features:
                    # ビリアルテンソルの対角成分から圧力を計算
                    virial = md_features['virial_tensor']
                    if virial.ndim == 3:
                        # (n_frames, n_clusters, 6) -> 対角成分の平均
                        pressure = -np.mean(virial[..., :3], axis=-1)
                    else:
                        pressure = np.abs(virial)
                    
                    # 圧力から仮想的な体積を推定（PV = nkT的な関係を仮定）
                    V = np.where(pressure > 1e-10, 1.0 / pressure, 1.0)
                    
                elif 'pressure' in md_features:
                    pressure = np.abs(md_features['pressure'])
                    V = np.where(pressure > 1e-10, 1.0 / pressure, 1.0)
                    
                elif 'volume' in md_features:
                    V = md_features['volume']
                    
                else:
                    # 体積情報がない場合、一定値と仮定
                    logger.warning("No volume/pressure data for K/V ratio calculation")
                    V = np.ones_like(K)
                
                # K/V比計算（ゼロ除算回避）
                kv_ratios = np.divide(K, V, out=np.ones_like(K), where=(V > 1e-10))
                
                # 物理的に妥当な範囲にクリップ
                kv_ratios = np.clip(kv_ratios, 0.01, 100.0)
                
                return kv_ratios
                
        except Exception as e:
            logger.warning(f"Failed to extract K/V ratios: {e}")
            return None
    
    def _extract_temperature_history(self, features: Dict) -> np.ndarray:
        """MD特徴から温度履歴を抽出または推定"""
        if 'temperature' in features:
            return features['temperature']
        
        elif 'kinetic_energy' in features:
            # 運動エネルギーから温度推定
            kb = 8.617e-5  # eV/K（ボルツマン定数）
            ke = features['kinetic_energy']
            
            # 3/2 kB T = KE/N （理想気体近似）
            # ここではシンプルに比例関係を仮定
            temperature = ke / (1.5 * kb)
            
            # 物理的に妥当な範囲にクリップ
            temperature = np.clip(temperature, 1.0, 5000.0)
            
            return temperature
        
        else:
            # デフォルト温度
            n_frames = next(iter(features.values())).shape[0]
            return np.full(n_frames, self.config.damage_temperature)
    
    def _compute_physical_damage_analysis(self,
                                         kv_ratios: np.ndarray,
                                         temperature_history: np.ndarray,
                                         defect_analysis: Optional[Dict]
                                         ) -> Dict[str, Any]:
        """物理ダメージの詳細解析"""
        # 平均温度を使用（時変温度対応も可能）
        avg_temperature = float(np.mean(temperature_history))
        
        # PhysicalDamageCalculatorを使用
        damage_result = self.material_analytics.compute_physical_damage(
            kv_ratios,
            temperature=avg_temperature
        )
        
        # 結果を辞書形式に変換
        physical_damage = {
            'cumulative_damage': damage_result.cumulative_damage,
            'damage_rate': damage_result.damage_rate,
            'failure_probability': damage_result.failure_probability,
            'remaining_life': damage_result.remaining_life,
            'critical_clusters': damage_result.critical_clusters,
            'max_damage': float(np.max(damage_result.cumulative_damage)),
            'avg_damage': float(np.mean(damage_result.cumulative_damage)),
            'temperature': avg_temperature
        }
        
        # 欠陥との相関を計算
        if defect_analysis and 'cumulative_charge' in defect_analysis:
            defect_charge = defect_analysis['cumulative_charge']
            if len(defect_charge) == len(damage_result.cumulative_damage):
                # 欠陥と物理ダメージの相関
                correlation = np.corrcoef(
                    defect_charge,
                    np.mean(damage_result.cumulative_damage, axis=1) if damage_result.cumulative_damage.ndim > 1 
                    else damage_result.cumulative_damage
                )[0, 1]
                physical_damage['defect_damage_correlation'] = float(correlation)
        
        # ダメージ進展の統計
        if damage_result.damage_rate is not None:
            max_rate = np.max(damage_result.damage_rate)
            avg_rate = np.mean(damage_result.damage_rate)
            physical_damage['max_damage_rate'] = float(max_rate)
            physical_damage['avg_damage_rate'] = float(avg_rate)
            
            # 加速度（2次微分）
            if len(damage_result.damage_rate) > 2:
                acceleration = np.gradient(damage_result.damage_rate.mean(axis=1) 
                                         if damage_result.damage_rate.ndim > 1 
                                         else damage_result.damage_rate)
                physical_damage['damage_acceleration'] = float(np.max(np.abs(acceleration)))
        
        return physical_damage
    
    def _compute_integrated_failure_prediction(self,
                                              features: Dict,
                                              physical_damage: Optional[Dict],
                                              defect_analysis: Optional[Dict]
                                              ) -> Dict[str, Any]:
        """物理ダメージを統合した破壊予測"""
        # 従来の破壊予測
        traditional_prediction = self.material_analytics.predict_failure(
            features.get('strain', np.zeros(1)),
            damage_history=features.get('damage'),
            defect_charge=defect_analysis.get('cumulative_charge') if defect_analysis else None
        )
        
        failure_prediction = {
            'failure_probability': traditional_prediction.failure_probability,
            'reliability_index': traditional_prediction.reliability_index,
            'failure_mode': traditional_prediction.failure_mode,
            'remaining_life_cycles': traditional_prediction.remaining_life_cycles
        }
        
        # 物理ダメージとの統合
        if physical_damage:
            # 物理ダメージによる破壊確率
            phys_failure_prob = physical_damage.get('failure_probability', 0.0)
            
            # 統合破壊確率（最大値を採用）
            integrated_prob = max(
                failure_prediction['failure_probability'],
                phys_failure_prob
            )
            
            failure_prediction['integrated_failure_probability'] = integrated_prob
            failure_prediction['physical_damage_probability'] = phys_failure_prob
            
            # 残存寿命の更新
            if physical_damage.get('remaining_life'):
                phys_remaining = physical_damage['remaining_life']
                if failure_prediction.get('remaining_life_cycles'):
                    # 最小値を採用（より悲観的な予測）
                    failure_prediction['integrated_remaining_life'] = min(
                        failure_prediction['remaining_life_cycles'],
                        phys_remaining
                    )
                else:
                    failure_prediction['integrated_remaining_life'] = phys_remaining
            
            # 破壊モードの統合
            if phys_failure_prob > 0.8:
                failure_prediction['integrated_failure_mode'] = 'physical_damage_dominant'
            elif phys_failure_prob > 0.5 and failure_prediction['failure_probability'] > 0.5:
                failure_prediction['integrated_failure_mode'] = 'combined_failure'
            else:
                failure_prediction['integrated_failure_mode'] = failure_prediction.get('failure_mode', 'safe')
        
        return failure_prediction
    
    def _compute_enhanced_anomaly_scores(self,
                                        lambda_structures: Dict,
                                        boundaries: Dict,
                                        topological_breaks: Dict,
                                        md_features: Dict,
                                        physical_damage: Optional[Dict]
                                        ) -> Dict[str, np.ndarray]:
        """物理ダメージを考慮した拡張異常スコア計算"""
        # 基本の異常スコア計算
        anomaly_scores = self.anomaly_detector.compute_multiscale_anomalies(
            lambda_structures, boundaries,
            topological_breaks, md_features, self.config
        )
        
        # 物理ダメージによる異常スコアの追加
        if physical_damage and self.config.use_physical_damage:
            n_frames = len(anomaly_scores.get('combined', []))
            
            # ダメージベースの異常スコア
            if 'cumulative_damage' in physical_damage:
                damage_data = physical_damage['cumulative_damage']
                
                # 形状調整
                if damage_data.ndim > 1:
                    damage_score = np.mean(damage_data, axis=1)
                else:
                    damage_score = damage_data
                
                # 長さ調整
                if len(damage_score) != n_frames:
                    # 補間または切り詰め
                    if len(damage_score) > n_frames:
                        damage_score = damage_score[:n_frames]
                    else:
                        # 線形補間で延長
                        x_old = np.linspace(0, 1, len(damage_score))
                        x_new = np.linspace(0, 1, n_frames)
                        damage_score = np.interp(x_new, x_old, damage_score)
                
                anomaly_scores['physical_damage'] = damage_score
                
                # 統合スコアの再計算
                w_damage = self.config.w_physical_damage
                combined = anomaly_scores['combined'] * (1 - w_damage) + damage_score * w_damage
                anomaly_scores['combined'] = combined
        
        return anomaly_scores
    
    # === 既存のヘルパーメソッド（変更なし） ===
    
    def _identify_defect_regions(self,
                                cluster_atoms: Optional[Dict[int, List[int]]],
                                backbone_indices: Optional[np.ndarray]
                                ) -> Optional[np.ndarray]:
        """欠陥領域の特定"""
        if backbone_indices is not None:
            if backbone_indices.dtype != np.int32:
                backbone_indices = np.array(backbone_indices, dtype=np.int32)
            print(f"  Using provided backbone_indices: {len(backbone_indices)} atoms")
            return backbone_indices
        
        if cluster_atoms is not None:
            defect_indices = []
            for cid, atoms in cluster_atoms.items():
                if str(cid) != "0":  # Cluster 0（健全領域）以外
                    if isinstance(atoms, (list, np.ndarray)):
                        defect_indices.extend([int(a) for a in atoms])
                    else:
                        defect_indices.extend(atoms)
            
            if defect_indices:
                backbone_indices = np.array(sorted(set(defect_indices)), dtype=np.int32)
                print(f"  Auto-detected {len(backbone_indices)} defect atoms from {len(cluster_atoms)-1} clusters")
                return backbone_indices
            else:
                print("  ⚠️ No defect atoms found in clusters")
                return None
        
        print("  ℹ️ No backbone_indices or cluster_atoms provided")
        return None
    
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
    
    def _preprocess_inputs(self, trajectory, atom_types, backbone_indices):
        """入力データの前処理とGPU転送"""
        if atom_types is not None and atom_types.dtype.kind == 'U':
            atom_type_names = np.unique(atom_types)
            type_map = {t: i for i, t in enumerate(atom_type_names)}
            atom_types = np.array([type_map[t] for t in atom_types], dtype=np.int32)
        
        if self.is_gpu and cp is not None:
            print("📊 Converting arrays to GPU...")
            trajectory = cp.asarray(trajectory)
            if backbone_indices is not None:
                backbone_indices = cp.asarray(backbone_indices)
            if atom_types is not None:
                atom_types = cp.asarray(atom_types)
        
        return trajectory, atom_types
    
    def _optimize_batch_plan(self, n_frames: int, batch_size: int) -> List[Tuple[int, int]]:
        """バッチ計画の最適化"""
        MIN_BATCH_SIZE = 1000
        batches = []
        current_pos = 0
        
        while current_pos < n_frames:
            batch_end = min(current_pos + batch_size, n_frames)
            remaining = batch_end - current_pos
            
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
                        cluster_definition_path: Optional[str],
                        cluster_atoms: Optional[Dict[int, List[int]]] = None
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
        """バッチ結果の統合（K/V比対応版）"""
        print("  Merging batch results...")
        n_frames, n_atoms, _ = original_shape
        
        if not batch_results:
            return self._create_empty_result(n_frames, n_atoms)
        
        # 結果配列の初期化
        merged_lambda = {}
        merged_features = {}
        merged_material_features = {}
        merged_kv_ratios = None
        
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
        
        # Material特徴の初期化
        if 'material_features' in first_batch and first_batch['material_features']:
            for key in first_batch['material_features'].keys():
                sample = first_batch['material_features'][key]
                if isinstance(sample, (np.ndarray, self.xp.ndarray)):
                    shape = (n_frames,) + sample.shape[1:] if sample.ndim > 1 else (n_frames,)
                    merged_material_features[key] = np.full(shape, np.nan, dtype=sample.dtype)
        
        # K/V比の初期化（NEW!）
        if 'kv_ratios' in first_batch and first_batch['kv_ratios'] is not None:
            sample = first_batch['kv_ratios']
            shape = (n_frames,) + sample.shape[1:] if sample.ndim > 1 else (n_frames,)
            merged_kv_ratios = np.full(shape, np.nan, dtype=sample.dtype)
        
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
            
            # Material特徴
            if 'material_features' in batch and batch['material_features']:
                for key, value in batch['material_features'].items():
                    if key in merged_material_features:
                        value = self.to_cpu(value) if hasattr(value, 'get') else value
                        actual_frames = min(len(value), end_idx - offset)
                        merged_material_features[key][offset:offset + actual_frames] = value[:actual_frames]
            
            # K/V比（NEW!）
            if 'kv_ratios' in batch and batch['kv_ratios'] is not None and merged_kv_ratios is not None:
                value = self.to_cpu(batch['kv_ratios']) if hasattr(batch['kv_ratios'], 'get') else batch['kv_ratios']
                actual_frames = min(len(value), end_idx - offset)
                merged_kv_ratios[offset:offset + actual_frames] = value[:actual_frames]
        
        # ウィンドウサイズの平均
        window_steps = int(np.mean([b.get('window', 100) for b in batch_results]))
        
        print(f"  ✅ Merged {n_frames} frames successfully")
        
        return MaterialLambda3Result(
            lambda_structures=merged_lambda,
            md_features=merged_features,
            material_features=merged_material_features if merged_material_features else None,
            kv_ratios=merged_kv_ratios,  # K/V比を保存
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
                              initial_window: int,
                              kv_ratios: Optional[np.ndarray] = None
                              ) -> MaterialLambda3Result:
        """単一軌道用の後処理（K/V比対応）"""
        partial_result = MaterialLambda3Result(
            lambda_structures=self._to_cpu_dict(lambda_structures),
            md_features=self._to_cpu_dict(md_features),
            material_features=self._to_cpu_dict(material_features) if material_features else None,
            kv_ratios=self.to_cpu(kv_ratios) if kv_ratios is not None else None,
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
                if boundary_locs[0] > 30:
                    patterns.append({
                        'type': 'initial_structure',
                        'start': 0,
                        'end': boundary_locs[0],
                        'duration': boundary_locs[0]
                    })
                
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
        print(f"=== Lambda³ Material Analysis (Physics-Integrated v3.0) ===")
        print(f"{'='*60}")
        print(f"Trajectory: {n_frames} frames, {n_atoms} atoms")
        print(f"Material: {self.config.material_type}")
        print(f"GPU Device: {self.device}")
        print(f"Physical Damage: {'ON' if self.config.use_physical_damage else 'OFF'}")
        
        try:
            mem_info = self.memory_manager.get_memory_info()
            print(f"Available GPU Memory: {mem_info.free / 1e9:.2f} GB")
        except:
            pass
    
    def _print_initialization_info(self):
        """初期化情報の表示"""
        if self.verbose:
            print(f"\n💎 Material Lambda³ GPU Detector v3.0 Initialized")
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
            print(f"   Physical Damage: {'ON' if self.config.use_physical_damage else 'OFF'}")
    
    def _print_summary(self, result: MaterialLambda3Result):
        """結果サマリーの表示"""
        print("\n" + "="*60)
        print("=== Analysis Complete ===")
        print("="*60)
        print(f"Total frames: {result.n_frames}")
        print(f"Computation time: {result.computation_time:.2f} seconds")
        
        if result.computation_time > 0:
            print(f"Speed: {result.n_frames / result.computation_time:.1f} frames/second")
        
        # 物理ダメージ解析結果（NEW!）
        if result.physical_damage:
            print(f"\n⚡ Physical Damage Analysis:")
            print(f"  Max damage: {result.physical_damage['max_damage']:.1%}")
            print(f"  Failure probability: {result.physical_damage['failure_probability']:.1%}")
            if result.physical_damage.get('remaining_life'):
                print(f"  Remaining life: {result.physical_damage['remaining_life']:.1f} cycles")
            print(f"  Critical clusters: {len(result.physical_damage.get('critical_clusters', []))}")
        
        # 統合破壊予測
        if result.failure_prediction:
            print(f"\n🔬 Integrated Failure Prediction:")
            if 'integrated_failure_probability' in result.failure_prediction:
                print(f"  Integrated probability: {result.failure_prediction['integrated_failure_probability']:.1%}")
            else:
                print(f"  Traditional probability: {result.failure_prediction['failure_probability']:.1%}")
            print(f"  Reliability index: {result.failure_prediction['reliability_index']:.2f}")
            print(f"  Failure mode: {result.failure_prediction.get('integrated_failure_mode', result.failure_prediction.get('failure_mode', 'unknown'))}")
        
        # 材料イベント
        if result.material_events:
            print(f"\n📊 Material Events: {len(result.material_events)}")
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
    """材料イベントを検出する便利関数（物理ダメージ対応版）"""
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

def analyze_with_physics(trajectory: np.ndarray,
                        atom_types: Optional[np.ndarray] = None,
                        material_type: str = 'SUJ2',
                        temperature: float = 300.0,
                        **kwargs) -> MaterialLambda3Result:
    """物理ダメージ解析を含む完全解析の便利関数"""
    config = MaterialConfig(
        material_type=material_type,
        use_physical_damage=True,
        damage_temperature=temperature
    )
    
    detector = MaterialLambda3DetectorGPU(config)
    return detector.analyze(trajectory, atom_types=atom_types, **kwargs)

# ===============================
# Test Function
# ===============================

if __name__ == "__main__":
    print("💎 Material Lambda³ Detector GPU v3.0 - Physics Integration Test")
    print("=" * 70)
    
    # テストデータ生成
    n_frames = 100
    n_atoms = 1000
    trajectory = np.random.randn(n_frames, n_atoms, 3).astype(np.float32) * 0.1
    
    # クラスター定義
    cluster_atoms = {
        0: list(range(0, 900)),      # 健全領域
        1: list(range(900, 950)),    # 欠陥クラスター1
        2: list(range(950, 1000)),   # 欠陥クラスター2
    }
    
    # 物理ダメージ解析を有効にした設定
    config = MaterialConfig(
        material_type='SUJ2',
        use_physical_damage=True,
        damage_temperature=300.0
    )
    
    # 検出器の初期化と実行
    detector = MaterialLambda3DetectorGPU(config)
    result = detector.analyze(
        trajectory=trajectory,
        cluster_atoms=cluster_atoms
    )
    
    print("\n✨ Physical Damage Integration Complete!")
    
    if result.physical_damage:
        print("\n📊 Physical Damage Summary:")
        print(f"   Max Damage: {result.physical_damage['max_damage']:.1%}")
        print(f"   Failure Probability: {result.physical_damage['failure_probability']:.1%}")
        print(f"   Temperature: {result.physical_damage['temperature']:.1f} K")
    
    print("\n🎉 Test completed successfully!")
    
