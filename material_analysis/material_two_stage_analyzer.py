#!/usr/bin/env python3
"""
Material Two-Stage Lambda³ Analyzer GPU v2.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用2段階Lambda³解析のGPU実装
マクロレベル→クラスターレベルの階層的解析で転位・亀裂を追跡！💎

Version 2.0: 物理ベース破損予測（RMSF発散・相転移）を統合
水の沸点研究から生まれた理論による革新的な破損予測！

by 環ちゃん - Material Edition v2.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
import warnings

logger = logging.getLogger('lambda3_gpu.material_analysis')
warnings.filterwarnings('ignore')

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from concurrent.futures import ThreadPoolExecutor

# Lambda³ GPU imports
from lambda3_gpu.core.gpu_utils import GPUBackend
from lambda3_gpu.material.cluster_structures_gpu import ClusterStructuresGPU, ClusterStructureResult
from lambda3_gpu.material.cluster_network_gpu import ClusterNetworkGPU
from lambda3_gpu.material.cluster_causality_analysis_gpu import MaterialCausalityAnalyzerGPU
from lambda3_gpu.material.cluster_confidence_analysis_gpu import MaterialConfidenceAnalyzerGPU
from lambda3_gpu.material.material_failure_physics_gpu import (
    MaterialFailurePhysicsGPU, 
    FailurePhysicsResult
)
from lambda3_gpu.material_analysis.material_lambda3_detector import MaterialLambda3Result
from lambda3_gpu.types import ArrayType, NDArray

# ===============================
# Configuration
# ===============================

@dataclass
class ClusterAnalysisConfig:
    """クラスターレベル解析の設定（物理解析拡張版）"""
    # 解析パラメータ（材料用に調整）
    sensitivity: float = 0.3  # 材料は低感度
    correlation_threshold: float = 0.2  # 相関閾値
    sync_threshold: float = 0.3  # 同期閾値
    
    # ウィンドウパラメータ（材料は短い時間スケール）
    min_window: int = 10  # 最小ウィンドウ
    max_window: int = 100  # 最大ウィンドウ
    base_window: int = 30  # 基本ウィンドウ
    base_lag_window: int = 50  # ラグウィンドウ
    
    # ネットワーク制約
    max_causal_links: int = 200  # 因果リンク数上限
    min_causality_strength: float = 0.3  # 最小因果強度
    
    # 材料特有パラメータ
    detect_dislocations: bool = True  # 転位検出
    detect_cracks: bool = True  # 亀裂検出
    detect_phase_transitions: bool = True  # 相変態検出
    
    # 物理ベース解析（NEW!）
    use_physics_prediction: bool = True  # 物理予測を使用
    physics_confidence_threshold: float = 0.7  # 物理予測の信頼度閾値
    temperature: float = 300.0  # 基礎温度 (K)
    
    # Bootstrap パラメータ
    use_confidence: bool = True
    n_bootstrap: int = 30  # 材料は少なめ
    confidence_level: float = 0.95
    
    # GPU設定
    gpu_batch_clusters: int = 30  # クラスターバッチサイズ
    parallel_events: bool = True
    adaptive_window: bool = True
    
    # イベント固有設定（材料イベント）
    event_sensitivities: Dict[str, float] = field(default_factory=lambda: {
        'elastic_deformation': 0.2,
        'plastic_deformation': 0.4,
        'dislocation_nucleation': 0.5,
        'crack_initiation': 0.6,
        'crack_propagation': 0.5,
        'phase_transition': 0.7,
        'fatigue_damage': 0.4,
        'local_melting': 0.8  # NEW! RMSF発散による局所融解
    })
    
    event_windows: Dict[str, int] = field(default_factory=lambda: {
        'elastic_deformation': 20,
        'plastic_deformation': 50,
        'dislocation_nucleation': 30,
        'crack_initiation': 40,
        'crack_propagation': 30,
        'phase_transition': 100,
        'fatigue_damage': 80,
        'local_melting': 60  # NEW!
    })

@dataclass
class ClusterEvent:
    """クラスターレベルイベント（物理情報追加）"""
    cluster_id: int
    cluster_name: str
    start_frame: int
    end_frame: int
    peak_strain: float  # 最大歪み
    peak_damage: float  # 最大損傷度
    propagation_delay: int
    role: str  # 'initiator', 'mediator', 'responder'
    adaptive_window: int = 50
    
    # 材料特有フィールド
    dislocation_density: Optional[float] = None
    coordination_defect: Optional[float] = None
    von_mises_stress: Optional[float] = None
    
    # 物理情報（NEW!）
    rmsf_value: Optional[float] = None  # RMSF値
    lindemann_ratio: Optional[float] = None  # Lindemann比
    local_temperature: Optional[float] = None  # 局所温度
    phase_state: Optional[str] = None  # 'solid', 'liquid', 'transition'

@dataclass
class ClusterLevelAnalysis:
    """クラスターレベル解析結果（物理予測追加）"""
    event_name: str
    macro_start: int
    macro_end: int
    cluster_events: List[ClusterEvent] = field(default_factory=list)
    causality_chain: List[Tuple[int, int, float]] = field(default_factory=list)
    initiator_clusters: List[int] = field(default_factory=list)
    propagation_paths: List[List[int]] = field(default_factory=list)
    
    # 材料ネットワーク
    strain_network: List[Dict] = field(default_factory=list)
    dislocation_network: List[Dict] = field(default_factory=list)
    damage_network: List[Dict] = field(default_factory=list)
    
    network_stats: Dict = field(default_factory=dict)
    confidence_results: List[Any] = field(default_factory=list)
    
    # 材料特性
    failure_probability: float = 0.0
    reliability_index: float = 5.0
    critical_stress_intensity: Optional[float] = None
    
    # 物理予測（NEW!）
    physics_prediction: Optional[Dict] = None
    failure_mechanism: Optional[str] = None  # 破損メカニズム
    time_to_failure: Optional[float] = None  # 破損までの時間 (ps)
    critical_rmsf_atoms: Optional[List[int]] = None  # 臨界RMSF原子
    
    gpu_time: float = 0.0

@dataclass
class MaterialTwoStageResult:
    """材料2段階解析の統合結果（物理予測統合版）"""
    macro_result: MaterialLambda3Result
    cluster_analyses: Dict[str, ClusterLevelAnalysis]
    global_cluster_importance: Dict[int, float]
    critical_clusters: List[int]  # 破壊危険クラスター
    suggested_reinforcement_points: List[int]  # 補強推奨箇所
    global_network_stats: Dict
    material_state: Dict  # 材料状態（健全/損傷/破壊）
    
    # 統合物理予測（NEW!）
    global_physics_prediction: Optional[Dict] = None
    predicted_failure_time: Optional[float] = None  # システム全体の破損予測時間
    dominant_failure_mechanism: Optional[str] = None  # 支配的な破損メカニズム
    
    total_gpu_time: float = 0.0

# ===============================
# Material Two-Stage Analyzer v2
# ===============================

class MaterialTwoStageAnalyzerGPU(GPUBackend):
    """材料用GPU版2段階解析器（物理予測統合版）"""
    
    def __init__(self, config: ClusterAnalysisConfig = None, material_type: str = 'SUJ2'):
        super().__init__()
        self.config = config or ClusterAnalysisConfig()
        self.material_type = material_type
        
        # 材料プロパティ設定
        self._set_material_properties()
        
        # GPU版コンポーネント初期化（材料プロパティは渡さない）
        self.cluster_structures = ClusterStructuresGPU()
        self.cluster_network = ClusterNetworkGPU()
        self.causality_analyzer = MaterialCausalityAnalyzerGPU()
        self.confidence_analyzer = MaterialConfidenceAnalyzerGPU()
        
        # 材料プロパティを後から設定
        for component in [self.cluster_structures, self.cluster_network,
                         self.causality_analyzer, self.confidence_analyzer]:
            component.material_props = self.material_props
        
        # 物理ベース破損予測器を追加
        if self.config.use_physics_prediction:
            self.failure_physics = MaterialFailurePhysicsGPU(
                material_type=material_type
            )
            logger.info("💫 Physics-based failure prediction enabled!")
        else:
            self.failure_physics = None
        
        # メモリマネージャ共有
        components = [self.cluster_structures, self.cluster_network,
                     self.causality_analyzer, self.confidence_analyzer]
        if self.failure_physics:
            components.append(self.failure_physics)
            
        for component in components:
            if hasattr(component, 'memory_manager'):
                component.memory_manager = self.memory_manager
            if hasattr(component, 'device'):
                component.device = self.device
    
    def _set_material_properties(self):
        """材料プロパティ設定（物理定数追加）"""
        if self.material_type == 'SUJ2':
            self.material_props = {
                'elastic_modulus': 210.0,  # GPa
                'yield_strength': 1.5,
                'ultimate_strength': 2.0,
                'fatigue_strength': 0.7,
                'fracture_toughness': 30.0,
                'poisson_ratio': 0.3,
                # 物理定数（NEW!）
                'lattice_constant': 2.87,  # Å
                'melting_temp': 1811,  # K
                'lindemann_criterion': 0.10
            }
        elif self.material_type == 'AL7075':
            self.material_props = {
                'elastic_modulus': 71.7,
                'yield_strength': 0.503,
                'ultimate_strength': 0.572,
                'fatigue_strength': 0.159,
                'fracture_toughness': 23.0,
                'poisson_ratio': 0.33,
                # 物理定数（NEW!）
                'lattice_constant': 4.05,
                'melting_temp': 933,
                'lindemann_criterion': 0.12
            }
        else:
            # デフォルト（鋼鉄）
            self.material_props = {
                'elastic_modulus': 210.0,
                'yield_strength': 1.0,
                'ultimate_strength': 1.5,
                'fatigue_strength': 0.5,
                'fracture_toughness': 20.0,
                'poisson_ratio': 0.3,
                'lattice_constant': 2.87,
                'melting_temp': 1800,
                'lindemann_criterion': 0.10
            }
    
    def analyze_trajectory(self,
                          trajectory: np.ndarray,
                          macro_result: MaterialLambda3Result,
                          detected_events: List[Tuple[int, int, str]],
                          cluster_atoms: Dict[int, List[int]],
                          atom_types: np.ndarray,
                          stress_history: Optional[np.ndarray] = None) -> MaterialTwoStageResult:
        """
        材料2段階Lambda³解析の実行（物理予測拡張版）
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子トラジェクトリ (n_frames, n_atoms, 3)
        macro_result : MaterialLambda3Result
            マクロレベル解析結果
        detected_events : List[Tuple[int, int, str]]
            検出イベントリスト [(start, end, event_type), ...]
        cluster_atoms : Dict[int, List[int]]
            クラスター定義
        atom_types : np.ndarray
            原子タイプ配列
        stress_history : np.ndarray, optional
            応力履歴（物理予測用）
            
        Returns
        -------
        MaterialTwoStageResult
            統合解析結果
        """
        start_time = time.time()
        
        n_clusters = len(cluster_atoms)
        
        print("\n" + "="*60)
        print("=== Material Two-Stage Lambda³ Analysis v2.0 (GPU) ===")
        print("="*60)
        print(f"Material: {self.material_type}")
        print(f"Events to analyze: {len(detected_events)}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Physics prediction: {'ENABLED ✨' if self.config.use_physics_prediction else 'DISABLED'}")
        print(f"GPU Device: {self.device}")
        
        # 入力検証
        if len(trajectory.shape) != 3:
            raise ValueError(f"Expected 3D trajectory, got shape {trajectory.shape}")
        
        # クラスター名設定
        cluster_names = {i: f"CLUSTER_{i}" for i in range(n_clusters)}
        
        # グローバル物理予測（NEW!）
        global_physics = None
        if self.config.use_physics_prediction and self.failure_physics:
            print("\n🔬 Running global physics-based failure analysis...")
            global_physics = self._run_global_physics_prediction(
                trajectory, stress_history
            )
            if global_physics:
                print(f"   Mechanism: {global_physics.failure_mechanism}")
                print(f"   Time to failure: {global_physics.time_to_failure:.1f} ps")
                print(f"   Confidence: {global_physics.confidence:.1%}")
        
        # 各イベントを解析
        cluster_analyses = {}
        all_important_clusters = {}
        
        # 並列処理
        if self.config.parallel_events and len(detected_events) > 1:
            cluster_analyses = self._analyze_events_parallel(
                trajectory, detected_events, cluster_atoms,
                cluster_names, atom_types, stress_history
            )
        else:
            cluster_analyses = self._analyze_events_sequential(
                trajectory, detected_events, cluster_atoms,
                cluster_names, atom_types, stress_history
            )
        
        # グローバル重要度計算（物理情報を考慮）
        for event_name, analysis in cluster_analyses.items():
            for event in analysis.cluster_events:
                cluster_id = event.cluster_id
                if cluster_id not in all_important_clusters:
                    all_important_clusters[cluster_id] = 0.0
                
                # 材料重要度スコア（歪みと損傷を考慮）
                importance = (
                    event.peak_strain * self.material_props['elastic_modulus'] +
                    event.peak_damage * 10.0
                )
                
                # 物理ベースの重要度補正（NEW!）
                if event.lindemann_ratio and event.lindemann_ratio > 0.1:
                    importance *= (1 + event.lindemann_ratio)
                
                all_important_clusters[cluster_id] += importance
        
        # 臨界クラスター特定（物理情報統合）
        critical_clusters = self._identify_critical_clusters_physics(
            all_important_clusters, cluster_analyses, global_physics
        )
        
        # 補強推奨箇所
        reinforcement_points = self._identify_reinforcement_points(
            all_important_clusters, cluster_analyses
        )
        
        # グローバル統計
        global_stats = self._compute_global_stats(cluster_analyses)
        
        # 材料状態評価（物理情報統合）
        material_state = self._evaluate_material_state_physics(
            cluster_analyses, critical_clusters, global_physics
        )
        
        # 統合物理予測（NEW!）
        global_physics_dict = None
        predicted_failure_time = None
        dominant_mechanism = None
        
        if global_physics:
            global_physics_dict = {
                'mechanism': global_physics.failure_mechanism,
                'time_to_failure': global_physics.time_to_failure,
                'critical_atoms': global_physics.failure_location,
                'lindemann_ratio': global_physics.rmsf_analysis.lindemann_ratio,
                'phase_state': global_physics.energy_balance.phase_state,
                'confidence': global_physics.confidence
            }
            predicted_failure_time = global_physics.time_to_failure
            dominant_mechanism = global_physics.failure_mechanism
        
        # 結果サマリー
        self._print_summary_physics(
            all_important_clusters, critical_clusters,
            reinforcement_points, material_state,
            dominant_mechanism, predicted_failure_time
        )
        
        total_time = time.time() - start_time
        
        return MaterialTwoStageResult(
            macro_result=macro_result,
            cluster_analyses=cluster_analyses,
            global_cluster_importance=all_important_clusters,
            critical_clusters=critical_clusters,
            suggested_reinforcement_points=reinforcement_points,
            global_network_stats=global_stats,
            material_state=material_state,
            global_physics_prediction=global_physics_dict,
            predicted_failure_time=predicted_failure_time,
            dominant_failure_mechanism=dominant_mechanism,
            total_gpu_time=total_time
        )
    
    def _run_global_physics_prediction(self,
                                      trajectory: np.ndarray,
                                      stress_history: Optional[np.ndarray]) -> Optional[FailurePhysicsResult]:
        """グローバル物理予測の実行"""
        if not self.failure_physics:
            return None
        
        try:
            result = self.failure_physics.analyze_failure_physics(
                trajectory=trajectory,
                stress_history=stress_history,
                temperature=self.config.temperature,
                loading_cycles=None  # TODO: サイクル情報があれば追加
            )
            return result
        except Exception as e:
            logger.warning(f"Global physics prediction failed: {e}")
            return None
    
    def _analyze_single_event_gpu(self,
                                 trajectory: np.ndarray,
                                 event_name: str,
                                 start_frame: int,
                                 end_frame: int,
                                 cluster_atoms: Dict[int, List[int]],
                                 cluster_names: Dict[int, str],
                                 atom_types: np.ndarray,
                                 stress_history: Optional[np.ndarray] = None) -> ClusterLevelAnalysis:
        """単一イベントのGPU解析（物理予測追加）"""
        event_start_time = time.time()
        
        # フレーム範囲検証
        if end_frame <= start_frame:
            end_frame = min(start_frame + 1, trajectory.shape[0])
        
        event_frames = end_frame - start_frame
        event_trajectory = trajectory[start_frame:end_frame]
        
        with self.memory_manager.batch_context(event_frames * len(cluster_atoms) * 3 * 4):
            
            # 1. クラスター構造計算
            structures = self.cluster_structures.compute_cluster_structures(
                event_trajectory, 0, event_frames - 1,
                cluster_atoms, atom_types,
                window_size=self.config.base_window
            )
            
            # 2. イベント物理予測（NEW!）
            event_physics = None
            physics_dict = None
            if self.config.use_physics_prediction and self.failure_physics:
                event_stress = None
                if stress_history is not None:
                    event_stress = stress_history[start_frame:end_frame]
                
                try:
                    event_physics = self.failure_physics.analyze_failure_physics(
                        trajectory=event_trajectory,
                        stress_history=event_stress,
                        temperature=self.config.temperature
                    )
                    
                    physics_dict = {
                        'mechanism': event_physics.failure_mechanism,
                        'time_to_failure': event_physics.time_to_failure,
                        'lindemann_ratio': event_physics.rmsf_analysis.lindemann_ratio,
                        'critical_atoms': event_physics.failure_location[:10],
                        'phase_state': event_physics.energy_balance.phase_state,
                        'confidence': event_physics.confidence
                    }
                    
                    # 高信頼度なら警告
                    if event_physics.confidence > self.config.physics_confidence_threshold:
                        logger.warning(f"⚠️ High-confidence failure prediction for {event_name}: "
                                     f"{event_physics.failure_mechanism}")
                        
                except Exception as e:
                    logger.debug(f"Event physics prediction failed: {e}")
            
            # 3. 異常検出
            anomaly_scores = self._detect_cluster_anomalies_gpu(
                structures, event_name
            )
            
            # 4. ネットワーク解析
            network_results = self.cluster_network.analyze_network(
                anomaly_scores,
                structures.cluster_coupling,
                structures.cluster_centers,
                structures.coordination_numbers,
                structures.local_strain
            )
            
            # 5. クラスターイベント構築（物理情報追加）
            cluster_events = self._build_cluster_events_physics(
                structures, anomaly_scores, cluster_names,
                start_frame, network_results, event_physics,
                cluster_atoms
            )
            
            # 6. 因果解析（従来通り）
            causality_chains = []
            initiators = []
            propagation_paths = []
            
            if self.config.detect_dislocations and network_results.dislocation_network:
                # 転位伝播解析
                causality_result = self.causality_analyzer.detect_causal_pairs(
                    anomaly_scores,
                    structures.coordination_numbers,
                    structures.local_strain,
                    threshold=self.config.min_causality_strength
                )
                
                causality_chains = [
                    (r.pair[0], r.pair[1], r.causality_strength)
                    for r in causality_result[:self.config.max_causal_links]
                ]
                
                initiators = self._find_dislocation_sources(
                    cluster_events, causality_chains
                )
                
                propagation_paths = self._trace_dislocation_paths(
                    initiators, causality_chains
                )
            
            # 7. 信頼性解析
            confidence_results = []
            if self.config.use_confidence and causality_chains:
                cluster_data = {
                    i: {
                        'strain': structures.local_strain[:, i],
                        'coordination': structures.coordination_numbers[:, i],
                        'anomaly': scores
                    }
                    for i, scores in anomaly_scores.items()
                }
                
                confidence_results = self.confidence_analyzer.analyze_material_reliability(
                    causality_chains[:10],
                    cluster_data,
                    analysis_type='strain',
                    n_bootstrap=self.config.n_bootstrap
                )
            
            # 8. 破壊確率計算
            failure_prob = self._compute_failure_probability(
                structures.local_strain
            )
            
            # 9. 信頼性指標
            reliability_idx = self._compute_reliability_index(
                structures.local_strain
            )
        
        gpu_time = time.time() - event_start_time
        
        # 結果構築
        network_stats = network_results.network_stats.copy()
        network_stats.update({
            'n_strain_links': len(network_results.strain_network),
            'n_dislocation_links': len(network_results.dislocation_network),
            'n_damage_links': len(network_results.damage_network),
            'mean_coordination': float(np.mean(structures.coordination_numbers))
        })
        
        # 物理予測から追加情報
        failure_mechanism = event_physics.failure_mechanism if event_physics else None
        time_to_failure = event_physics.time_to_failure if event_physics else None
        critical_rmsf_atoms = event_physics.failure_location[:10] if event_physics else None
        
        return ClusterLevelAnalysis(
            event_name=event_name,
            macro_start=start_frame,
            macro_end=end_frame,
            cluster_events=cluster_events,
            causality_chain=causality_chains,
            initiator_clusters=initiators,
            propagation_paths=propagation_paths[:5],
            strain_network=network_results.strain_network,
            dislocation_network=network_results.dislocation_network,
            damage_network=network_results.damage_network,
            network_stats=network_stats,
            confidence_results=confidence_results,
            failure_probability=failure_prob,
            reliability_index=reliability_idx,
            physics_prediction=physics_dict,
            failure_mechanism=failure_mechanism,
            time_to_failure=time_to_failure,
            critical_rmsf_atoms=critical_rmsf_atoms,
            gpu_time=gpu_time
        )
    
    def _build_cluster_events_physics(self,
                                     structures: ClusterStructureResult,
                                     anomaly_scores: Dict[int, np.ndarray],
                                     cluster_names: Dict[int, str],
                                     start_frame: int,
                                     network_results,
                                     event_physics: Optional[FailurePhysicsResult],
                                     cluster_atoms: Dict[int, List[int]] = None) -> List[ClusterEvent]:  # 追加！
        """クラスターイベント構築（物理情報追加）"""
        events = []
        
        # 物理情報の準備
        rmsf_field = None
        local_temps = None
        if event_physics:
            rmsf_field = event_physics.rmsf_analysis.rmsf_field
            local_temps = event_physics.energy_balance.local_temperature
        
        for cluster_id, scores in anomaly_scores.items():
            # 基本情報（従来通り）
            peak_idx = np.argmax(scores)
            peak_score = scores[peak_idx]
            
            strain_values = structures.local_strain[:, cluster_id]
            von_mises_strain = np.sqrt(np.sum(strain_values**2, axis=-1))
            peak_strain = float(np.max(von_mises_strain))
            
            peak_damage = peak_strain * self.material_props['elastic_modulus'] / \
                         self.material_props['ultimate_strength']
            
            coord_defect = np.abs(structures.coordination_numbers[:, cluster_id] - 12.0)
            dislocation_density = 1e14 * np.mean(coord_defect)**2
            
            role = self._determine_cluster_role(cluster_id, network_results)
            
            # 物理情報の追加（NEW!）
            rmsf_value = None
            lindemann_ratio = None
            local_temperature = None
            phase_state = None
            
            if event_physics and cluster_id < len(cluster_atoms):
                # クラスターの代表原子のRMSF
                cluster_atom_indices = list(cluster_atoms[cluster_id])[:10]
                if rmsf_field is not None and len(cluster_atom_indices) > 0:
                    # 最新時間窓のRMSF
                    last_window_rmsf = rmsf_field[-1] if len(rmsf_field) > 0 else None
                    if last_window_rmsf is not None:
                        cluster_rmsf = np.mean([last_window_rmsf[i] for i in cluster_atom_indices 
                                               if i < len(last_window_rmsf)])
                        rmsf_value = float(cluster_rmsf)
                        lindemann_ratio = rmsf_value / self.material_props['lattice_constant']
                
                # 局所温度
                if local_temps is not None and cluster_id < local_temps.shape[1]:
                    local_temperature = float(np.mean(local_temps[:, cluster_id]))
                    
                    # 相状態判定
                    melting_temp = self.material_props['melting_temp']
                    if local_temperature < 0.8 * melting_temp:
                        phase_state = 'solid'
                    elif local_temperature > 0.95 * melting_temp:
                        phase_state = 'liquid'
                    else:
                        phase_state = 'transition'
            
            event = ClusterEvent(
                cluster_id=cluster_id,
                cluster_name=cluster_names.get(cluster_id, f"C{cluster_id}"),
                start_frame=start_frame + peak_idx,
                end_frame=start_frame + min(peak_idx + 10, len(scores) - 1),
                peak_strain=peak_strain,
                peak_damage=float(peak_damage),
                propagation_delay=peak_idx,
                role=role,
                adaptive_window=self.config.base_window,
                dislocation_density=float(dislocation_density),
                coordination_defect=float(np.mean(coord_defect)),
                von_mises_stress=peak_strain * self.material_props['elastic_modulus'],
                # 物理情報
                rmsf_value=rmsf_value,
                lindemann_ratio=lindemann_ratio,
                local_temperature=local_temperature,
                phase_state=phase_state
            )
            
            events.append(event)
        
        return events
    
    def _identify_critical_clusters_physics(self,
                                       importance_scores: Dict[int, float],
                                       cluster_analyses: Dict,
                                       global_physics: Optional[FailurePhysicsResult]) -> List[int]:
        """臨界クラスター特定（物理情報統合）"""
        critical = []
        
        # 従来の方法（そのまま）
        if importance_scores:
            sorted_clusters = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for cluster_id, score in sorted_clusters[:10]:
                for analysis in cluster_analyses.values():
                    for event in analysis.cluster_events:
                        if event.cluster_id == cluster_id:
                            # 損傷基準
                            if event.peak_damage > 0.7:
                                critical.append(cluster_id)
                                break
                            # Lindemann基準（NEW!）
                            if event.lindemann_ratio and event.lindemann_ratio > 0.1:
                                critical.append(cluster_id)
                                break
        
        # 物理予測からの臨界原子（修正！）
        if global_physics and global_physics.failure_location and importance_scores:  # ← ここ追加
            for atom_id in global_physics.failure_location[:5]:
                estimated_cluster = atom_id % max(len(importance_scores), 1)  # ← ゼロ除算防止
                if estimated_cluster not in critical:
                    critical.append(estimated_cluster)
        
        return list(set(critical))[:15]
        
    def _evaluate_material_state_physics(self,
                                        cluster_analyses: Dict,
                                        critical_clusters: List,
                                        global_physics: Optional[FailurePhysicsResult]) -> Dict:
        """材料状態評価（物理情報統合）"""
        # 従来の評価
        max_damage = 0.0
        mean_strain = 0.0
        n_events = 0
        max_lindemann = 0.0  # NEW!
        
        for analysis in cluster_analyses.values():
            for event in analysis.cluster_events:
                max_damage = max(max_damage, event.peak_damage)
                mean_strain += event.peak_strain
                n_events += 1
                
                # Lindemann比の最大値（NEW!）
                if event.lindemann_ratio:
                    max_lindemann = max(max_lindemann, event.lindemann_ratio)
        
        if n_events > 0:
            mean_strain /= n_events
        
        # 物理ベースの状態判定（NEW!）
        if global_physics:
            phase = global_physics.energy_balance.phase_state
            if phase == 'liquid':
                state = "failed_melting"
                health = 0.0
            elif phase == 'transition':
                state = "critical_transition"
                health = 0.3
            elif max_lindemann > 0.1:
                state = "incipient_melting"
                health = 0.4
            elif max_damage > 0.8:
                state = "critical_damage"
                health = 0.2
            elif max_damage > 0.5:
                state = "moderate_damage"
                health = 0.5
            elif max_damage > 0.2:
                state = "minor_damage"
                health = 0.8
            else:
                state = "healthy"
                health = 1.0
        else:
            # 従来の判定
            if max_damage > 0.8:
                state = "critical_damage"
                health = 0.2
            elif max_damage > 0.5:
                state = "moderate_damage"
                health = 0.5
            elif max_damage > 0.2:
                state = "minor_damage"
                health = 0.8
            else:
                state = "healthy"
                health = 1.0
        
        return {
            'state': state,
            'health_index': health,
            'max_damage': float(max_damage),
            'mean_strain': float(mean_strain),
            'max_lindemann_ratio': float(max_lindemann),  # NEW!
            'n_critical_clusters': len(critical_clusters)
        }
    
    def _print_summary_physics(self,
                              importance_scores: Dict,
                              critical_clusters: List,
                              reinforcement_points: List,
                              material_state: Dict,
                              dominant_mechanism: Optional[str],
                              predicted_failure_time: Optional[float]):
        """解析サマリー表示（物理予測追加）"""
        print("\n💎 Material Analysis Complete! (Physics-Enhanced)")
        print(f"   Material state: {material_state['state']}")
        print(f"   Health index: {material_state['health_index']:.1%}")
        print(f"   Key clusters identified: {len(importance_scores)}")
        print(f"   Critical clusters: {len(critical_clusters)}")
        print(f"   Reinforcement points: {len(reinforcement_points)}")
        print(f"   Max damage: {material_state['max_damage']:.1%}")
        print(f"   Mean strain: {material_state['mean_strain']:.3f}")
        
        # 物理予測情報（NEW!）
        if 'max_lindemann_ratio' in material_state:
            print(f"   Max Lindemann ratio: {material_state['max_lindemann_ratio']:.3f}")
        if dominant_mechanism:
            print(f"   🔬 Predicted failure mechanism: {dominant_mechanism}")
        if predicted_failure_time and predicted_failure_time < float('inf'):
            print(f"   ⏰ Time to failure: {predicted_failure_time:.1f} ps")
        
    def _analyze_events_parallel(self,
                            trajectory: np.ndarray,
                            detected_events: List[Tuple[int, int, str]],
                            cluster_atoms: Dict[int, List[int]],
                            cluster_names: Dict[int, str],
                            atom_types: np.ndarray,
                            stress_history: Optional[np.ndarray] = None) -> Dict:
        """イベントの並列解析"""
        print("\n⚡ Processing material events in parallel on GPU...")
        
        cluster_analyses = {}
        
        with ThreadPoolExecutor(max_workers=min(4, len(detected_events))) as executor:
            futures = []
            
            for start, end, event_name in detected_events:
                future = executor.submit(
                    self._analyze_single_event_gpu,
                    trajectory, event_name, start, end,
                    cluster_atoms, cluster_names, atom_types, stress_history
                )
                futures.append((event_name, future))
            
            for event_name, future in futures:
                try:
                    analysis = future.result()
                    cluster_analyses[event_name] = analysis
                    print(f"  ✓ {event_name} complete (GPU time: {analysis.gpu_time:.2f}s)")
                except Exception as e:
                    logger.error(f"  ✗ {event_name} failed: {str(e)}")
        
        return cluster_analyses
    
    def _analyze_events_sequential(self,
                              trajectory: np.ndarray,
                              detected_events: List[Tuple[int, int, str]],
                              cluster_atoms: Dict[int, List[int]],
                              cluster_names: Dict[int, str],
                              atom_types: np.ndarray,
                              stress_history: Optional[np.ndarray] = None) -> Dict[str, ClusterLevelAnalysis]:
        """イベントの逐次解析"""
        print("\n⚙️ Processing material events sequentially on GPU...")
        
        cluster_analyses = {}
        
        for start, end, event_name in detected_events:
            print(f"\n  → Analyzing {event_name}...")
            analysis = self._analyze_single_event_gpu(
                trajectory, event_name, start, end,
                cluster_atoms, cluster_names, atom_types,
                stress_history  # ← 追加！
            )
            cluster_analyses[event_name] = analysis
            print(f"    GPU time: {analysis.gpu_time:.2f}s")
        
        return cluster_analyses
      
    def _detect_cluster_anomalies_gpu(self,
                                 structures: ClusterStructureResult,
                                 event_type: str) -> Dict[int, np.ndarray]:
        """クラスター異常検出"""
        n_frames, n_clusters = structures.cluster_rho_t.shape
        
        sensitivity = self.config.event_sensitivities.get(
            event_type, self.config.sensitivity
        )
        
        cluster_anomaly_scores = {}
        
        # 各クラスターの異常スコア計算
        for cluster_id in range(n_clusters):
            # 歪み異常
            strain_values = structures.local_strain[:, cluster_id]  # (n_frames, 3, 3)
            # ↓ ここを修正！
            von_mises = np.sqrt(np.sum(strain_values**2, axis=(1,2)))  # axis=(1,2)に変更
            strain_anomaly = self._compute_anomaly_score(von_mises)
            
            # 配位数異常（そのまま）
            coord_values = structures.coordination_numbers[:, cluster_id]
            ideal_coord = 12.0 if hasattr(structures, 'crystal_structure') else 8.0
            coord_anomaly = np.abs(coord_values - ideal_coord) / ideal_coord
            
            # 統合スコア
            combined = (strain_anomaly + coord_anomaly) / 2
            
            if np.max(combined) > sensitivity:
                cluster_anomaly_scores[cluster_id] = combined
        
        # 最低限のクラスターを確保
        if len(cluster_anomaly_scores) < min(5, n_clusters):
            for cluster_id in range(min(5, n_clusters)):
                if cluster_id not in cluster_anomaly_scores:
                    cluster_anomaly_scores[cluster_id] = np.ones(n_frames) * 0.5
        
        return cluster_anomaly_scores
    
    def _determine_cluster_role(self, cluster_id: int, network_results) -> str:
        """クラスターの役割決定"""
        # ハブクラスター判定
        if hasattr(network_results, 'network_stats'):
            stats = network_results.network_stats
            if 'hub_clusters' in stats:
                hub_ids = [h[0] for h in stats['hub_clusters']]
                if cluster_id in hub_ids:
                    return "initiator"
        
        # ネットワークでの役割
        if hasattr(network_results, 'critical_clusters'):
            if cluster_id in network_results.critical_clusters:
                return "critical"
        
        return "participant"
    
    def _find_dislocation_sources(self,
                                 cluster_events: List[ClusterEvent],
                                 causality_chains: List) -> List[int]:
        """転位発生源の特定"""
        sources = []
        
        # 高転位密度クラスター
        for event in cluster_events:
            if event.dislocation_density and event.dislocation_density > 1e12:
                sources.append(event.cluster_id)
        
        # 因果ネットワークのソース
        if causality_chains:
            out_degree = {}
            for from_c, to_c, _ in causality_chains:
                out_degree[from_c] = out_degree.get(from_c, 0) + 1
            
            top_sources = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:3]
            for cluster_id, _ in top_sources:
                if cluster_id not in sources:
                    sources.append(cluster_id)
        
        return sources[:5]
    
    def _trace_dislocation_paths(self,
                                sources: List[int],
                                causality_chains: List) -> List[List[int]]:
        """転位伝播経路の追跡"""
        paths = []
        
        # グラフ構築
        graph = {}
        for from_c, to_c, weight in causality_chains:
            if from_c not in graph:
                graph[from_c] = []
            graph[from_c].append((to_c, weight))
        
        # 各ソースから探索
        for source in sources:
            path = self._dfs_path(source, graph, max_depth=5)
            if len(path) > 1:
                paths.append(path)
        
        return paths
    
    def _find_crack_initiation_sites(self,
                                    cluster_events: List[ClusterEvent],
                                    damage_network: List) -> List[int]:
        """亀裂開始点の特定"""
        sites = []
        
        # 高損傷クラスター
        for event in cluster_events:
            if event.peak_damage > 0.5:  # 50%損傷
                sites.append(event.cluster_id)
        
        return sites[:5]
    
    def _trace_crack_paths(self,
                         initiation_sites: List[int],
                         damage_network: List) -> List[List[int]]:
        """亀裂伝播経路の追跡"""
        paths = []
        
        # ネットワークから経路構築
        for site in initiation_sites:
            path = [site]
            current = site
            
            for link in damage_network:
                if hasattr(link, 'from_cluster') and link.from_cluster == current:
                    path.append(link.to_cluster)
                    current = link.to_cluster
                    
                    if len(path) >= 5:
                        break
            
            if len(path) > 1:
                paths.append(path)
        
        return paths
    
    def _dfs_path(self, start: int, graph: Dict, max_depth: int = 5) -> List[int]:
        """深さ優先探索で経路取得"""
        path = [start]
        current = start
        visited = {start}
        
        for _ in range(max_depth):
            if current not in graph:
                break
            
            # 最大重みの隣接ノード選択
            neighbors = [(n, w) for n, w in graph[current] if n not in visited]
            if not neighbors:
                break
            
            next_node = max(neighbors, key=lambda x: x[1])[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return path
    
    def _compute_anomaly_score(self, values: np.ndarray) -> np.ndarray:
        """異常スコア計算"""
        mean = np.mean(values)
        std = np.std(values) + 1e-10
        return np.abs(values - mean) / std
    
    def _compute_failure_probability(self, strain_tensor: np.ndarray) -> float:
        """破壊確率計算"""
        if strain_tensor is None or strain_tensor.size == 0:
            return 0.0
        
        # von Mises応力
        von_mises_strain = np.mean(np.abs(strain_tensor))
        von_mises_stress = von_mises_strain * self.material_props['elastic_modulus']
        
        # 破壊確率（簡易版）
        if von_mises_stress > self.material_props['yield_strength']:
            ratio = von_mises_stress / self.material_props['ultimate_strength']
            return min(1.0, ratio)
        
        return 0.0
    
    def _compute_reliability_index(self, strain_tensor: np.ndarray) -> float:
        """信頼性指標計算"""
        if strain_tensor is None or strain_tensor.size == 0:
            return 5.0
        
        mean_strain = np.mean(np.abs(strain_tensor))
        std_strain = np.std(np.abs(strain_tensor))
        critical_strain = self.material_props['yield_strength'] / self.material_props['elastic_modulus']
        
        if std_strain > 0:
            beta = (critical_strain - mean_strain) / std_strain
            return float(beta)
        
        return 5.0
    
    def _identify_critical_clusters(self,
                                   importance_scores: Dict[int, float],
                                   cluster_analyses: Dict) -> List[int]:
        """臨界クラスター特定"""
        critical = []
        
        # 高重要度クラスター
        if importance_scores:
            sorted_clusters = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for cluster_id, score in sorted_clusters[:10]:
                # 破壊確率チェック
                for analysis in cluster_analyses.values():
                    for event in analysis.cluster_events:
                        if event.cluster_id == cluster_id:
                            if event.peak_damage > 0.7:  # 70%損傷
                                critical.append(cluster_id)
                                break
        
        return list(set(critical))
    
    def _identify_reinforcement_points(self,
                                      importance_scores: Dict[int, float],
                                      cluster_analyses: Dict) -> List[int]:
        """補強推奨箇所の特定"""
        reinforcement = []
        
        # 中程度の重要度で損傷初期のクラスター
        if importance_scores:
            for cluster_id, score in importance_scores.items():
                for analysis in cluster_analyses.values():
                    for event in analysis.cluster_events:
                        if event.cluster_id == cluster_id:
                            if 0.3 < event.peak_damage < 0.7:  # 30-70%損傷
                                reinforcement.append(cluster_id)
                                break
        
        return list(set(reinforcement))[:10]
    
    def _evaluate_material_state(self,
                                cluster_analyses: Dict,
                                critical_clusters: List) -> Dict:
        """材料状態評価"""
        max_damage = 0.0
        mean_strain = 0.0
        n_events = 0
        
        for analysis in cluster_analyses.values():
            for event in analysis.cluster_events:
                max_damage = max(max_damage, event.peak_damage)
                mean_strain += event.peak_strain
                n_events += 1
        
        if n_events > 0:
            mean_strain /= n_events
        
        # 状態判定
        if max_damage > 0.8:
            state = "critical_damage"
            health = 0.2
        elif max_damage > 0.5:
            state = "moderate_damage"
            health = 0.5
        elif max_damage > 0.2:
            state = "minor_damage"
            health = 0.8
        else:
            state = "healthy"
            health = 1.0
        
        return {
            'state': state,
            'health_index': health,
            'max_damage': float(max_damage),
            'mean_strain': float(mean_strain),
            'n_critical_clusters': len(critical_clusters)
        }
    
    def _compute_global_stats(self, cluster_analyses: Dict) -> Dict:
        """グローバル統計計算"""
        total_strain_links = sum(
            len(a.strain_network) for a in cluster_analyses.values()
        )
        total_dislocation_links = sum(
            len(a.dislocation_network) for a in cluster_analyses.values()
        )
        total_damage_links = sum(
            len(a.damage_network) for a in cluster_analyses.values()
        )
        
        max_failure_prob = max(
            (a.failure_probability for a in cluster_analyses.values()),
            default=0.0
        )
        
        min_reliability = min(
            (a.reliability_index for a in cluster_analyses.values()),
            default=5.0
        )
        
        total_gpu_time = sum(a.gpu_time for a in cluster_analyses.values())
        
        return {
            'total_strain_links': total_strain_links,
            'total_dislocation_links': total_dislocation_links,
            'total_damage_links': total_damage_links,
            'max_failure_probability': max_failure_prob,
            'min_reliability_index': min_reliability,
            'total_gpu_time': total_gpu_time,
            'events_analyzed': len(cluster_analyses)
        }
    
    def _print_summary(self,
                      importance_scores: Dict,
                      critical_clusters: List,
                      reinforcement_points: List,
                      material_state: Dict):
        """解析サマリー表示"""
        print("\n💎 Material Analysis Complete!")
        print(f"   Material state: {material_state['state']}")
        print(f"   Health index: {material_state['health_index']:.1%}")
        print(f"   Key clusters identified: {len(importance_scores)}")
        print(f"   Critical clusters: {len(critical_clusters)}")
        print(f"   Reinforcement points: {len(reinforcement_points)}")
        print(f"   Max damage: {material_state['max_damage']:.1%}")
        print(f"   Mean strain: {material_state['mean_strain']:.3f}")

# ===============================
# Utility Functions
# ===============================

def perform_material_two_stage_analysis_gpu(
    trajectory: np.ndarray,
    macro_result: MaterialLambda3Result,
    detected_events: List[Tuple[int, int, str]],
    cluster_atoms: Dict[int, List[int]],
    atom_types: np.ndarray,
    material_type: str = 'SUJ2',
    config: ClusterAnalysisConfig = None
) -> MaterialTwoStageResult:
    """
    材料2段階解析の便利なラッパー関数
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ
    macro_result : MaterialLambda3Result
        マクロ解析結果
    detected_events : List[Tuple[int, int, str]]
        検出イベント
    cluster_atoms : Dict[int, List[int]]
        クラスター定義
    atom_types : np.ndarray
        原子タイプ
    material_type : str
        材料タイプ ('SUJ2', 'AL7075', etc.)
    config : ClusterAnalysisConfig
        設定
    """
    analyzer = MaterialTwoStageAnalyzerGPU(config, material_type)
    return analyzer.analyze_trajectory(
        trajectory, macro_result, detected_events,
        cluster_atoms, atom_types
    )
