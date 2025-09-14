#!/usr/bin/env python3
"""
Material Lambda³ Detector GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用Lambda³検出器のGPU実装
金属・セラミックス・ポリマーの疲労・破壊を高速検出！💎

MD版をベースに材料クラスター解析に特化

by 環ちゃん - Material Edition v1.0
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import logging

logger = logging.getLogger('lambda3_gpu.material_analysis')

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

# Lambda³ GPU imports
from ..core.gpu_utils import GPUBackend
from ..material.cluster_structures_gpu import ClusterStructuresGPU
from ..material.cluster_network_gpu import ClusterNetworkGPU
from ..material.cluster_causality_analysis_gpu import MaterialCausalityAnalyzerGPU
from ..material.cluster_confidence_analysis_gpu import MaterialConfidenceAnalyzerGPU

# 検出モジュール（MD版から流用）
from ..detection.anomaly_detection_gpu import AnomalyDetectorGPU
from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
from ..detection.topology_breaks_gpu import TopologyBreaksDetectorGPU

# ===============================
# Configuration
# ===============================

@dataclass
class MaterialConfig:
    """材料解析設定"""
    # Lambda³パラメータ
    adaptive_window: bool = True
    sensitivity: float = 1.5  # 材料は低めに
    min_boundary_gap: int = 50  # 材料は短い時間スケール
    
    # 材料特有の設定
    use_coordination: bool = True  # 配位数解析
    use_strain: bool = True  # 歪み解析
    use_damage: bool = True  # 損傷解析
    detect_dislocations: bool = True  # 転位検出
    detect_cracks: bool = True  # 亀裂検出
    
    # 材料パラメータ
    material_type: str = 'SUJ2'  # 'SUJ2', 'AL7075', 'TI6AL4V'
    crystal_structure: str = 'BCC'  # 'BCC', 'FCC', 'HCP'
    cutoff_distance: float = 3.0  # Å
    strain_threshold: float = 0.01  # 1%歪み
    damage_threshold: float = 0.5  # 50%損傷
    
    # GPU設定
    gpu_batch_size: int = 5000  # 材料は小さめ
    mixed_precision: bool = False
    benchmark_mode: bool = False
    
    # 異常検出の重み
    w_strain: float = 0.4  # 歪み異常の重み
    w_coordination: float = 0.3  # 配位数異常の重み
    w_damage: float = 0.3  # 損傷異常の重み

@dataclass
class MaterialLambda3Result:
    """材料Lambda³解析結果"""
    # Core Lambda³構造
    cluster_structures: Dict[str, np.ndarray]  # ClusterStructureResult
    structural_boundaries: Dict[str, Any]
    topological_breaks: Dict[str, np.ndarray]
    
    # 材料特有の特徴
    material_features: Dict[str, np.ndarray]
    coordination_defects: Optional[np.ndarray] = None
    strain_tensors: Optional[np.ndarray] = None
    damage_accumulation: Optional[np.ndarray] = None
    
    # ネットワーク解析
    strain_network: Optional[List] = None
    dislocation_network: Optional[List] = None
    damage_network: Optional[List] = None
    
    # 解析結果
    anomaly_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    detected_structures: List[Dict] = field(default_factory=list)
    critical_clusters: List[int] = field(default_factory=list)
    
    # メタデータ
    n_frames: int = 0
    n_atoms: int = 0
    n_clusters: int = 0
    window_steps: int = 0
    computation_time: float = 0.0
    gpu_info: Optional[Dict] = None
    critical_events: List = field(default_factory=list)
    
    # 材料特性
    material_properties: Optional[Dict] = None
    failure_probability: float = 0.0
    reliability_index: float = 0.0

# ===============================
# Material Features GPU
# ===============================

class MaterialFeaturesGPU(GPUBackend):
    """材料特徴抽出のGPU実装"""
    
    def __init__(self, force_cpu: bool = False):
        super().__init__(force_cpu=force_cpu)
    
    def extract_material_features(self,
                                 trajectory: np.ndarray,
                                 cluster_atoms: Dict[int, List[int]],
                                 atom_types: np.ndarray,
                                 config: MaterialConfig) -> Dict[str, np.ndarray]:
        """
        材料特徴を抽出
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子トラジェクトリ (n_frames, n_atoms, 3)
        cluster_atoms : Dict[int, List[int]]
            クラスター定義
        atom_types : np.ndarray
            原子タイプ
        config : MaterialConfig
            設定
            
        Returns
        -------
        Dict[str, np.ndarray]
            材料特徴の辞書
        """
        features = {}
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        
        # 配位数計算
        if config.use_coordination:
            features['coordination'] = self._compute_coordination_numbers(
                trajectory, cluster_atoms, config.cutoff_distance
            )
        
        # 歪み計算（簡易版）
        if config.use_strain:
            features['strain'] = self._compute_strain_tensors(
                trajectory, cluster_atoms
            )
        
        # 損傷度計算（簡易版）
        if config.use_damage:
            features['damage'] = self._compute_damage_scores(
                trajectory, cluster_atoms, atom_types
            )
        
        return features
    
    def _compute_coordination_numbers(self,
                                     trajectory: np.ndarray,
                                     cluster_atoms: Dict[int, List[int]],
                                     cutoff: float) -> np.ndarray:
        """配位数計算"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        coordination = self.zeros((n_frames, n_clusters))
        
        for frame in range(n_frames):
            positions = self.to_gpu(trajectory[frame])
            
            for cluster_id, atoms in cluster_atoms.items():
                if cluster_id >= n_clusters:
                    continue
                    
                coord_sum = 0
                for atom_i in atoms[:10]:  # サンプリング
                    pos_i = positions[atom_i]
                    distances = self.xp.linalg.norm(positions - pos_i, axis=1)
                    neighbors = self.xp.sum((distances > 0) & (distances < cutoff))
                    coord_sum += neighbors
                
                coordination[frame, cluster_id] = coord_sum / min(len(atoms), 10)
        
        return self.to_cpu(coordination)
    
    def _compute_strain_tensors(self,
                               trajectory: np.ndarray,
                               cluster_atoms: Dict[int, List[int]]) -> np.ndarray:
        """歪みテンソル計算（簡易版）"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        strain = self.zeros((n_frames, n_clusters))
        
        if n_frames < 2:
            return self.to_cpu(strain)
        
        ref_positions = self.to_gpu(trajectory[0])
        
        for frame in range(1, n_frames):
            current_positions = self.to_gpu(trajectory[frame])
            
            for cluster_id, atoms in cluster_atoms.items():
                if cluster_id >= n_clusters or len(atoms) < 4:
                    continue
                
                # 簡易歪み：相対変位の平均
                sample_atoms = atoms[:10]
                ref_pos = ref_positions[sample_atoms]
                curr_pos = current_positions[sample_atoms]
                
                displacement = self.xp.linalg.norm(curr_pos - ref_pos, axis=1)
                strain[frame, cluster_id] = self.xp.mean(displacement)
        
        return self.to_cpu(strain)
    
    def _compute_damage_scores(self,
                              trajectory: np.ndarray,
                              cluster_atoms: Dict[int, List[int]],
                              atom_types: np.ndarray) -> np.ndarray:
        """損傷度計算（簡易版）"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        damage = self.zeros((n_frames, n_clusters))
        
        # 初期構造からの変位で損傷を推定
        ref_positions = self.to_gpu(trajectory[0])
        
        for frame in range(n_frames):
            current_positions = self.to_gpu(trajectory[frame])
            
            for cluster_id, atoms in cluster_atoms.items():
                if cluster_id >= n_clusters:
                    continue
                
                ref_pos = ref_positions[atoms]
                curr_pos = current_positions[atoms]
                
                # RMSDベースの損傷度
                rmsd = self.xp.sqrt(self.xp.mean((curr_pos - ref_pos) ** 2))
                damage[frame, cluster_id] = self.xp.tanh(rmsd / 2.0)  # 0-1に正規化
        
        return self.to_cpu(damage)

# ===============================
# Material Lambda³ Detector GPU
# ===============================

class MaterialLambda3DetectorGPU(GPUBackend):
    """材料用Lambda³検出器（GPU版）"""
    
    def __init__(self, config: MaterialConfig = None, device: str = 'auto'):
        """
        Parameters
        ----------
        config : MaterialConfig
            設定パラメータ
        device : str
            'auto', 'gpu', 'cpu'のいずれか
        """
        # GPUBackendの初期化
        if device == 'cpu':
            super().__init__(device='cpu', force_cpu=True)
        else:
            super().__init__(device=device, force_cpu=False)
        
        self.config = config or MaterialConfig()
        self.verbose = True
        
        # force_cpuフラグ
        force_cpu_flag = not self.is_gpu
        
        # 材料版コンポーネント
        self.structure_computer = ClusterStructuresGPU()
        self.feature_extractor = MaterialFeaturesGPU(force_cpu_flag)
        self.network_analyzer = ClusterNetworkGPU()
        self.causality_analyzer = MaterialCausalityAnalyzerGPU()
        self.confidence_analyzer = MaterialConfidenceAnalyzerGPU()
        
        # 検出コンポーネント（MD版から流用）
        self.anomaly_detector = AnomalyDetectorGPU(force_cpu_flag)
        self.boundary_detector = BoundaryDetectorGPU(force_cpu_flag)
        self.topology_detector = TopologyBreaksDetectorGPU(force_cpu_flag)
        
        # 材料プロパティ設定
        self._set_material_properties()
        
        # メモリマネージャとデバイスを共有
        for component in [self.structure_computer, self.feature_extractor,
                         self.network_analyzer, self.causality_analyzer,
                         self.confidence_analyzer, self.anomaly_detector,
                         self.boundary_detector, self.topology_detector]:
            if hasattr(component, 'memory_manager'):
                component.memory_manager = self.memory_manager
            if hasattr(component, 'device'):
                component.device = self.device
        
        self._print_initialization_info()
    
    def _set_material_properties(self):
        """材料プロパティを設定"""
        if self.config.material_type == 'SUJ2':
            self.material_props = {
                'elastic_modulus': 210.0,  # GPa
                'yield_strength': 1.5,
                'ultimate_strength': 2.0,
                'fatigue_strength': 0.7,
                'fracture_toughness': 30.0
            }
        elif self.config.material_type == 'AL7075':
            self.material_props = {
                'elastic_modulus': 71.7,
                'yield_strength': 0.503,
                'ultimate_strength': 0.572,
                'fatigue_strength': 0.159,
                'fracture_toughness': 23.0
            }
        else:
            # デフォルト（鋼鉄）
            self.material_props = {
                'elastic_modulus': 210.0,
                'yield_strength': 1.0,
                'ultimate_strength': 1.5,
                'fatigue_strength': 0.5,
                'fracture_toughness': 20.0
            }
    
    def analyze(self,
               trajectory: np.ndarray,
               cluster_atoms: Dict[int, List[int]],
               atom_types: np.ndarray) -> MaterialLambda3Result:
        """
        材料トラジェクトリのLambda³解析
        
        Parameters
        ----------
        trajectory : np.ndarray
            原子トラジェクトリ (n_frames, n_atoms, 3)
        cluster_atoms : Dict[int, List[int]]
            クラスター定義
        atom_types : np.ndarray
            原子タイプ配列
            
        Returns
        -------
        MaterialLambda3Result
            解析結果
        """
        start_time = time.time()
        
        # GPU変換
        if self.is_gpu and cp is not None:
            print("📊 Converting arrays to GPU...")
            trajectory = cp.asarray(trajectory)
            atom_types = cp.asarray(atom_types) if isinstance(atom_types, np.ndarray) else atom_types
        
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        print(f"\n{'='*60}")
        print(f"=== Material Lambda³ Analysis (GPU) ===")
        print(f"{'='*60}")
        print(f"Trajectory: {n_frames} frames, {n_atoms} atoms")
        print(f"Clusters: {n_clusters}")
        print(f"Material: {self.config.material_type}")
        print(f"GPU Device: {self.device}")
        
        # バッチ処理判定
        batch_size = min(self.config.gpu_batch_size, n_frames)
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        if n_batches > 1:
            print(f"Processing in {n_batches} batches of {batch_size} frames")
            result = self._analyze_batched(trajectory, cluster_atoms, atom_types, batch_size)
        else:
            result = self._analyze_single_trajectory(trajectory, cluster_atoms, atom_types)
        
        computation_time = time.time() - start_time
        result.computation_time = computation_time
        
        self._print_summary(result)
        
        return result
    
    def _analyze_single_trajectory(self,
                                 trajectory: np.ndarray,
                                 cluster_atoms: Dict[int, List[int]],
                                 atom_types: np.ndarray) -> MaterialLambda3Result:
        """単一軌道の解析"""
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        # 1. 材料特徴抽出
        print("\n1. Extracting material features...")
        material_features = self.feature_extractor.extract_material_features(
            trajectory, cluster_atoms, atom_types, self.config
        )
        
        # 2. クラスター構造計算
        print("\n2. Computing cluster structures...")
        window_size = self._compute_initial_window(n_frames)
        
        # CPU配列に変換（ClusterStructuresGPUの期待する形式）
        if self.is_gpu and hasattr(trajectory, 'get'):
            trajectory_cpu = trajectory.get()
            atom_types_cpu = atom_types.get() if hasattr(atom_types, 'get') else atom_types
        else:
            trajectory_cpu = trajectory
            atom_types_cpu = atom_types
        
        cluster_result = self.structure_computer.compute_cluster_structures(
            trajectory_cpu,
            0,
            n_frames - 1,
            cluster_atoms,
            atom_types_cpu,
            window_size,
            self.config.cutoff_distance
        )
        
        # Lambda構造を辞書形式に変換
        lambda_structures = {
            'lambda_f': cluster_result.cluster_lambda_f,
            'lambda_f_mag': cluster_result.cluster_lambda_f_mag,
            'rho_t': cluster_result.cluster_rho_t,
            'coupling': cluster_result.cluster_coupling
        }
        
        # 3. 構造境界検出
        print("\n3. Detecting structural boundaries...")
        structural_boundaries = self.boundary_detector.detect_structural_boundaries(
            lambda_structures, window_size // 3
        )
        
        # 4. トポロジカル破れ検出
        print("\n4. Detecting topological breaks...")
        topological_breaks = self.topology_detector.detect_topological_breaks(
            lambda_structures, window_size // 2
        )
        
        # 5. 異常スコア計算
        print("\n5. Computing anomaly scores...")
        anomaly_scores = self._compute_material_anomalies(
            lambda_structures,
            material_features,
            structural_boundaries,
            topological_breaks
        )
        
        # 6. ネットワーク解析
        print("\n6. Analyzing material network...")
        network_result = self.network_analyzer.analyze_network(
            {i: anomaly_scores['combined'][i::n_clusters] for i in range(n_clusters)},
            cluster_result.cluster_coupling,
            cluster_result.cluster_centers,
            cluster_result.coordination_numbers,
            cluster_result.local_strain,
            cluster_result.element_composition
        )
        
        # 7. 臨界イベント検出
        critical_events = self._detect_critical_events(anomaly_scores)
        
        # 結果構築
        result = MaterialLambda3Result(
            cluster_structures=self._to_cpu_dict(lambda_structures),
            structural_boundaries=structural_boundaries,
            topological_breaks=self._to_cpu_dict(topological_breaks),
            material_features=self._to_cpu_dict(material_features),
            coordination_defects=cluster_result.coordination_numbers,
            strain_tensors=cluster_result.local_strain,
            damage_accumulation=material_features.get('damage'),
            strain_network=network_result.strain_network,
            dislocation_network=network_result.dislocation_network,
            damage_network=network_result.damage_network,
            anomaly_scores=self._to_cpu_dict(anomaly_scores),
            detected_structures=self._detect_structural_patterns(
                lambda_structures, structural_boundaries, window_size
            ),
            critical_clusters=network_result.critical_clusters,
            critical_events=critical_events,
            n_frames=n_frames,
            n_atoms=n_atoms,
            n_clusters=n_clusters,
            window_steps=window_size,
            computation_time=0.0,
            gpu_info=self._get_gpu_info(),
            material_properties=self.material_props,
            failure_probability=self._estimate_failure_probability(
                cluster_result.local_strain
            ),
            reliability_index=self._compute_reliability_index(
                cluster_result.local_strain
            )
        )
        
        return result
    
    def _analyze_batched(self,
                        trajectory: np.ndarray,
                        cluster_atoms: Dict[int, List[int]],
                        atom_types: np.ndarray,
                        batch_size: int) -> MaterialLambda3Result:
        """バッチ処理による解析"""
        print("\n⚡ Running batched GPU analysis for materials...")
        
        n_frames = trajectory.shape[0]
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        # バッチ結果を蓄積
        batch_results = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_frames)
            
            print(f"  Batch {batch_idx + 1}/{n_batches}: frames {start_idx}-{end_idx}")
            
            batch_trajectory = trajectory[start_idx:end_idx]
            
            # バッチ解析
            batch_result = self._analyze_single_trajectory(
                batch_trajectory, cluster_atoms, atom_types
            )
            batch_result.offset = start_idx
            batch_results.append(batch_result)
            
            # メモリクリア
            self.memory_manager.clear_cache()
        
        # 結果マージ
        print("\n[Step 2] Merging batch results...")
        merged_result = self._merge_batch_results(batch_results, trajectory.shape)
        
        return merged_result
    
    def _compute_material_anomalies(self,
                                   lambda_structures: Dict,
                                   material_features: Dict,
                                   structural_boundaries: Dict,
                                   topological_breaks: Dict) -> Dict[str, np.ndarray]:
        """材料特有の異常スコア計算"""
        scores = {}
        
        # Lambda異常
        if 'lambda_f_mag' in lambda_structures:
            lambda_anomaly = self._compute_anomaly_score(
                lambda_structures['lambda_f_mag']
            )
            scores['lambda'] = lambda_anomaly
        
        # 歪み異常
        if 'strain' in material_features:
            strain_anomaly = self._compute_anomaly_score(
                material_features['strain']
            )
            scores['strain'] = strain_anomaly
        
        # 配位数異常
        if 'coordination' in material_features:
            ideal_coord = 12.0 if self.config.crystal_structure == 'FCC' else 8.0
            coord_defect = np.abs(material_features['coordination'] - ideal_coord)
            scores['coordination'] = coord_defect / ideal_coord
        
        # 損傷異常
        if 'damage' in material_features:
            scores['damage'] = material_features['damage']
        
        # 統合スコア
        combined = np.zeros_like(scores.get('lambda', np.zeros(1)))
        weights_sum = 0.0
        
        if 'strain' in scores:
            combined += self.config.w_strain * scores['strain']
            weights_sum += self.config.w_strain
        
        if 'coordination' in scores:
            combined += self.config.w_coordination * scores['coordination']
            weights_sum += self.config.w_coordination
        
        if 'damage' in scores:
            combined += self.config.w_damage * scores['damage']
            weights_sum += self.config.w_damage
        
        if weights_sum > 0:
            combined /= weights_sum
        
        scores['combined'] = combined
        
        return scores
    
    def _compute_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """簡易異常スコア計算"""
        if data.size == 0:
            return np.zeros(1)
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10
        
        z_scores = np.abs((data - mean) / std)
        
        return z_scores
    
    def _estimate_failure_probability(self, strain_tensor: np.ndarray) -> float:
        """破壊確率推定"""
        if strain_tensor is None or strain_tensor.size == 0:
            return 0.0
        
        # von Mises応力推定
        von_mises_strain = np.mean(np.abs(strain_tensor))
        critical_strain = self.material_props['yield_strength'] / self.material_props['elastic_modulus']
        
        # 簡易確率
        if von_mises_strain > critical_strain:
            return min(1.0, von_mises_strain / critical_strain)
        
        return 0.0
    
    def _compute_reliability_index(self, strain_tensor: np.ndarray) -> float:
        """信頼性指標計算"""
        if strain_tensor is None or strain_tensor.size == 0:
            return 5.0  # 高信頼性
        
        # 簡易β指標
        mean_strain = np.mean(np.abs(strain_tensor))
        std_strain = np.std(np.abs(strain_tensor))
        critical_strain = self.material_props['yield_strength'] / self.material_props['elastic_modulus']
        
        if std_strain > 0:
            beta = (critical_strain - mean_strain) / std_strain
            return float(beta)
        
        return 5.0
    
    # ========================================
    # ヘルパーメソッド（MD版から流用）
    # ========================================
    
    def _compute_initial_window(self, n_frames: int) -> int:
        """初期ウィンドウサイズ"""
        return min(50, n_frames // 10)  # 材料は短め
    
    def _detect_structural_patterns(self, lambda_structures: Dict,
                                   boundaries: Dict, window: int) -> List[Dict]:
        """構造パターン検出"""
        patterns = []
        
        if isinstance(boundaries, dict) and 'boundary_locations' in boundaries:
            boundary_locs = boundaries['boundary_locations']
            
            for i in range(len(boundary_locs) - 1):
                duration = boundary_locs[i+1] - boundary_locs[i]
                if duration > 20:  # 材料は短い閾値
                    pattern_type = 'elastic' if duration < 100 else 'plastic'
                    patterns.append({
                        'type': pattern_type,
                        'start': boundary_locs[i],
                        'end': boundary_locs[i+1],
                        'duration': duration
                    })
        
        return patterns
    
    def _detect_critical_events(self, anomaly_scores: Dict) -> List:
        """臨界イベント検出"""
        events = []
        
        if 'combined' in anomaly_scores:
            scores = anomaly_scores['combined']
            threshold = np.mean(scores) + 1.5 * np.std(scores)  # 材料は低め
            
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
    
    def _merge_batch_results(self, batch_results: List, original_shape: Tuple) -> MaterialLambda3Result:
        """バッチ結果のマージ（簡易版）"""
        if not batch_results:
            return self._create_empty_result(original_shape[0], original_shape[1])
        
        # 最初のバッチの構造をコピー
        merged = batch_results[0]
        
        # 他のバッチの重要な情報をマージ
        for batch in batch_results[1:]:
            merged.critical_clusters.extend(batch.critical_clusters)
            merged.critical_events.extend(batch.critical_events)
        
        merged.n_frames = original_shape[0]
        
        return merged
    
    def _create_empty_result(self, n_frames: int, n_atoms: int) -> MaterialLambda3Result:
        """空の結果作成"""
        return MaterialLambda3Result(
            cluster_structures={},
            structural_boundaries={},
            topological_breaks={},
            material_features={},
            n_frames=n_frames,
            n_atoms=n_atoms,
            n_clusters=0
        )
    
    def _to_cpu_dict(self, data_dict: Dict) -> Dict:
        """GPU配列をCPUに転送"""
        cpu_dict = {}
        for key, value in data_dict.items():
            if hasattr(value, 'get'):
                cpu_dict[key] = value.get()
            elif isinstance(value, dict):
                cpu_dict[key] = self._to_cpu_dict(value)
            else:
                cpu_dict[key] = value
        return cpu_dict
    
    def _get_gpu_info(self) -> Dict:
        """GPU情報取得"""
        return {
            'device_name': str(self.device),
            'computation_mode': 'single_batch'
        }
    
    def _print_initialization_info(self):
        """初期化情報表示"""
        if self.verbose:
            print(f"\n💎 Material Lambda³ Detector Initialized")
            print(f"   Material: {self.config.material_type}")
            print(f"   Crystal: {self.config.crystal_structure}")
            print(f"   Device: {self.device}")
            print(f"   GPU Mode: {self.is_gpu}")
            print(f"   Batch Size: {self.config.gpu_batch_size} frames")
    
    def _print_summary(self, result: MaterialLambda3Result):
        """結果サマリー表示"""
        print("\n" + "="*60)
        print("=== Material Analysis Complete ===")
        print("="*60)
        print(f"Total frames: {result.n_frames}")
        print(f"Total clusters: {result.n_clusters}")
        print(f"Computation time: {result.computation_time:.2f} seconds")
        
        if result.computation_time > 0:
            print(f"Speed: {result.n_frames / result.computation_time:.1f} frames/second")
        
        print(f"\nMaterial properties:")
        print(f"  Failure probability: {result.failure_probability:.1%}")
        print(f"  Reliability index β: {result.reliability_index:.2f}")
        
        print(f"\nDetected features:")
        print(f"  Critical clusters: {len(result.critical_clusters)}")
        print(f"  Critical events: {len(result.critical_events)}")
        
        if result.strain_network:
            print(f"  Strain network links: {len(result.strain_network)}")
        if result.dislocation_network:
            print(f"  Dislocation links: {len(result.dislocation_network)}")
        if result.damage_network:
            print(f"  Damage links: {len(result.damage_network)}")
