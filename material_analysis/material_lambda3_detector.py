#!/usr/bin/env python3
"""
Material Lambda³ Detector GPU - Enhanced with Topological Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用Lambda³検出器のGPU実装
金属・セラミックス・ポリマーの疲労・破壊を高速検出！💎
トポロジカル欠陥の蓄積と構造一貫性を追跡

MD版をベースに材料クラスター解析に特化

by 環ちゃん - Material Edition v2.0
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import logging
from scipy.spatial import cKDTree

logger = logging.getLogger('lambda3_gpu.material_analysis')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False

# Lambda³ GPU imports
from lambda3_gpu.core.gpu_utils import GPUBackend
try:
    from lambda3_gpu.material_analysis.cuda_kernels import MaterialCUDAKernels
    HAS_CUDA_KERNELS = True
except ImportError:
    MaterialCUDAKernels = None
    HAS_CUDA_KERNELS = False
from lambda3_gpu.material.cluster_structures_gpu import ClusterStructuresGPU
from lambda3_gpu.material.cluster_network_gpu import ClusterNetworkGPU
from lambda3_gpu.material.cluster_causality_analysis_gpu import MaterialCausalityAnalyzerGPU
from lambda3_gpu.material.cluster_confidence_analysis_gpu import MaterialConfidenceAnalyzerGPU

# 検出モジュール（MD版から流用）
from lambda3_gpu.detection.anomaly_detection_gpu import AnomalyDetectorGPU
from lambda3_gpu.detection.boundary_detection_gpu import BoundaryDetectorGPU
from lambda3_gpu.detection.topology_breaks_gpu import TopologyBreaksDetectorGPU

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
    use_topological: bool = True  # トポロジカル解析（新規追加）
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

    # CUDA最適化設定
    use_cuda_kernels: bool = True  # CUDAカーネル使用
    cuda_kernel_fallback: bool = True  # 失敗時は標準実装にフォールバック

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
    
    # トポロジカル解析結果（新規追加）
    topological_charge: Optional[np.ndarray] = None
    topological_charge_cumulative: Optional[np.ndarray] = None
    structural_coherence: Optional[np.ndarray] = None
    
    # ネットワーク解析
    strain_network: Optional[List] = None
    dislocation_network: Optional[List] = None
    damage_network: Optional[List] = None
    
    # 解析結果
    anomaly_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    detected_structures: List[Dict] = field(default_factory=list)
    critical_clusters: List[int] = field(default_factory=list)
    material_events: List[Tuple[int, int, str]] = field(default_factory=list)  # 追加
    
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
    """材料特徴抽出のGPU実装（CUDA最適化版）"""
    
    def __init__(self, force_cpu: bool = False, use_cuda: bool = True):
        super().__init__(force_cpu=force_cpu)
        
        # CUDAカーネル初期化
        self.cuda_kernels = None
        if use_cuda and HAS_CUDA_KERNELS and not force_cpu:
            try:
                from lambda3_gpu.material_analysis.cuda_kernels import MaterialCUDAKernels
                self.cuda_kernels = MaterialCUDAKernels()
                if self.cuda_kernels.compiled:
                    logger.info("🚀 CUDA kernels enabled for material features")
            except Exception as e:
                logger.warning(f"CUDA kernels disabled: {e}")
    
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
            if config.use_cuda_kernels and self.cuda_kernels and self.cuda_kernels.compiled:
                features['coordination'] = self._compute_coordination_cuda(
                    trajectory, cluster_atoms, config.cutoff_distance
                )
            else:
                features['coordination'] = self._compute_coordination_numbers(
                    trajectory, cluster_atoms, config.cutoff_distance
                )
        
        # 歪み計算
        if config.use_strain:
            if config.use_cuda_kernels and self.cuda_kernels and self.cuda_kernels.compiled:
                features['strain'] = self._compute_strain_tensors_cuda(
                    trajectory, cluster_atoms
                )
            else:
                features['strain'] = self._compute_strain_tensors(
                    trajectory, cluster_atoms
                )
        
        # 損傷度計算
        if config.use_damage:
            if config.use_cuda_kernels and self.cuda_kernels and self.cuda_kernels.compiled:
                features['damage'] = self._compute_damage_scores_cuda(
                    trajectory, cluster_atoms, atom_types
                )
            else:
                features['damage'] = self._compute_damage_scores(
                    trajectory, cluster_atoms, atom_types
                )
        
        return features
    
    # ========================================
    # CUDA版メソッド
    # ========================================
    
    def _compute_coordination_cuda(self,
                                  trajectory: np.ndarray,
                                  cluster_atoms: Dict[int, List[int]],
                                  cutoff: float) -> np.ndarray:
        """CUDA版配位数計算"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        coordination = np.zeros((n_frames, n_clusters))
        
        for frame in range(n_frames):
            positions_gpu = cp.asarray(trajectory[frame])
            coord_gpu = self.cuda_kernels.compute_coordination_cuda(
                positions_gpu, cluster_atoms, cutoff
            )
            coordination[frame] = cp.asnumpy(coord_gpu)
        
        return coordination
    
    def _compute_strain_tensors_cuda(self,
                                    trajectory: np.ndarray,
                                    cluster_atoms: Dict[int, List[int]]) -> np.ndarray:
        """CUDA版歪みテンソル計算"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        strain = np.zeros((n_frames, n_clusters))
        
        if n_frames < 2:
            return strain
        
        ref_pos_gpu = cp.asarray(trajectory[0])
        
        for frame in range(1, n_frames):
            curr_pos_gpu = cp.asarray(trajectory[frame])
            strain_gpu = self.cuda_kernels.compute_strain_tensors_cuda(
                ref_pos_gpu, curr_pos_gpu, cluster_atoms, n_frames
            )
            # Voigt記法から主歪みの最大値を抽出
            max_strain = cp.max(cp.abs(strain_gpu), axis=1)
            strain[frame] = cp.asnumpy(max_strain)
        
        return strain
    
    def _compute_damage_scores_cuda(self,
                                   trajectory: np.ndarray,
                                   cluster_atoms: Dict[int, List[int]],
                                   atom_types: np.ndarray) -> np.ndarray:
        """CUDA版損傷度計算"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        damage = np.zeros((n_frames, n_clusters))
        
        ref_pos_gpu = cp.asarray(trajectory[0])
        
        for frame in range(n_frames):
            curr_pos_gpu = cp.asarray(trajectory[frame])
            damage_gpu = self.cuda_kernels.compute_damage_cuda(
                ref_pos_gpu, curr_pos_gpu, cluster_atoms
            )
            damage[frame] = cp.asnumpy(damage_gpu)
        
        return damage
    
    # ========================================
    # 標準版メソッド（フォールバック用）
    # ========================================
    
    def _compute_coordination_numbers(self,
                                     trajectory: np.ndarray,
                                     cluster_atoms: Dict[int, List[int]],
                                     cutoff: float) -> np.ndarray:
        """配位数計算（標準版）"""
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
        """歪みテンソル計算（標準版）"""
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
        """損傷度計算（標準版）"""
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
        self.structure_computer = ClusterStructuresGPU()
        self.feature_extractor = MaterialFeaturesGPU(
            force_cpu=force_cpu_flag,
            use_cuda=config.use_cuda_kernels  # CUDAカーネル設定を渡す
        )
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
    
    # ========================================
    # トポロジカル解析メソッド（新規追加）
    # ========================================
    
    def _compute_topological_charge(self, 
                                   trajectory: np.ndarray,
                                   cluster_atoms: Dict[int, List[int]],
                                   atom_types: np.ndarray,
                                   cutoff: float = 3.5) -> Dict[str, np.ndarray]:
        """
        材料のトポロジカルチャージ計算
        転位・空孔・格子間原子のトポロジカル欠陥を定量化
        """
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        
        # 各クラスターのトポロジカルチャージ
        Q_lambda = np.zeros((n_frames-1, n_clusters))
        
        for frame in range(n_frames-1):
            for cid, atoms in cluster_atoms.items():
                if cid >= n_clusters:
                    continue
                    
                # Burgers回路による転位検出（簡易版）
                burgers_vector = self._compute_burgers_vector(
                    trajectory[frame], atoms, cutoff
                )
                
                # 配位数欠陥の変化率
                coord_change = self._compute_coordination_change(
                    trajectory[frame], trajectory[frame+1], atoms, cutoff
                )
                
                # トポロジカルチャージ = 転位密度 + 配位欠陥変化
                Q_lambda[frame, cid] = (
                    np.linalg.norm(burgers_vector) / (len(atoms) + 1) +
                    abs(coord_change) * 0.1  # 重み付け
                )
        
        # 累積チャージ（欠陥の蓄積）
        Q_cumulative = np.cumsum(np.sum(Q_lambda, axis=1))
        
        return {
            'Q_lambda': Q_lambda,
            'Q_cumulative': Q_cumulative
        }
    
    def _compute_burgers_vector(self,
                               positions: np.ndarray,
                               atoms: List[int],
                               cutoff: float) -> np.ndarray:
        """Burgers vectorの計算（簡易版）"""
        if len(atoms) < 10:
            return np.zeros(3)
        
        # サンプル原子周りの回路を計算
        center_atom = atoms[len(atoms)//2]
        center_pos = positions[center_atom]
        
        # 近傍原子
        distances = np.linalg.norm(positions[atoms] - center_pos, axis=1)
        neighbors = [atoms[i] for i in range(len(atoms)) 
                    if 0 < distances[i] < cutoff]
        
        if len(neighbors) < 6:
            return np.zeros(3)
        
        # 簡易Burgers回路（6原子の閉路）
        circuit = neighbors[:6]
        burgers = np.zeros(3)
        
        for i in range(len(circuit)):
            j = (i + 1) % len(circuit)
            burgers += positions[circuit[j]] - positions[circuit[i]]
        
        # 理想的な閉路なら0になるはず
        return burgers
    
    def _compute_coordination_change(self,
                                    pos1: np.ndarray,
                                    pos2: np.ndarray,
                                    atoms: List[int],
                                    cutoff: float) -> float:
        """配位数変化の計算"""
        if len(atoms) < 5:
            return 0.0
        
        # サンプル原子の配位数変化
        sample_atoms = atoms[:min(5, len(atoms))]
        coord_change = 0.0
        
        for atom in sample_atoms:
            # frame1での配位数
            dist1 = np.linalg.norm(pos1 - pos1[atom], axis=1)
            coord1 = np.sum((dist1 > 0) & (dist1 < cutoff))
            
            # frame2での配位数
            dist2 = np.linalg.norm(pos2 - pos2[atom], axis=1)
            coord2 = np.sum((dist2 > 0) & (dist2 < cutoff))
            
            coord_change += abs(coord2 - coord1)
        
        return coord_change / len(sample_atoms)
    
    def _compute_structural_coherence(self,
                                     coordination: np.ndarray,
                                     strain: np.ndarray,
                                     ideal_coord: int = 8) -> np.ndarray:
        """
        構造一貫性の計算
        熱ゆらぎと永続的構造変化を区別
        """
        n_frames, n_clusters = coordination.shape
        coherence = np.ones(n_frames)
        
        window = min(50, n_frames // 4)  # 時間平均ウィンドウ
        
        for i in range(window, n_frames - window):
            cluster_coherence = []
            
            for c in range(n_clusters):
                # 配位数の時間的一貫性
                local_coord = coordination[i-window:i+window, c]
                coord_std = np.std(local_coord)
                coord_mean = np.mean(local_coord)
                
                # 理想配位数からのずれの一貫性
                if coord_std < 0.5 and abs(coord_mean - ideal_coord) < 1:
                    # 安定して理想構造を保持
                    cluster_coherence.append(1.0)
                elif coord_std > 2.0:
                    # 激しく揺らいでいる（熱的）
                    cluster_coherence.append(0.3)
                else:
                    # 構造変化中
                    cluster_coherence.append(1.0 - coord_std / 3.0)
                
                # 歪みによる補正
                if strain[i, c] > 0.05:  # 5%以上の歪み
                    cluster_coherence[-1] *= (1.0 - min(strain[i, c], 1.0))
            
            coherence[i] = np.mean(cluster_coherence)
        
        # エッジ処理
        coherence[:window] = coherence[window]
        coherence[-window:] = coherence[-window-1]
        
        return coherence
    
    # ========================================
    # メイン解析メソッド
    # ========================================
    
    def analyze(self,
               trajectory: np.ndarray,
               atom_types: np.ndarray,
               cluster_atoms: Optional[Dict[int, List[int]]] = None,
               strain_field: Optional[np.ndarray] = None,
               **kwargs) -> MaterialLambda3Result:
        """
        材料トラジェクトリのLambda³解析
        """
        start_time = time.time()
        
        # cluster_atomsがなければ自動生成
        if cluster_atoms is None:
            n_atoms = trajectory.shape[1]
            cluster_atoms = {0: list(range(n_atoms))}
        
        # atom_typesが文字列の場合、数値に変換
        if atom_types.dtype.kind == 'U' or atom_types.dtype.kind == 'S':
            # 文字列を数値にマッピング
            unique_types = np.unique(atom_types)
            type_to_id = {atype: i for i, atype in enumerate(unique_types)}
            atom_types_numeric = np.array([type_to_id[atype] for atype in atom_types])
            
            # 元の文字列型も保存しておく
            self.atom_type_labels = unique_types
            print(f"📝 Atom types mapped: {dict(zip(unique_types, range(len(unique_types))))}")
        else:
            atom_types_numeric = atom_types
        
        # GPU変換
        if self.is_gpu and cp is not None:
            print("📊 Converting arrays to GPU...")
            trajectory = cp.asarray(trajectory)
            atom_types = cp.asarray(atom_types_numeric)  # 数値配列をGPUへ
        else:
            atom_types = atom_types_numeric
        
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
        
        # CPU配列に変換
        if self.is_gpu and hasattr(trajectory, 'get'):
            trajectory_cpu = trajectory.get()
            atom_types_cpu = atom_types.get() if hasattr(atom_types, 'get') else atom_types
        else:
            trajectory_cpu = trajectory
            atom_types_cpu = atom_types
        
        cluster_result = self.structure_computer.compute_cluster_structures(
            trajectory_cpu, 0, n_frames - 1, cluster_atoms,
            atom_types_cpu, window_size, self.config.cutoff_distance
        )
        
        # 2.5. トポロジカル解析
        if self.config.use_topological:
            print("\n2.5. Computing topological charges...")
            topo_charge = self._compute_topological_charge(
                trajectory_cpu, cluster_atoms, atom_types_cpu, self.config.cutoff_distance
            )
            
            print("\n2.6. Computing structural coherence...")
            structural_coherence = self._compute_structural_coherence(
                material_features.get('coordination', np.zeros((n_frames, n_clusters))),
                material_features.get('strain', np.zeros((n_frames, n_clusters))),
                ideal_coord=8 if self.config.crystal_structure == 'BCC' else 12
            )
        else:
            topo_charge = {
                'Q_lambda': np.zeros((n_frames-1, n_clusters)),
                'Q_cumulative': np.zeros(n_frames-1)
            }
            structural_coherence = np.ones(n_frames)
        
        # Lambda構造（全クラスターの集約）
        if isinstance(cluster_result.cluster_lambda_f_mag, list):
            # 複数クラスター：最大値で集約（最悪ケースを追跡）
            lambda_structures = {
                'lambda_F': np.max(cluster_result.cluster_lambda_f, axis=0),
                'lambda_F_mag': np.max(cluster_result.cluster_lambda_f_mag, axis=0),
                'rho_T': np.max(cluster_result.cluster_rho_t, axis=0),
                'coupling': np.mean(cluster_result.cluster_coupling, axis=0),  # 結合は平均
                'Q_lambda': topo_charge['Q_lambda'],
                'Q_cumulative': topo_charge['Q_cumulative'],
                'structural_coherence': structural_coherence
            }
            # 元データも保存（Two-Stage用）
            lambda_structures['_per_cluster_data'] = {
                'lambda_f_mag': cluster_result.cluster_lambda_f_mag,
                'rho_t': cluster_result.cluster_rho_t
            }
        else:
            # 単一クラスター
            lambda_structures = {
                'lambda_F': cluster_result.cluster_lambda_f,
                'lambda_F_mag': cluster_result.cluster_lambda_f_mag,
                'rho_T': cluster_result.cluster_rho_t,
                'coupling': cluster_result.cluster_coupling,
                'Q_lambda': topo_charge['Q_lambda'],
                'Q_cumulative': topo_charge['Q_cumulative'],
                'structural_coherence': structural_coherence
            }
        
        # 3. 構造境界検出
        print("\n3. Detecting structural boundaries...")
        structural_boundaries = self.boundary_detector.detect_structural_boundaries(
            lambda_structures, window_size // 3
        )
        
        # 4. トポロジカル破れ検出（全体で1回だけ）
        print("\n4. Detecting topological breaks...")
        topological_breaks = self.topology_detector.detect_topological_breaks(
            lambda_structures, window_size // 2
        )
        
        # 5. 異常スコア計算
        print("\n5. Computing anomaly scores...")
        anomaly_scores = self._compute_material_anomalies(
            lambda_structures, material_features,
            structural_boundaries, topological_breaks
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
        
        # 8. 材料イベント分類
        material_events = self._classify_material_events(
            critical_events, anomaly_scores, structural_coherence
        )
        
        # 結果構築
        result = MaterialLambda3Result(
            cluster_structures=self._to_cpu_dict(lambda_structures),
            structural_boundaries=structural_boundaries,
            topological_breaks=self._to_cpu_dict(topological_breaks),
            material_features=self._to_cpu_dict(material_features),
            coordination_defects=cluster_result.coordination_numbers,
            strain_tensors=cluster_result.local_strain,
            damage_accumulation=material_features.get('damage'),
            topological_charge=topo_charge.get('Q_lambda'),
            topological_charge_cumulative=topo_charge.get('Q_cumulative'),
            structural_coherence=structural_coherence,
            strain_network=network_result.strain_network,
            dislocation_network=network_result.dislocation_network,
            damage_network=network_result.damage_network,
            anomaly_scores=self._to_cpu_dict(anomaly_scores),
            detected_structures=self._detect_structural_patterns(
                lambda_structures, structural_boundaries, window_size
            ),
            critical_clusters=network_result.critical_clusters,
            critical_events=critical_events,
            material_events=material_events,
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
       
    def _classify_material_events(self,
                                 critical_events: List,
                                 anomaly_scores: Dict,
                                 structural_coherence: np.ndarray) -> List[Tuple[int, int, str]]:
        """
        材料イベントの分類
        構造一貫性を使って熱ゆらぎと真の構造変化を区別
        """
        classified_events = []
        
        for event in critical_events:
            if isinstance(event, tuple) and len(event) >= 2:
                start, end = event[0], event[1]
                
                # 構造一貫性の平均値
                coherence_mean = np.mean(structural_coherence[start:end+1])
                
                # 異常スコアの最大値
                if 'strain' in anomaly_scores:
                    strain_max = np.max(anomaly_scores['strain'][start:end+1])
                else:
                    strain_max = 0
                
                if 'damage' in anomaly_scores:
                    damage_max = np.max(anomaly_scores['damage'][start:end+1])
                else:
                    damage_max = 0
                
                # イベント分類
                if coherence_mean < 0.5:  # 構造一貫性が低い
                    if damage_max > 0.7:
                        event_type = 'crack_initiation'
                    elif strain_max > 2.0:
                        event_type = 'plastic_deformation'
                    else:
                        event_type = 'dislocation_nucleation'
                elif coherence_mean > 0.8:  # 構造一貫性が高い
                    event_type = 'elastic_deformation'
                else:
                    event_type = 'transition_state'
                
                classified_events.append((start, end, event_type))
        
        return classified_events
    
    def _analyze_batched(self,
                        trajectory: np.ndarray,
                        cluster_atoms: Dict[int, List[int]],
                        atom_types: np.ndarray,
                        batch_size: int) -> MaterialLambda3Result:
        """バッチ処理による解析"""
        # 既存の実装と同じ
        print("\n⚡ Running batched GPU analysis for materials...")
        
        n_frames = trajectory.shape[0]
        n_batches = (n_frames + batch_size - 1) // batch_size
        
        batch_results = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_frames)
            
            print(f"  Batch {batch_idx + 1}/{n_batches}: frames {start_idx}-{end_idx}")
            
            batch_trajectory = trajectory[start_idx:end_idx]
            
            batch_result = self._analyze_single_trajectory(
                batch_trajectory, cluster_atoms, atom_types
            )
            batch_result.offset = start_idx
            batch_results.append(batch_result)
            
            self.memory_manager.clear_cache()
        
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
        
        # トポロジカル異常（新規追加）
        if 'Q_cumulative' in lambda_structures:
            # 累積チャージの加速度（欠陥生成率）
            q_cum = lambda_structures['Q_cumulative']
            if len(q_cum) > 2:
                q_acceleration = np.abs(np.gradient(np.gradient(q_cum)))
                # サイズを合わせる
                scores['topological'] = np.pad(q_acceleration, 
                                              (0, len(scores.get('lambda', [])) - len(q_acceleration)),
                                              mode='edge')
            else:
                scores['topological'] = np.zeros_like(scores.get('lambda', np.zeros(1)))
        
        # 統合スコア
        combined = np.zeros_like(scores.get('lambda', np.zeros(1)))
        weights_sum = 0.0
        
        for key, weight in [('strain', self.config.w_strain),
                            ('coordination', self.config.w_coordination),
                            ('damage', self.config.w_damage),
                            ('topological', 0.2)]:  # トポロジカルの重み
            if key in scores:
                combined += weight * scores[key]
                weights_sum += weight
        
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
            merged.material_events.extend(batch.material_events)
        
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
            print(f"   Topological Analysis: {self.config.use_topological}")
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
        print(f"  Material events: {len(result.material_events)}")
        
        # イベントタイプ別統計
        if result.material_events:
            event_types = {}
            for event in result.material_events:
                if len(event) >= 3:
                    event_type = event[2]
                    event_types[event_type] = event_types.get(event_type, 0) + 1
            
            print(f"\nEvent classification:")
            for etype, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {etype}: {count}")
        
        # トポロジカル解析結果
        if result.topological_charge_cumulative is not None:
            final_charge = result.topological_charge_cumulative[-1] if len(result.topological_charge_cumulative) > 0 else 0
            print(f"\nTopological analysis:")
            print(f"  Final cumulative charge: {final_charge:.3f}")
            print(f"  Mean structural coherence: {np.mean(result.structural_coherence):.3f}")
        
        if result.strain_network:
            print(f"\nNetwork analysis:")
            print(f"  Strain network links: {len(result.strain_network)}")
        if result.dislocation_network:
            print(f"  Dislocation links: {len(result.dislocation_network)}")
        if result.damage_network:
            print(f"  Damage links: {len(result.damage_network)}")

# ===============================
# Module Level Functions
# ===============================

def detect_material_events(
    trajectory: np.ndarray,
    atom_types: np.ndarray,
    config: Optional[MaterialConfig] = None,
    cluster_atoms: Optional[Dict[int, List[int]]] = None,
    **kwargs
) -> List[Tuple[int, int, str]]:
    """
    材料イベントを検出する便利関数
    
    MaterialLambda3DetectorGPUのラッパー関数
    """
    config = config or MaterialConfig()
    detector = MaterialLambda3DetectorGPU(config)
    
    # クラスター定義（指定なければ全原子を1クラスターに）
    if cluster_atoms is None:
        n_atoms = trajectory.shape[1]
        cluster_atoms = {0: list(range(n_atoms))}
    
    # 解析実行
    result = detector.analyze(
        trajectory=trajectory,
        atom_types=atom_types,
        cluster_atoms=cluster_atoms,
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
