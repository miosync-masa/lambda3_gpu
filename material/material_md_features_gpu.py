#!/usr/bin/env python3
"""
Material MD Features Extraction (GPU Version)
============================================

材料解析用MD特徴抽出のGPU実装
欠陥領域に焦点を当てた効率的な特徴計算

タンパク質のbackbone_indicesの代わりに、
材料の欠陥領域を自動検出して計算量を削減！

CUDAカーネルによる高速化対応

Version: 2.0.0
Author: 環ちゃん
"""

import numpy as np
import json
import logging
from typing import Dict, Optional, List, Tuple, Union, Set
from dataclasses import dataclass
from pathlib import Path
import warnings

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.spatial.distance import cdist as cp_cdist
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cp_cdist = None

# Spatial analysis
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    cKDTree = None

# Local imports
from ..structures.md_features_gpu import MDFeaturesGPU, MDFeatureConfig
from ..core import GPUBackend, GPUMemoryManager

logger = logging.getLogger(__name__)

# ===============================
# CUDA Kernels for Material Features
# ===============================

# 配位数計算カーネル（材料特有）
COORDINATION_KERNEL = r'''
extern "C" __global__
void calculate_coordination_kernel(
    const float* __restrict__ positions,  // (n_atoms, 3)
    int* __restrict__ coord_numbers,      // (n_atoms,)
    const float cutoff_sq,
    const int n_atoms
) {
    unsigned int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (atom_i >= n_atoms) return;
    
    float xi = positions[atom_i * 3 + 0];
    float yi = positions[atom_i * 3 + 1];
    float zi = positions[atom_i * 3 + 2];
    
    int coord = 0;
    for (int j = 0; j < n_atoms; j++) {
        if (atom_i == j) continue;
        
        float dx = positions[j * 3 + 0] - xi;
        float dy = positions[j * 3 + 1] - yi;
        float dz = positions[j * 3 + 2] - zi;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        
        if (dist_sq < cutoff_sq) {
            coord++;
        }
    }
    
    coord_numbers[atom_i] = coord;
}
'''

# 欠陥検出カーネル（配位数の理想値からのずれ）
DEFECT_DETECTION_KERNEL = r'''
extern "C" __global__
void detect_defects_kernel(
    const int* __restrict__ coord_numbers,  // (n_atoms,)
    bool* __restrict__ defect_mask,         // (n_atoms,)
    const int ideal_coord,
    const int tolerance,
    const int n_atoms
) {
    unsigned int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (atom_i >= n_atoms) return;
    
    int coord = coord_numbers[atom_i];
    int deviation = abs(coord - ideal_coord);
    
    defect_mask[atom_i] = (deviation > tolerance);
}
'''

# 局所歪み計算カーネル
LOCAL_STRAIN_KERNEL = r'''
extern "C" __global__
void calculate_local_strain_kernel(
    const float* __restrict__ ref_positions,   // (n_atoms, 3)
    const float* __restrict__ curr_positions,  // (n_atoms, 3)
    float* __restrict__ local_strain,          // (n_atoms,)
    const int n_atoms
) {
    unsigned int atom_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (atom_i >= n_atoms) return;
    
    float dx = curr_positions[atom_i * 3 + 0] - ref_positions[atom_i * 3 + 0];
    float dy = curr_positions[atom_i * 3 + 1] - ref_positions[atom_i * 3 + 1];
    float dz = curr_positions[atom_i * 3 + 2] - ref_positions[atom_i * 3 + 2];
    
    local_strain[atom_i] = sqrtf(dx*dx + dy*dy + dz*dz);
}
'''

# ===============================
# Configuration
# ===============================

@dataclass
class MaterialMDFeatureConfig(MDFeatureConfig):
    """材料用MD特徴抽出の設定"""
    # 基本設定（継承）
    use_contacts: bool = False      # 接触マップ（材料では通常不要）
    use_rmsd: bool = True           # RMSD（欠陥領域のみ）
    use_rg: bool = True             # Radius of gyration（欠陥領域のみ）
    use_dihedrals: bool = False     # 二面角（材料では不要）
    
    # 材料特有の設定
    auto_detect_defects: bool = True    # 欠陥領域の自動検出
    defect_expansion_radius: float = 5.0  # 欠陥周辺領域の半径 [Å]
    coordination_cutoff: float = 3.5      # 配位数計算のカットオフ [Å]
    ideal_coordination: int = 8           # 理想配位数（BCC=8, FCC=12）
    crystal_structure: str = 'BCC'       # 結晶構造
    max_defect_atoms: int = sys.maxsize  # 欠陥原子数の上限（メモリ制限に応じて変更）
    min_defect_atoms: int = 100          # 欠陥原子数の下限
    
    # 欠陥検出パラメータ
    coord_tolerance: int = 1              # 配位数の許容誤差
    use_cluster_definition: bool = True   # クラスター定義を使用
    cluster_definition_path: Optional[str] = None
    
    # 材料特有の特徴
    calculate_defect_density: bool = True
    calculate_local_strain: bool = True
    calculate_coordination_distribution: bool = True

# ===============================
# Helper Functions
# ===============================

def _ensure_numpy(arr):
    """CuPy配列をNumPy配列に変換（必要な場合のみ）"""
    if hasattr(arr, 'get'):
        return arr.get()
    return arr

# ===============================
# Material MD Features GPU Class
# ===============================

class MaterialMDFeaturesGPU(MDFeaturesGPU):
    """
    材料解析用MD特徴抽出のGPU実装
    
    欠陥領域を自動検出して、計算効率と精度のバランスを取る
    """
    
    def __init__(self,
                 config: Optional[MaterialMDFeatureConfig] = None,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        config : MaterialMDFeatureConfig
            材料用特徴抽出設定
        memory_manager : GPUMemoryManager
            メモリ管理
        """
        # 材料用設定を使用
        material_config = config or MaterialMDFeatureConfig()
        
        # 基底クラス初期化（基底クラスのCUDAカーネルも継承）
        super().__init__(config=material_config, memory_manager=memory_manager, **kwargs)
        
        # 材料特有の設定を保持
        self.material_config = material_config
        
        # 結晶構造別の理想配位数
        self.ideal_coordination_map = {
            'BCC': 8,   # Body-Centered Cubic
            'FCC': 12,  # Face-Centered Cubic
            'HCP': 12,  # Hexagonal Close-Packed
            'SC': 6,    # Simple Cubic
        }
        
        # 材料特有のCUDAカーネルをコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_material_kernels()
        
        if not HAS_SCIPY:
            logger.warning("scipy not available - defect detection will be limited")
    
    def _compile_material_kernels(self):
        """材料特有のカスタムカーネルをコンパイル（基底クラスのカーネルに追加）"""
        try:
            self.coord_kernel = cp.RawKernel(COORDINATION_KERNEL, 'calculate_coordination_kernel')
            self.defect_kernel = cp.RawKernel(DEFECT_DETECTION_KERNEL, 'detect_defects_kernel')
            self.strain_kernel = cp.RawKernel(LOCAL_STRAIN_KERNEL, 'calculate_local_strain_kernel')
            logger.debug("Material CUDA kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile material kernels: {e}")
            self.coord_kernel = None
            self.defect_kernel = None
            self.strain_kernel = None
    
    def extract_md_features(self,
                           trajectory: np.ndarray,
                           backbone_indices: Optional[np.ndarray] = None,
                           cluster_definition_path: Optional[str] = None,
                           atom_types: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        材料用MD特徴量を抽出（欠陥領域フォーカス版）
        
        Parameters
        ----------
        trajectory : np.ndarray
            トラジェクトリ (n_frames, n_atoms, 3)
        backbone_indices : np.ndarray, optional
            重要原子のインデックス（Noneの場合は自動検出）
        cluster_definition_path : str, optional
            クラスター定義ファイルパス
        atom_types : np.ndarray, optional
            原子タイプ配列
            
        Returns
        -------
        dict
            抽出された特徴量
        """
        n_frames, n_atoms, _ = trajectory.shape
        
        logger.info(f"🔬 Material MD Feature Extraction")
        logger.info(f"   Trajectory: {n_frames} frames, {n_atoms} atoms")
        logger.info(f"   Crystal structure: {self.material_config.crystal_structure}")
        
        # 欠陥領域の特定
        if backbone_indices is None and self.material_config.auto_detect_defects:
            logger.info("🎯 Auto-detecting defect regions...")
            
            # クラスター定義優先
            if cluster_definition_path or self.material_config.cluster_definition_path:
                path = cluster_definition_path or self.material_config.cluster_definition_path
                backbone_indices = self._get_defect_region_from_clusters(
                    trajectory[0], path
                )
                logger.info(f"   Loaded {len(backbone_indices)} atoms from cluster definition")
                
            # 配位数ベースの検出
            else:
                backbone_indices = self._detect_defects_by_coordination(
                    trajectory[0],
                    self.material_config.coordination_cutoff,
                    self.ideal_coordination_map.get(
                        self.material_config.crystal_structure,
                        self.material_config.ideal_coordination
                    )
                )
                logger.info(f"   Detected {len(backbone_indices)} defect region atoms")
        
        elif backbone_indices is None:
            # 自動検出無効の場合は全原子（非推奨）
            logger.warning("⚠️ Using all atoms - this may be slow!")
            backbone_indices = np.arange(n_atoms)
        
        # 欠陥原子数の制限
        if len(backbone_indices) > self.material_config.max_defect_atoms:
            logger.warning(f"Limiting defect atoms from {len(backbone_indices)} "
                         f"to {self.material_config.max_defect_atoms}")
            # ランダムサンプリング
            np.random.seed(42)
            backbone_indices = np.random.choice(
                backbone_indices,
                self.material_config.max_defect_atoms,
                replace=False
            )
        
        # 欠陥原子が少なすぎる場合の警告
        if len(backbone_indices) < self.material_config.min_defect_atoms:
            logger.warning(f"Only {len(backbone_indices)} defect atoms found "
                         f"(minimum recommended: {self.material_config.min_defect_atoms})")
        
        # 基本的なMD特徴を計算（欠陥領域のみ）
        logger.info(f"📊 Computing features for {len(backbone_indices)} atoms...")
        
        # 欠陥領域のトラジェクトリ
        defect_trajectory = trajectory[:, backbone_indices]
        
        # 基底クラスのメソッドを呼ぶ（RMSD, Rg等）
        features = super().extract_md_features(defect_trajectory, None)
        
        # 材料特有の特徴を追加
        if self.material_config.calculate_defect_density:
            features['defect_density'] = len(backbone_indices) / n_atoms
            features['n_defect_atoms'] = len(backbone_indices)
        
        if self.material_config.calculate_coordination_distribution:
            features['coordination_distribution'] = self._calculate_coordination_distribution(
                trajectory, backbone_indices
            )
        
        if self.material_config.calculate_local_strain:
            features['local_strain'] = self._calculate_local_strain(
                trajectory, backbone_indices
            )
        
        # 欠陥インデックスも保存（後続処理用）
        features['defect_indices'] = backbone_indices
        
        # 統計情報表示
        self._print_material_stats(features, n_atoms)
        
        return features
    
    def _get_defect_region_from_clusters(self,
                                        positions: np.ndarray,
                                        cluster_path: str) -> np.ndarray:
        """
        クラスター定義から欠陥領域を取得
        
        Parameters
        ----------
        positions : np.ndarray
            原子位置 (n_atoms, 3)
        cluster_path : str
            クラスター定義ファイルパス
            
        Returns
        -------
        np.ndarray
            欠陥領域の原子インデックス
        """
        if not Path(cluster_path).exists():
            logger.warning(f"Cluster file not found: {cluster_path}")
            return np.array([])
        
        # クラスター定義読み込み
        with open(cluster_path, 'r') as f:
            clusters = json.load(f)
        
        defect_atoms = set()
        
        # 欠陥クラスター（ID=0以外）の原子を収集
        for cluster_id, atoms in clusters.items():
            # 文字列キーを処理
            cid = str(cluster_id)
            if cid != "0" and cid != 0:  # 0は通常、完全結晶領域
                if isinstance(atoms, list):
                    defect_atoms.update(atoms)
        
        if not defect_atoms:
            logger.warning("No defect atoms found in cluster definition")
            return np.array([])
        
        # 欠陥周辺領域も含める
        if HAS_SCIPY and self.material_config.defect_expansion_radius > 0:
            expanded_atoms = self._expand_defect_region(
                positions, 
                list(defect_atoms),
                self.material_config.defect_expansion_radius
            )
            return np.array(sorted(expanded_atoms))
        
        return np.array(sorted(defect_atoms))
    
    def _detect_defects_by_coordination(self,
                                       positions: np.ndarray,
                                       cutoff: float,
                                       ideal_coord: int) -> np.ndarray:
        """
        配位数から欠陥を自動検出（CUDAカーネル使用）
        
        Parameters
        ----------
        positions : np.ndarray
            原子位置 (n_atoms, 3)
        cutoff : float
            配位数計算のカットオフ距離
        ideal_coord : int
            理想配位数
            
        Returns
        -------
        np.ndarray
            欠陥領域の原子インデックス
        """
        n_atoms = positions.shape[0]
        
        # GPU高速版
        if self.is_gpu and self.coord_kernel is not None:
            logger.debug("Using CUDA kernel for coordination calculation")
            
            # GPU転送
            positions_gpu = cp.asarray(positions, dtype=cp.float32)
            coord_numbers_gpu = cp.zeros(n_atoms, dtype=cp.int32)
            
            # カーネル実行パラメータ
            threads_per_block = 256
            blocks = (n_atoms + threads_per_block - 1) // threads_per_block
            
            # 配位数計算（CUDAカーネル）
            self.coord_kernel(
                (blocks,), (threads_per_block,),
                (positions_gpu, coord_numbers_gpu, cutoff * cutoff, n_atoms)
            )
            
            # 欠陥検出（CUDAカーネル）
            if self.defect_kernel is not None:
                defect_mask_gpu = cp.zeros(n_atoms, dtype=cp.bool_)
                tolerance = self.material_config.coord_tolerance
                
                self.defect_kernel(
                    (blocks,), (threads_per_block,),
                    (coord_numbers_gpu, defect_mask_gpu, ideal_coord, tolerance, n_atoms)
                )
                
                defect_atoms = cp.where(defect_mask_gpu)[0]
            else:
                # フォールバック
                coord_numbers = cp.asnumpy(coord_numbers_gpu)
                tolerance = self.material_config.coord_tolerance
                defect_mask = np.abs(coord_numbers - ideal_coord) > tolerance
                defect_atoms = np.where(defect_mask)[0]
            
            # CPUに転送
            if isinstance(defect_atoms, cp.ndarray):
                defect_atoms = cp.asnumpy(defect_atoms)
            
            logger.info(f"   Found {len(defect_atoms)} atoms with non-ideal coordination (GPU)")
            
        # CPU版（scipy使用）
        elif HAS_SCIPY:
            logger.debug("Using scipy for coordination calculation")

            positions_cpu = _ensure_numpy(positions) 
            tree = cKDTree(positions_cpu)
            neighbors = tree.query_ball_tree(tree, cutoff)
            coord_numbers = np.array([len(n) - 1 for n in neighbors])  # -1 for self
            
            # 欠陥原子の特定
            tolerance = self.material_config.coord_tolerance
            defect_mask = np.abs(coord_numbers - ideal_coord) > tolerance
            defect_atoms = np.where(defect_mask)[0]
            
            logger.info(f"   Found {len(defect_atoms)} atoms with non-ideal coordination (CPU)")
            
            # 配位数分布の表示
            unique_coords, counts = np.unique(coord_numbers[defect_mask], return_counts=True)
            for coord, count in zip(unique_coords, counts):
                if count > 10:  # 主要な配位数のみ表示
                    logger.debug(f"     Coordination {coord}: {count} atoms")
        
        else:
            logger.warning("Neither CUDA nor scipy available - using simple defect detection")
            # 簡易版：ランダムサンプリング
            n_sample = min(1000, n_atoms // 10)
            defect_atoms = np.random.choice(n_atoms, n_sample, replace=False)
        
        # 欠陥が少なすぎる場合の処理
        if len(defect_atoms) < self.material_config.min_defect_atoms:
            logger.warning(f"Too few defects found ({len(defect_atoms)}), "
                         f"expanding tolerance...")
            if self.is_gpu and self.coord_kernel:
                # GPU版での再計算（tolerance=0）
                defect_mask = coord_numbers_gpu != ideal_coord
                defect_atoms = cp.asnumpy(cp.where(defect_mask)[0])
            elif HAS_SCIPY:
                defect_mask = coord_numbers != ideal_coord
                defect_atoms = np.where(defect_mask)[0]
        
        # 欠陥周辺領域も含める
        if self.material_config.defect_expansion_radius > 0:
            expanded_atoms = self._expand_defect_region(
                positions,
                defect_atoms,
                self.material_config.defect_expansion_radius
            )
            return np.array(sorted(expanded_atoms))
        
        return defect_atoms
    
    def _expand_defect_region(self,
                         positions: np.ndarray,
                         defect_atoms: Union[List[int], np.ndarray],
                         radius: float) -> Set[int]:
        """
        欠陥領域を周辺原子まで拡張
        """
        if not HAS_SCIPY:
            return set(defect_atoms)
        
        # CuPy配列の場合はNumPyに変換
        if hasattr(positions, 'get'):  # CuPy配列かチェック
            positions_cpu = positions.get()
        else:
            positions_cpu = positions
        
        # defect_atomsもCuPy配列の可能性あり
        if hasattr(defect_atoms, 'get'):
            defect_atoms = defect_atoms.get()
        
        tree = cKDTree(positions_cpu)  # CPU版を使用
        expanded = set(defect_atoms)
        
        # 各欠陥原子の周辺を追加
        for atom in defect_atoms:
            if atom < len(positions_cpu):
                neighbors = tree.query_ball_point(positions_cpu[atom], r=radius)
                expanded.update(neighbors)
        
        logger.debug(f"   Expanded from {len(defect_atoms)} to {len(expanded)} atoms")
        
        return expanded
    
    def _calculate_coordination_distribution(self,
                                           trajectory: np.ndarray,
                                           defect_indices: np.ndarray) -> np.ndarray:
        """
        欠陥領域の配位数分布を時系列で計算
        
        Parameters
        ----------
        trajectory : np.ndarray
            全体のトラジェクトリ
        defect_indices : np.ndarray
            欠陥原子のインデックス
            
        Returns
        -------
        np.ndarray
            配位数分布の時系列 (n_frames, max_coord+1)
        """
        if not HAS_SCIPY:
            return np.array([])
    
        # defect_indicesもNumPyに変換！
        defect_indices = _ensure_numpy(defect_indices)  # ← 追加！
        
        n_frames = trajectory.shape[0]
        max_coord = 20
        coord_dist = np.zeros((n_frames, max_coord + 1))
        
        cutoff = self.material_config.coordination_cutoff
        
        # フレームごとに配位数分布を計算
        for frame in range(0, n_frames, 10):
            positions = trajectory[frame]
            positions_cpu = _ensure_numpy(positions)
            tree = cKDTree(positions_cpu)
            
            for atom_idx in defect_indices[:100]:
                if atom_idx < len(positions_cpu):
                    neighbors = tree.query_ball_point(positions_cpu[atom_idx], cutoff)
                    coord = len(neighbors) - 1
                    if 0 <= coord <= max_coord:
                        coord_dist[frame, coord] += 1
        
        return coord_dist
        
    def _calculate_local_strain(self,
                              trajectory: np.ndarray,
                              defect_indices: np.ndarray) -> np.ndarray:
        """
        欠陥領域の局所歪みを計算（CUDAカーネル対応）
        
        Parameters
        ----------
        trajectory : np.ndarray
            全体のトラジェクトリ
        defect_indices : np.ndarray
            欠陥原子のインデックス
            
        Returns
        -------
        np.ndarray
            局所歪みの時系列 (n_frames,)
        """
        n_frames = trajectory.shape[0]
        local_strain = np.zeros(n_frames)
        
        if n_frames < 2:
            return local_strain
        
        # 初期構造を参照
        ref_positions = trajectory[0, defect_indices]
        
        # GPU高速版
        if self.is_gpu and self.strain_kernel is not None:
            logger.debug("Using CUDA kernel for strain calculation")
            
            ref_positions_gpu = cp.asarray(ref_positions, dtype=cp.float32)
            n_defects = len(defect_indices)
            
            # カーネルパラメータ
            threads_per_block = 256
            blocks = (n_defects + threads_per_block - 1) // threads_per_block
            
            for frame in range(1, n_frames):
                curr_positions_gpu = cp.asarray(
                    trajectory[frame, defect_indices], dtype=cp.float32
                )
                strain_per_atom_gpu = cp.zeros(n_defects, dtype=cp.float32)
                
                # CUDAカーネル実行
                self.strain_kernel(
                    (blocks,), (threads_per_block,),
                    (ref_positions_gpu, curr_positions_gpu, strain_per_atom_gpu, n_defects)
                )
                
                # 平均歪み
                local_strain[frame] = float(cp.mean(strain_per_atom_gpu))
        
        # CPU版
        else:
            for frame in range(1, n_frames):
                curr_positions = trajectory[frame, defect_indices]
                
                # 簡易歪み計算（RMSD的なアプローチ）
                displacement = curr_positions - ref_positions
                strain = np.sqrt(np.mean(displacement ** 2))
                local_strain[frame] = strain
        
        return local_strain
    
    def _print_material_stats(self, features: Dict[str, np.ndarray], n_atoms: int):
        """材料特有の統計情報を表示"""
        logger.info("\n📊 Material feature summary:")
        
        # 欠陥密度
        if 'defect_density' in features:
            density = features['defect_density']
            n_defects = features.get('n_defect_atoms', 0)
            logger.info(f"   Defect density: {density:.1%} ({n_defects}/{n_atoms} atoms)")
        
        # RMSD統計（欠陥領域）
        if 'rmsd' in features:
            rmsd = features['rmsd']
            logger.info(f"   Defect RMSD: min={np.min(rmsd):.3f}, "
                       f"max={np.max(rmsd):.3f}, mean={np.mean(rmsd):.3f} Å")
        
        # Rg統計（欠陥領域）
        if 'radius_of_gyration' in features:
            rg = features['radius_of_gyration']
            logger.info(f"   Defect Rg: min={np.min(rg):.3f}, "
                       f"max={np.max(rg):.3f}, mean={np.mean(rg):.3f} Å")
        
        # 局所歪み
        if 'local_strain' in features and features['local_strain'].size > 0:
            strain = features['local_strain']
            max_strain = np.max(strain)
            logger.info(f"   Max local strain: {max_strain:.3f} Å")
            if max_strain > 1.0:
                logger.warning(f"   ⚠️ High strain detected: {max_strain:.3f} Å")
        
        # 配位数分布
        if 'coordination_distribution' in features:
            coord_dist = features['coordination_distribution']
            if coord_dist.size > 0:
                # 最も頻度の高い配位数
                avg_dist = np.mean(coord_dist, axis=0)
                if avg_dist.sum() > 0:
                    most_common_coord = np.argmax(avg_dist)
                    logger.info(f"   Most common coordination: {most_common_coord}")

# ===============================
# Convenience Functions
# ===============================

def extract_material_md_features(
    trajectory: np.ndarray,
    cluster_definition_path: Optional[str] = None,
    atom_types: Optional[np.ndarray] = None,
    crystal_structure: str = 'BCC',
    auto_detect: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    材料MD特徴抽出の便利関数
    
    Parameters
    ----------
    trajectory : np.ndarray
        トラジェクトリ (n_frames, n_atoms, 3)
    cluster_definition_path : str, optional
        クラスター定義ファイル
    atom_types : np.ndarray, optional
        原子タイプ
    crystal_structure : str
        結晶構造 ('BCC', 'FCC', 'HCP')
    auto_detect : bool
        欠陥領域の自動検出を有効化
        
    Returns
    -------
    Dict[str, np.ndarray]
        抽出された特徴量
    """
    config = MaterialMDFeatureConfig(
        crystal_structure=crystal_structure,
        auto_detect_defects=auto_detect,
        cluster_definition_path=cluster_definition_path
    )
    
    extractor = MaterialMDFeaturesGPU(config=config)
    
    return extractor.extract_md_features(
        trajectory=trajectory,
        cluster_definition_path=cluster_definition_path,
        atom_types=atom_types
    )

def get_defect_region_indices(
    positions: np.ndarray,
    crystal_structure: str = 'BCC',
    cutoff: float = 3.5,
    expansion_radius: float = 5.0
) -> np.ndarray:
    """
    欠陥領域の原子インデックスを取得する便利関数
    
    Parameters
    ----------
    positions : np.ndarray
        原子位置 (n_atoms, 3)
    crystal_structure : str
        結晶構造
    cutoff : float
        配位数計算のカットオフ
    expansion_radius : float
        欠陥周辺の拡張半径
        
    Returns
    -------
    np.ndarray
        欠陥領域の原子インデックス
    """
    config = MaterialMDFeatureConfig(
        crystal_structure=crystal_structure,
        coordination_cutoff=cutoff,
        defect_expansion_radius=expansion_radius
    )
    
    extractor = MaterialMDFeaturesGPU(config=config)
    
    ideal_coord = extractor.ideal_coordination_map.get(crystal_structure, 8)
    
    return extractor._detect_defects_by_coordination(
        positions, cutoff, ideal_coord
    )
