"""
MD Features Extraction (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MD特徴量の抽出をGPUで高速化！
RMSD、Rg、接触マップとか全部速いよ〜！💕

by 環ちゃん
"""
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
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

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, GPUTimer, handle_gpu_errors
from ..core import get_optimal_block_size

logger = logging.getLogger('lambda3_gpu.structures.md_features')

# ===============================
# CUDA Kernels for MD Features
# ===============================

# RMSD計算カーネル
RMSD_KERNEL = r'''
extern "C" __global__
void calculate_rmsd_kernel(
    const float* __restrict__ coords1,  // (n_atoms, 3)
    const float* __restrict__ coords2,  // (n_atoms, 3)
    float* __restrict__ rmsd,          // スカラー出力
    const int n_atoms
) {
    // 共有メモリで部分和を計算
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 各スレッドが1原子の差分二乗を計算
    float sum = 0.0f;
    if (i < n_atoms) {
        float dx = coords1[i*3 + 0] - coords2[i*3 + 0];
        float dy = coords1[i*3 + 1] - coords2[i*3 + 1];
        float dz = coords1[i*3 + 2] - coords2[i*3 + 2];
        sum = dx*dx + dy*dy + dz*dz;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 共有メモリ内でリダクション
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // ブロックごとの結果を書き込み
    if (tid == 0) {
        atomicAdd(rmsd, sdata[0]);
    }
}
'''

# Radius of Gyration計算カーネル
RG_KERNEL = r'''
extern "C" __global__
void calculate_rg_kernel(
    const float* __restrict__ coords,   // (n_atoms, 3)
    float* __restrict__ center,         // (3,) 重心
    float* __restrict__ rg,            // スカラー出力
    const int n_atoms
) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 1: 重心計算（別カーネルで事前計算済みと仮定）
    
    // Step 2: 各原子の重心からの距離二乗
    float sum = 0.0f;
    if (i < n_atoms) {
        float dx = coords[i*3 + 0] - center[0];
        float dy = coords[i*3 + 1] - center[1];
        float dz = coords[i*3 + 2] - center[2];
        sum = dx*dx + dy*dy + dz*dz;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // リダクション
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(rg, sdata[0]);
    }
}
'''

# ===============================
# Configuration
# ===============================

@dataclass
class MDFeatureConfig:
    """MD特徴抽出の設定"""
    use_contacts: bool = False      # 接触マップ（メモリ集約的）
    use_rmsd: bool = True          # RMSD
    use_rg: bool = True            # Radius of gyration
    use_dihedrals: bool = True     # 二面角
    contact_cutoff: float = 8.0    # 接触判定距離（Å）
    rmsd_reference: int = 0        # RMSD参照フレーム
    batch_size: Optional[int] = None

# ===============================
# MD Features GPU Class
# ===============================

class MDFeaturesGPU(GPUBackend):
    """
    MD特徴抽出のGPU実装クラス
    超高速でMD特徴量を計算するよ〜！
    """
    
    def __init__(self,
                 config: Optional[MDFeatureConfig] = None,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        config : MDFeatureConfig
            特徴抽出設定
        memory_manager : GPUMemoryManager
            メモリ管理
        """
        super().__init__(**kwargs)
        self.config = config or MDFeatureConfig()
        self.memory_manager = memory_manager or GPUMemoryManager()
        
        # CUDAカーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.rmsd_kernel = cp.RawKernel(RMSD_KERNEL, 'calculate_rmsd_kernel')
            self.rg_kernel = cp.RawKernel(RG_KERNEL, 'calculate_rg_kernel')
            logger.debug("MD feature kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.rmsd_kernel = None
            self.rg_kernel = None
    
    @handle_gpu_errors
    def extract_md_features(self,
                          trajectory: np.ndarray,
                          backbone_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        MD特徴量を抽出（GPU高速版）
        
        Parameters
        ----------
        trajectory : np.ndarray
            トラジェクトリ (n_frames, n_atoms, 3)
        backbone_indices : np.ndarray, optional
            バックボーン原子のインデックス
            
        Returns
        -------
        dict
            抽出された特徴量
        """
        with self.timer('extract_md_features'):
            n_frames, n_atoms, _ = trajectory.shape
            features = {}
            
            logger.info(f"🚀 Extracting MD features on {'GPU' if self.is_gpu else 'CPU'}")
            logger.info(f"   Trajectory: {n_frames} frames, {n_atoms} atoms")
            
            # バッチサイズ決定
            if self.config.batch_size is None:
                batch_size = self.memory_manager.estimate_batch_size(
                    (1, n_atoms, 3), dtype=np.float32
                )
                batch_size = min(batch_size, n_frames)
            else:
                batch_size = self.config.batch_size
            
            # 1. RMSD計算
            if self.config.use_rmsd:
                with self.timer('rmsd'):
                    features['rmsd'] = self._calculate_rmsd_batched(
                        trajectory, batch_size
                    )
            
            # 2. Radius of gyration
            if self.config.use_rg:
                with self.timer('radius_of_gyration'):
                    features['radius_of_gyration'] = self._calculate_rg_batched(
                        trajectory, batch_size
                    )
            
            # 3. Center of mass
            with self.timer('com_positions'):
                features['com_positions'] = self._calculate_com_batched(
                    trajectory, batch_size
                )
            
            # 4. 接触マップ（オプション）
            if self.config.use_contacts and backbone_indices is not None:
                with self.timer('contacts'):
                    logger.warning("Contact map calculation is memory intensive!")
                    features['contacts'] = self._calculate_contacts_batched(
                        trajectory, backbone_indices, batch_size
                    )
            
            # 5. 二面角（オプション）
            if self.config.use_dihedrals and backbone_indices is not None:
                with self.timer('dihedrals'):
                    features['dihedrals'] = self._calculate_dihedrals_batched(
                        trajectory, backbone_indices, batch_size
                    )
            
            # 統計情報
            self._print_feature_stats(features)
            
            return features
    
    def _calculate_rmsd_batched(self, 
                               trajectory: np.ndarray,
                               batch_size: int) -> np.ndarray:
        """バッチ処理でRMSD計算"""
        n_frames = trajectory.shape[0]
        rmsd_values = np.zeros(n_frames, dtype=np.float32)
        
        # 参照構造
        ref_frame = self.config.rmsd_reference
        ref_coords = trajectory[ref_frame]
        
        if self.is_gpu:
            ref_coords_gpu = self.to_gpu(ref_coords)
        
        # バッチ処理
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            batch = trajectory[i:end]
            
            if self.is_gpu:
                batch_gpu = self.to_gpu(batch)
                batch_rmsd = self._calculate_rmsd_gpu(batch_gpu, ref_coords_gpu)
                rmsd_values[i:end] = self.to_cpu(batch_rmsd)
            else:
                batch_rmsd = self._calculate_rmsd_cpu(batch, ref_coords)
                rmsd_values[i:end] = batch_rmsd
            
            if i % (batch_size * 10) == 0:
                logger.debug(f"RMSD progress: {i}/{n_frames} frames")
        
        return rmsd_values
    
    def _calculate_rmsd_gpu(self,
                          coords_batch: cp.ndarray,
                          ref_coords: cp.ndarray) -> cp.ndarray:
        """GPU版RMSD計算"""
        n_frames = coords_batch.shape[0]
        rmsd_values = cp.zeros(n_frames, dtype=cp.float32)
        
        # ベクトル化計算
        for i in range(n_frames):
            diff = coords_batch[i] - ref_coords
            rmsd_values[i] = cp.sqrt(cp.mean(diff * diff))
        
        return rmsd_values
    
    def _calculate_rmsd_cpu(self,
                          coords_batch: np.ndarray,
                          ref_coords: np.ndarray) -> np.ndarray:
        """CPU版RMSD計算"""
        n_frames = coords_batch.shape[0]
        rmsd_values = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(n_frames):
            diff = coords_batch[i] - ref_coords
            rmsd_values[i] = np.sqrt(np.mean(diff * diff))
        
        return rmsd_values
    
    def _calculate_rg_batched(self,
                            trajectory: np.ndarray,
                            batch_size: int) -> np.ndarray:
        """バッチ処理でRadius of gyration計算"""
        n_frames = trajectory.shape[0]
        rg_values = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            batch = trajectory[i:end]
            
            if self.is_gpu:
                batch_gpu = self.to_gpu(batch)
                batch_rg = self._calculate_rg_gpu(batch_gpu)
                rg_values[i:end] = self.to_cpu(batch_rg)
            else:
                batch_rg = self._calculate_rg_cpu(batch)
                rg_values[i:end] = batch_rg
        
        return rg_values
    
    def _calculate_rg_gpu(self, coords_batch: cp.ndarray) -> cp.ndarray:
        """GPU版Rg計算"""
        n_frames, n_atoms, _ = coords_batch.shape
        rg_values = cp.zeros(n_frames, dtype=cp.float32)
        
        for i in range(n_frames):
            # 重心
            center = cp.mean(coords_batch[i], axis=0)
            
            # 重心からの距離
            diff = coords_batch[i] - center
            rg_squared = cp.mean(cp.sum(diff * diff, axis=1))
            rg_values[i] = cp.sqrt(rg_squared)
        
        return rg_values
    
    def _calculate_rg_cpu(self, coords_batch: np.ndarray) -> np.ndarray:
        """CPU版Rg計算"""
        n_frames, n_atoms, _ = coords_batch.shape
        rg_values = np.zeros(n_frames, dtype=np.float32)
        
        for i in range(n_frames):
            center = np.mean(coords_batch[i], axis=0)
            diff = coords_batch[i] - center
            rg_squared = np.mean(np.sum(diff * diff, axis=1))
            rg_values[i] = np.sqrt(rg_squared)
        
        return rg_values
    
    def _calculate_com_batched(self,
                             trajectory: np.ndarray,
                             batch_size: int) -> np.ndarray:
        """バッチ処理でCOM計算"""
        n_frames = trajectory.shape[0]
        com_positions = np.zeros((n_frames, 3), dtype=np.float32)
        
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            batch = trajectory[i:end]
            
            if self.is_gpu:
                batch_gpu = self.to_gpu(batch)
                batch_com = cp.mean(batch_gpu, axis=1)
                com_positions[i:end] = self.to_cpu(batch_com)
            else:
                com_positions[i:end] = np.mean(batch, axis=1)
        
        return com_positions
    
    def _calculate_contacts_batched(self,
                                  trajectory: np.ndarray,
                                  backbone_indices: np.ndarray,
                                  batch_size: int) -> np.ndarray:
        """バッチ処理で接触マップ計算（メモリ効率重視）"""
        n_frames = trajectory.shape[0]
        n_backbone = len(backbone_indices)
        
        # メモリ節約のため、接触数のみ保存
        contact_counts = np.zeros(n_frames, dtype=np.int32)
        
        logger.warning(f"Contact calculation for {n_backbone} backbone atoms")
        
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            batch = trajectory[i:end, backbone_indices]
            
            if self.is_gpu:
                batch_gpu = self.to_gpu(batch)
                batch_contacts = self._count_contacts_gpu(batch_gpu)
                contact_counts[i:end] = self.to_cpu(batch_contacts)
            else:
                batch_contacts = self._count_contacts_cpu(batch)
                contact_counts[i:end] = batch_contacts
        
        return contact_counts
    
    def _count_contacts_gpu(self, coords_batch: cp.ndarray) -> cp.ndarray:
        """GPU版接触数カウント"""
        n_frames, n_atoms, _ = coords_batch.shape
        contact_counts = cp.zeros(n_frames, dtype=cp.int32)
        cutoff_sq = self.config.contact_cutoff ** 2
        
        for i in range(n_frames):
            # 距離行列計算
            distances_sq = cp.sum((coords_batch[i, :, None, :] - 
                                 coords_batch[i, None, :, :]) ** 2, axis=2)
            
            # 対角要素を除外して接触をカウント
            cp.fill_diagonal(distances_sq, cutoff_sq + 1)
            contact_counts[i] = cp.sum(distances_sq < cutoff_sq) // 2
        
        return contact_counts
    
    def _count_contacts_cpu(self, coords_batch: np.ndarray) -> np.ndarray:
        """CPU版接触数カウント"""
        n_frames, n_atoms, _ = coords_batch.shape
        contact_counts = np.zeros(n_frames, dtype=np.int32)
        cutoff_sq = self.config.contact_cutoff ** 2
        
        for i in range(n_frames):
            count = 0
            for j in range(n_atoms):
                for k in range(j+1, n_atoms):
                    dist_sq = np.sum((coords_batch[i, j] - coords_batch[i, k]) ** 2)
                    if dist_sq < cutoff_sq:
                        count += 1
            contact_counts[i] = count
        
        return contact_counts
    
    def _calculate_dihedrals_batched(self,
                                   trajectory: np.ndarray,
                                   backbone_indices: np.ndarray,
                                   batch_size: int) -> np.ndarray:
        """バッチ処理で二面角計算"""
        n_frames = trajectory.shape[0]
        
        # φ/ψ角のインデックス（簡略化）
        n_dihedrals = max(0, len(backbone_indices) - 3)
        if n_dihedrals == 0:
            return np.array([])
        
        dihedral_values = np.zeros((n_frames, n_dihedrals), dtype=np.float32)
        
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            batch = trajectory[i:end, backbone_indices]
            
            if self.is_gpu:
                batch_gpu = self.to_gpu(batch)
                batch_dihedrals = self._calculate_dihedrals_gpu(batch_gpu)
                dihedral_values[i:end] = self.to_cpu(batch_dihedrals)
            else:
                batch_dihedrals = self._calculate_dihedrals_cpu(batch)
                dihedral_values[i:end] = batch_dihedrals
        
        return dihedral_values
    
    def _calculate_dihedrals_gpu(self, coords_batch: cp.ndarray) -> cp.ndarray:
        """GPU版二面角計算"""
        n_frames, n_atoms, _ = coords_batch.shape
        n_dihedrals = n_atoms - 3
        dihedrals = cp.zeros((n_frames, n_dihedrals), dtype=cp.float32)
        
        for frame in range(n_frames):
            for i in range(n_dihedrals):
                # 4原子の座標
                p0 = coords_batch[frame, i]
                p1 = coords_batch[frame, i+1]
                p2 = coords_batch[frame, i+2]
                p3 = coords_batch[frame, i+3]
                
                # ベクトル計算
                b1 = p1 - p0
                b2 = p2 - p1
                b3 = p3 - p2
                
                # 法線ベクトル
                n1 = cp.cross(b1, b2)
                n2 = cp.cross(b2, b3)
                
                # 二面角
                x = cp.dot(n1, n2)
                y = cp.dot(cp.cross(n1, b2/cp.linalg.norm(b2)), n2)
                dihedrals[frame, i] = cp.arctan2(y, x)
        
        return dihedrals
    
    def _calculate_dihedrals_cpu(self, coords_batch: np.ndarray) -> np.ndarray:
        """CPU版二面角計算（同じロジック）"""
        n_frames, n_atoms, _ = coords_batch.shape
        n_dihedrals = n_atoms - 3
        dihedrals = np.zeros((n_frames, n_dihedrals), dtype=np.float32)
        
        for frame in range(n_frames):
            for i in range(n_dihedrals):
                p0 = coords_batch[frame, i]
                p1 = coords_batch[frame, i+1]
                p2 = coords_batch[frame, i+2]
                p3 = coords_batch[frame, i+3]
                
                b1 = p1 - p0
                b2 = p2 - p1
                b3 = p3 - p2
                
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                
                x = np.dot(n1, n2)
                y = np.dot(np.cross(n1, b2/np.linalg.norm(b2)), n2)
                dihedrals[frame, i] = np.arctan2(y, x)
        
        return dihedrals
    
    def _print_feature_stats(self, features: Dict[str, np.ndarray]):
        """特徴量の統計情報を出力"""
        logger.info("\n📊 Extracted features summary:")
        
        for name, values in features.items():
            if values.size == 0:
                continue
                
            if values.ndim == 1:
                logger.info(f"   {name}: shape={values.shape}, "
                          f"range=[{np.min(values):.3f}, {np.max(values):.3f}], "
                          f"mean={np.mean(values):.3f}")
            else:
                logger.info(f"   {name}: shape={values.shape}")

# ===============================
# Standalone Functions
# ===============================

def extract_md_features_gpu(trajectory: np.ndarray,
                          config: Optional[MDFeatureConfig] = None,
                          backbone_indices: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """MD特徴抽出のスタンドアロン関数"""
    extractor = MDFeaturesGPU(config)
    return extractor.extract_md_features(trajectory, backbone_indices)

def calculate_rmsd_gpu(coords1: Union[np.ndarray, cp.ndarray],
                     coords2: Union[np.ndarray, cp.ndarray],
                     backend: Optional[GPUBackend] = None) -> float:
    """RMSD計算のスタンドアロン関数"""
    if backend is None:
        backend = GPUBackend()
    
    coords1_gpu = backend.to_gpu(coords1)
    coords2_gpu = backend.to_gpu(coords2)
    
    diff = coords1_gpu - coords2_gpu
    rmsd = backend.xp.sqrt(backend.xp.mean(diff * diff))
    
    return float(backend.to_cpu(rmsd))

def calculate_radius_of_gyration_gpu(coords: Union[np.ndarray, cp.ndarray],
                                   backend: Optional[GPUBackend] = None) -> float:
    """Rg計算のスタンドアロン関数"""
    if backend is None:
        backend = GPUBackend()
    
    coords_gpu = backend.to_gpu(coords)
    center = backend.xp.mean(coords_gpu, axis=0)
    diff = coords_gpu - center
    rg_squared = backend.xp.mean(backend.xp.sum(diff * diff, axis=1))
    
    return float(backend.xp.sqrt(rg_squared))

def calculate_contacts_gpu(coords: Union[np.ndarray, cp.ndarray],
                         cutoff: float = 8.0,
                         backend: Optional[GPUBackend] = None) -> int:
    """接触数計算のスタンドアロン関数"""
    if backend is None:
        backend = GPUBackend()
    
    coords_gpu = backend.to_gpu(coords)
    n_atoms = coords_gpu.shape[0]
    
    # 距離行列
    if backend.is_gpu and HAS_GPU:
        distances = cp_cdist(coords_gpu, coords_gpu)
    else:
        from scipy.spatial.distance import cdist
        distances = cdist(coords_gpu, coords_gpu)
    
    # 対角要素を除外
    backend.xp.fill_diagonal(distances, cutoff + 1)
    
    # 接触カウント
    contacts = backend.xp.sum(distances < cutoff) // 2
    
    return int(contacts)

def calculate_dihedrals_gpu(coords: Union[np.ndarray, cp.ndarray],
                          indices: List[Tuple[int, int, int, int]],
                          backend: Optional[GPUBackend] = None) -> np.ndarray:
    """二面角計算のスタンドアロン関数"""
    if backend is None:
        backend = GPUBackend()
    
    coords_gpu = backend.to_gpu(coords)
    dihedrals = []
    
    for i, j, k, l in indices:
        p0 = coords_gpu[i]
        p1 = coords_gpu[j]
        p2 = coords_gpu[k]
        p3 = coords_gpu[l]
        
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        
        n1 = backend.xp.cross(b1, b2)
        n2 = backend.xp.cross(b2, b3)
        
        x = backend.xp.dot(n1, n2)
        y = backend.xp.dot(backend.xp.cross(n1, b2/backend.xp.linalg.norm(b2)), n2)
        
        dihedral = backend.xp.arctan2(y, x)
        dihedrals.append(float(dihedral))
    
    return np.array(dihedrals)
