#!/usr/bin/env python3
"""
Cluster-Level Lambda³ Structure Computation for Materials (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用：クラスターレベルのLambda³構造をGPUで計算！
金属結晶・ポリマー・セラミックスに対応！💎

残基版(residue_structures_gpu.py)をベースに材料解析に特化

by 環ちゃん - Material Edition v1.0
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# GPU imports
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors
from .cluster_id_mapping import ClusterIDMapper

# cluster_com_kernelのインポート（エラー時はNone）
try:
    from ..core import cluster_com_kernel
except ImportError:
    cluster_com_kernel = None

logger = logging.getLogger('lambda3_gpu.material.structures')

# ===============================
# Helper Functions
# ===============================

def get_optimal_block_size(n_elements: int, max_block_size: int = 1024) -> int:
    """最適なブロックサイズを計算"""
    if n_elements <= 32:
        return 32
    elif n_elements <= 64:
        return 64
    elif n_elements <= 128:
        return 128
    elif n_elements <= 256:
        return 256
    elif n_elements <= 512:
        return 512
    else:
        return min(1024, max_block_size)

# ===============================
# Data Classes (Material版)
# ===============================

@dataclass
class ClusterStructureResult:
    """クラスターレベル構造計算の結果（材料特化版）"""
    cluster_lambda_f: np.ndarray         # (n_frames, n_clusters, 3) - パディング済み
    cluster_lambda_f_mag: np.ndarray     # (n_frames, n_clusters) - パディング済み
    cluster_rho_t: np.ndarray           # (n_frames, n_clusters)
    cluster_coupling: np.ndarray        # (n_frames, n_clusters, n_clusters)
    cluster_centers: np.ndarray         # (n_frames, n_clusters, 3)
    
    # Material特有の追加メトリクス
    coordination_numbers: np.ndarray    # (n_frames, n_clusters) 配位数
    local_strain: np.ndarray           # (n_frames, n_clusters, 3, 3) 歪みテンソル
    element_composition: Dict[int, Dict[str, int]]  # 各クラスターの元素組成
    bond_angles: Optional[np.ndarray] = None  # ボンド角度分布
    
    @property
    def n_frames(self) -> int:
        """全配列で統一されたフレーム数"""
        return self.cluster_centers.shape[0]
    
    @property
    def n_clusters(self) -> int:
        return self.cluster_centers.shape[1]
    
    def validate_shapes(self) -> bool:
        """形状の整合性を確認"""
        n_frames = self.n_frames
        n_clusters = self.n_clusters
        
        expected_shapes = {
            'cluster_lambda_f': (n_frames, n_clusters, 3),
            'cluster_lambda_f_mag': (n_frames, n_clusters),
            'cluster_rho_t': (n_frames, n_clusters),
            'cluster_coupling': (n_frames, n_clusters, n_clusters),
            'cluster_centers': (n_frames, n_clusters, 3),
            'coordination_numbers': (n_frames, n_clusters),
            'local_strain': (n_frames, n_clusters, 3, 3)
        }
        
        for name, expected_shape in expected_shapes.items():
            actual_shape = getattr(self, name).shape
            if actual_shape != expected_shape:
                logger.warning(f"Shape mismatch in {name}: "
                             f"expected {expected_shape}, got {actual_shape}")
                return False
        
        return True
    
    def get_summary_stats(self) -> Dict[str, float]:
        """統計サマリーを辞書で返す（材料特化版）"""
        return {
            'mean_lambda_f': float(np.nanmean(self.cluster_lambda_f_mag)),
            'max_lambda_f': float(np.nanmax(self.cluster_lambda_f_mag)),
            'mean_rho_t': float(np.nanmean(self.cluster_rho_t)),
            'mean_coupling': float(np.nanmean(self.cluster_coupling)),
            'mean_coordination': float(np.nanmean(self.coordination_numbers)),
            'max_strain': float(np.nanmax(np.linalg.norm(self.local_strain, axis=(2,3)))),
            'n_frames': self.n_frames,
            'n_clusters': self.n_clusters
        }

# ===============================
# CUDA Kernels (Material版)
# ===============================

CLUSTER_TENSION_KERNEL = r'''
extern "C" __global__
void compute_cluster_tension_kernel(
    const float* __restrict__ cluster_centers,  // (n_frames, n_clusters, 3)
    float* __restrict__ rho_t,                 // (n_frames, n_clusters)
    const int n_frames,
    const int n_clusters,
    const int window_size
) {
    const int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame_id = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (cluster_id >= n_clusters || frame_id >= n_frames) return;
    
    const int window_half = window_size / 2;
    const int start_frame = max(0, frame_id - window_half);
    const int end_frame = min(n_frames, frame_id + window_half + 1);
    const int local_window = end_frame - start_frame;
    
    // 局所平均
    float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
    
    for (int f = start_frame; f < end_frame; f++) {
        const int idx = (f * n_clusters + cluster_id) * 3;
        mean_x += cluster_centers[idx + 0];
        mean_y += cluster_centers[idx + 1];
        mean_z += cluster_centers[idx + 2];
    }
    
    mean_x /= local_window;
    mean_y /= local_window;
    mean_z /= local_window;
    
    // 局所分散（トレース）
    float var_sum = 0.0f;
    
    for (int f = start_frame; f < end_frame; f++) {
        const int idx = (f * n_clusters + cluster_id) * 3;
        const float dx = cluster_centers[idx + 0] - mean_x;
        const float dy = cluster_centers[idx + 1] - mean_y;
        const float dz = cluster_centers[idx + 2] - mean_z;
        
        var_sum += dx * dx + dy * dy + dz * dz;
    }
    
    rho_t[frame_id * n_clusters + cluster_id] = var_sum / local_window;
}
'''

COORDINATION_NUMBER_KERNEL = r'''
extern "C" __global__
void compute_coordination_kernel(
    const float* __restrict__ positions,      // (n_atoms, 3)
    const int* __restrict__ cluster_mapping,  // (n_atoms,) -> cluster_id
    float* __restrict__ coordination,         // (n_clusters,)
    const float cutoff_sq,                    // カットオフ距離の2乗
    const int n_atoms,
    const int n_clusters
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    __shared__ float local_coord[256];
    local_coord[tid] = 0.0f;
    
    // このクラスターに属する原子を探す
    for (int i = tid; i < n_atoms; i += blockDim.x) {
        if (cluster_mapping[i] == cluster_id) {
            float xi = positions[i * 3 + 0];
            float yi = positions[i * 3 + 1];
            float zi = positions[i * 3 + 2];
            
            int neighbors = 0;
            
            // 他の原子との距離を計算
            for (int j = 0; j < n_atoms; j++) {
                if (i != j) {
                    float xj = positions[j * 3 + 0];
                    float yj = positions[j * 3 + 1];
                    float zj = positions[j * 3 + 2];
                    
                    float dist_sq = (xi-xj)*(xi-xj) + 
                                   (yi-yj)*(yi-yj) + 
                                   (zi-zj)*(zi-zj);
                    
                    if (dist_sq < cutoff_sq) {
                        neighbors++;
                    }
                }
            }
            
            local_coord[tid] += neighbors;
        }
    }
    
    __syncthreads();
    
    // リダクション
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_coord[tid] += local_coord[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        coordination[cluster_id] = local_coord[0];
    }
}
'''

# ===============================
# ClusterStructuresGPU Class
# ===============================

class ClusterStructuresGPU(GPUBackend):
    """
    材料用クラスターレベルLambda³構造計算のGPU実装
    単一フレームイベント対応版！
    """
    
    def __init__(self,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.memory_manager = memory_manager or GPUMemoryManager()

        # 🆕 追加
        self.id_mapper = None
        
        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.tension_kernel = cp.RawKernel(
                CLUSTER_TENSION_KERNEL, 'compute_cluster_tension_kernel'
            )
            self.coord_kernel = cp.RawKernel(
                COORDINATION_NUMBER_KERNEL, 'compute_coordination_kernel'
            )
            logger.debug("Material structure kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.tension_kernel = None
            self.coord_kernel = None
    
    @handle_gpu_errors
    def compute_cluster_structures(self,
                                  trajectory: np.ndarray,
                                  start_frame: int,
                                  end_frame: int,
                                  cluster_atoms: Dict[int, List[int]],
                                  atom_types: np.ndarray,
                                  window_size: int = 50,
                                  cutoff: float = 3.0) -> ClusterStructureResult:
        """
        クラスターレベルのLambda³構造を計算（単一フレーム対応版）
        
        重要：end_frameは包含的（inclusive）として扱う！
        
        Parameters
        ----------
        trajectory : トラジェクトリ
        start_frame, end_frame : フレーム範囲（包含的）
        cluster_atoms : クラスターID → 原子リストのマッピング
        atom_types : 各原子の元素タイプ (文字列配列)
        window_size : ρT計算用窓サイズ
        cutoff : 配位数計算用カットオフ距離 (Å)
        """
        with self.timer('compute_cluster_structures'):
            # 包含的範囲として処理！
            actual_end = end_frame + 1  # スライスのために+1
            n_frames = actual_end - start_frame
            
            logger.info(f"⚙️ Computing cluster-level Lambda³ for materials")
            logger.info(f"   Frames: {start_frame}-{end_frame} ({n_frames} frames)")
            logger.info(f"   Clusters: {len(cluster_atoms)}")
            logger.info(f"   Cutoff: {cutoff} Å")
            
            n_clusters = len(cluster_atoms)
            
            # エッジケース：空のフレーム範囲
            if n_frames <= 0:
                logger.warning(f"Empty frame range: {start_frame}-{end_frame}")
                return self._create_empty_result(n_clusters)
            
            # 入力検証
            if not self._validate_inputs(trajectory, cluster_atoms, atom_types):
                logger.error("Input validation failed")
                return self._create_empty_result(n_clusters)
            
            # 1. クラスター重心計算（包含的スライス）
            with self.timer('cluster_centers'):
                cluster_centers = self._compute_cluster_centers(
                    trajectory[start_frame:actual_end], 
                    cluster_atoms
                )
            
            # 2. クラスターレベルΛF（形状統一版）
            with self.timer('cluster_lambda_f'):
                cluster_lambda_f, cluster_lambda_f_mag = self._compute_cluster_lambda_f(
                    cluster_centers
                )
            
            # 3. クラスターレベルρT
            with self.timer('cluster_rho_t'):
                cluster_rho_t = self._compute_cluster_rho_t(
                    cluster_centers, window_size
                )
            
            # 4. クラスター間カップリング
            with self.timer('cluster_coupling'):
                cluster_coupling = self._compute_cluster_coupling(
                    cluster_centers
                )
            
            # 5. 配位数計算（Material特有！）
            with self.timer('coordination_numbers'):
                coordination_numbers = self._compute_coordination_numbers(
                    trajectory[start_frame:actual_end],
                    cluster_atoms,
                    cutoff
                )
            
            # 6. 局所歪みテンソル（Material特有！）
            with self.timer('local_strain'):
                local_strain = self._compute_local_strain(
                    trajectory[start_frame:actual_end],
                    cluster_atoms
                )
            
            # 7. 元素組成（Material特有！）
            element_composition = self._compute_element_composition(
                cluster_atoms, atom_types
            )
            
            # 結果をCPUに転送
            result = ClusterStructureResult(
                cluster_lambda_f=self.to_cpu(cluster_lambda_f),
                cluster_lambda_f_mag=self.to_cpu(cluster_lambda_f_mag),
                cluster_rho_t=self.to_cpu(cluster_rho_t),
                cluster_coupling=self.to_cpu(cluster_coupling),
                cluster_centers=self.to_cpu(cluster_centers),
                coordination_numbers=self.to_cpu(coordination_numbers),
                local_strain=self.to_cpu(local_strain),
                element_composition=element_composition
            )
            
            # 形状検証
            if not result.validate_shapes():
                logger.warning("Shape validation failed! But continuing...")
            
            self._print_statistics_safe(result)
            
            return result
    
    def _validate_inputs(self, trajectory, cluster_atoms, atom_types):
        """入力データの妥当性を検証"""
        n_atoms = trajectory.shape[1]
        
        # クラスターマッピングの検証
        for cluster_id, atoms in cluster_atoms.items():
            if not atoms:
                logger.warning(f"Cluster {cluster_id} has no atoms")
                return False
            if max(atoms) >= n_atoms:
                logger.error(f"Cluster {cluster_id} has invalid atom indices")
                return False
        
        # 元素タイプの検証
        if len(atom_types) != n_atoms:
            logger.error(f"atom_types size ({len(atom_types)}) != n_atoms ({n_atoms})")
            return False
        
        return True
    
    def _create_empty_result(self, n_clusters: int) -> ClusterStructureResult:
        """空の結果を生成（エラー回避用）"""
        return ClusterStructureResult(
            cluster_lambda_f=np.zeros((0, n_clusters, 3), dtype=np.float32),
            cluster_lambda_f_mag=np.zeros((0, n_clusters), dtype=np.float32),
            cluster_rho_t=np.zeros((0, n_clusters), dtype=np.float32),
            cluster_coupling=np.zeros((0, n_clusters, n_clusters), dtype=np.float32),
            cluster_centers=np.zeros((0, n_clusters, 3), dtype=np.float32),
            coordination_numbers=np.zeros((0, n_clusters), dtype=np.float32),
            local_strain=np.zeros((0, n_clusters, 3, 3), dtype=np.float32),
            element_composition={}
        )
    
    # _compute_cluster_centers メソッドの修正（重要！）
    def _compute_cluster_centers(self,
                                trajectory: np.ndarray,
                                cluster_atoms: Dict[int, List[int]]) -> cp.ndarray:
        """クラスター重心計算（細分化ID対応版）"""
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        # 🆕 IDマッパー初期化
        if self.id_mapper is None:
            self.id_mapper = ClusterIDMapper(cluster_atoms)
        
        if self.is_gpu and HAS_GPU and cluster_com_kernel is not None:
            # GPU版：カスタムカーネル使用
            trajectory_gpu = self.to_gpu(trajectory, dtype=cp.float32)
            cluster_centers = cluster_com_kernel(trajectory_gpu, cluster_atoms)
        else:
            # CPU版フォールバック（修正版）
            cluster_centers = np.zeros((n_frames, n_clusters, 3), dtype=np.float32)
            
            # 🆕 クラスターIDとインデックスのマッピングを使用
            for cluster_id, idx in self.id_mapper.iterate_with_idx():
                atoms = cluster_atoms[cluster_id]
                if len(atoms) > 0 and max(atoms) < n_atoms:
                    cluster_centers[:, idx] = np.mean(trajectory[:, atoms], axis=1)
            
            cluster_centers = self.to_gpu(cluster_centers)
        
        return cluster_centers
    
    def _compute_cluster_lambda_f(self, cluster_centers: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """クラスターレベルΛF計算（形状統一版）"""
        n_frames, n_clusters, _ = cluster_centers.shape
        
        # 単一フレームの特別処理
        if n_frames <= 1:
            # 単一フレーム：変化なし（ゼロ）
            cluster_lambda_f = self.xp.zeros((n_frames, n_clusters, 3), dtype=self.xp.float32)
            cluster_lambda_f_mag = self.xp.zeros((n_frames, n_clusters), dtype=self.xp.float32)
            logger.debug(f"Single frame detected, returning zero lambda_f")
            return cluster_lambda_f, cluster_lambda_f_mag
        
        # 通常処理：フレーム間差分
        cluster_lambda_f = self.xp.diff(cluster_centers, axis=0)
        cluster_lambda_f_mag = self.xp.linalg.norm(cluster_lambda_f, axis=2)
        
        # パディング：最初のフレームを補完（diffで失われた分）
        zero_frame = self.xp.zeros((1, n_clusters, 3), dtype=cluster_lambda_f.dtype)
        cluster_lambda_f = self.xp.concatenate([zero_frame, cluster_lambda_f], axis=0)
        
        zero_mag = self.xp.zeros((1, n_clusters), dtype=cluster_lambda_f_mag.dtype)
        cluster_lambda_f_mag = self.xp.concatenate([zero_mag, cluster_lambda_f_mag], axis=0)
        
        return cluster_lambda_f, cluster_lambda_f_mag
    
    def _compute_cluster_rho_t(self,
                              cluster_centers: cp.ndarray,
                              window_size: int) -> cp.ndarray:
        """クラスターレベルρT計算（単一フレーム対応）"""
        n_frames, n_clusters, _ = cluster_centers.shape
        
        # 単一フレームの特別処理
        if n_frames == 1:
            # 単一フレーム：ゼロ分散
            logger.debug("Single frame: returning zero rho_t")
            return self.zeros((n_frames, n_clusters), dtype=self.xp.float32)
        
        # カスタムカーネル使用
        if self.is_gpu and self.tension_kernel is not None:
            rho_t = self.zeros((n_frames, n_clusters), dtype=cp.float32)
            
            # 2Dグリッド設定
            block_size = (16, 16)
            grid_size = (
                (n_clusters + block_size[0] - 1) // block_size[0],
                (n_frames + block_size[1] - 1) // block_size[1]
            )
            
            self.tension_kernel(
                grid_size, block_size,
                (cluster_centers.ravel(), rho_t, n_frames, n_clusters, window_size)
            )
            
            return rho_t
        else:
            # CPU/GPU汎用版
            cluster_rho_t = self.zeros((n_frames, n_clusters))
            
            for frame in range(n_frames):
                for cluster_id in range(n_clusters):
                    local_start = max(0, frame - window_size // 2)
                    local_end = min(n_frames, frame + window_size // 2 + 1)
                    
                    local_centers = cluster_centers[local_start:local_end, cluster_id]
                    if len(local_centers) > 1:
                        cov = self.xp.cov(local_centers.T)
                        if not self.xp.any(self.xp.isnan(cov)) and not self.xp.all(cov == 0):
                            cluster_rho_t[frame, cluster_id] = self.xp.trace(cov)
            
            return cluster_rho_t
    
    def _compute_cluster_coupling(self,
                                 cluster_centers: cp.ndarray) -> cp.ndarray:
        """クラスター間カップリング計算"""
        n_frames, n_clusters, _ = cluster_centers.shape
        
        coupling = self.zeros((n_frames, n_clusters, n_clusters))
        
        for frame in range(n_frames):
            # 距離行列
            distances = self.xp.linalg.norm(
                cluster_centers[frame, :, None, :] - cluster_centers[frame, None, :, :],
                axis=2
            )
            # カップリング（材料用に調整：距離が近いほど強い）
            coupling[frame] = self.xp.exp(-distances / 5.0)  # 5Åの特性長
            self.xp.fill_diagonal(coupling[frame], 0)  # 自己結合を除外
        
        return coupling

    # _compute_coordination_numbers メソッドの修正
    def _compute_coordination_numbers(self,
                                     trajectory: np.ndarray,
                                     cluster_atoms: Dict[int, List[int]],
                                     cutoff: float) -> cp.ndarray:
        """配位数計算（細分化ID対応版）"""
        n_frames, n_atoms, _ = trajectory.shape
        n_clusters = len(cluster_atoms)
        
        # 🆕 IDマッパー確認
        if self.id_mapper is None:
            self.id_mapper = ClusterIDMapper(cluster_atoms)
        
        coordination = self.zeros((n_frames, n_clusters), dtype=self.xp.float32)
        cutoff_sq = cutoff * cutoff
        
        if self.is_gpu and self.coord_kernel is not None:
            # カスタムカーネル使用（修正版）
            # クラスターマッピング作成（IDではなくインデックスを使用）
            cluster_mapping = -self.xp.ones(n_atoms, dtype=cp.int32)
            
            # 🆕 実際のクラスターIDをインデックスにマップ
            for cluster_id, atoms in cluster_atoms.items():
                idx = self.id_mapper.to_idx(cluster_id)
                for atom_id in atoms:
                    cluster_mapping[atom_id] = idx  # インデックスを格納
            
            # 各フレームで配位数計算
            for frame in range(n_frames):
                positions = self.to_gpu(trajectory[frame], dtype=cp.float32)
                coord_frame = self.zeros(n_clusters, dtype=cp.float32)
                
                # カーネル実行
                block_size = 256
                grid_size = n_clusters
                
                self.coord_kernel(
                    (grid_size,), (block_size,),
                    (positions.ravel(), cluster_mapping, coord_frame, 
                     cutoff_sq, n_atoms, n_clusters)
                )
                
                coordination[frame] = coord_frame
        else:
            # CPU版フォールバック（修正版）
            for frame in range(n_frames):
                positions = self.to_gpu(trajectory[frame])
                
                # 🆕 IDとインデックスのマッピングを使用
                for cluster_id, idx in self.id_mapper.iterate_with_idx():
                    atoms = cluster_atoms[cluster_id]
                    total_coord = 0
                    for atom_i in atoms:
                        pos_i = positions[atom_i]
                        distances = self.xp.linalg.norm(positions - pos_i, axis=1)
                        neighbors = self.xp.sum((distances > 0) & (distances < cutoff))
                        total_coord += neighbors
                    
                    # インデックスで配列に格納
                    coordination[frame, idx] = total_coord / len(atoms) if atoms else 0
        
        return coordination
    
    # _compute_local_strain メソッドの修正
    def _compute_local_strain(self,
                             trajectory: np.ndarray,
                             cluster_atoms: Dict[int, List[int]]) -> cp.ndarray:
        """局所歪みテンソル計算（細分化ID対応版）"""
        n_frames = trajectory.shape[0]
        n_clusters = len(cluster_atoms)
        
        # 🆕 IDマッパー確認
        if self.id_mapper is None:
            self.id_mapper = ClusterIDMapper(cluster_atoms)
        
        strain = self.zeros((n_frames, n_clusters, 3, 3), dtype=self.xp.float32)
        
        # 初期構造を参照
        ref_positions = self.to_gpu(trajectory[0])
        
        for frame in range(n_frames):
            current_positions = self.to_gpu(trajectory[frame])
            
            # 🆕 IDとインデックスのマッピングを使用
            for cluster_id, idx in self.id_mapper.iterate_with_idx():
                atoms = cluster_atoms[cluster_id]
                if len(atoms) < 4:  # 最低4原子必要
                    continue
                
                try:
                    # 変形勾配テンソル
                    F = self._compute_deformation_gradient(
                        ref_positions[atoms],
                        current_positions[atoms]
                    )
                    
                    # Green-Lagrange歪みテンソル
                    strain[frame, idx] = 0.5 * (F.T @ F - self.xp.eye(3))  # idxで格納
                except Exception as e:
                    logger.debug(f"Strain computation failed for cluster {cluster_id}: {e}")
                    # ゼロ歪みのまま
        
        return strain
    
    def _compute_element_composition(self,
                                    cluster_atoms: Dict[int, List[int]],
                                    atom_types: np.ndarray) -> Dict:
        """元素組成の計算"""
        composition = {}
        
        for cluster_id, atoms in cluster_atoms.items():
            cluster_comp = {}
            for atom_id in atoms:
                if atom_id < len(atom_types):
                    element = str(atom_types[atom_id])  # 文字列として扱う
                    cluster_comp[element] = cluster_comp.get(element, 0) + 1
            composition[cluster_id] = cluster_comp
        
        return composition
    
    def _compute_deformation_gradient(self, ref_pos, current_pos):
        """変形勾配テンソルの計算"""
        # 簡易版：最小二乗法
        ref_centered = ref_pos - self.xp.mean(ref_pos, axis=0)
        current_centered = current_pos - self.xp.mean(current_pos, axis=0)
        
        # F = current @ ref^(-1)
        H = current_centered.T @ ref_centered
        C = ref_centered.T @ ref_centered
        
        try:
            # 正則化して数値的安定性を向上
            C_reg = C + 1e-6 * self.xp.eye(3)
            F = H @ self.xp.linalg.inv(C_reg)
        except:
            F = self.xp.eye(3)
        
        return F
    
    def _print_statistics_safe(self, result: ClusterStructureResult):
        """統計情報を安全に表示"""
        logger.info(f"  ✅ Structure computation complete")
        logger.info(f"  Frames: {result.n_frames}, Clusters: {result.n_clusters}")
        
        # 統計サマリー
        stats = result.get_summary_stats()
        if result.n_frames > 0:
            logger.info(f"  Mean ΛF: {stats['mean_lambda_f']:.3f}")
            logger.info(f"  Mean ρT: {stats['mean_rho_t']:.3f}")
            logger.info(f"  Mean Coupling: {stats['mean_coupling']:.3f}")
            logger.info(f"  Mean Coordination: {stats['mean_coordination']:.2f}")
            logger.info(f"  Max Strain: {stats['max_strain']:.4f}")
        else:
            logger.info(f"  No frames to analyze")

# ===============================
# Convenience Functions
# ===============================

def compute_cluster_structures_gpu(trajectory: np.ndarray,
                                  start_frame: int,
                                  end_frame: int,
                                  cluster_atoms: Dict[int, List[int]],
                                  atom_types: np.ndarray,
                                  window_size: int = 50,
                                  cutoff: float = 3.0) -> ClusterStructureResult:
    """クラスター構造計算のラッパー関数（包含的範囲）"""
    calculator = ClusterStructuresGPU()
    return calculator.compute_cluster_structures(
        trajectory, start_frame, end_frame, cluster_atoms, 
        atom_types, window_size, cutoff
    )

def compute_cluster_lambda_f_gpu(cluster_centers: Union[np.ndarray, cp.ndarray],
                                backend: Optional[GPUBackend] = None) -> Tuple[np.ndarray, np.ndarray]:
    """クラスターΛF計算のスタンドアロン関数（パディング済み）"""
    calculator = ClusterStructuresGPU() if backend is None else ClusterStructuresGPU(device=backend.device)
    centers_gpu = calculator.to_gpu(cluster_centers)
    lambda_f, lambda_f_mag = calculator._compute_cluster_lambda_f(centers_gpu)
    return calculator.to_cpu(lambda_f), calculator.to_cpu(lambda_f_mag)

def compute_cluster_rho_t_gpu(cluster_centers: Union[np.ndarray, cp.ndarray],
                             window_size: int = 50,
                             backend: Optional[GPUBackend] = None) -> np.ndarray:
    """クラスターρT計算のスタンドアロン関数"""
    calculator = ClusterStructuresGPU() if backend is None else ClusterStructuresGPU(device=backend.device)
    centers_gpu = calculator.to_gpu(cluster_centers)
    rho_t = calculator._compute_cluster_rho_t(centers_gpu, window_size)
    return calculator.to_cpu(rho_t)

def compute_cluster_coupling_gpu(cluster_centers: Union[np.ndarray, cp.ndarray],
                                backend: Optional[GPUBackend] = None) -> np.ndarray:
    """クラスター間カップリング計算のスタンドアロン関数"""
    calculator = ClusterStructuresGPU() if backend is None else ClusterStructuresGPU(device=backend.device)
    centers_gpu = calculator.to_gpu(cluster_centers)
    coupling = calculator._compute_cluster_coupling(centers_gpu)
    return calculator.to_cpu(coupling)

# ===============================
# Batch Processing
# ===============================

class ClusterStructureBatchProcessor:
    """大規模トラジェクトリのバッチ処理（材料用）"""
    
    def __init__(self,
                 calculator: ClusterStructuresGPU,
                 batch_size: Optional[int] = None):
        self.calculator = calculator
        self.batch_size = batch_size
    
    def process_trajectory(self,
                         trajectory: np.ndarray,
                         cluster_atoms: Dict[int, List[int]],
                         atom_types: np.ndarray,
                         window_size: int = 50,
                         cutoff: float = 3.0,
                         overlap: int = 100) -> ClusterStructureResult:
        """バッチ処理でトラジェクトリ全体を処理"""
        n_frames = trajectory.shape[0]
        
        if self.batch_size is None:
            # 自動決定
            self.batch_size = self.calculator.memory_manager.estimate_batch_size(
                trajectory.shape, dtype=np.float32
            )
        
        logger.info(f"Processing {n_frames} frames in batches of {self.batch_size}")
        
        # TODO: 実装が必要
        # バッチごとに処理して結果を結合
        raise NotImplementedError("Batch processing implementation needed")

# ===============================
# Cluster Definition Utilities
# ===============================

def create_nearest_neighbor_clusters(positions: np.ndarray,
                                    cutoff: float = 3.0,
                                    min_size: int = 12,
                                    max_size: int = 14) -> Dict[int, List[int]]:
    """最近接原子でクラスターを定義（材料用）"""
    n_atoms = len(positions)
    clusters = {}
    assigned = np.zeros(n_atoms, dtype=bool)
    cluster_id = 0
    
    for atom_id in range(n_atoms):
        if assigned[atom_id]:
            continue
        
        # この原子を中心にクラスター作成
        distances = np.linalg.norm(positions - positions[atom_id], axis=1)
        neighbors = np.where((distances < cutoff) & (~assigned))[0]
        
        if len(neighbors) >= min_size:
            # サイズ制限
            if len(neighbors) > max_size:
                neighbors = neighbors[np.argsort(distances[neighbors])[:max_size]]
            
            clusters[cluster_id] = neighbors.tolist()
            assigned[neighbors] = True
            cluster_id += 1
    
    # 未割り当て原子を最近接クラスターに追加
    for atom_id in np.where(~assigned)[0]:
        if clusters:
            # 最近接クラスター中心を探す
            min_dist = np.inf
            best_cluster = 0
            
            for cid, atoms in clusters.items():
                center = np.mean(positions[atoms], axis=0)
                dist = np.linalg.norm(positions[atom_id] - center)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cid
            
            clusters[best_cluster].append(atom_id)
    
    logger.info(f"Created {len(clusters)} clusters with cutoff={cutoff}Å")
    
    return clusters

def create_grid_based_clusters(positions: np.ndarray,
                              grid_size: float = 1.0) -> Dict[int, List[int]]:
    """グリッドベースでクラスターを定義（材料用）"""
    # グリッドインデックス計算
    grid_indices = np.floor(positions / grid_size).astype(int)
    
    # ユニークなグリッドセルを特定
    unique_cells = np.unique(grid_indices, axis=0)
    
    clusters = {}
    for i, cell in enumerate(unique_cells):
        mask = np.all(grid_indices == cell, axis=1)
        atoms = np.where(mask)[0]
        if len(atoms) > 0:
            clusters[i] = atoms.tolist()
    
    logger.info(f"Created {len(clusters)} grid-based clusters with size={grid_size}Å")
    
    return clusters
