#!/usr/bin/env python3
"""
Extended CUDA Kernels for Material Lambda³ Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料解析用の高速CUDAカーネル集（拡張版）
material_analytics_gpu.py最適化用の追加カーネル実装

歪みテンソル、損傷度、配位数に加えて
欠陥チャージ、構造一貫性、Burgersベクトルを全並列計算！💎

RTX 4070 Ti SUPERの力を100%引き出す本気実装 v2.0

by 環ちゃん - CUDA Extended Edition
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger('lambda3_gpu.material.cuda')

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    cp = None
    HAS_CUDA = False

# ===============================
# Original Kernels (from v1.0)
# ===============================

STRAIN_TENSOR_KERNEL_CODE = r'''
extern "C" __global__
void compute_strain_tensor_kernel(
    const float* __restrict__ ref_positions,    // (n_atoms, 3)
    const float* __restrict__ curr_positions,   // (n_atoms, 3)
    const int* __restrict__ cluster_atoms,      // (n_clusters, max_atoms_per_cluster)
    const int* __restrict__ cluster_sizes,      // (n_clusters,)
    float* __restrict__ strain_tensor,          // (n_clusters, 6) Voigt記法
    const int n_atoms,
    const int n_clusters,
    const int max_atoms_per_cluster
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    const int cluster_size = cluster_sizes[cluster_id];
    if (cluster_size < 4) return;
    
    // 共有メモリで高速化
    __shared__ float ref_centroid[3];
    __shared__ float curr_centroid[3];
    __shared__ float covariance[9];  // 3x3行列
    __shared__ float ref_matrix[9];  // ref^T * ref
    
    // 初期化
    if (tid < 3) {
        ref_centroid[tid] = 0.0f;
        curr_centroid[tid] = 0.0f;
    }
    if (tid < 9) {
        covariance[tid] = 0.0f;
        ref_matrix[tid] = 0.0f;
    }
    __syncthreads();
    
    // 重心計算（並列リダクション）
    int atoms_per_thread = (cluster_size + blockDim.x - 1) / blockDim.x;
    int start_atom = tid * atoms_per_thread;
    int end_atom = min(start_atom + atoms_per_thread, cluster_size);
    
    float local_ref[3] = {0.0f, 0.0f, 0.0f};
    float local_curr[3] = {0.0f, 0.0f, 0.0f};
    
    for (int i = start_atom; i < end_atom; i++) {
        int atom_idx = cluster_atoms[cluster_id * max_atoms_per_cluster + i];
        if (atom_idx < 0 || atom_idx >= n_atoms) continue;
        
        for (int d = 0; d < 3; d++) {
            local_ref[d] += ref_positions[atom_idx * 3 + d];
            local_curr[d] += curr_positions[atom_idx * 3 + d];
        }
    }
    
    // 共有メモリに集約
    for (int d = 0; d < 3; d++) {
        atomicAdd(&ref_centroid[d], local_ref[d]);
        atomicAdd(&curr_centroid[d], local_curr[d]);
    }
    __syncthreads();
    
    if (tid < 3) {
        ref_centroid[tid] /= cluster_size;
        curr_centroid[tid] /= cluster_size;
    }
    __syncthreads();
    
    // 変形勾配テンソルF計算
    for (int i = start_atom; i < end_atom; i++) {
        int atom_idx = cluster_atoms[cluster_id * max_atoms_per_cluster + i];
        if (atom_idx < 0 || atom_idx >= n_atoms) continue;
        
        float ref[3], curr[3];
        for (int d = 0; d < 3; d++) {
            ref[d] = ref_positions[atom_idx * 3 + d] - ref_centroid[d];
            curr[d] = curr_positions[atom_idx * 3 + d] - curr_centroid[d];
        }
        
        // F = curr * ref^T の要素を計算
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                atomicAdd(&covariance[i * 3 + j], curr[i] * ref[j]);
                atomicAdd(&ref_matrix[j * 3 + i], ref[j] * ref[i]);
            }
        }
    }
    __syncthreads();
    
    // Green-Lagrange歪みテンソル計算（スレッド0が代表）
    if (tid == 0) {
        // 簡易逆行列（3x3の場合は解析的に計算可能）
        float det = ref_matrix[0] * (ref_matrix[4] * ref_matrix[8] - ref_matrix[5] * ref_matrix[7])
                  - ref_matrix[1] * (ref_matrix[3] * ref_matrix[8] - ref_matrix[5] * ref_matrix[6])
                  + ref_matrix[2] * (ref_matrix[3] * ref_matrix[7] - ref_matrix[4] * ref_matrix[6]);
        
        if (fabsf(det) < 1e-10f) {
            for (int i = 0; i < 6; i++) {
                strain_tensor[cluster_id * 6 + i] = 0.0f;
            }
            return;
        }
        
        float F[9];
        float inv_det = 1.0f / det;
        
        for (int i = 0; i < 9; i++) {
            F[i] = covariance[i] * inv_det;
        }
        
        // Green-Lagrange歪み E = 0.5 * (F^T * F - I)
        float E[6];  // Voigt記法
        
        E[0] = 0.5f * (F[0]*F[0] + F[3]*F[3] + F[6]*F[6] - 1.0f);  // E_xx
        E[1] = 0.5f * (F[1]*F[1] + F[4]*F[4] + F[7]*F[7] - 1.0f);  // E_yy
        E[2] = 0.5f * (F[2]*F[2] + F[5]*F[5] + F[8]*F[8] - 1.0f);  // E_zz
        E[3] = F[0]*F[1] + F[3]*F[4] + F[6]*F[7];  // E_xy
        E[4] = F[0]*F[2] + F[3]*F[5] + F[6]*F[8];  // E_xz
        E[5] = F[1]*F[2] + F[4]*F[5] + F[7]*F[8];  // E_yz
        
        for (int i = 0; i < 6; i++) {
            strain_tensor[cluster_id * 6 + i] = E[i];
        }
    }
}
'''

COORDINATION_NUMBER_KERNEL_CODE = r'''
extern "C" __global__
void compute_coordination_kernel(
    const float* __restrict__ positions,        // (n_atoms, 3)
    const int* __restrict__ cluster_atoms,      // (n_clusters, max_atoms)
    const int* __restrict__ cluster_sizes,
    float* __restrict__ coordination_numbers,   // (n_clusters,)
    const float cutoff_squared,
    const int n_atoms,
    const int n_clusters,
    const int max_atoms_per_cluster
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    const int cluster_size = cluster_sizes[cluster_id];
    if (cluster_size == 0) return;
    
    __shared__ float total_coordination;
    if (tid == 0) total_coordination = 0.0f;
    __syncthreads();
    
    int n_samples = min(cluster_size, 50);
    int atoms_per_thread = (n_samples + blockDim.x - 1) / blockDim.x;
    int start_sample = tid * atoms_per_thread;
    int end_sample = min(start_sample + atoms_per_thread, n_samples);
    
    float local_coord = 0.0f;
    
    for (int s = start_sample; s < end_sample; s++) {
        int atom_i = cluster_atoms[cluster_id * max_atoms_per_cluster + s];
        if (atom_i < 0 || atom_i >= n_atoms) continue;
        
        float xi = positions[atom_i * 3 + 0];
        float yi = positions[atom_i * 3 + 1];
        float zi = positions[atom_i * 3 + 2];
        
        int neighbors = 0;
        
        for (int j = 0; j < n_atoms; j++) {
            if (j == atom_i) continue;
            
            float dx = positions[j * 3 + 0] - xi;
            float dy = positions[j * 3 + 1] - yi;
            float dz = positions[j * 3 + 2] - zi;
            
            float dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq < cutoff_squared) {
                neighbors++;
            }
        }
        
        local_coord += neighbors;
    }
    
    atomicAdd(&total_coordination, local_coord);
    __syncthreads();
    
    if (tid == 0) {
        coordination_numbers[cluster_id] = total_coordination / n_samples;
    }
}
'''

DAMAGE_SCORE_KERNEL_CODE = r'''
extern "C" __global__
void compute_damage_kernel(
    const float* __restrict__ ref_positions,
    const float* __restrict__ curr_positions,
    const int* __restrict__ cluster_atoms,
    const int* __restrict__ cluster_sizes,
    float* __restrict__ damage_scores,          // (n_clusters,)
    const int n_atoms,
    const int n_clusters,
    const int max_atoms_per_cluster
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    const int cluster_size = cluster_sizes[cluster_id];
    if (cluster_size == 0) return;
    
    __shared__ float total_rmsd;
    if (tid == 0) total_rmsd = 0.0f;
    __syncthreads();
    
    int atoms_per_thread = (cluster_size + blockDim.x - 1) / blockDim.x;
    int start_atom = tid * atoms_per_thread;
    int end_atom = min(start_atom + atoms_per_thread, cluster_size);
    
    float local_rmsd = 0.0f;
    
    for (int i = start_atom; i < end_atom; i++) {
        int atom_idx = cluster_atoms[cluster_id * max_atoms_per_cluster + i];
        if (atom_idx < 0 || atom_idx >= n_atoms) continue;
        
        float dx = curr_positions[atom_idx * 3 + 0] - ref_positions[atom_idx * 3 + 0];
        float dy = curr_positions[atom_idx * 3 + 1] - ref_positions[atom_idx * 3 + 1];
        float dz = curr_positions[atom_idx * 3 + 2] - ref_positions[atom_idx * 3 + 2];
        
        local_rmsd += dx*dx + dy*dy + dz*dz;
    }
    
    atomicAdd(&total_rmsd, local_rmsd);
    __syncthreads();
    
    if (tid == 0) {
        float rmsd = sqrtf(total_rmsd / cluster_size);
        damage_scores[cluster_id] = tanhf(rmsd / 2.0f);  // 0-1に正規化
    }
}
'''

# ===============================
# NEW: Extended Kernels for MaterialAnalyticsGPU
# ===============================

DEFECT_CHARGE_KERNEL_CODE = r'''
extern "C" __global__
void compute_defect_charge_kernel(
    const float* __restrict__ curr_positions,   // (n_atoms, 3)
    const float* __restrict__ next_positions,   // (n_atoms, 3)
    const int* __restrict__ cluster_atoms,      // (n_clusters, max_atoms)
    const int* __restrict__ cluster_sizes,
    float* __restrict__ defect_charge,          // (n_clusters,)
    const float lattice_constant,
    const int ideal_coordination,
    const float cutoff_sq,
    const int n_atoms,
    const int n_clusters,
    const int max_atoms_per_cluster
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    const int cluster_size = cluster_sizes[cluster_id];
    if (cluster_size < 5) {
        if (tid == 0) defect_charge[cluster_id] = 0.0f;
        return;
    }
    
    __shared__ float burgers_magnitude;
    __shared__ float coord_change_total;
    __shared__ int sample_count;
    
    if (tid == 0) {
        burgers_magnitude = 0.0f;
        coord_change_total = 0.0f;
        sample_count = 0;
    }
    __syncthreads();
    
    // 1. Burgersベクトル計算（簡易版）
    if (tid < cluster_size) {
        int atom_idx = cluster_atoms[cluster_id * max_atoms_per_cluster + tid];
        if (atom_idx >= 0 && atom_idx < n_atoms) {
            float disp[3];
            for (int d = 0; d < 3; d++) {
                disp[d] = next_positions[atom_idx * 3 + d] - curr_positions[atom_idx * 3 + d];
            }
            
            // クラスター変位の寄与を加算
            float disp_mag = sqrtf(disp[0]*disp[0] + disp[1]*disp[1] + disp[2]*disp[2]);
            atomicAdd(&burgers_magnitude, disp_mag);
        }
    }
    __syncthreads();
    
    // 2. 配位数変化の計算（サンプリング）
    int n_samples = min(10, cluster_size);
    if (tid < n_samples) {
        int atom_i = cluster_atoms[cluster_id * max_atoms_per_cluster + tid];
        if (atom_i >= 0 && atom_i < n_atoms) {
            // 現在フレームの配位数
            int coord_curr = 0;
            float xi_c = curr_positions[atom_i * 3 + 0];
            float yi_c = curr_positions[atom_i * 3 + 1];
            float zi_c = curr_positions[atom_i * 3 + 2];
            
            for (int j = 0; j < n_atoms; j++) {
                if (j == atom_i) continue;
                float dx = curr_positions[j * 3 + 0] - xi_c;
                float dy = curr_positions[j * 3 + 1] - yi_c;
                float dz = curr_positions[j * 3 + 2] - zi_c;
                if (dx*dx + dy*dy + dz*dz < cutoff_sq) coord_curr++;
            }
            
            // 次フレームの配位数
            int coord_next = 0;
            float xi_n = next_positions[atom_i * 3 + 0];
            float yi_n = next_positions[atom_i * 3 + 1];
            float zi_n = next_positions[atom_i * 3 + 2];
            
            for (int j = 0; j < n_atoms; j++) {
                if (j == atom_i) continue;
                float dx = next_positions[j * 3 + 0] - xi_n;
                float dy = next_positions[j * 3 + 1] - yi_n;
                float dz = next_positions[j * 3 + 2] - zi_n;
                if (dx*dx + dy*dy + dz*dz < cutoff_sq) coord_next++;
            }
            
            float coord_change = fabsf(coord_next - coord_curr) / (float)ideal_coordination;
            atomicAdd(&coord_change_total, coord_change);
            atomicAdd(&sample_count, 1);
        }
    }
    __syncthreads();
    
    // 3. 欠陥チャージ計算（正規化）
    if (tid == 0) {
        float burgers_normalized = burgers_magnitude / (lattice_constant * cluster_size);
        float coord_change_normalized = (sample_count > 0) ? 
            coord_change_total / sample_count : 0.0f;
        
        // tanhで飽和させて爆発を防ぐ
        float raw_charge = 0.3f * burgers_normalized + 0.2f * coord_change_normalized;
        defect_charge[cluster_id] = tanhf(raw_charge);
    }
}
'''

STRUCTURAL_COHERENCE_KERNEL_CODE = r'''
extern "C" __global__
void compute_coherence_kernel(
    const float* __restrict__ coordination,     // (n_frames, n_clusters)
    const float* __restrict__ strain,          // (n_frames, n_clusters)  
    float* __restrict__ coherence,             // (n_frames,)
    const int window,
    const int ideal_coord,
    const int n_frames,
    const int n_clusters
) {
    const int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    
    // エッジ処理
    if (frame < window || frame >= n_frames - window) {
        if (frame < window) coherence[frame] = coherence[window];
        else coherence[frame] = coherence[n_frames - window - 1];
        return;
    }
    
    float total_coherence = 0.0f;
    
    for (int c = 0; c < n_clusters; c++) {
        // ウィンドウ内の統計量計算
        float coord_mean = 0.0f;
        float coord_sq_mean = 0.0f;
        
        for (int w = -window; w <= window; w++) {
            int idx = (frame + w) * n_clusters + c;
            float coord = coordination[idx];
            coord_mean += coord;
            coord_sq_mean += coord * coord;
        }
        
        int window_size = 2 * window + 1;
        coord_mean /= window_size;
        coord_sq_mean /= window_size;
        
        // 標準偏差
        float coord_std = sqrtf(fmaxf(0.0f, coord_sq_mean - coord_mean * coord_mean));
        
        // 構造安定性評価
        float cluster_coherence = 1.0f;
        
        if (coord_std < 0.5f && fabsf(coord_mean - ideal_coord) < 1.0f) {
            // 安定して理想構造を保持
            cluster_coherence = 1.0f;
        } else if (coord_std > 2.0f) {
            // 激しく揺らいでいる
            cluster_coherence = 0.3f;
        } else {
            // 構造変化中
            cluster_coherence = 1.0f - fminf(coord_std / 3.0f, 1.0f);
        }
        
        // 歪みによる補正
        float strain_val = strain[frame * n_clusters + c];
        if (strain_val > 0.05f) {  // 5%以上の歪み
            cluster_coherence *= (1.0f - fminf(strain_val, 1.0f));
        }
        
        total_coherence += cluster_coherence;
    }
    
    coherence[frame] = total_coherence / n_clusters;
}
'''

BURGERS_VECTOR_KERNEL_CODE = r'''
extern "C" __global__
void compute_burgers_kernel(
    const float* __restrict__ positions,        // (n_atoms, 3)
    const int* __restrict__ cluster_atoms,      // (n_clusters, max_atoms)
    const int* __restrict__ cluster_sizes,
    float* __restrict__ burgers_vectors,        // (n_clusters, 3)
    const float cutoff,
    const int n_atoms,
    const int n_clusters,
    const int max_atoms_per_cluster
) {
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (cluster_id >= n_clusters) return;
    
    const int cluster_size = cluster_sizes[cluster_id];
    if (cluster_size < 10) {
        if (tid == 0) {
            burgers_vectors[cluster_id * 3 + 0] = 0.0f;
            burgers_vectors[cluster_id * 3 + 1] = 0.0f;
            burgers_vectors[cluster_id * 3 + 2] = 0.0f;
        }
        return;
    }
    
    __shared__ float burgers[3];
    
    if (tid < 3) {
        burgers[tid] = 0.0f;
    }
    __syncthreads();
    
    // クラスター中心原子
    int center_idx = cluster_size / 2;
    int center_atom = cluster_atoms[cluster_id * max_atoms_per_cluster + center_idx];
    
    if (center_atom < 0 || center_atom >= n_atoms) {
        if (tid == 0) {
            burgers_vectors[cluster_id * 3 + 0] = 0.0f;
            burgers_vectors[cluster_id * 3 + 1] = 0.0f;
            burgers_vectors[cluster_id * 3 + 2] = 0.0f;
        }
        return;
    }
    
    float center_pos[3];
    if (tid == 0) {
        center_pos[0] = positions[center_atom * 3 + 0];
        center_pos[1] = positions[center_atom * 3 + 1];
        center_pos[2] = positions[center_atom * 3 + 2];
    }
    __syncthreads();
    
    // 最近傍6原子で閉回路を構成（簡易版）
    if (tid < 6 && tid < cluster_size) {
        int atom_i = cluster_atoms[cluster_id * max_atoms_per_cluster + tid];
        int atom_j = cluster_atoms[cluster_id * max_atoms_per_cluster + ((tid + 1) % 6)];
        
        if (atom_i >= 0 && atom_i < n_atoms && atom_j >= 0 && atom_j < n_atoms) {
            for (int d = 0; d < 3; d++) {
                float delta = positions[atom_j * 3 + d] - positions[atom_i * 3 + d];
                atomicAdd(&burgers[d], delta);
            }
        }
    }
    __syncthreads();
    
    // 結果を出力
    if (tid == 0) {
        burgers_vectors[cluster_id * 3 + 0] = burgers[0];
        burgers_vectors[cluster_id * 3 + 1] = burgers[1];
        burgers_vectors[cluster_id * 3 + 2] = burgers[2];
    }
}
'''

ROLLING_STATS_KERNEL_CODE = r'''
extern "C" __global__
void compute_rolling_stats_kernel(
    const float* __restrict__ data,             // (n_frames, n_features)
    float* __restrict__ mean,                   // (n_frames, n_features)
    float* __restrict__ std,                    // (n_frames, n_features)
    const int window,
    const int n_frames,
    const int n_features
) {
    const int frame = blockIdx.x;
    const int feature = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (frame >= n_frames || feature >= n_features) return;
    
    int start = max(0, frame - window);
    int end = min(n_frames - 1, frame + window);
    int window_size = end - start + 1;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int f = start; f <= end; f++) {
        float val = data[f * n_features + feature];
        sum += val;
        sum_sq += val * val;
    }
    
    float mean_val = sum / window_size;
    float std_val = sqrtf(fmaxf(0.0f, sum_sq / window_size - mean_val * mean_val));
    
    mean[frame * n_features + feature] = mean_val;
    std[frame * n_features + feature] = std_val;
}
'''

# ===============================
# Extended Kernel Manager Class
# ===============================

class MaterialCUDAKernels:
    """材料解析用CUDAカーネル管理クラス（拡張版）"""
    
    def __init__(self):
        # Original kernels
        self.strain_kernel = None
        self.coord_kernel = None
        self.damage_kernel = None
        
        # Extended kernels for MaterialAnalyticsGPU
        self.defect_charge_kernel = None
        self.coherence_kernel = None
        self.burgers_kernel = None
        self.rolling_stats_kernel = None
        
        self.compiled = False
        
        if HAS_CUDA:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """全CUDAカーネルのコンパイル"""
        try:
            # === Original 3 kernels ===
            self.strain_kernel = cp.RawKernel(
                STRAIN_TENSOR_KERNEL_CODE,
                'compute_strain_tensor_kernel',
                options=('--use_fast_math',)
            )
            
            self.coord_kernel = cp.RawKernel(
                COORDINATION_NUMBER_KERNEL_CODE,
                'compute_coordination_kernel',
                options=('--use_fast_math',)
            )
            
            self.damage_kernel = cp.RawKernel(
                DAMAGE_SCORE_KERNEL_CODE,
                'compute_damage_kernel',
                options=('--use_fast_math',)
            )
            
            # === NEW: Extended kernels ===
            self.defect_charge_kernel = cp.RawKernel(
                DEFECT_CHARGE_KERNEL_CODE,
                'compute_defect_charge_kernel',
                options=('--use_fast_math',)
            )
            
            self.coherence_kernel = cp.RawKernel(
                STRUCTURAL_COHERENCE_KERNEL_CODE,
                'compute_coherence_kernel',
                options=('--use_fast_math',)
            )
            
            self.burgers_kernel = cp.RawKernel(
                BURGERS_VECTOR_KERNEL_CODE,
                'compute_burgers_kernel',
                options=('--use_fast_math',)
            )
            
            self.rolling_stats_kernel = cp.RawKernel(
                ROLLING_STATS_KERNEL_CODE,
                'compute_rolling_stats_kernel',
                options=('--use_fast_math',)
            )
            
            self.compiled = True
            logger.info("✅ All 7 CUDA kernels compiled successfully!")
            logger.info("   - Original: strain, coordination, damage")
            logger.info("   - Extended: defect_charge, coherence, burgers, rolling_stats")
            
        except Exception as e:
            logger.warning(f"⚠️ CUDA kernel compilation failed: {e}")
            logger.warning("   Falling back to standard GPU computation")
            self.compiled = False
    
    # ===============================
    # Original Methods (unchanged)
    # ===============================
    
    def compute_strain_tensors_cuda(self,
                                   ref_positions: cp.ndarray,
                                   curr_positions: cp.ndarray,
                                   cluster_atoms: Dict,
                                   n_frames: int = None) -> cp.ndarray:
        """CUDAカーネルで歪みテンソル計算"""
        if not self.compiled or self.strain_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_atoms = ref_positions.shape[0]
        n_clusters = len(cluster_atoms)
        
        # クラスター配列の準備
        max_atoms = max(len(atoms) for atoms in cluster_atoms.values())
        cluster_array = np.full((n_clusters, max_atoms), -1, dtype=np.int32)
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        
        for cid, atoms in cluster_atoms.items():
            cluster_array[cid, :len(atoms)] = atoms
            cluster_sizes[cid] = len(atoms)
        
        # GPU転送
        cluster_array_gpu = cp.asarray(cluster_array)
        cluster_sizes_gpu = cp.asarray(cluster_sizes)
        ref_pos_flat = ref_positions.flatten().astype(cp.float32)
        
        # 結果配列
        strain_result = cp.zeros((n_clusters, 6), dtype=cp.float32)
        
        # カーネル実行
        threads_per_block = 256
        blocks = n_clusters
        
        self.strain_kernel(
            (blocks,), (threads_per_block,),
            (ref_pos_flat, curr_positions.flatten().astype(cp.float32),
             cluster_array_gpu, cluster_sizes_gpu,
             strain_result, n_atoms, n_clusters, max_atoms)
        )
        
        return strain_result
    
    def compute_coordination_cuda(self,
                                 positions: cp.ndarray,
                                 cluster_atoms: Dict,
                                 cutoff: float) -> cp.ndarray:
        """CUDAカーネルで配位数計算"""
        if not self.compiled or self.coord_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_atoms = positions.shape[0]
        n_clusters = len(cluster_atoms)
        
        # クラスター配列の準備
        max_atoms = max(len(atoms) for atoms in cluster_atoms.values())
        cluster_array = np.full((n_clusters, max_atoms), -1, dtype=np.int32)
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        
        for cid, atoms in cluster_atoms.items():
            cluster_array[cid, :len(atoms)] = atoms
            cluster_sizes[cid] = len(atoms)
        
        # GPU転送
        cluster_array_gpu = cp.asarray(cluster_array)
        cluster_sizes_gpu = cp.asarray(cluster_sizes)
        positions_flat = positions.flatten().astype(cp.float32)
        
        # 結果配列
        coord_result = cp.zeros(n_clusters, dtype=cp.float32)
        
        # カーネル実行
        threads_per_block = 256
        blocks = n_clusters
        
        self.coord_kernel(
            (blocks,), (threads_per_block,),
            (positions_flat, cluster_array_gpu, cluster_sizes_gpu,
             coord_result, cutoff * cutoff, n_atoms, n_clusters, max_atoms)
        )
        
        return coord_result
    
    def compute_damage_cuda(self,
                          ref_positions: cp.ndarray,
                          curr_positions: cp.ndarray,
                          cluster_atoms: Dict) -> cp.ndarray:
        """CUDAカーネルで損傷度計算"""
        if not self.compiled or self.damage_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_atoms = ref_positions.shape[0]
        n_clusters = len(cluster_atoms)
        
        # クラスター配列の準備
        max_atoms = max(len(atoms) for atoms in cluster_atoms.values())
        cluster_array = np.full((n_clusters, max_atoms), -1, dtype=np.int32)
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        
        for cid, atoms in cluster_atoms.items():
            cluster_array[cid, :len(atoms)] = atoms
            cluster_sizes[cid] = len(atoms)
        
        # GPU転送
        cluster_array_gpu = cp.asarray(cluster_array)
        cluster_sizes_gpu = cp.asarray(cluster_sizes)
        ref_pos_flat = ref_positions.flatten().astype(cp.float32)
        curr_pos_flat = curr_positions.flatten().astype(cp.float32)
        
        # 結果配列
        damage_result = cp.zeros(n_clusters, dtype=cp.float32)
        
        # カーネル実行
        threads_per_block = 256
        blocks = n_clusters
        
        self.damage_kernel(
            (blocks,), (threads_per_block,),
            (ref_pos_flat, curr_pos_flat, cluster_array_gpu, cluster_sizes_gpu,
             damage_result, n_atoms, n_clusters, max_atoms)
        )
        
        return damage_result
    
    # ===============================
    # NEW: Extended Methods for MaterialAnalyticsGPU
    # ===============================
    
    def compute_defect_charge_cuda(self,
                                  curr_positions: cp.ndarray,
                                  next_positions: cp.ndarray,
                                  cluster_atoms: Dict,
                                  lattice_constant: float,
                                  ideal_coordination: int,
                                  cutoff: float) -> cp.ndarray:
        """
        CUDAカーネルで欠陥チャージ計算（MaterialAnalyticsGPU用）
        
        Returns
        -------
        cp.ndarray
            (n_clusters,) の欠陥チャージ（0-1正規化済み）
        """
        if not self.compiled or self.defect_charge_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_atoms = curr_positions.shape[0]
        n_clusters = len(cluster_atoms)
        
        # クラスター配列の準備
        max_atoms = max(len(atoms) for atoms in cluster_atoms.values())
        cluster_array = np.full((n_clusters, max_atoms), -1, dtype=np.int32)
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        
        for cid, atoms in cluster_atoms.items():
            cluster_array[cid, :len(atoms)] = atoms
            cluster_sizes[cid] = len(atoms)
        
        # GPU転送
        cluster_array_gpu = cp.asarray(cluster_array)
        cluster_sizes_gpu = cp.asarray(cluster_sizes)
        curr_pos_flat = curr_positions.flatten().astype(cp.float32)
        next_pos_flat = next_positions.flatten().astype(cp.float32)
        
        # 結果配列
        defect_charge = cp.zeros(n_clusters, dtype=cp.float32)
        
        # カーネル実行
        threads_per_block = 256
        blocks = n_clusters
        
        self.defect_charge_kernel(
            (blocks,), (threads_per_block,),
            (curr_pos_flat, next_pos_flat, cluster_array_gpu, cluster_sizes_gpu,
             defect_charge, lattice_constant, ideal_coordination, 
             cutoff * cutoff, n_atoms, n_clusters, max_atoms)
        )
        
        return defect_charge
    
    def compute_structural_coherence_cuda(self,
                                         coordination: cp.ndarray,
                                         strain: cp.ndarray,
                                         window: int,
                                         ideal_coord: int) -> cp.ndarray:
        """
        CUDAカーネルで構造一貫性計算
        
        Returns
        -------
        cp.ndarray
            (n_frames,) の構造一貫性
        """
        if not self.compiled or self.coherence_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_frames, n_clusters = coordination.shape
        coherence = cp.zeros(n_frames, dtype=cp.float32)
        
        # カーネル実行
        threads_per_block = 256
        blocks = (n_frames + threads_per_block - 1) // threads_per_block
        
        self.coherence_kernel(
            (blocks,), (threads_per_block,),
            (coordination.astype(cp.float32), strain.astype(cp.float32),
             coherence, window, ideal_coord, n_frames, n_clusters)
        )
        
        return coherence
    
    def compute_burgers_vectors_cuda(self,
                                    positions: cp.ndarray,
                                    cluster_atoms: Dict,
                                    cutoff: float) -> cp.ndarray:
        """
        CUDAカーネルでBurgersベクトル計算
        
        Returns
        -------
        cp.ndarray
            (n_clusters, 3) のBurgersベクトル
        """
        if not self.compiled or self.burgers_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_atoms = positions.shape[0]
        n_clusters = len(cluster_atoms)
        
        # クラスター配列の準備
        max_atoms = max(len(atoms) for atoms in cluster_atoms.values())
        cluster_array = np.full((n_clusters, max_atoms), -1, dtype=np.int32)
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
        
        for cid, atoms in cluster_atoms.items():
            cluster_array[cid, :len(atoms)] = atoms
            cluster_sizes[cid] = len(atoms)
        
        # GPU転送
        cluster_array_gpu = cp.asarray(cluster_array)
        cluster_sizes_gpu = cp.asarray(cluster_sizes)
        positions_flat = positions.flatten().astype(cp.float32)
        
        # 結果配列
        burgers_vectors = cp.zeros((n_clusters, 3), dtype=cp.float32)
        
        # カーネル実行
        threads_per_block = 256
        blocks = n_clusters
        
        self.burgers_kernel(
            (blocks,), (threads_per_block,),
            (positions_flat, cluster_array_gpu, cluster_sizes_gpu,
             burgers_vectors.ravel(), cutoff, n_atoms, n_clusters, max_atoms)
        )
        
        return burgers_vectors
    
    def compute_rolling_stats_cuda(self,
                                  data: cp.ndarray,
                                  window: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        CUDAカーネルで移動平均・標準偏差計算
        
        Returns
        -------
        tuple
            (mean, std) の両方 (n_frames, n_features)
        """
        if not self.compiled or self.rolling_stats_kernel is None:
            raise RuntimeError("CUDA kernels not compiled")
        
        n_frames, n_features = data.shape
        mean = cp.zeros_like(data, dtype=cp.float32)
        std = cp.zeros_like(data, dtype=cp.float32)
        
        # カーネル実行（2Dグリッド）
        threads_per_block = (16, 16)
        blocks_x = (n_frames + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (n_features + threads_per_block[1] - 1) // threads_per_block[1]
        
        self.rolling_stats_kernel(
            (blocks_x, blocks_y), threads_per_block,
            (data.astype(cp.float32), mean, std,
             window, n_frames, n_features)
        )
        
        return mean, std
    
    def get_kernel_status(self) -> Dict[str, bool]:
        """各カーネルのコンパイル状態を取得"""
        return {
            'strain_tensor': self.strain_kernel is not None,
            'coordination': self.coord_kernel is not None,
            'damage_score': self.damage_kernel is not None,
            'defect_charge': self.defect_charge_kernel is not None,
            'structural_coherence': self.coherence_kernel is not None,
            'burgers_vector': self.burgers_kernel is not None,
            'rolling_stats': self.rolling_stats_kernel is not None,
        }

# ===============================
# Export
# ===============================

__all__ = [
    'MaterialCUDAKernels',
    # Original kernel codes
    'STRAIN_TENSOR_KERNEL_CODE',
    'COORDINATION_NUMBER_KERNEL_CODE',
    'DAMAGE_SCORE_KERNEL_CODE',
    # Extended kernel codes
    'DEFECT_CHARGE_KERNEL_CODE',
    'STRUCTURAL_COHERENCE_KERNEL_CODE',
    'BURGERS_VECTOR_KERNEL_CODE',
    'ROLLING_STATS_KERNEL_CODE',
]

# ===============================
# Test/Demo
# ===============================

if __name__ == "__main__":
    print("💎 Extended Material CUDA Kernels Module v2.0")
    print("   RTX 4070 Ti SUPER optimized!")
    print("   7 kernels for ultimate performance!")
    
    if HAS_CUDA:
        kernels = MaterialCUDAKernels()
        if kernels.compiled:
            print("\n✅ Compilation status:")
            for name, status in kernels.get_kernel_status().items():
                symbol = "✓" if status else "✗"
                print(f"   {symbol} {name}")
        else:
            print("   ⚠️ Compilation failed")
    else:
        print("   ❌ CUDA not available")
        
