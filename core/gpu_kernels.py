"""
CUDA Kernels for Lambda³ GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

高速なCUDAカーネル集だよ〜！💕
めちゃくちゃ速い処理を実現しちゃう！

by 環ちゃん
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TYPE_CHECKING
from ..types import ArrayType, NDArray

# GPU imports
try:
    import cupy as cp
    from cupy import cuda
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cuda = None

logger = logging.getLogger('lambda3_gpu.core.kernels')

# ===============================
# Kernel Code Templates
# ===============================

# 残基COM計算カーネル
RESIDUE_COM_KERNEL = r'''
extern "C" __global__
void compute_residue_com_kernel(
    const float* __restrict__ trajectory,      // (n_frames, n_atoms, 3)
    const int* __restrict__ atom_indices,      // 各残基の原子インデックス
    const int* __restrict__ residue_starts,    // 各残基の開始インデックス
    const int* __restrict__ residue_sizes,     // 各残基の原子数
    float* __restrict__ com_output,           // (n_frames, n_residues, 3)
    const int n_frames,
    const int n_atoms,
    const int n_residues
) {
    // グリッド・ストライド・ループ
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total_work = n_frames * n_residues;
    
    for (int i = idx; i < total_work; i += stride) {
        const int frame = i / n_residues;
        const int res_id = i % n_residues;
        
        // 残基の原子範囲
        const int start = residue_starts[res_id];
        const int size = residue_sizes[res_id];
        
        // COM計算（共有メモリ使用で高速化）
        float com_x = 0.0f, com_y = 0.0f, com_z = 0.0f;
        
        for (int j = 0; j < size; j++) {
            const int atom_idx = atom_indices[start + j];
            const int traj_idx = (frame * n_atoms + atom_idx) * 3;
            
            com_x += trajectory[traj_idx + 0];
            com_y += trajectory[traj_idx + 1];
            com_z += trajectory[traj_idx + 2];
        }
        
        // 平均を計算して出力
        const float inv_size = 1.0f / size;
        const int out_idx = (frame * n_residues + res_id) * 3;
        com_output[out_idx + 0] = com_x * inv_size;
        com_output[out_idx + 1] = com_y * inv_size;
        com_output[out_idx + 2] = com_z * inv_size;
    }
}
'''

# テンション場計算カーネル（高速版）
TENSION_FIELD_KERNEL = r'''
extern "C" __global__
void compute_tension_field_kernel(
    const float* __restrict__ positions,       // (n_frames, 3)
    float* __restrict__ rho_T,                // (n_frames,)
    const int n_frames,
    const int window_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_frames) return;
    
    const int start = max(0, idx - window_size);
    const int end = min(n_frames, idx + window_size + 1);
    const int local_size = end - start;
    const float inv_size = 1.0f / local_size;
    
    // 局所平均計算
    float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
    
    for (int i = start; i < end; i++) {
        const int pos_idx = i * 3;
        mean_x += positions[pos_idx + 0];
        mean_y += positions[pos_idx + 1];
        mean_z += positions[pos_idx + 2];
    }
    
    mean_x *= inv_size;
    mean_y *= inv_size;
    mean_z *= inv_size;
    
    // 共分散行列のトレース計算
    float cov_xx = 0.0f, cov_yy = 0.0f, cov_zz = 0.0f;
    
    for (int i = start; i < end; i++) {
        const int pos_idx = i * 3;
        const float dx = positions[pos_idx + 0] - mean_x;
        const float dy = positions[pos_idx + 1] - mean_y;
        const float dz = positions[pos_idx + 2] - mean_z;
        
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_zz += dz * dz;
    }
    
    rho_T[idx] = (cov_xx + cov_yy + cov_zz) * inv_size;
}
'''

# 異常検出カーネル（適応的z-score）
ANOMALY_DETECTION_KERNEL = r'''
extern "C" __global__
void detect_anomalies_kernel(
    const float* __restrict__ series,
    float* __restrict__ anomaly_scores,
    const int n_points,
    const int window_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    const int start = max(0, idx - window_size);
    const int end = min(n_points, idx + window_size + 1);
    const int local_size = end - start;
    const float inv_size = 1.0f / local_size;
    
    // 局所統計量計算（Welford's algorithm使用）
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    
    for (int i = start; i < end; i++) {
        const float delta = series[i] - local_mean;
        local_mean += delta * inv_size;
        local_m2 += delta * (series[i] - local_mean);
    }
    
    const float local_var = local_m2 * inv_size;
    const float local_std = sqrtf(local_var);
    
    // Z-score計算
    if (local_std > 1e-10f) {
        anomaly_scores[idx] = fabsf(series[idx] - local_mean) / local_std;
    } else {
        anomaly_scores[idx] = 0.0f;
    }
}
'''

# 距離行列計算カーネル（最適化版）
DISTANCE_MATRIX_KERNEL = r'''
extern "C" __global__
void compute_distance_matrix_kernel(
    const float* __restrict__ positions,      // (n_points, 3)
    float* __restrict__ distances,           // (n_points, n_points)
    const int n_points
) {
    // 共有メモリを使用して高速化
    extern __shared__ float shared_pos[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int block_size = blockDim.x;
    
    const int row = by * block_size + ty;
    const int col = bx * block_size + tx;
    
    // タイルごとに処理
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n_points + block_size - 1) / block_size; tile++) {
        // 共有メモリにロード
        const int idx_a = row;
        const int idx_b = tile * block_size + tx;
        
        if (idx_a < n_points && idx_b < n_points) {
            shared_pos[ty * block_size * 3 + tx * 3 + 0] = positions[idx_b * 3 + 0];
            shared_pos[ty * block_size * 3 + tx * 3 + 1] = positions[idx_b * 3 + 1];
            shared_pos[ty * block_size * 3 + tx * 3 + 2] = positions[idx_b * 3 + 2];
        }
        
        __syncthreads();
        
        // 距離計算
        if (row < n_points && col < n_points) {
            const int shared_idx = ty * block_size * 3 + tx * 3;
            const float dx = positions[row * 3 + 0] - shared_pos[shared_idx + 0];
            const float dy = positions[row * 3 + 1] - shared_pos[shared_idx + 1];
            const float dz = positions[row * 3 + 2] - shared_pos[shared_idx + 2];
            sum = dx * dx + dy * dy + dz * dz;
        }
        
        __syncthreads();
    }
    
    // 結果を書き込み
    if (row < n_points && col < n_points) {
        distances[row * n_points + col] = sqrtf(sum);
    }
}
'''

# トポロジカルチャージ計算カーネル
TOPOLOGICAL_CHARGE_KERNEL = r'''
extern "C" __global__
void compute_topological_charge_kernel(
    const float* __restrict__ lambda_F,        // (n_steps, 3)
    const float* __restrict__ lambda_F_mag,    // (n_steps,)
    float* __restrict__ Q_lambda,             // (n_steps,)
    const int n_steps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_steps - 1 || idx == 0) {
        if (idx < n_steps) Q_lambda[idx] = 0.0f;
        return;
    }
    
    const float mag_curr = lambda_F_mag[idx];
    const float mag_prev = lambda_F_mag[idx - 1];
    
    if (mag_curr > 1e-10f && mag_prev > 1e-10f) {
        // 正規化ベクトル
        const float inv_mag_prev = 1.0f / mag_prev;
        const float inv_mag_curr = 1.0f / mag_curr;
        
        const float v1_x = lambda_F[(idx - 1) * 3 + 0] * inv_mag_prev;
        const float v1_y = lambda_F[(idx - 1) * 3 + 1] * inv_mag_prev;
        const float v1_z = lambda_F[(idx - 1) * 3 + 2] * inv_mag_prev;
        
        const float v2_x = lambda_F[idx * 3 + 0] * inv_mag_curr;
        const float v2_y = lambda_F[idx * 3 + 1] * inv_mag_curr;
        const float v2_z = lambda_F[idx * 3 + 2] * inv_mag_curr;
        
        // 内積
        const float cos_angle = v1_x * v2_x + v1_y * v2_y + v1_z * v2_z;
        const float clamped_cos = fmaxf(-1.0f, fminf(1.0f, cos_angle));
        const float angle = acosf(clamped_cos);
        
        // 2D回転方向（z成分の外積）
        const float cross_z = v1_x * v2_y - v1_y * v2_x;
        const float signed_angle = (cross_z >= 0) ? angle : -angle;
        
        Q_lambda[idx] = signed_angle / (2.0f * 3.14159265359f);
    } else {
        Q_lambda[idx] = 0.0f;
    }
}
'''

# ===============================
# Kernel Manager Class
# ===============================

class CUDAKernels:
    """
    CUDAカーネル管理クラス
    カーネルのコンパイルと実行を管理するよ〜！
    """
    
    def __init__(self):
        self.kernels = {}
        self.is_initialized = False
        
        if HAS_GPU and cp.cuda.is_available():
            self._compile_kernels()
    
    def _compile_kernels(self):
        """全カーネルをコンパイル"""
        try:
            # 残基COM計算
            self.kernels['residue_com'] = cp.RawKernel(
                RESIDUE_COM_KERNEL, 'compute_residue_com_kernel'
            )
            
            # テンション場
            self.kernels['tension_field'] = cp.RawKernel(
                TENSION_FIELD_KERNEL, 'compute_tension_field_kernel'
            )
            
            # 異常検出
            self.kernels['anomaly_detection'] = cp.RawKernel(
                ANOMALY_DETECTION_KERNEL, 'detect_anomalies_kernel'
            )
            
            # 距離行列
            self.kernels['distance_matrix'] = cp.RawKernel(
                DISTANCE_MATRIX_KERNEL, 'compute_distance_matrix_kernel'
            )
            
            # トポロジカルチャージ
            self.kernels['topological_charge'] = cp.RawKernel(
                TOPOLOGICAL_CHARGE_KERNEL, 'compute_topological_charge_kernel'
            )
            
            self.is_initialized = True
            logger.info("CUDA kernels compiled successfully!")
            
        except Exception as e:
            logger.error(f"Failed to compile CUDA kernels: {e}")
            self.is_initialized = False
    
    def get_kernel(self, name: str):
        """カーネルを取得"""
        if not self.is_initialized:
            raise RuntimeError("CUDA kernels not initialized")
        
        if name not in self.kernels:
            raise KeyError(f"Kernel '{name}' not found")
        
        return self.kernels[name]

# ===============================
# Kernel Wrapper Functions
# ===============================

def residue_com_kernel(trajectory: NDArray,
                      residue_mapping: dict,
                      block_size: int = 256) -> NDArray:
    """
    残基COM計算カーネルのラッパー
    
    Parameters
    ----------
    trajectory : cp.ndarray
        トラジェクトリ (n_frames, n_atoms, 3)
    residue_mapping : dict
        残基ID -> 原子インデックスリスト
    block_size : int
        CUDAブロックサイズ
        
    Returns
    -------
    cp.ndarray
        残基COM (n_frames, n_residues, 3)
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")
    
    n_frames, n_atoms, _ = trajectory.shape
    n_residues = len(residue_mapping)
    
    # 残基情報を準備
    all_indices = []
    residue_starts = []
    residue_sizes = []
    
    current_start = 0
    for res_id in sorted(residue_mapping.keys()):
        atoms = residue_mapping[res_id]
        all_indices.extend(atoms)
        residue_starts.append(current_start)
        residue_sizes.append(len(atoms))
        current_start += len(atoms)
    
    # GPU配列に変換
    atom_indices_gpu = cp.array(all_indices, dtype=cp.int32)
    residue_starts_gpu = cp.array(residue_starts, dtype=cp.int32)
    residue_sizes_gpu = cp.array(residue_sizes, dtype=cp.int32)
    
    # 出力配列
    com_output = cp.zeros((n_frames, n_residues, 3), dtype=cp.float32)
    
    # カーネル実行
    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel('residue_com')
    
    grid_size = (n_frames * n_residues + block_size - 1) // block_size
    
    kernel(
        (grid_size,), (block_size,),
        (trajectory.ravel(), atom_indices_gpu, residue_starts_gpu,
         residue_sizes_gpu, com_output,
         n_frames, n_atoms, n_residues)
    )
    
    return com_output

def tension_field_kernel(positions: NDArray,
                        window_size: int,
                        block_size: int = 256) -> NDArray:
    """
    テンション場計算カーネルのラッパー
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")
    
    n_frames = positions.shape[0]
    rho_T = cp.zeros(n_frames, dtype=cp.float32)
    
    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel('tension_field')
    
    grid_size = (n_frames + block_size - 1) // block_size
    
    kernel(
        (grid_size,), (block_size,),
        (positions.ravel(), rho_T, n_frames, window_size)
    )
    
    return rho_T

def anomaly_detection_kernel(series: NDArray,
                           window_size: int,
                           block_size: int = 256) -> NDArray:
    """
    異常検出カーネルのラッパー
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")
    
    n_points = len(series)
    anomaly_scores = cp.zeros(n_points, dtype=cp.float32)
    
    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel('anomaly_detection')
    
    grid_size = (n_points + block_size - 1) // block_size
    
    kernel(
        (grid_size,), (block_size,),
        (series, anomaly_scores, n_points, window_size)
    )
    
    return anomaly_scores

def distance_matrix_kernel(positions: NDArray,
                         block_size: int = 16) -> NDArray:
    """
    距離行列計算カーネルのラッパー
    共有メモリ使用で高速化！
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")
    
    n_points = positions.shape[0]
    distances = cp.zeros((n_points, n_points), dtype=cp.float32)
    
    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel('distance_matrix')
    
    grid_size = ((n_points + block_size - 1) // block_size,
                 (n_points + block_size - 1) // block_size)
    
    # 共有メモリサイズ
    shared_mem_size = block_size * block_size * 3 * 4  # float32
    
    kernel(
        grid_size, (block_size, block_size),
        (positions.ravel(), distances, n_points),
        shared_mem=shared_mem_size
    )
    
    return distances

def topological_charge_kernel(lambda_F: NDArray,
                            lambda_F_mag: NDArray,
                            block_size: int = 256) -> NDArray:
    """
    トポロジカルチャージ計算カーネルのラッパー
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")
    
    n_steps = len(lambda_F_mag)
    Q_lambda = cp.zeros(n_steps, dtype=cp.float32)
    
    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel('topological_charge')
    
    grid_size = (n_steps + block_size - 1) // block_size
    
    kernel(
        (grid_size,), (block_size,),
        (lambda_F.ravel(), lambda_F_mag, Q_lambda, n_steps)
    )
    
    return Q_lambda

# ===============================
# Utility Kernels
# ===============================

# 要素ごとの演算カーネル（汎用）
ELEMENTWISE_KERNEL_TEMPLATE = '''
extern "C" __global__
void {kernel_name}(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n_elements,
    const float param1,
    const float param2
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {{
        {operation}
    }}
}}
'''

def create_elementwise_kernel(name: str, operation: str):
    """
    要素ごとの演算カーネルを動的に生成
    
    使用例:
        sigmoid_kernel = create_elementwise_kernel(
            'sigmoid',
            'output[idx] = 1.0f / (1.0f + expf(-input[idx]))'
        )
    """
    if not HAS_GPU:
        return None
    
    kernel_code = ELEMENTWISE_KERNEL_TEMPLATE.format(
        kernel_name=name,
        operation=operation
    )
    
    return cp.RawKernel(kernel_code, name)

# ===============================
# Performance Testing
# ===============================

def benchmark_kernels(n_frames: int = 10000, 
                     n_atoms: int = 1000,
                     n_residues: int = 100):
    """
    カーネルのベンチマーク
    """
    if not HAS_GPU:
        logger.warning("GPU not available for benchmarking")
        return
    
    import time
    
    logger.info(f"\n{'='*60}")
    logger.info("CUDA Kernel Benchmarks")
    logger.info(f"{'='*60}")
    logger.info(f"Frames: {n_frames}, Atoms: {n_atoms}, Residues: {n_residues}")
    
    # ダミーデータ
    trajectory = cp.random.randn(n_frames, n_atoms, 3, dtype=cp.float32)
    positions = cp.random.randn(n_frames, 3, dtype=cp.float32)
    residue_mapping = {
        i: list(range(i * (n_atoms // n_residues), 
                      (i + 1) * (n_atoms // n_residues)))
        for i in range(n_residues)
    }
    
    # 1. 残基COM計算
    start = time.time()
    com = residue_com_kernel(trajectory[:100], residue_mapping)  # 100フレームのみ
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    logger.info(f"\nResidue COM: {elapsed:.4f} seconds")
    logger.info(f"  Throughput: {100 * n_atoms / elapsed:.2f} atoms/sec")
    
    # 2. テンション場
    start = time.time()
    rho_t = tension_field_kernel(positions, window_size=50)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    logger.info(f"\nTension Field: {elapsed:.4f} seconds")
    logger.info(f"  Throughput: {n_frames / elapsed:.2f} frames/sec")
    
    # 3. 異常検出
    series = cp.random.randn(n_frames, dtype=cp.float32)
    start = time.time()
    anomalies = anomaly_detection_kernel(series, window_size=50)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    logger.info(f"\nAnomaly Detection: {elapsed:.4f} seconds")
    logger.info(f"  Throughput: {n_frames / elapsed:.2f} points/sec")
    
    logger.info(f"\n{'='*60}\n")

# グローバルカーネルマネージャー
_kernel_manager = None

def get_kernel_manager() -> CUDAKernels:
    """グローバルカーネルマネージャーを取得"""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = CUDAKernels()
    return _kernel_manager

# ===============================
# Export Kernels
# ===============================

__all__ = [
    'CUDAKernels',
    'residue_com_kernel',
    'tension_field_kernel', 
    'anomaly_detection_kernel',
    'distance_matrix_kernel',
    'topological_charge_kernel',
    'create_elementwise_kernel',
    'benchmark_kernels',
    'get_kernel_manager'
]
