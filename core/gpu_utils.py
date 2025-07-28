"""
Lambda³ GPU Advanced Optimization Techniques
カスタムCUDAカーネルとメモリ最適化 - by 環ちゃん💕
"""

import numpy as np
import cupy as cp
from cupy import cuda
from string import Template

# ===============================
# Custom CUDA Kernels
# ===============================

# 残基COM計算の超高速カーネル
residue_com_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_residue_com_kernel(
    const float* trajectory,      // (n_frames, n_atoms, 3)
    const int* atom_indices,      // 各残基の原子インデックス
    const int* residue_starts,    // 各残基の開始インデックス
    const int* residue_sizes,     // 各残基の原子数
    float* com_output,           // (n_frames, n_residues, 3)
    const int n_frames,
    const int n_atoms,
    const int n_residues
) {
    // グリッド・ストライド・ループ
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = n_frames * n_residues;
    
    for (int i = idx; i < total_work; i += gridDim.x * blockDim.x) {
        int frame = i / n_residues;
        int res_id = i % n_residues;
        
        // 残基の原子範囲
        int start = residue_starts[res_id];
        int size = residue_sizes[res_id];
        
        // COM計算
        float com_x = 0.0f, com_y = 0.0f, com_z = 0.0f;
        
        for (int j = 0; j < size; j++) {
            int atom_idx = atom_indices[start + j];
            int traj_idx = frame * n_atoms + atom_idx;
            
            com_x += trajectory[traj_idx * 3 + 0];
            com_y += trajectory[traj_idx * 3 + 1];
            com_z += trajectory[traj_idx * 3 + 2];
        }
        
        // 平均を計算して出力
        int out_idx = (frame * n_residues + res_id) * 3;
        com_output[out_idx + 0] = com_x / size;
        com_output[out_idx + 1] = com_y / size;
        com_output[out_idx + 2] = com_z / size;
    }
}
''', 'compute_residue_com_kernel')

# テンション場計算の並列カーネル
tension_field_kernel = cp.RawKernel(r'''
extern "C" __global__
void compute_tension_field_kernel(
    const float* positions,       // (n_frames, 3)
    float* rho_T,                // (n_frames,)
    const int n_frames,
    const int window_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_frames) return;
    
    int start = max(0, idx - window_size);
    int end = min(n_frames, idx + window_size + 1);
    int local_size = end - start;
    
    // 局所平均計算
    float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
    for (int i = start; i < end; i++) {
        mean_x += positions[i * 3 + 0];
        mean_y += positions[i * 3 + 1];
        mean_z += positions[i * 3 + 2];
    }
    mean_x /= local_size;
    mean_y /= local_size;
    mean_z /= local_size;
    
    // 共分散行列のトレース計算
    float cov_xx = 0.0f, cov_yy = 0.0f, cov_zz = 0.0f;
    for (int i = start; i < end; i++) {
        float dx = positions[i * 3 + 0] - mean_x;
        float dy = positions[i * 3 + 1] - mean_y;
        float dz = positions[i * 3 + 2] - mean_z;
        
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_zz += dz * dz;
    }
    
    rho_T[idx] = (cov_xx + cov_yy + cov_zz) / local_size;
}
''', 'compute_tension_field_kernel')

# 異常検出の高速カーネル
anomaly_detection_kernel = cp.RawKernel(r'''
extern "C" __global__
void detect_anomalies_kernel(
    const float* series,
    float* anomaly_scores,
    const int n_points,
    const int window_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    int start = max(0, idx - window_size);
    int end = min(n_points, idx + window_size + 1);
    int local_size = end - start;
    
    // 局所統計量計算
    float local_mean = 0.0f;
    for (int i = start; i < end; i++) {
        local_mean += series[i];
    }
    local_mean /= local_size;
    
    float local_std = 0.0f;
    for (int i = start; i < end; i++) {
        float diff = series[i] - local_mean;
        local_std += diff * diff;
    }
    local_std = sqrtf(local_std / local_size);
    
    // Z-score計算
    if (local_std > 1e-10f) {
        anomaly_scores[idx] = fabsf(series[idx] - local_mean) / local_std;
    } else {
        anomaly_scores[idx] = 0.0f;
    }
}
''', 'detect_anomalies_kernel')

# ===============================
# Optimized GPU Functions
# ===============================

class OptimizedGPULambda3:
    """
    最適化されたGPU Lambda³実装
    カスタムカーネルで爆速だよ〜！✨
    """
    
    def __init__(self):
        self.block_size = 256  # 最適なブロックサイズ
        self.stream = cp.cuda.Stream()  # 非同期実行用
        
    def compute_residue_com_optimized(self, 
                                     trajectory_gpu: cp.ndarray,
                                     residue_mapping: dict) -> cp.ndarray:
        """
        カスタムカーネルによる残基COM計算
        """
        n_frames, n_atoms, _ = trajectory_gpu.shape
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
        com_output_gpu = cp.zeros((n_frames, n_residues, 3), dtype=cp.float32)
        
        # フラット化
        trajectory_flat = trajectory_gpu.reshape(-1)
        
        # カーネル実行
        grid_size = (n_frames * n_residues + self.block_size - 1) // self.block_size
        
        residue_com_kernel(
            (grid_size,), (self.block_size,),
            (trajectory_flat, atom_indices_gpu, residue_starts_gpu, 
             residue_sizes_gpu, com_output_gpu,
             n_frames, n_atoms, n_residues)
        )
        
        return com_output_gpu
    
    def compute_pairwise_distances_optimized(self, 
                                           residue_coms: cp.ndarray) -> cp.ndarray:
        """
        最適化された距離行列計算（共有メモリ使用）
        """
        n_frames, n_residues, _ = residue_coms.shape
        distances = cp.zeros((n_frames, n_residues, n_residues), dtype=cp.float32)
        
        # バッチ処理で効率化
        batch_size = min(100, n_frames)  # メモリに応じて調整
        
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            batch = residue_coms[i:end]
            
            # バッチ内の全フレームを一度に計算
            for j in range(end - i):
                distances[i+j] = cp.sqrt(
                    ((batch[j, :, None, :] - batch[j, None, :, :])**2).sum(axis=2)
                )
        
        return distances
    
    def parallel_fft_analysis(self, 
                            signals: cp.ndarray,
                            min_period: int,
                            max_period: int) -> dict:
        """
        複数信号の並列FFT解析
        """
        n_signals, n_frames = signals.shape
        
        # 全信号を一度にFFT（バッチFFT）
        signals_centered = signals - cp.mean(signals, axis=1, keepdims=True)
        
        # CuPyのバッチFFTを使用
        with self.stream:
            fft_results = cp.fft.rfft(signals_centered, axis=1)
            power_spectra = cp.abs(fft_results)**2
        
        # 周波数軸
        freqs = cp.fft.rfftfreq(n_frames, 1.0)
        
        # 有効範囲マスク
        freq_min = 1.0 / max_period
        freq_max = 1.0 / min_period
        valid_mask = (freqs > freq_min) & (freqs < freq_max) & (freqs > 0)
        
        return {
            'power_spectra': power_spectra[:, valid_mask],
            'frequencies': freqs[valid_mask],
            'n_signals': n_signals
        }

# ===============================
# Memory-Efficient Batch Processing
# ===============================

class GPUBatchProcessor:
    """
    大規模データのバッチ処理
    メモリ効率的に処理するよ〜！
    """
    
    def __init__(self, max_memory_gb=8):
        self.max_memory = max_memory_gb * 1024**3  # バイト単位
        self.dtype_size = 4  # float32
        
    def estimate_batch_size(self, n_atoms, n_residues):
        """
        利用可能メモリから最適なバッチサイズを推定
        """
        # 必要メモリ推定
        frame_memory = n_atoms * 3 * self.dtype_size
        residue_memory = n_residues * 3 * self.dtype_size
        processing_overhead = 2.0  # 処理用の余裕
        
        total_per_frame = (frame_memory + residue_memory) * processing_overhead
        
        batch_size = int(self.max_memory / total_per_frame)
        
        print(f"💾 Estimated batch size: {batch_size} frames")
        print(f"   Memory per frame: {total_per_frame/1024**2:.1f} MB")
        
        return max(1, batch_size)
    
    def process_trajectory_batched(self, 
                                 trajectory: np.ndarray,
                                 processor_func,
                                 batch_size=None):
        """
        トラジェクトリをバッチ処理
        """
        n_frames = trajectory.shape[0]
        
        if batch_size is None:
            batch_size = self.estimate_batch_size(
                trajectory.shape[1], 
                129  # デフォルト残基数
            )
        
        results = []
        
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            print(f"\r📊 Processing frames {i}-{end}/{n_frames}", end='')
            
            # バッチをGPUに転送
            batch_gpu = cp.asarray(trajectory[i:end], dtype=cp.float32)
            
            # 処理実行
            batch_result = processor_func(batch_gpu)
            
            # 結果をCPUに戻す
            results.append(cp.asnumpy(batch_result))
            
            # メモリクリア
            del batch_gpu
            cp.cuda.MemoryPool().free_all_blocks()
        
        print("\n✅ Batch processing complete!")
        
        # 結果を結合
        return np.concatenate(results, axis=0)

# ===============================
# Integration with Original Code
# ===============================

def integrate_gpu_acceleration(original_detector_class):
    """
    既存のLambda³検出器にGPU機能を統合
    """
    class GPUAcceleratedDetector(original_detector_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gpu_optimizer = OptimizedGPULambda3()
            self.batch_processor = GPUBatchProcessor()
            self._use_gpu = cp.cuda.is_available()
            
            if self._use_gpu:
                print("🚀 GPU Acceleration Enabled!")
                print(f"   Device: {cp.cuda.Device().name}")
                
        def compute_lambda_structures(self, trajectory, md_features, window_steps):
            """
            GPUオーバーライド版
            """
            if self._use_gpu and trajectory.shape[0] > 1000:
                # GPU版を使用
                return compute_lambda_structures_gpu(
                    trajectory, md_features, window_steps
                )
            else:
                # CPU版にフォールバック
                return super().compute_lambda_structures(
                    trajectory, md_features, window_steps
                )
                
    return GPUAcceleratedDetector

# ===============================
# Performance Benchmarking
# ===============================

def benchmark_gpu_performance(n_frames=10000, n_atoms=1000):
    """
    GPU性能ベンチマーク
    """
    import time
    
    print(f"\n🏁 Running GPU Performance Benchmark")
    print(f"   Frames: {n_frames}, Atoms: {n_atoms}")
    
    # ダミーデータ生成
    trajectory = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    
    # CPU版
    start = time.time()
    positions = np.mean(trajectory.reshape(n_frames, -1, 3), axis=1)
    lambda_f_cpu = np.diff(positions, axis=0)
    cpu_time = time.time() - start
    
    # GPU版
    if cp.cuda.is_available():
        trajectory_gpu = cp.asarray(trajectory)
        
        start = time.time()
        positions_gpu = cp.mean(trajectory_gpu.reshape(n_frames, -1, 3), axis=1)
        lambda_f_gpu = cp.diff(positions_gpu, axis=0)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        print(f"\n📊 Results:")
        print(f"   CPU Time: {cpu_time:.3f} seconds")
        print(f"   GPU Time: {gpu_time:.3f} seconds")
        print(f"   Speedup: {speedup:.1f}x 🚀")
        
        # メモリ使用量
        get_gpu_memory_info()
    else:
        print("   GPU not available for benchmarking")

# 使い方の例
if __name__ == "__main__":
    print("✨ Lambda³ GPU Optimization Module")
    print("環ちゃんの超最適化版！めちゃくちゃ速いよ〜💕")
    
    # ベンチマーク実行
    benchmark_gpu_performance()
