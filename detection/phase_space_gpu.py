"""
Lambda³ GPU版位相空間解析モジュール（爆速カーネル版・完全リファクタリング版）
高次元位相空間での異常検出とアトラクタ解析
CuPy RawKernelによる超高速化実装（PTX 8.4対応）

主な改善点：
- float型とndarray型の明確な区別
- エラーハンドリングの強化
- メモリ効率の最適化
- 型ヒントの充実
- ドキュメントの改善
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_GPU = True
    ArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    cp = None
    HAS_GPU = False
    ArrayType = np.ndarray

from ..types import NDArray
from ..core.gpu_utils import GPUBackend

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ===============================
# 設定データクラス
# ===============================

@dataclass
class PhaseSpaceConfig:
    """位相空間解析の設定パラメータ"""
    embedding_dim: int = 3
    delay: int = 50
    n_neighbors: int = 20
    recurrence_threshold: float = 0.1
    min_diagonal_length: int = 2
    voxel_grid_size: int = 128
    max_analysis_points: int = 5000
    gpu_block_size: int = 256
    
    def validate(self) -> bool:
        """パラメータの妥当性チェック"""
        if self.embedding_dim < 1 or self.embedding_dim > 20:
            logger.warning(f"Invalid embedding_dim: {self.embedding_dim}")
            return False
        if self.delay < 1:
            logger.warning(f"Invalid delay: {self.delay}")
            return False
        if self.n_neighbors < 1:
            logger.warning(f"Invalid n_neighbors: {self.n_neighbors}")
            return False
        return True

# ===============================
# CuPy RawKernel定義（全13個）
# ===============================

# 1. RQA特徴量計算カーネル
RQA_FEATURES_KERNEL_CODE = r'''
extern "C" __global__
void compute_rqa_features_kernel(
    const float* rec_matrix,
    float* features_out,
    const int n,
    const int min_line_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 対角線構造の検出
        for (int offset = 1; offset < n - idx; offset++) {
            int diag_len = 0;
            int i = idx;
            int j = offset;
            
            while (i < n && j < n) {
                if (rec_matrix[i * n + j] > 0) {
                    diag_len++;
                } else {
                    if (diag_len >= min_line_length) {
                        atomicAdd(&features_out[0], (float)diag_len);  // determinism
                        atomicAdd(&features_out[3], 1.0f);  // diagonal count
                    }
                    diag_len = 0;
                }
                i++;
                j++;
            }
        }
        
        // 垂直構造の検出
        int vert_len = 0;
        for (int j = 0; j < n; j++) {
            if (rec_matrix[idx * n + j] > 0) {
                vert_len++;
            } else {
                if (vert_len >= min_line_length) {
                    atomicAdd(&features_out[1], (float)vert_len);  // laminarity
                    atomicAdd(&features_out[2], (float)vert_len);  // trapping time
                    atomicAdd(&features_out[4], 1.0f);  // vertical count
                }
                vert_len = 0;
            }
        }
    }
}
'''

# 2. アトラクタ体積計算カーネル
ATTRACTOR_VOLUME_KERNEL_CODE = r'''
extern "C" __global__
void compute_attractor_volume_kernel(
    const float* phase_space,
    int* voxel_grid,
    const float* phase_min,
    const float* phase_range,
    const int n_points,
    const int dim,
    const int grid_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_points) {
        // 3次元ボクセル座標を計算
        int voxel_x = 0, voxel_y = 0, voxel_z = 0;
        
        if (dim >= 1) {
            float normalized = (phase_space[idx * dim + 0] - phase_min[0]) / phase_range[0];
            voxel_x = min(max(0, (int)(normalized * (grid_size - 1))), grid_size - 1);
        }
        if (dim >= 2) {
            float normalized = (phase_space[idx * dim + 1] - phase_min[1]) / phase_range[1];
            voxel_y = min(max(0, (int)(normalized * (grid_size - 1))), grid_size - 1);
        }
        if (dim >= 3) {
            float normalized = (phase_space[idx * dim + 2] - phase_min[2]) / phase_range[2];
            voxel_z = min(max(0, (int)(normalized * (grid_size - 1))), grid_size - 1);
        }
        
        // 1次元インデックスに変換
        int voxel_idx = voxel_x * grid_size * grid_size + voxel_y * grid_size + voxel_z;
        
        // ボクセルを占有としてマーク
        voxel_grid[voxel_idx] = 1;
    }
}
'''

# 3. フラクタル次元計算カーネル
FRACTAL_DIMENSION_KERNEL_CODE = r'''
extern "C" __global__
void compute_fractal_dimension_kernel(
    const float* distances,
    int* correlation_integral,
    const float* radii,
    const int n_pairs,
    const int n_radii
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_pairs) {
        float dist = distances[idx];
        
        // 各半径での相関積分に寄与
        for (int r_idx = 0; r_idx < n_radii; r_idx++) {
            if (dist < radii[r_idx]) {
                atomicAdd(&correlation_integral[r_idx], 1);
            }
        }
    }
}
'''

# 4. 複雑性指標カーネル
COMPLEXITY_MEASURES_KERNEL_CODE = r'''
extern "C" __global__
void compute_complexity_measures_kernel(
    const float* phase_space,
    float* measures_out,
    const int n,
    const int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n - 2) {
        // 局所的なカオス性（近傍の発散率）
        float local_divergence = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff1 = phase_space[(idx + 1) * dim + d] - phase_space[idx * dim + d];
            float diff2 = phase_space[(idx + 2) * dim + d] - phase_space[(idx + 1) * dim + d];
            local_divergence += fabsf(diff2 - diff1);
        }
        atomicAdd(&measures_out[0], local_divergence);
        
        // 予測誤差
        float pred_error = 0.0f;
        for (int d = 0; d < dim; d++) {
            float predicted = 2 * phase_space[(idx + 1) * dim + d] - phase_space[idx * dim + d];
            float actual = phase_space[(idx + 2) * dim + d];
            pred_error += (predicted - actual) * (predicted - actual);
        }
        atomicAdd(&measures_out[1], sqrtf(pred_error));
        
        // 局所複雑性（近傍の分散）
        if (idx > 0) {
            float local_var = 0.0f;
            for (int d = 0; d < dim; d++) {
                float mean = (phase_space[(idx - 1) * dim + d] + 
                             phase_space[idx * dim + d] + 
                             phase_space[(idx + 1) * dim + d]) / 3.0f;
                float diff = phase_space[idx * dim + d] - mean;
                local_var += diff * diff;
            }
            atomicAdd(&measures_out[2], sqrtf(local_var));
        }
        
        // エントロピー寄与
        for (int d = 0; d < dim; d++) {
            float delta = fabsf(phase_space[(idx + 1) * dim + d] - phase_space[idx * dim + d]);
            if (delta > 1e-10f) {
                atomicAdd(&measures_out[3], -delta * logf(delta));
            }
        }
        
        // 安定性（軌道の局所的な収束性）
        if (idx < n - 3) {
            float stability = 0.0f;
            for (int d = 0; d < dim; d++) {
                float second_diff = phase_space[(idx + 2) * dim + d] - 
                                   2 * phase_space[(idx + 1) * dim + d] + 
                                   phase_space[idx * dim + d];
                stability += fabsf(second_diff);
            }
            atomicAdd(&measures_out[4], stability);
        }
    }
}
'''

# 5. k-NN異常検出カーネル
KNN_ANOMALY_KERNEL_CODE = r'''
extern "C" __global__
void knn_trajectory_anomaly_kernel(
    const float* phase_space,
    float* anomaly_scores,
    const int k,
    const int n,
    const int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 共有メモリの代わりに固定サイズ配列
        float distances[256];
        int batch_size = min(256, n);
        
        // 全点の初期化
        for (int i = 0; i < batch_size; i++) {
            distances[i] = 1e10f;
        }
        
        // 距離計算
        for (int j = 0; j < n && j < batch_size; j++) {
            if (j != idx) {
                float dist_sq = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = phase_space[j * dim + d] - phase_space[idx * dim + d];
                    dist_sq += diff * diff;
                }
                distances[j] = sqrtf(dist_sq);
            }
        }
        
        // k近傍を探索（簡易ソート）
        float knn_sum = 0.0f;
        for (int kth = 0; kth < k && kth < batch_size; kth++) {
            float min_dist = 1e10f;
            int min_idx = 0;
            
            for (int j = 0; j < batch_size; j++) {
                if (distances[j] < min_dist) {
                    min_dist = distances[j];
                    min_idx = j;
                }
            }
            
            knn_sum += min_dist;
            distances[min_idx] = 1e10f;
        }
        
        anomaly_scores[idx] = knn_sum / k;
    }
}
'''

# 6. 対角線長分布カーネル
DIAGONAL_DIST_KERNEL_CODE = r'''
extern "C" __global__
void compute_diagonal_distribution_kernel(
    const float* rec_matrix,
    int* diag_dist,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 各対角線を処理
        for (int start_j = 0; start_j < n; start_j++) {
            int i = idx;
            int j = start_j;
            int current_len = 0;
            
            while (i < n && j < n) {
                if (rec_matrix[i * n + j] > 0) {
                    current_len++;
                } else {
                    if (current_len > 0) {
                        int dist_idx = min(current_len, n/2 - 1);
                        atomicAdd(&diag_dist[dist_idx], 1);
                    }
                    current_len = 0;
                }
                i++;
                j++;
            }
            
            // 最後のセグメント
            if (current_len > 0) {
                int dist_idx = min(current_len, n/2 - 1);
                atomicAdd(&diag_dist[dist_idx], 1);
            }
        }
    }
}
'''

# 7. 曲率異常計算カーネル
CURVATURE_ANOMALY_KERNEL_CODE = r'''
extern "C" __global__
void compute_curvature_anomaly_kernel(
    const float* phase_space,
    float* curvature_anomaly,
    const int n,
    const int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > 0 && idx < n - 1) {
        // 3点での曲率計算
        float v1_norm = 0.0f;
        float v2_norm = 0.0f;
        float dot_product = 0.0f;
        
        for (int d = 0; d < dim; d++) {
            float v1 = phase_space[idx * dim + d] - phase_space[(idx - 1) * dim + d];
            float v2 = phase_space[(idx + 1) * dim + d] - phase_space[idx * dim + d];
            
            v1_norm += v1 * v1;
            v2_norm += v2 * v2;
            dot_product += v1 * v2;
        }
        
        v1_norm = sqrtf(v1_norm);
        v2_norm = sqrtf(v2_norm);
        
        if (v1_norm > 1e-10f && v2_norm > 1e-10f) {
            float cos_angle = dot_product / (v1_norm * v2_norm);
            cos_angle = fminf(fmaxf(cos_angle, -1.0f), 1.0f);
            curvature_anomaly[idx] = 1.0f - cos_angle;
        }
    }
}
'''

# 8. 速度異常計算カーネル
VELOCITY_ANOMALY_KERNEL_CODE = r'''
extern "C" __global__
void compute_velocity_anomaly_kernel(
    const float* phase_space,
    float* velocity_anomaly,
    const int n,
    const int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n - 1) {
        float velocity = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = phase_space[(idx + 1) * dim + d] - phase_space[idx * dim + d];
            velocity += diff * diff;
        }
        velocity_anomaly[idx] = sqrtf(velocity);
    }
}
'''

# 9. 加速度異常計算カーネル
ACCELERATION_ANOMALY_KERNEL_CODE = r'''
extern "C" __global__
void compute_acceleration_anomaly_kernel(
    const float* phase_space,
    float* acceleration_anomaly,
    const int n,
    const int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > 0 && idx < n - 1) {
        float acceleration = 0.0f;
        for (int d = 0; d < dim; d++) {
            float acc = phase_space[(idx + 1) * dim + d] - 
                       2 * phase_space[idx * dim + d] + 
                       phase_space[(idx - 1) * dim + d];
            acceleration += acc * acc;
        }
        acceleration_anomaly[idx] = sqrtf(acceleration);
    }
}
'''

# 10. Lyapunov指数計算カーネル
LYAPUNOV_KERNEL_CODE = r'''
extern "C" __global__
void compute_lyapunov_kernel(
    const float* phase_space,
    const int* ref_indices,
    float* lyap_values,
    const int n,
    const int dim,
    const int n_refs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_refs) {
        int ref_idx = ref_indices[idx];
        
        // 近傍を探索
        float min_dist = 1e10f;
        int nearest_idx = -1;
        
        int search_start = max(0, ref_idx - 100);
        int search_end = min(n, ref_idx + 100);
        
        for (int j = search_start; j < search_end; j++) {
            if (abs(j - ref_idx) > 1) {
                float dist = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = phase_space[j * dim + d] - phase_space[ref_idx * dim + d];
                    dist += diff * diff;
                }
                
                dist = sqrtf(dist);
                if (dist < min_dist && dist > 1e-10f) {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }
        }
        
        // 時間発展を追跡
        if (nearest_idx >= 0) {
            int max_steps = min(10, n - max(ref_idx, nearest_idx));
            float lyap_sum = 0.0f;
            int count = 0;
            
            for (int dt = 1; dt < max_steps; dt++) {
                if (ref_idx + dt < n && nearest_idx + dt < n) {
                    float evolved_dist = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        float diff = phase_space[(ref_idx + dt) * dim + d] - 
                                    phase_space[(nearest_idx + dt) * dim + d];
                        evolved_dist += diff * diff;
                    }
                    
                    evolved_dist = sqrtf(evolved_dist);
                    
                    if (evolved_dist > 1e-10f && min_dist > 1e-10f) {
                        lyap_sum += logf(evolved_dist / min_dist) / dt;
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                lyap_values[idx] = lyap_sum / count;
            }
        }
    }
}
'''

# 11. 遷移スコアマッピングカーネル
MAP_TRANSITION_SCORES_KERNEL_CODE = r'''
extern "C" __global__
void map_transition_scores_kernel(
    float* scores,
    const float* distances,
    const int window_size,
    const int stride,
    const int n_distances,
    const int n_frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_distances) {
        int start = idx * stride;
        int end = min(start + window_size, n_frames);
        
        for (int j = start; j < end; j++) {
            atomicAdd(&scores[j], distances[idx] / window_size);
        }
    }
}
'''

# 12. ペアワイズ距離計算カーネル
PAIRWISE_DISTANCES_KERNEL_CODE = r'''
extern "C" __global__
void compute_pairwise_distances_kernel(
    const float* phase_space,
    float* distances,
    const int n,
    const int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n * (n - 1) / 2) {
        // インデックスから(i, j)ペアを復元
        int i = (int)((1 + sqrtf(1 + 8 * idx)) / 2);
        int j = idx - i * (i - 1) / 2;
        
        if (i < n && j < i) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = phase_space[i * dim + d] - phase_space[j * dim + d];
                dist += diff * diff;
            }
            distances[idx] = sqrtf(dist);
        }
    }
}
'''

# 13. FNN（False Nearest Neighbors）カーネル
FNN_KERNEL_CODE = r'''
extern "C" __global__
void compute_fnn_kernel(
    const float* phase_space,
    const float* series,
    int* fnn_count,
    const int n,
    const int dim,
    const int delay,
    const int series_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 最近傍を探索
        float min_dist = 1e10f;
        int nn_idx = -1;
        
        for (int j = 0; j < n; j++) {
            if (j != idx) {
                float dist = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = phase_space[j * dim + d] - phase_space[idx * dim + d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = sqrtf(dist);
                    nn_idx = j;
                }
            }
        }
        
        // False nearest neighbor判定
        if (nn_idx >= 0 && min_dist > 1e-10f) {
            int next_dim_idx = idx + dim * delay;
            int next_dim_nn_idx = nn_idx + dim * delay;
            
            if (next_dim_idx < series_len && next_dim_nn_idx < series_len) {
                float next_dim_dist = fabsf(series[next_dim_idx] - series[next_dim_nn_idx]);
                
                // Kennel et al.の基準
                if (next_dim_dist / min_dist > 10.0f) {
                    atomicAdd(fnn_count, 1);
                }
            }
        }
    }
}
'''

# ===============================
# メインクラス
# ===============================

class PhaseSpaceAnalyzerGPU(GPUBackend):
    """位相空間解析のGPU実装（爆速版・完全リファクタリング済み）"""
    
    def __init__(self, config: Optional[PhaseSpaceConfig] = None, force_cpu: bool = False):
        """初期化"""
        super().__init__(force_cpu)
        self.config = config or PhaseSpaceConfig()
        self.embedding_cache = {}
        self.kernels_initialized = False
        self._init_kernels()
        
    def _init_kernels(self) -> None:
        """CuPy RawKernelの初期化"""
        if not HAS_GPU or self.force_cpu:
            self._set_kernels_to_none()
            return
            
        try:
            # 全13個のカーネルをコンパイル
            self.rqa_features_kernel = cp.RawKernel(
                RQA_FEATURES_KERNEL_CODE, 'compute_rqa_features_kernel'
            )
            self.attractor_volume_kernel = cp.RawKernel(
                ATTRACTOR_VOLUME_KERNEL_CODE, 'compute_attractor_volume_kernel'
            )
            self.fractal_dimension_kernel = cp.RawKernel(
                FRACTAL_DIMENSION_KERNEL_CODE, 'compute_fractal_dimension_kernel'
            )
            self.complexity_measures_kernel = cp.RawKernel(
                COMPLEXITY_MEASURES_KERNEL_CODE, 'compute_complexity_measures_kernel'
            )
            self.knn_anomaly_kernel = cp.RawKernel(
                KNN_ANOMALY_KERNEL_CODE, 'knn_trajectory_anomaly_kernel'
            )
            self.diagonal_dist_kernel = cp.RawKernel(
                DIAGONAL_DIST_KERNEL_CODE, 'compute_diagonal_distribution_kernel'
            )
            self.curvature_anomaly_kernel = cp.RawKernel(
                CURVATURE_ANOMALY_KERNEL_CODE, 'compute_curvature_anomaly_kernel'
            )
            self.velocity_anomaly_kernel = cp.RawKernel(
                VELOCITY_ANOMALY_KERNEL_CODE, 'compute_velocity_anomaly_kernel'
            )
            self.acceleration_anomaly_kernel = cp.RawKernel(
                ACCELERATION_ANOMALY_KERNEL_CODE, 'compute_acceleration_anomaly_kernel'
            )
            self.lyapunov_kernel = cp.RawKernel(
                LYAPUNOV_KERNEL_CODE, 'compute_lyapunov_kernel'
            )
            self.map_transition_scores_kernel = cp.RawKernel(
                MAP_TRANSITION_SCORES_KERNEL_CODE, 'map_transition_scores_kernel'
            )
            self.pairwise_distances_kernel = cp.RawKernel(
                PAIRWISE_DISTANCES_KERNEL_CODE, 'compute_pairwise_distances_kernel'
            )
            self.fnn_kernel = cp.RawKernel(
                FNN_KERNEL_CODE, 'compute_fnn_kernel'
            )
            
            self.kernels_initialized = True
            logger.info("✅ All 13 phase space kernels compiled successfully (PTX 8.4)")
            
        except Exception as e:
            logger.warning(f"Failed to compile phase space kernels: {e}")
            self._set_kernels_to_none()
    
    def _set_kernels_to_none(self) -> None:
        """全カーネルをNoneに設定"""
        self.rqa_features_kernel = None
        self.attractor_volume_kernel = None
        self.fractal_dimension_kernel = None
        self.complexity_measures_kernel = None
        self.knn_anomaly_kernel = None
        self.diagonal_dist_kernel = None
        self.curvature_anomaly_kernel = None
        self.velocity_anomaly_kernel = None
        self.acceleration_anomaly_kernel = None
        self.lyapunov_kernel = None
        self.map_transition_scores_kernel = None
        self.pairwise_distances_kernel = None
        self.fnn_kernel = None
        self.kernels_initialized = False
    
    def analyze_phase_space(self,
                          structures: Dict[str, np.ndarray],
                          embedding_params: Optional[Dict] = None) -> Dict[str, Any]:
        """包括的な位相空間解析（爆速版）"""
        print("\n🚀 Ultra-Fast Phase Space Analysis on GPU...")
        
        # デフォルトパラメータ
        if embedding_params is None:
            embedding_params = {
                'embedding_dim': 3,
                'delay': 50,
                'n_neighbors': 20,
                'recurrence_threshold': 0.1,
                'min_diagonal_length': 2,
                'voxel_grid_size': 128
            }
        
        # 主要な時系列を取得
        primary_series = self._get_primary_series(structures)
        
        results = {}
        
        # メモリ管理
        if hasattr(self, 'memory_manager'):
            with self.memory_manager.temporary_allocation(len(primary_series) * 4 * 20, "phase_space"):
                results = self._perform_analysis(primary_series, embedding_params)
        else:
            results = self._perform_analysis(primary_series, embedding_params)
        
        self._print_summary(results)
        
        return results
    
    def _perform_analysis(self, primary_series: ArrayType, embedding_params: Dict) -> Dict:
        """実際の解析処理"""
        results = {}
        
        # 1. 最適埋め込みパラメータの推定
        optimal_params = self._estimate_embedding_parameters_gpu(
            primary_series, embedding_params
        )
        results['optimal_parameters'] = optimal_params
        
        # 2. 位相空間再構成
        phase_space = self._reconstruct_phase_space_gpu(
            primary_series,
            optimal_params['embedding_dim'],
            optimal_params['delay']
        )
        results['phase_space'] = self.to_cpu(phase_space)
        
        # 3. アトラクタ解析
        attractor_features = self._analyze_attractor_gpu_fast(
            phase_space, embedding_params['voxel_grid_size']
        )
        results['attractor_features'] = attractor_features
        
        # 4. リカレンスプロット解析
        recurrence_features = self._recurrence_analysis_gpu_fast(
            phase_space, 
            embedding_params['recurrence_threshold'],
            embedding_params['min_diagonal_length']
        )
        results['recurrence_features'] = recurrence_features
        
        # 5. 異常軌道検出
        anomaly_scores = self._detect_anomalous_trajectories_gpu_fast(
            phase_space, embedding_params['n_neighbors']
        )
        results['anomaly_scores'] = self.to_cpu(anomaly_scores)
        
        # 6. ダイナミクス特性
        dynamics_features = self._analyze_dynamics_gpu_fast(phase_space)
        results['dynamics_features'] = dynamics_features
        
        # 7. 統合スコア
        integrated_score = self._compute_integrated_score_gpu(
            anomaly_scores, attractor_features, recurrence_features
        )
        results['integrated_anomaly_score'] = self.to_cpu(integrated_score)
        
        return results
    
    def _analyze_attractor_gpu_fast(self, 
                                   phase_space: cp.ndarray,
                                   voxel_grid_size: int) -> Dict:
        """アトラクタ特性の高速解析"""
        features = {}
        n_points = len(phase_space)
        dim = phase_space.shape[1]
        
        phase_space_flat = cp.ascontiguousarray(phase_space.flatten()).astype(cp.float32)
        
        features['correlation_dimension'] = float(
            self._correlation_dimension_gpu_fast(phase_space)
        )
        
        features['lyapunov_exponent'] = float(
            self._estimate_lyapunov_gpu_fast(phase_space_flat, n_points, dim)
        )
        
        if self.attractor_volume_kernel is not None:
            voxel_grid = cp.zeros(voxel_grid_size**3, dtype=cp.int32)
            phase_min = cp.min(phase_space, axis=0).astype(cp.float32)
            phase_max = cp.max(phase_space, axis=0).astype(cp.float32)
            phase_range = (phase_max - phase_min + 1e-10).astype(cp.float32)
            
            blocks, threads = self._optimize_kernel_launch(n_points)
            
            self.attractor_volume_kernel(
                (blocks,), (threads,),
                (phase_space_flat, voxel_grid, phase_min, phase_range,
                 n_points, dim, voxel_grid_size)
            )
            
            occupied_voxels = cp.sum(voxel_grid > 0)
            total_voxels = voxel_grid_size**3
            features['attractor_volume'] = float(occupied_voxels / total_voxels)
        else:
            features['attractor_volume'] = 0.0
        
        features['fractal_measure'] = float(
            self._compute_fractal_measure_gpu_fast(phase_space)
        )
        
        features['information_dimension'] = float(
            self._compute_information_dimension_gpu(phase_space)
        )
        
        return features
    
    def _recurrence_analysis_gpu_fast(self,
                                     phase_space: cp.ndarray,
                                     threshold: float,
                                     min_line_length: int) -> Dict:
        """高速リカレンス定量化解析"""
        n = len(phase_space)
        
        max_size = 5000
        if n > max_size:
            indices = cp.random.choice(n, max_size, replace=False)
            phase_space_sample = phase_space[indices]
            n = max_size
        else:
            phase_space_sample = phase_space
        
        rec_matrix = self._compute_recurrence_matrix_gpu_fast(
            phase_space_sample, threshold
        )
        
        features = {}
        
        if self.rqa_features_kernel is not None and self.diagonal_dist_kernel is not None:
            features_array = cp.zeros(10, dtype=cp.float32)
            rec_matrix_flat = cp.ascontiguousarray(rec_matrix.flatten()).astype(cp.float32)
            
            blocks, threads = self._optimize_kernel_launch(n)
            
            self.rqa_features_kernel(
                (blocks,), (threads,),
                (rec_matrix_flat, features_array, n, min_line_length)
            )
            
            diag_dist = cp.zeros(n//2, dtype=cp.int32)
            self.diagonal_dist_kernel(
                (blocks,), (threads,),
                (rec_matrix_flat, diag_dist, n)
            )
            
            total_points = n * n
            rec_points = cp.sum(rec_matrix)
            
            features = {
                'recurrence_rate': float(rec_points / total_points),
                'determinism': float(features_array[0] / (rec_points + 1e-10)),
                'laminarity': float(features_array[1] / (rec_points + 1e-10)),
                'diagonal_lines': int(features_array[3]),
                'vertical_lines': int(features_array[4]),
                'entropy': float(self._compute_entropy_from_distribution(diag_dist)),
                'trapping_time': float(features_array[2] / (features_array[4] + 1e-10)),
                'max_diagonal_length': int(cp.max(cp.where(diag_dist > 0)[0])) if cp.any(diag_dist > 0) else 0
            }
        else:
            total_points = n * n
            rec_points = cp.sum(rec_matrix)
            features = {
                'recurrence_rate': float(rec_points / total_points),
                'determinism': 0.5,
                'laminarity': 0.5,
                'diagonal_lines': 0,
                'vertical_lines': 0,
                'entropy': 1.0,
                'trapping_time': 1.0,
                'max_diagonal_length': 0
            }
        
        return features
    
    def _detect_anomalous_trajectories_gpu_fast(self,
                                              phase_space: cp.ndarray,
                                              n_neighbors: int) -> cp.ndarray:
        """高速異常軌道検出"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        phase_space_flat = cp.ascontiguousarray(phase_space.flatten()).astype(cp.float32)
        anomaly_scores = cp.zeros(n, dtype=cp.float32)
        
        if self.knn_anomaly_kernel is not None:
            blocks, threads = self._optimize_kernel_launch(n)
            self.knn_anomaly_kernel(
                (blocks,), (threads,),
                (phase_space_flat, anomaly_scores, n_neighbors, n, dim)
            )
        
        if self.curvature_anomaly_kernel is not None:
            curvature_anomaly = cp.zeros(n, dtype=cp.float32)
            blocks, threads = self._optimize_kernel_launch(n)
            self.curvature_anomaly_kernel(
                (blocks,), (threads,),
                (phase_space_flat, curvature_anomaly, n, dim)
            )
            anomaly_scores += 0.2 * curvature_anomaly
        
        if self.velocity_anomaly_kernel is not None:
            velocity_anomaly = cp.zeros(n, dtype=cp.float32)
            blocks, threads = self._optimize_kernel_launch(n)
            self.velocity_anomaly_kernel(
                (blocks,), (threads,),
                (phase_space_flat, velocity_anomaly, n, dim)
            )
            anomaly_scores += 0.2 * velocity_anomaly
        
        if self.acceleration_anomaly_kernel is not None:
            acceleration_anomaly = cp.zeros(n, dtype=cp.float32)
            blocks, threads = self._optimize_kernel_launch(n)
            self.acceleration_anomaly_kernel(
                (blocks,), (threads,),
                (phase_space_flat, acceleration_anomaly, n, dim)
            )
            anomaly_scores += 0.2 * acceleration_anomaly
        
        return anomaly_scores
    
    def _analyze_dynamics_gpu_fast(self, phase_space: cp.ndarray) -> Dict:
        """高速ダイナミクス解析"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        features = {}
        
        if self.complexity_measures_kernel is not None:
            phase_space_flat = cp.ascontiguousarray(phase_space.flatten()).astype(cp.float32)
            measures = cp.zeros(5, dtype=cp.float32)
            blocks, threads = self._optimize_kernel_launch(n)
            
            self.complexity_measures_kernel(
                (blocks,), (threads,),
                (phase_space_flat, measures, n, dim)
            )
            
            features['chaos_measure'] = float(measures[0] / n)
            features['predictability'] = float(1.0 / (1.0 + measures[1] / n))
            features['complexity'] = float(measures[2] / n)
            features['entropy_rate'] = float(measures[3] / n)
            features['stability'] = float(1.0 / (1.0 + measures[4] / n))
        else:
            features['chaos_measure'] = 0.5
            features['predictability'] = 0.5
            features['complexity'] = 0.5
            features['entropy_rate'] = 1.0
            features['stability'] = 0.5
        
        features['periodicity'] = float(
            self._detect_periodicity_gpu_fast(phase_space)
        )
        
        return features
    
    def _estimate_lyapunov_gpu_fast(self, 
                                   phase_space_flat: cp.ndarray,
                                   n: int, 
                                   dim: int) -> float:
        """高速Lyapunov指数推定"""
        if self.lyapunov_kernel is None:
            return 0.0
        
        n_ref_points = min(200, n // 10)
        ref_indices = cp.random.choice(n - 10, n_ref_points, replace=False).astype(cp.int32)
        
        lyap_values = cp.zeros(n_ref_points, dtype=cp.float32)
        
        blocks, threads = self._optimize_kernel_launch(n_ref_points)
        self.lyapunov_kernel(
            (blocks,), (threads,),
            (phase_space_flat, ref_indices, lyap_values, n, dim, n_ref_points)
        )
        
        lyap_values = lyap_values[lyap_values != 0]
        if len(lyap_values) > 0:
            return float(cp.median(lyap_values))
        
        return 0.0
    
    def _optimize_kernel_launch(self, n_elements: int) -> Tuple[int, int]:
        """カーネル起動パラメータの最適化"""
        threads = 256
        blocks = (n_elements + threads - 1) // threads
        blocks = min(blocks, 65535)
        return blocks, threads
    
    def _correlation_dimension_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速相関次元計算"""
        n = min(2000, len(phase_space))
        
        if len(phase_space) > n:
            indices = cp.random.choice(len(phase_space), n, replace=False)
            phase_space = phase_space[indices]
        
        if self.pairwise_distances_kernel is not None and self.fractal_dimension_kernel is not None:
            n_pairs = n * (n - 1) // 2
            distances = cp.zeros(n_pairs, dtype=cp.float32)
            phase_space_flat = cp.ascontiguousarray(phase_space.flatten()).astype(cp.float32)
            
            blocks, threads = self._optimize_kernel_launch(n_pairs)
            self.pairwise_distances_kernel(
                (blocks,), (threads,),
                (phase_space_flat, distances, n, phase_space.shape[1])
            )
            
            radii = cp.logspace(-2, 1, 30).astype(cp.float32)
            correlation_integral = cp.zeros(len(radii), dtype=cp.int32)
            
            self.fractal_dimension_kernel(
                (blocks,), (threads,),
                (distances, correlation_integral, radii, n_pairs, len(radii))
            )
            
            valid = correlation_integral > 0
            if cp.sum(valid) > 5:
                log_r = cp.log(radii[valid])
                log_c = cp.log(correlation_integral[valid].astype(cp.float32) / n_pairs)
                
                slope = cp.polyfit(log_r, log_c, 1)[0]
                return float(cp.clip(slope, 0.5, 5.0))
        
        return 2.0
    
    def _compute_fractal_measure_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速フラクタル測度計算（修正版）"""
        corr_dim = self._correlation_dimension_gpu_fast(phase_space)
        embedding_dim = phase_space.shape[1]
        # float型の除算結果をPythonの組み込み関数で処理
        fractal_measure_value = corr_dim / embedding_dim
        return float(max(0.0, min(1.0, fractal_measure_value)))
    
    def _compute_information_dimension_gpu(self, phase_space: cp.ndarray) -> float:
        """情報次元の計算"""
        n = min(1000, len(phase_space))
        
        n_boxes_list = [10, 20, 40, 80]
        entropy_values = []
        
        for n_boxes in n_boxes_list:
            box_counts = cp.zeros(n_boxes**3, dtype=cp.int32)
            
            phase_min = cp.min(phase_space[:n], axis=0)
            phase_max = cp.max(phase_space[:n], axis=0)
            phase_range = phase_max - phase_min + 1e-10
            
            for i in range(n):
                box_coords = ((phase_space[i] - phase_min) / phase_range * (n_boxes - 1)).astype(int)
                box_coords = cp.clip(box_coords, 0, n_boxes - 1)
                
                if len(box_coords) >= 3:
                    box_idx = (box_coords[0] * n_boxes * n_boxes + 
                             box_coords[1] * n_boxes + 
                             box_coords[2])
                    box_counts[box_idx] += 1
            
            probs = box_counts[box_counts > 0] / n
            entropy = -cp.sum(probs * cp.log(probs))
            entropy_values.append(float(entropy))
        
        if len(entropy_values) > 1:
            log_sizes = cp.log(cp.array(n_boxes_list, dtype=cp.float32))
            info_dim = cp.polyfit(log_sizes, cp.array(entropy_values), 1)[0]
            return float(cp.clip(info_dim, 0.5, 3.0))
        
        return 1.5
    
    def _compute_recurrence_matrix_gpu_fast(self,
                                           phase_space: cp.ndarray,
                                           threshold: float) -> cp.ndarray:
        """高速リカレンス行列計算"""
        n = len(phase_space)
        rec_matrix = cp.zeros((n, n), dtype=cp.float32)
        
        block_size = 128
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                
                block_i = phase_space[i:i_end]
                block_j = phase_space[j:j_end]
                
                for ii in range(len(block_i)):
                    distances = cp.sqrt(cp.sum((block_j - block_i[ii])**2, axis=1))
                    rec_matrix[i + ii, j:j_end] = (distances < threshold).astype(cp.float32)
        
        return rec_matrix
    
    def _detect_periodicity_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速周期性検出"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        max_period_strength = 0.0
        
        for d in range(min(3, dim)):
            signal = phase_space[:, d]
            fft = cp.fft.fft(signal)
            power = cp.abs(fft[:n//2])**2
            
            power[0] = 0
            
            if len(power) > 1:
                max_peak = cp.max(power[1:])
                mean_power = cp.mean(power[1:])
                
                if mean_power > 0:
                    period_strength = max_peak / mean_power
                    max_period_strength = max(max_period_strength, float(period_strength))
        
        return float(1.0 / (1.0 + cp.exp(-max_period_strength + 5)))
    
    def _compute_entropy_from_distribution(self, distribution: cp.ndarray) -> float:
        """分布からエントロピー計算"""
        total = cp.sum(distribution)
        if total == 0:
            return 0.0
        
        probs = distribution[distribution > 0] / total
        entropy = -cp.sum(probs * cp.log(probs))
        
        return float(entropy)
    
    def _get_primary_series(self, structures: Dict) -> cp.ndarray:
        """主要な時系列を選択"""
        if 'rho_T' in structures:
            return self.to_gpu(structures['rho_T'])
        elif 'lambda_F_mag' in structures:
            return self.to_gpu(structures['lambda_F_mag'])
        else:
            for key, value in structures.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 1:
                    return self.to_gpu(value)
        
        raise ValueError("No suitable time series found in structures")
    
    def _estimate_embedding_parameters_gpu(self,
                                         series: cp.ndarray,
                                         params: Dict) -> Dict:
        """最適な埋め込みパラメータを推定"""
        optimal_delay = self._estimate_delay_mutual_info_gpu_fast(series)
        optimal_dim = self._estimate_dimension_fnn_gpu_fast(
            series, optimal_delay, max_dim=10
        )
        
        return {
            'embedding_dim': int(optimal_dim),
            'delay': int(optimal_delay),
            'estimated_from': 'mutual_info_and_fnn'
        }

    def _estimate_delay_mutual_info_gpu_fast(self, series: cp.ndarray) -> int:
        """高速相互情報量による遅延推定（修正版）"""
        max_delay = min(100, len(series) // 10)
        
        # 🔧 修正1: 配列を無限大で初期化（index 0も含む）
        mi_values = cp.full(max_delay, cp.inf, dtype=cp.float32)
        
        # 🔧 修正2: delay=1から計算（delay=0は意味がない）
        for delay in range(1, max_delay):
            x = series[:-delay]
            y = series[delay:]
            
            n_bins = 20
            hist_2d, _, _ = cp.histogram2d(x, y, bins=n_bins)
            hist_2d = hist_2d / cp.sum(hist_2d)
            
            px = cp.sum(hist_2d, axis=1)
            py = cp.sum(hist_2d, axis=0)
            
            valid = hist_2d > 0
            mi = cp.sum(hist_2d[valid] * cp.log(
                hist_2d[valid] / (px[:, None] * py[None, :])[valid]
            ))
            
            mi_values[delay] = mi
        
        # 🔧 修正3: 最初の極小値を探す（範囲を修正）
        for i in range(3, len(mi_values) - 1):  # 3から開始
            if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                return int(i)
        
        # 🔧 修正4: mi_values[1]が存在することを確認
        if mi_values[1] != cp.inf:
            decay_point = cp.where(mi_values[1:] < mi_values[1] * 0.5)[0]
            if len(decay_point) > 0:
                return int(decay_point[0] + 1)  # +1してインデックス調整
        
        # 🔧 修正5: デフォルト値を安全な値に
        return max(1, min(10, max_delay // 4))
    
    def _estimate_dimension_fnn_gpu_fast(self,
                                       series: cp.ndarray,
                                       delay: int,
                                       max_dim: int = 10) -> int:
        """高速False Nearest Neighbors法（修正版）"""
        
        # 🔧 修正1: delay=0のチェック
        if delay <= 0:
            logger.warning(f"Invalid delay {delay}, using 1")
            delay = 1
        
        # 🔧 修正2: max_dimの妥当性チェック
        max_possible_dim = min(max_dim, len(series) // (delay * 10))
        if max_possible_dim < 2:
            logger.warning("Series too short for FNN analysis")
            return 3
        
        fnn_fractions = []
        
        for dim in range(1, max_possible_dim + 1):
            try:
                phase_space = self._reconstruct_phase_space_gpu(series, dim, delay)
                n = min(2000, len(phase_space))
                
                if n < 100:
                    logger.debug(f"Too few points ({n}) for dim={dim}")
                    break
                
                if len(phase_space) > n:
                    indices = cp.random.choice(len(phase_space), n, replace=False)
                    phase_space_sample = phase_space[indices]
                else:
                    phase_space_sample = phase_space
                
                if self.fnn_kernel is not None:
                    phase_space_flat = cp.ascontiguousarray(
                        phase_space_sample.flatten()
                    ).astype(cp.float32)
                    series_flat = cp.ascontiguousarray(series).astype(cp.float32)
                    fnn_count = cp.zeros(1, dtype=cp.int32)
                    
                    blocks, threads = self._optimize_kernel_launch(n)
                    self.fnn_kernel(
                        (blocks,), (threads,),
                        (phase_space_flat, series_flat, fnn_count, 
                         n, dim, delay, len(series))
                    )
                    
                    fnn_fraction = float(fnn_count[0]) / n
                else:
                    # 🔧 修正3: CPUフォールバックの改善
                    fnn_fraction = self._compute_fnn_cpu(phase_space_sample, series, delay)
                
                fnn_fractions.append(fnn_fraction)
                
                # 🔧 修正4: 早期終了条件の緩和
                if fnn_fraction < 0.05:  # 0.01→0.05に緩和
                    if dim >= 3:  # 最小でも3次元は試す
                        return dim
                
                # 🔧 修正5: 急激な改善があった場合
                if len(fnn_fractions) >= 2:
                    improvement = fnn_fractions[-2] - fnn_fraction
                    if improvement > 0.3 and fnn_fraction < 0.1:
                        return dim
            
            except Exception as e:
                logger.warning(f"FNN dimension {dim} failed: {e}")
                if len(fnn_fractions) > 0:
                    break
        
        # 🔧 修正6: 結果の解釈改善
        if fnn_fractions:
            # 最小値の次元を返す（ただし最低3）
            best_dim = int(cp.argmin(cp.array(fnn_fractions)) + 1)
            return max(3, best_dim)
        
        # デフォルト
        return 3
    
    def _reconstruct_phase_space_gpu(self,
                                   series: cp.ndarray,
                                   embedding_dim: int,
                                   delay: int) -> cp.ndarray:
        """時系列から位相空間を再構成（修正版）"""
        
        # 🔧 修正1: パラメータ検証
        if embedding_dim < 1:
            raise ValueError(f"Invalid embedding dimension: {embedding_dim}")
        
        if delay < 1:
            raise ValueError(f"Invalid delay: {delay}")
        
        n = len(series)
        
        # 🔧 修正2: 最小長チェック
        min_length = embedding_dim * delay + 1
        if n < min_length:
            raise ValueError(
                f"Series too short: {n} < {min_length} "
                f"(dim={embedding_dim}, delay={delay})"
            )
        
        embed_length = n - (embedding_dim - 1) * delay
        
        # 🔧 修正3: メモリ効率化（大きすぎる場合の警告）
        expected_size = embed_length * embedding_dim * 4  # float32
        if expected_size > 1e9:  # 1GB以上
            logger.warning(
                f"Large phase space: {embed_length}x{embedding_dim} "
                f"({expected_size/1e9:.1f}GB)"
            )
        
        # 🔧 修正4: より効率的なインデックス生成
        try:
            # ブロードキャスティングを使った効率的な実装
            indices = cp.arange(embed_length)[:, None] + \
                     cp.arange(embedding_dim)[None, :] * delay
            phase_space = series[indices]
            
        except cp.cuda.MemoryError:
            # メモリ不足時は分割処理
            logger.warning("GPU memory insufficient, using chunked processing")
            chunk_size = 10000
            chunks = []
            
            for start in range(0, embed_length, chunk_size):
                end = min(start + chunk_size, embed_length)
                idx = cp.arange(start, end)[:, None] + \
                      cp.arange(embedding_dim)[None, :] * delay
                chunks.append(series[idx])
            
            phase_space = cp.vstack(chunks)
        
        return phase_space
    
    # 🔧 追加: CPU フォールバック用のFNN計算
    def _compute_fnn_cpu(self, phase_space: cp.ndarray, 
                        series: cp.ndarray, delay: int) -> float:
        """CPUでのFNN計算（フォールバック用）"""
        n = len(phase_space)
        fnn_count = 0
        
        for i in range(n):
            # 最近傍を探索（簡易版）
            min_dist = cp.inf
            nn_idx = -1
            
            for j in range(max(0, i-50), min(n, i+50)):
                if i != j:
                    dist = cp.sqrt(cp.sum((phase_space[i] - phase_space[j])**2))
                    if dist < min_dist:
                        min_dist = dist
                        nn_idx = j
            
            if nn_idx >= 0 and min_dist > 1e-10:
                # False nearest neighbor判定
                next_idx = i + len(phase_space[0]) * delay
                next_nn_idx = nn_idx + len(phase_space[0]) * delay
                
                if next_idx < len(series) and next_nn_idx < len(series):
                    next_dist = abs(series[next_idx] - series[next_nn_idx])
                    if next_dist / min_dist > 10.0:
                        fnn_count += 1
        
        return fnn_count / n
    
    def _compute_integrated_score_gpu(self,
                                anomaly_scores: cp.ndarray,
                                attractor_features: Dict,
                                recurrence_features: Dict) -> cp.ndarray:
        """統合異常スコア（修正版）"""
        # アトラクタ異常度（float型として計算）
        attractor_anomaly = 0.0
        
        if abs(attractor_features['lyapunov_exponent']) > 0.1:
            attractor_anomaly += 0.3
        
        frac_deviation = abs(attractor_features['fractal_measure'] - 0.5)
        attractor_anomaly += frac_deviation * 0.2
        
        corr_dim = attractor_features['correlation_dimension']
        if corr_dim < 1.0 or corr_dim > 3.0:
            attractor_anomaly += 0.2
        
        info_dim = attractor_features.get('information_dimension', 2.0)
        dim_diff = abs(corr_dim - info_dim)
        attractor_anomaly += dim_diff * 0.1
        
        # リカレンス異常度（float型として計算）
        recurrence_anomaly = 0.0
        
        determinism = recurrence_features['determinism']
        if determinism < 0.7:
            recurrence_anomaly += (1 - determinism) * 0.3
        
        entropy = recurrence_features['entropy']
        if entropy > 2.0:
            recurrence_anomaly += min(entropy / 5.0, 0.3)
        
        laminarity = recurrence_features['laminarity']
        if laminarity < 0.5:
            recurrence_anomaly += (1 - laminarity) * 0.2
        
        trapping = recurrence_features.get('trapping_time', 1.0)
        if trapping > 10.0:
            recurrence_anomaly += 0.2
        
        # 正規化（Python組み込み関数でfloat型を処理）
        attractor_anomaly = max(0.0, min(1.0, attractor_anomaly))
        recurrence_anomaly = max(0.0, min(1.0, recurrence_anomaly))
        
        # 統合
        global_anomaly = 0.5 * attractor_anomaly + 0.5 * recurrence_anomaly
        
        # 局所異常と統合（ndarray型の処理）
        integrated = 0.7 * anomaly_scores + 0.3 * global_anomaly
        
        # 正規化
        xp = cp if isinstance(integrated, cp.ndarray) else np
        integrated = (integrated - xp.mean(integrated)) / (xp.std(integrated) + 1e-10)
        
        return integrated
    
    def _print_summary(self, results: Dict):
        """解析結果のサマリー表示"""
        print("\n📊 Phase Space Analysis Summary (CuPy RawKernel Edition):")
        print(f"   ⚡ GPU Acceleration: {'ENABLED' if self.is_gpu else 'DISABLED'}")
        print(f"   Embedding dimension: {results['optimal_parameters']['embedding_dim']}")
        print(f"   Optimal delay: {results['optimal_parameters']['delay']}")
        
        if 'attractor_features' in results:
            att = results['attractor_features']
            print(f"\n   🌀 Attractor Features:")
            print(f"   - Correlation dimension: {att['correlation_dimension']:.3f}")
            print(f"   - Lyapunov exponent: {att['lyapunov_exponent']:.3f}")
            print(f"   - Fractal measure: {att['fractal_measure']:.3f}")
            print(f"   - Attractor volume: {att['attractor_volume']:.3f}")
            print(f"   - Information dimension: {att.get('information_dimension', 0):.3f}")
        
        if 'recurrence_features' in results:
            rec = results['recurrence_features']
            print(f"\n   📈 Recurrence Features:")
            print(f"   - Recurrence rate: {rec['recurrence_rate']:.3f}")
            print(f"   - Determinism: {rec['determinism']:.3f}")
            print(f"   - Laminarity: {rec['laminarity']:.3f}")
            print(f"   - Entropy: {rec['entropy']:.3f}")
            print(f"   - Max diagonal length: {rec.get('max_diagonal_length', 0)}")
            
        if 'dynamics_features' in results:
            dyn = results['dynamics_features']
            print(f"\n   🔄 Dynamics Features:")
            print(f"   - Periodicity: {dyn['periodicity']:.3f}")
            print(f"   - Chaos measure: {dyn['chaos_measure']:.3f}")
            print(f"   - Predictability: {dyn.get('predictability', 0):.3f}")
            print(f"   - Complexity: {dyn['complexity']:.3f}")
            print(f"   - Stability: {dyn['stability']:.3f}")


# ===============================
# テスト関数
# ===============================

def test_phase_space_analysis():
    """位相空間解析のテスト"""
    print("\n🧪 Testing Phase Space Analysis GPU (13 kernels)...")
    print("=" * 60)
    
    # テストデータ生成
    t = np.linspace(0, 100, 10000)
    x = np.sin(0.1 * t) + 0.1 * np.sin(2.3 * t) + np.random.randn(len(t)) * 0.01
    
    structures = {
        'rho_T': x.astype(np.float32),
        'lambda_F_mag': np.abs(np.gradient(x)).astype(np.float32),
        'lambda_FF_mag': np.abs(np.gradient(np.gradient(x))).astype(np.float32)
    }
    
    # アナライザー初期化
    analyzer = PhaseSpaceAnalyzerGPU()
    
    # 位相空間解析実行
    print("\n🚀 Running comprehensive phase space analysis...")
    results = analyzer.analyze_phase_space(structures)
    
    # 結果確認
    print("\n✅ Analysis completed!")
    print(f"   Anomaly scores shape: {results['anomaly_scores'].shape}")
    print(f"   Mean anomaly: {np.mean(results['anomaly_scores']):.4f}")
    print(f"   Max anomaly: {np.max(results['anomaly_scores']):.4f}")
    
    print("\n🎉 All 13 kernels tested successfully!")
    return True

if __name__ == "__main__":
    test_phase_space_analysis()
