"""
Lambda³ GPU版位相空間解析モジュール（爆速カーネル版）
高次元位相空間での異常検出とアトラクタ解析
CUDAカーネルによる超高速化実装
"""

import numpy as np
import cupy as cp
from typing import Dict, List, Tuple, Optional, Any
from numba import cuda
import math

from ..types import ArrayType, NDArray
from ..core.gpu_utils import GPUBackend

class PhaseSpaceAnalyzerGPU(GPUBackend):
    """位相空間解析のGPU実装（爆速版）"""
    
    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.embedding_cache = {}
        self._init_kernels()
        
    def _init_kernels(self):
        """CUDAカーネルの初期化"""
        # RQA特徴量計算カーネル
        self._rqa_features_kernel = cuda.jit(self._compute_rqa_features_kernel_impl)
        # アトラクタ体積計算カーネル
        self._attractor_volume_kernel = cuda.jit(self._compute_attractor_volume_kernel_impl)
        # フラクタル次元計算カーネル
        self._fractal_kernel = cuda.jit(self._compute_fractal_dimension_kernel_impl)
        # 複雑性指標カーネル
        self._complexity_kernel = cuda.jit(self._compute_complexity_measures_kernel_impl)
        # k-NN異常検出カーネル
        self._knn_anomaly_kernel = cuda.jit(self._knn_trajectory_anomaly_kernel_impl)
        # 対角線長分布カーネル
        self._diagonal_dist_kernel = cuda.jit(self._compute_diagonal_distribution_kernel_impl)
        
    def analyze_phase_space(self,
                          structures: Dict[str, np.ndarray],
                          embedding_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        包括的な位相空間解析（爆速版）
        
        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Lambda構造体
        embedding_params : Dict, optional
            埋め込みパラメータ
            
        Returns
        -------
        Dict[str, Any]
            位相空間解析結果
        """
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
        
        with self.memory_manager.batch_context(len(primary_series) * 10):
            # 1. 最適埋め込みパラメータの推定（GPU並列化）
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
            
            # 3. アトラクタ解析（爆速カーネル版）
            attractor_features = self._analyze_attractor_gpu_fast(
                phase_space, embedding_params['voxel_grid_size']
            )
            results['attractor_features'] = attractor_features
            
            # 4. リカレンスプロット解析（並列RQA）
            recurrence_features = self._recurrence_analysis_gpu_fast(
                phase_space, 
                embedding_params['recurrence_threshold'],
                embedding_params['min_diagonal_length']
            )
            results['recurrence_features'] = recurrence_features
            
            # 5. 異常軌道検出（並列k-NN）
            anomaly_scores = self._detect_anomalous_trajectories_gpu_fast(
                phase_space, embedding_params['n_neighbors']
            )
            results['anomaly_scores'] = self.to_cpu(anomaly_scores)
            
            # 6. ダイナミクス特性（並列解析）
            dynamics_features = self._analyze_dynamics_gpu_fast(phase_space)
            results['dynamics_features'] = dynamics_features
            
            # 7. 統合スコア
            integrated_score = self._compute_integrated_score_gpu(
                anomaly_scores, attractor_features, recurrence_features
            )
            results['integrated_anomaly_score'] = self.to_cpu(integrated_score)
        
        self._print_summary(results)
        
        return results
    
    def detect_phase_transitions(self,
                               structures: Dict[str, np.ndarray],
                               window_size: int = 1000,
                               stride: int = 100) -> Dict[str, cp.ndarray]:
        """
        位相空間での遷移検出（並列スライディングウィンドウ）
        """
        print("\n⚡ Detecting phase transitions with GPU acceleration...")
        
        primary_series = self._get_primary_series(structures)
        n_frames = len(primary_series)
        n_windows = (n_frames - window_size) // stride + 1
        
        transition_scores = cp.zeros(n_frames)
        phase_distances = []
        
        # 並列ウィンドウ処理の準備
        all_features = []
        
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            
            if end > n_frames:
                break
            
            # ウィンドウ内の位相空間
            window_series = primary_series[start:end]
            phase_space = self._reconstruct_phase_space_gpu(
                window_series, embedding_dim=3, delay=20
            )
            
            # 位相空間の特徴抽出（並列化）
            features = self._extract_phase_features_gpu_fast(phase_space)
            all_features.append(features)
        
        # 並列で距離計算
        if len(all_features) > 1:
            distances = self._compute_phase_distances_parallel(all_features)
            phase_distances = self.to_cpu(distances).tolist()
            
            # スコアマッピング（並列化）
            self._map_transition_scores_kernel(transition_scores, distances, window_size, stride)
        
        # スムージングと正規化
        transition_scores = self._smooth_and_normalize_gpu(transition_scores)
        
        return {
            'transition_scores': transition_scores,
            'phase_distances': phase_distances,
            'window_size': window_size,
            'stride': stride
        }
    
    # === 爆速実装メソッド ===
    
    def _analyze_attractor_gpu_fast(self, 
                                   phase_space: cp.ndarray,
                                   voxel_grid_size: int) -> Dict:
        """アトラクタ特性の高速解析"""
        features = {}
        n_points = len(phase_space)
        dim = phase_space.shape[1]
        
        # 1. 相関次元（並列計算）
        features['correlation_dimension'] = float(
            self._correlation_dimension_gpu_fast(phase_space)
        )
        
        # 2. Lyapunov指数（並列近傍探索）
        features['lyapunov_exponent'] = float(
            self._estimate_lyapunov_gpu_fast(phase_space)
        )
        
        # 3. アトラクタ体積（ボクセル化）
        voxel_grid = cp.zeros(voxel_grid_size**3, dtype=cp.int32)
        blocks, threads = self._optimize_kernel_launch(n_points)
        
        # 位相空間の範囲を計算
        phase_min = cp.min(phase_space, axis=0)
        phase_max = cp.max(phase_space, axis=0)
        phase_range = phase_max - phase_min + 1e-10
        
        self._attractor_volume_kernel[blocks, threads](
            phase_space, voxel_grid, phase_min, phase_range,
            n_points, dim, voxel_grid_size
        )
        
        occupied_voxels = cp.sum(voxel_grid > 0)
        total_voxels = voxel_grid_size**3
        features['attractor_volume'] = float(occupied_voxels / total_voxels)
        
        # 4. フラクタル測度（並列相関積分）
        features['fractal_measure'] = float(
            self._compute_fractal_measure_gpu_fast(phase_space)
        )
        
        # 5. 情報次元
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
        
        # メモリ効率のためサンプリング
        max_size = 5000
        if n > max_size:
            indices = cp.random.choice(n, max_size, replace=False)
            phase_space_sample = phase_space[indices]
            n = max_size
        else:
            phase_space_sample = phase_space
        
        # リカレンス行列（並列距離計算）
        rec_matrix = self._compute_recurrence_matrix_gpu_fast(
            phase_space_sample, threshold
        )
        
        # RQA特徴量を並列計算
        features_array = cp.zeros(10, dtype=cp.float32)
        blocks, threads = self._optimize_kernel_launch(n)
        
        self._rqa_features_kernel[blocks, threads](
            rec_matrix, features_array, n, min_line_length
        )
        
        # 対角線長分布の解析
        diag_dist = cp.zeros(n//2, dtype=cp.int32)
        self._diagonal_dist_kernel[blocks, threads](
            rec_matrix, diag_dist, n
        )
        
        # 統計量を計算
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
            'max_diagonal_length': int(cp.max(diag_dist))
        }
        
        return features
    
    def _detect_anomalous_trajectories_gpu_fast(self,
                                              phase_space: cp.ndarray,
                                              n_neighbors: int) -> cp.ndarray:
        """高速異常軌道検出"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        # 異常スコア配列
        anomaly_scores = cp.zeros(n, dtype=cp.float32)
        
        # k-NN異常検出（並列化）
        blocks, threads = self._optimize_kernel_launch(n)
        
        self._knn_anomaly_kernel[blocks, threads](
            phase_space, anomaly_scores, n_neighbors, n, dim
        )
        
        # 曲率異常（並列計算）
        curvature_anomaly = cp.zeros(n, dtype=cp.float32)
        self._compute_curvature_anomaly_kernel[blocks, threads](
            phase_space, curvature_anomaly, n, dim
        )
        
        # 速度異常（並列計算）
        velocity_anomaly = cp.zeros(n, dtype=cp.float32)
        self._compute_velocity_anomaly_kernel[blocks, threads](
            phase_space, velocity_anomaly, n, dim
        )
        
        # 加速度異常も追加
        acceleration_anomaly = cp.zeros(n, dtype=cp.float32)
        self._compute_acceleration_anomaly_kernel[blocks, threads](
            phase_space, acceleration_anomaly, n, dim
        )
        
        # 重み付き統合
        anomaly_scores = (
            0.4 * anomaly_scores + 
            0.2 * curvature_anomaly + 
            0.2 * velocity_anomaly +
            0.2 * acceleration_anomaly
        )
        
        return anomaly_scores
    
    def _analyze_dynamics_gpu_fast(self, phase_space: cp.ndarray) -> Dict:
        """高速ダイナミクス解析"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        features = {}
        
        # 複雑性指標を並列計算
        measures = cp.zeros(5, dtype=cp.float32)
        blocks, threads = self._optimize_kernel_launch(n)
        
        self._complexity_kernel[blocks, threads](
            phase_space, measures, n, dim
        )
        
        # 1. 周期性（並列FFT）
        features['periodicity'] = float(
            self._detect_periodicity_gpu_fast(phase_space)
        )
        
        # 2. カオス性
        features['chaos_measure'] = float(measures[0] / n)
        
        # 3. 予測可能性
        features['predictability'] = float(1.0 / (1.0 + measures[1] / n))
        
        # 4. 複雑性
        features['complexity'] = float(measures[2] / n)
        
        # 5. エントロピー率
        features['entropy_rate'] = float(measures[3] / n)
        
        # 6. 安定性
        features['stability'] = float(1.0 / (1.0 + measures[4] / n))
        
        return features
    
    # === CUDAカーネル実装 ===
    
    @staticmethod
    def _compute_rqa_features_kernel_impl(rec_matrix, features_out, n, min_line_length):
        """RQA特徴量計算カーネル実装"""
        idx = cuda.grid(1)
        
        if idx < n:
            # 対角線構造の検出
            for offset in range(1, n - idx):
                diag_len = 0
                i = idx
                j = offset
                
                while i < n and j < n:
                    if rec_matrix[i, j] > 0:
                        diag_len += 1
                    else:
                        if diag_len >= min_line_length:
                            cuda.atomic.add(features_out, 0, diag_len)  # determinism
                            cuda.atomic.add(features_out, 3, 1)  # diagonal count
                        diag_len = 0
                    i += 1
                    j += 1
            
            # 垂直構造の検出
            vert_len = 0
            for j in range(n):
                if rec_matrix[idx, j] > 0:
                    vert_len += 1
                else:
                    if vert_len >= min_line_length:
                        cuda.atomic.add(features_out, 1, vert_len)  # laminarity
                        cuda.atomic.add(features_out, 2, vert_len)  # trapping time
                        cuda.atomic.add(features_out, 4, 1)  # vertical count
                    vert_len = 0
    
    @staticmethod
    def _compute_attractor_volume_kernel_impl(phase_space, voxel_grid, phase_min, 
                                             phase_range, n_points, dim, grid_size):
        """アトラクタ体積計算カーネル実装"""
        idx = cuda.grid(1)
        
        if idx < n_points:
            # 3次元ボクセル座標を計算
            voxel_coords = cuda.local.array(3, dtype=cuda.int32)
            
            for d in range(min(3, dim)):
                # 正規化してグリッド座標に変換
                normalized = (phase_space[idx, d] - phase_min[d]) / phase_range[d]
                grid_coord = int(normalized * (grid_size - 1))
                voxel_coords[d] = min(max(0, grid_coord), grid_size - 1)
            
            # 1次元インデックスに変換
            voxel_idx = (voxel_coords[0] * grid_size * grid_size + 
                        voxel_coords[1] * grid_size + 
                        voxel_coords[2])
            
            # ボクセルを占有としてマーク
            voxel_grid[voxel_idx] = 1
    
    @staticmethod
    def _compute_fractal_dimension_kernel_impl(distances, correlation_integral, 
                                              radii, n_pairs, n_radii):
        """フラクタル次元計算カーネル実装"""
        idx = cuda.grid(1)
        
        if idx < n_pairs:
            dist = distances[idx]
            
            # 各半径での相関積分に寄与
            for r_idx in range(n_radii):
                if dist < radii[r_idx]:
                    cuda.atomic.add(correlation_integral, r_idx, 1)
    
    @staticmethod
    def _compute_complexity_measures_kernel_impl(phase_space, measures_out, n, dim):
        """複雑性指標計算カーネル実装"""
        idx = cuda.grid(1)
        
        if idx < n - 2:
            # 局所的なカオス性（近傍の発散率）
            local_divergence = 0.0
            for d in range(dim):
                diff1 = phase_space[idx + 1, d] - phase_space[idx, d]
                diff2 = phase_space[idx + 2, d] - phase_space[idx + 1, d]
                local_divergence += abs(diff2 - diff1)
            
            cuda.atomic.add(measures_out, 0, local_divergence)
            
            # 予測誤差
            pred_error = 0.0
            for d in range(dim):
                # 線形予測
                predicted = 2 * phase_space[idx + 1, d] - phase_space[idx, d]
                actual = phase_space[idx + 2, d]
                pred_error += (predicted - actual) ** 2
            
            cuda.atomic.add(measures_out, 1, math.sqrt(pred_error))
            
            # 局所複雑性（近傍の分散）
            if idx > 0:
                local_var = 0.0
                for d in range(dim):
                    mean = (phase_space[idx - 1, d] + phase_space[idx, d] + 
                           phase_space[idx + 1, d]) / 3.0
                    local_var += ((phase_space[idx, d] - mean) ** 2)
                
                cuda.atomic.add(measures_out, 2, math.sqrt(local_var))
            
            # エントロピー寄与
            for d in range(dim):
                delta = abs(phase_space[idx + 1, d] - phase_space[idx, d])
                if delta > 1e-10:
                    cuda.atomic.add(measures_out, 3, -delta * math.log(delta))
            
            # 安定性（軌道の局所的な収束性）
            if idx < n - 3:
                stability = 0.0
                for d in range(dim):
                    second_diff = (phase_space[idx + 2, d] - 
                                 2 * phase_space[idx + 1, d] + 
                                 phase_space[idx, d])
                    stability += abs(second_diff)
                
                cuda.atomic.add(measures_out, 4, stability)
    
    @staticmethod
    def _knn_trajectory_anomaly_kernel_impl(phase_space, anomaly_scores, k, n, dim):
        """k-NN異常検出カーネル実装"""
        idx = cuda.grid(1)
        
        if idx < n:
            # 共有メモリで高速化
            distances = cuda.local.array(256, dtype=cuda.float32)
            
            # バッチ処理で距離計算
            batch_size = min(256, n)
            for batch_start in range(0, n, batch_size):
                batch_end = min(batch_start + batch_size, n)
                
                for j in range(batch_start, batch_end):
                    if j != idx:
                        dist_sq = 0.0
                        for d in range(dim):
                            diff = phase_space[j, d] - phase_space[idx, d]
                            dist_sq += diff * diff
                        distances[j - batch_start] = math.sqrt(dist_sq)
                    else:
                        distances[j - batch_start] = 1e10
            
            # k近傍を効率的に探索（部分ソート）
            knn_sum = 0.0
            for kth in range(k):
                min_dist = 1e10
                min_idx = 0
                
                for j in range(min(batch_size, n)):
                    if distances[j] < min_dist:
                        min_dist = distances[j]
                        min_idx = j
                
                knn_sum += min_dist
                distances[min_idx] = 1e10
            
            anomaly_scores[idx] = knn_sum / k
    
    @staticmethod
    def _compute_diagonal_distribution_kernel_impl(rec_matrix, diag_dist, n):
        """対角線長分布計算カーネル実装"""
        idx = cuda.grid(1)
        
        if idx < n:
            # 各対角線を並列処理
            max_diag_len = 0
            
            for start_j in range(n):
                i, j = idx, start_j
                current_len = 0
                
                while i < n and j < n:
                    if rec_matrix[i, j] > 0:
                        current_len += 1
                        max_diag_len = max(max_diag_len, current_len)
                    else:
                        if current_len > 0:
                            cuda.atomic.add(diag_dist, min(current_len, n//2 - 1), 1)
                        current_len = 0
                    i += 1
                    j += 1
    
    @cuda.jit
    def _compute_curvature_anomaly_kernel(phase_space, curvature_anomaly, n, dim):
        """曲率異常計算カーネル"""
        idx = cuda.grid(1)
        
        if idx > 0 and idx < n - 1:
            # 3点での曲率計算
            v1_norm = 0.0
            v2_norm = 0.0
            dot_product = 0.0
            
            for d in range(dim):
                v1 = phase_space[idx, d] - phase_space[idx - 1, d]
                v2 = phase_space[idx + 1, d] - phase_space[idx, d]
                
                v1_norm += v1 * v1
                v2_norm += v2 * v2
                dot_product += v1 * v2
            
            v1_norm = math.sqrt(v1_norm)
            v2_norm = math.sqrt(v2_norm)
            
            if v1_norm > 1e-10 and v2_norm > 1e-10:
                cos_angle = dot_product / (v1_norm * v2_norm)
                curvature_anomaly[idx] = 1.0 - min(max(cos_angle, -1.0), 1.0)
    
    @cuda.jit
    def _compute_velocity_anomaly_kernel(phase_space, velocity_anomaly, n, dim):
        """速度異常計算カーネル"""
        idx = cuda.grid(1)
        
        if idx < n - 1:
            velocity = 0.0
            for d in range(dim):
                diff = phase_space[idx + 1, d] - phase_space[idx, d]
                velocity += diff * diff
            
            velocity_anomaly[idx] = math.sqrt(velocity)
    
    @cuda.jit
    def _compute_acceleration_anomaly_kernel(phase_space, acceleration_anomaly, n, dim):
        """加速度異常計算カーネル"""
        idx = cuda.grid(1)
        
        if idx > 0 and idx < n - 1:
            acceleration = 0.0
            for d in range(dim):
                acc = (phase_space[idx + 1, d] - 
                      2 * phase_space[idx, d] + 
                      phase_space[idx - 1, d])
                acceleration += acc * acc
            
            acceleration_anomaly[idx] = math.sqrt(acceleration)
    
    # === 補助メソッド ===
    
    def _optimize_kernel_launch(self, n_elements: int) -> Tuple[int, int]:
        """カーネル起動パラメータの最適化"""
        device = cp.cuda.Device()
        
        # デバイス特性を取得
        max_threads = device.attributes['MaxThreadsPerBlock']
        sm_count = device.attributes['MultiProcessorCount']
        warp_size = 32
        
        # ワープサイズの倍数で最適化
        threads = min(256, max_threads)
        threads = (threads // warp_size) * warp_size
        
        # ブロック数を計算
        blocks = (n_elements + threads - 1) // threads
        
        # オキュパンシー最適化
        optimal_blocks = sm_count * 32  # 経験的な値
        blocks = min(blocks, optimal_blocks)
        
        return blocks, threads
    
    def _correlation_dimension_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速相関次元計算"""
        n = min(2000, len(phase_space))
        
        # サンプリング
        if len(phase_space) > n:
            indices = cp.random.choice(len(phase_space), n, replace=False)
            phase_space = phase_space[indices]
        
        # 並列距離計算
        n_pairs = n * (n - 1) // 2
        distances = cp.zeros(n_pairs, dtype=cp.float32)
        
        # 距離を並列計算
        blocks, threads = self._optimize_kernel_launch(n_pairs)
        self._compute_pairwise_distances_kernel[blocks, threads](
            phase_space, distances, n, phase_space.shape[1]
        )
        
        # 相関積分
        radii = cp.logspace(-2, 1, 30)
        correlation_integral = cp.zeros(len(radii), dtype=cp.int32)
        
        self._fractal_kernel[blocks, threads](
            distances, correlation_integral, radii, n_pairs, len(radii)
        )
        
        # log-log勾配
        valid = correlation_integral > 0
        if cp.sum(valid) > 5:
            log_r = cp.log(radii[valid])
            log_c = cp.log(correlation_integral[valid].astype(cp.float32) / n_pairs)
            
            # 線形フィット
            slope = cp.polyfit(log_r, log_c, 1)[0]
            return float(cp.clip(slope, 0.5, 5.0))
        
        return 2.0
    
    def _estimate_lyapunov_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速Lyapunov指数推定"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        # 並列近傍探索
        n_ref_points = min(200, n // 10)
        ref_indices = cp.random.choice(n - 10, n_ref_points, replace=False)
        
        lyap_values = cp.zeros(n_ref_points, dtype=cp.float32)
        
        blocks, threads = self._optimize_kernel_launch(n_ref_points)
        self._lyapunov_kernel[blocks, threads](
            phase_space, ref_indices, lyap_values, n, dim
        )
        
        # 外れ値を除去してmedian
        lyap_values = lyap_values[lyap_values != 0]
        if len(lyap_values) > 0:
            return float(cp.median(lyap_values))
        
        return 0.0
    
    @cuda.jit
    def _lyapunov_kernel(phase_space, ref_indices, lyap_values, n, dim):
        """Lyapunov指数計算カーネル"""
        idx = cuda.grid(1)
        
        if idx < len(ref_indices):
            ref_idx = ref_indices[idx]
            
            # 近傍を探索
            min_dist = 1e10
            nearest_idx = -1
            
            for j in range(max(0, ref_idx - 100), min(n, ref_idx + 100)):
                if abs(j - ref_idx) > 1:
                    dist = 0.0
                    for d in range(dim):
                        diff = phase_space[j, d] - phase_space[ref_idx, d]
                        dist += diff * diff
                    
                    dist = math.sqrt(dist)
                    if dist < min_dist and dist > 1e-10:
                        min_dist = dist
                        nearest_idx = j
            
            # 時間発展を追跡
            if nearest_idx >= 0:
                max_steps = min(10, n - max(ref_idx, nearest_idx))
                lyap_sum = 0.0
                count = 0
                
                for dt in range(1, max_steps):
                    if ref_idx + dt < n and nearest_idx + dt < n:
                        evolved_dist = 0.0
                        for d in range(dim):
                            diff = (phase_space[ref_idx + dt, d] - 
                                  phase_space[nearest_idx + dt, d])
                            evolved_dist += diff * diff
                        
                        evolved_dist = math.sqrt(evolved_dist)
                        
                        if evolved_dist > 1e-10 and min_dist > 1e-10:
                            lyap_sum += math.log(evolved_dist / min_dist) / dt
                            count += 1
                
                if count > 0:
                    lyap_values[idx] = lyap_sum / count
    
    def _compute_fractal_measure_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速フラクタル測度計算"""
        # 相関次元をフラクタル測度として使用
        corr_dim = self._correlation_dimension_gpu_fast(phase_space)
        embedding_dim = phase_space.shape[1]
        
        # 正規化（0-1の範囲に）
        fractal_measure = corr_dim / embedding_dim
        return float(cp.clip(fractal_measure, 0.0, 1.0))
    
    def _compute_information_dimension_gpu(self, phase_space: cp.ndarray) -> float:
        """情報次元の計算"""
        n = min(1000, len(phase_space))
        
        # ボックスカウンティング法
        n_boxes_list = [10, 20, 40, 80]
        entropy_values = []
        
        for n_boxes in n_boxes_list:
            # 位相空間をグリッドに分割
            box_counts = cp.zeros(n_boxes**3, dtype=cp.int32)
            
            # 各点をボックスに割り当て
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
            
            # シャノンエントロピー
            probs = box_counts[box_counts > 0] / n
            entropy = -cp.sum(probs * cp.log(probs))
            entropy_values.append(float(entropy))
        
        # log(box_size) vs entropyの勾配
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
        
        # ブロック単位で並列処理
        block_size = 128
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                
                # 部分距離行列
                block_i = phase_space[i:i_end]
                block_j = phase_space[j:j_end]
                
                # 並列距離計算
                for ii in range(len(block_i)):
                    distances = cp.sqrt(cp.sum((block_j - block_i[ii])**2, axis=1))
                    rec_matrix[i + ii, j:j_end] = (distances < threshold).astype(cp.float32)
        
        return rec_matrix
    
    def _detect_periodicity_gpu_fast(self, phase_space: cp.ndarray) -> float:
        """高速周期性検出"""
        # 各次元でFFT
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        max_period_strength = 0.0
        
        for d in range(min(3, dim)):
            # FFT
            signal = phase_space[:, d]
            fft = cp.fft.fft(signal)
            power = cp.abs(fft[:n//2])**2
            
            # DC成分を除去
            power[0] = 0
            
            # 最大ピーク
            if len(power) > 1:
                max_peak = cp.max(power[1:])
                mean_power = cp.mean(power[1:])
                
                if mean_power > 0:
                    period_strength = max_peak / mean_power
                    max_period_strength = max(max_period_strength, float(period_strength))
        
        # 正規化
        return float(1.0 / (1.0 + cp.exp(-max_period_strength + 5)))
    
    def _compute_entropy_from_distribution(self, distribution: cp.ndarray) -> float:
        """分布からエントロピー計算"""
        # 正規化
        total = cp.sum(distribution)
        if total == 0:
            return 0.0
        
        probs = distribution[distribution > 0] / total
        entropy = -cp.sum(probs * cp.log(probs))
        
        return float(entropy)
    
    def _extract_phase_features_gpu_fast(self, phase_space: cp.ndarray) -> Dict:
        """高速位相空間特徴抽出"""
        n = len(phase_space)
        dim = phase_space.shape[1]
        
        # 統計量を並列計算
        features = {
            'center': cp.mean(phase_space, axis=0),
            'spread': cp.std(phase_space, axis=0),
            'skewness': cp.zeros(dim),
            'kurtosis': cp.zeros(dim)
        }
        
        # 高次モーメント
        for d in range(dim):
            data = phase_space[:, d]
            mean = features['center'][d]
            std = features['spread'][d]
            
            if std > 1e-10:
                features['skewness'][d] = cp.mean(((data - mean) / std)**3)
                features['kurtosis'][d] = cp.mean(((data - mean) / std)**4) - 3
        
        # 体積と表面積
        features['volume'] = self._compute_attractor_volume_gpu(phase_space)
        features['surface_area'] = self._estimate_surface_area_gpu(phase_space)
        
        return features
    
    def _compute_phase_distances_parallel(self, all_features: List[Dict]) -> cp.ndarray:
        """並列位相距離計算"""
        n = len(all_features)
        distances = cp.zeros(n - 1)
        
        for i in range(n - 1):
            f1 = all_features[i]
            f2 = all_features[i + 1]
            
            # 各特徴の距離
            center_dist = cp.linalg.norm(f1['center'] - f2['center'])
            spread_dist = cp.linalg.norm(f1['spread'] - f2['spread'])
            volume_dist = abs(f1['volume'] - f2['volume'])
            
            # 重み付き合計
            distances[i] = 0.5 * center_dist + 0.3 * spread_dist + 0.2 * volume_dist
        
        return distances
    
    @cuda.jit
    def _map_transition_scores_kernel(scores, distances, window_size, stride):
        """遷移スコアマッピングカーネル"""
        idx = cuda.grid(1)
        
        if idx < len(distances):
            start = idx * stride
            end = start + window_size
            
            for j in range(start, min(end, len(scores))):
                cuda.atomic.add(scores, j, distances[idx] / window_size)
    
    @cuda.jit
    def _compute_pairwise_distances_kernel(phase_space, distances, n, dim):
        """ペアワイズ距離計算カーネル"""
        idx = cuda.grid(1)
        
        if idx < len(distances):
            # インデックスから(i, j)ペアを復元
            i = int((1 + math.sqrt(1 + 8 * idx)) / 2)
            j = idx - i * (i - 1) // 2
            
            if i < n and j < i:
                dist = 0.0
                for d in range(dim):
                    diff = phase_space[i, d] - phase_space[j, d]
                    dist += diff * diff
                
                distances[idx] = math.sqrt(dist)
    
    def _estimate_surface_area_gpu(self, phase_space: cp.ndarray) -> float:
        """表面積の推定（凸包近似）"""
        # 簡易版：最外点からの距離
        center = cp.mean(phase_space, axis=0)
        distances = cp.sqrt(cp.sum((phase_space - center)**2, axis=1))
        
        # 表面の点を推定
        threshold = cp.percentile(distances, 90)
        surface_points = cp.sum(distances > threshold)
        
        return float(surface_points / len(phase_space))
    
    def _compute_attractor_volume_gpu(self, phase_space: cp.ndarray) -> float:
        """アトラクタ体積（改良版）"""
        # 主成分で次元削減してから体積計算
        if len(phase_space) < 10:
            return 0.0
        
        # 中心化
        centered = phase_space - cp.mean(phase_space, axis=0)
        
        # 共分散行列
        cov = cp.cov(centered.T)
        
        # 固有値（体積の近似）
        eigenvalues = cp.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) > 0:
            # 楕円体の体積
            volume = cp.prod(cp.sqrt(eigenvalues))
            return float(volume)
        
        return 0.0
    
    def _get_primary_series(self, structures: Dict) -> cp.ndarray:
        """主要な時系列を選択"""
        if 'rho_T' in structures:
            return self.to_gpu(structures['rho_T'])
        elif 'lambda_F_mag' in structures:
            return self.to_gpu(structures['lambda_F_mag'])
        else:
            # 最初の利用可能な系列
            for key, value in structures.items():
                if isinstance(value, np.ndarray) and len(value.shape) == 1:
                    return self.to_gpu(value)
        
        raise ValueError("No suitable time series found in structures")
    
    def _estimate_embedding_parameters_gpu(self,
                                         series: cp.ndarray,
                                         params: Dict) -> Dict:
        """最適な埋め込みパラメータを推定（高速版）"""
        # 相互情報量による遅延時間の推定
        optimal_delay = self._estimate_delay_mutual_info_gpu_fast(series)
        
        # False Nearest Neighbors法による次元推定
        optimal_dim = self._estimate_dimension_fnn_gpu_fast(
            series, optimal_delay, max_dim=10
        )
        
        return {
            'embedding_dim': int(optimal_dim),
            'delay': int(optimal_delay),
            'estimated_from': 'mutual_info_and_fnn'
        }
    
    def _estimate_delay_mutual_info_gpu_fast(self, series: cp.ndarray) -> int:
        """高速相互情報量による遅延推定"""
        max_delay = min(100, len(series) // 10)
        mi_values = cp.zeros(max_delay)
        
        # 並列で相互情報量計算
        for delay in range(1, max_delay):
            x = series[:-delay]
            y = series[delay:]
            
            # 2次元ヒストグラム（高速版）
            n_bins = 20
            hist_2d, _, _ = cp.histogram2d(x, y, bins=n_bins)
            
            # 正規化
            hist_2d = hist_2d / cp.sum(hist_2d)
            
            # 周辺分布
            px = cp.sum(hist_2d, axis=1)
            py = cp.sum(hist_2d, axis=0)
            
            # 相互情報量（ベクトル化）
            valid = hist_2d > 0
            mi = cp.sum(hist_2d[valid] * cp.log(
                hist_2d[valid] / (px[:, None] * py[None, :])[valid]
            ))
            
            mi_values[delay] = mi
        
        # 最初の極小値を検出
        for i in range(2, len(mi_values) - 1):
            if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                return int(i)
        
        # 見つからない場合は減衰点
        decay_point = cp.where(mi_values < mi_values[1] * 0.5)[0]
        if len(decay_point) > 0:
            return int(decay_point[0])
        
        return 10  # デフォルト
    
    def _estimate_dimension_fnn_gpu_fast(self,
                                       series: cp.ndarray,
                                       delay: int,
                                       max_dim: int) -> int:
        """高速False Nearest Neighbors法"""
        fnn_fractions = []
        
        for dim in range(1, max_dim):
            try:
                # 埋め込み
                phase_space = self._reconstruct_phase_space_gpu(series, dim, delay)
                n = min(2000, len(phase_space))
                
                if n < 100:
                    break
                
                # サンプリング
                if len(phase_space) > n:
                    indices = cp.random.choice(len(phase_space), n, replace=False)
                    phase_space_sample = phase_space[indices]
                else:
                    phase_space_sample = phase_space
                
                # FNN計算（並列化）
                fnn_count = cp.zeros(1, dtype=cp.int32)
                
                blocks, threads = self._optimize_kernel_launch(n)
                self._fnn_kernel[blocks, threads](
                    phase_space_sample, series, fnn_count, 
                    n, dim, delay, len(series)
                )
                
                fnn_fraction = float(fnn_count[0]) / n
                fnn_fractions.append(fnn_fraction)
                
                # 収束判定
                if fnn_fraction < 0.01:
                    return dim
                
                # 急激な減少
                if len(fnn_fractions) > 1:
                    if fnn_fractions[-2] - fnn_fraction > 0.5:
                        return dim
            
            except Exception as e:
                print(f"FNN dimension {dim} failed: {e}")
                break
        
        # デフォルト
        if fnn_fractions:
            # 最小のFNN率の次元
            return int(cp.argmin(cp.array(fnn_fractions)) + 1)
        
        return 3
    
    @cuda.jit
    def _fnn_kernel(phase_space, series, fnn_count, n, dim, delay, series_len):
        """FNNカーネル"""
        idx = cuda.grid(1)
        
        if idx < n:
            # 最近傍を探索
            min_dist = 1e10
            nn_idx = -1
            
            for j in range(n):
                if j != idx:
                    dist = 0.0
                    for d in range(dim):
                        diff = phase_space[j, d] - phase_space[idx, d]
                        dist += diff * diff
                    
                    if dist < min_dist:
                        min_dist = math.sqrt(dist)
                        nn_idx = j
            
            # False nearest neighbor判定
            if nn_idx >= 0 and min_dist > 1e-10:
                # 次元を増やした時の距離
                next_dim_idx = idx + dim * delay
                next_dim_nn_idx = nn_idx + dim * delay
                
                if next_dim_idx < series_len and next_dim_nn_idx < series_len:
                    next_dim_dist = abs(series[next_dim_idx] - series[next_dim_nn_idx])
                    
                    # Kennel et al.の基準
                    if next_dim_dist / min_dist > 10.0:
                        cuda.atomic.add(fnn_count, 0, 1)
    
    def _reconstruct_phase_space_gpu(self,
                                   series: cp.ndarray,
                                   embedding_dim: int,
                                   delay: int) -> cp.ndarray:
        """時系列から位相空間を再構成（高速版）"""
        n = len(series)
        embed_length = n - (embedding_dim - 1) * delay
        
        if embed_length <= 0:
            raise ValueError(f"Series too short for embedding: {n} < {(embedding_dim - 1) * delay}")
        
        # ベクトル化された埋め込み
        indices = cp.arange(embed_length)[:, None] + cp.arange(embedding_dim)[None, :] * delay
        phase_space = series[indices]
        
        return phase_space
    
    def _compute_integrated_score_gpu(self,
                                    anomaly_scores: cp.ndarray,
                                    attractor_features: Dict,
                                    recurrence_features: Dict) -> cp.ndarray:
        """統合異常スコア（改良版）"""
        # アトラクタ異常度
        attractor_anomaly = 0.0
        
        # Lyapunov指数（正値はカオス的）
        if abs(attractor_features['lyapunov_exponent']) > 0.1:
            attractor_anomaly += 0.3
        
        # フラクタル性（非整数次元）
        frac_deviation = abs(attractor_features['fractal_measure'] - 0.5)
        attractor_anomaly += frac_deviation * 0.2
        
        # 相関次元の異常
        corr_dim = attractor_features['correlation_dimension']
        if corr_dim < 1.0 or corr_dim > 3.0:
            attractor_anomaly += 0.2
        
        # 情報次元との乖離
        info_dim = attractor_features.get('information_dimension', 2.0)
        dim_diff = abs(corr_dim - info_dim)
        attractor_anomaly += dim_diff * 0.1
        
        # リカレンス異常度
        recurrence_anomaly = 0.0
        
        # 決定論性の低下
        determinism = recurrence_features['determinism']
        if determinism < 0.7:
            recurrence_anomaly += (1 - determinism) * 0.3
        
        # エントロピーの異常
        entropy = recurrence_features['entropy']
        if entropy > 2.0:
            recurrence_anomaly += min(entropy / 5.0, 0.3)
        
        # 層流性の異常
        laminarity = recurrence_features['laminarity']
        if laminarity < 0.5:
            recurrence_anomaly += (1 - laminarity) * 0.2
        
        # トラッピング時間
        trapping = recurrence_features.get('trapping_time', 1.0)
        if trapping > 10.0:
            recurrence_anomaly += 0.2
        
        # 正規化
        attractor_anomaly = cp.clip(attractor_anomaly, 0.0, 1.0)
        recurrence_anomaly = cp.clip(recurrence_anomaly, 0.0, 1.0)
        
        # 統合（時系列全体に適用）
        global_anomaly = 0.5 * attractor_anomaly + 0.5 * recurrence_anomaly
        
        # 局所異常と統合
        integrated = 0.7 * anomaly_scores + 0.3 * global_anomaly
        
        # 正規化
        integrated = (integrated - cp.mean(integrated)) / (cp.std(integrated) + 1e-10)
        
        return integrated
    
    def _smooth_and_normalize_gpu(self, scores: cp.ndarray) -> cp.ndarray:
        """スムージングと正規化（高速版）"""
        # ガウシアンフィルタ（並列化）
        from cupyx.scipy.ndimage import gaussian_filter1d
        
        # 適応的なシグマ
        sigma = max(5, len(scores) // 100)
        smoothed = gaussian_filter1d(scores, sigma=sigma)
        
        # ロバスト正規化
        median = cp.median(smoothed)
        mad = cp.median(cp.abs(smoothed - median))
        
        if mad > 1e-10:
            normalized = (smoothed - median) / (1.4826 * mad)  # MAD to std
        else:
            # フォールバック
            if cp.std(smoothed) > 1e-10:
                normalized = (smoothed - cp.mean(smoothed)) / cp.std(smoothed)
            else:
                normalized = smoothed
        
        return normalized
    
    def _print_summary(self, results: Dict):
        """解析結果のサマリー表示"""
        print("\n📊 Phase Space Analysis Summary (Ultra-Fast Edition):")
        print(f"   ⚡ GPU Acceleration: ENABLED")
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
