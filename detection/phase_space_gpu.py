"""
Lambda³ GPU版位相空間解析モジュール
高次元位相空間での異常検出とアトラクタ解析
"""

import numpy as np
import cupy as cp
from typing import Dict, List, Tuple, Optional, Any
from numba import cuda
from cupyx.scipy.spatial import distance_matrix

from ..types import ArrayType, NDArray
from ..core.gpu_utils import GPUBackend

class PhaseSpaceAnalyzerGPU(GPUBackend):
    """位相空間解析のGPU実装"""
    
    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.embedding_cache = {}
        
    def analyze_phase_space(self,
                          structures: Dict[str, np.ndarray],
                          embedding_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        包括的な位相空間解析
        
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
        print("\n🌌 Comprehensive Phase Space Analysis on GPU...")
        
        # デフォルトパラメータ
        if embedding_params is None:
            embedding_params = {
                'embedding_dim': 3,
                'delay': 50,
                'n_neighbors': 20,
                'recurrence_threshold': 0.1
            }
        
        # 主要な時系列を取得
        primary_series = self._get_primary_series(structures)
        
        results = {}
        
        with self.memory_manager.batch_context(len(primary_series) * 10):
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
            attractor_features = self._analyze_attractor_gpu(phase_space)
            results['attractor_features'] = attractor_features
            
            # 4. リカレンスプロット解析
            recurrence_features = self._recurrence_analysis_gpu(
                phase_space, embedding_params['recurrence_threshold']
            )
            results['recurrence_features'] = recurrence_features
            
            # 5. 異常軌道検出
            anomaly_scores = self._detect_anomalous_trajectories_gpu(
                phase_space, embedding_params['n_neighbors']
            )
            results['anomaly_scores'] = self.to_cpu(anomaly_scores)
            
            # 6. ダイナミクス特性
            dynamics_features = self._analyze_dynamics_gpu(phase_space)
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
        位相空間での遷移検出（スライディングウィンドウ）
        """
        print("\n🔄 Detecting phase transitions in sliding windows...")
        
        primary_series = self._get_primary_series(structures)
        n_frames = len(primary_series)
        n_windows = (n_frames - window_size) // stride + 1
        
        transition_scores = cp.zeros(n_frames)
        phase_distances = []
        
        # 各ウィンドウで位相空間を構築し比較
        prev_features = None
        
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
            
            # 位相空間の特徴抽出
            features = self._extract_phase_features_gpu(phase_space)
            
            # 前のウィンドウとの距離
            if prev_features is not None:
                distance = self._phase_space_distance_gpu(prev_features, features)
                phase_distances.append(float(distance))
                
                # スコアに反映
                for j in range(start, end):
                    transition_scores[j] += distance / window_size
            
            prev_features = features
        
        # スムージングと正規化
        transition_scores = self._smooth_and_normalize_gpu(transition_scores)
        
        return {
            'transition_scores': transition_scores,
            'phase_distances': phase_distances,
            'window_size': window_size,
            'stride': stride
        }
    
    # === Private methods ===
    
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
        """最適な埋め込みパラメータを推定"""
        # 相互情報量による遅延時間の推定
        optimal_delay = self._estimate_delay_mutual_info_gpu(series)
        
        # False Nearest Neighbors法による次元推定
        optimal_dim = self._estimate_dimension_fnn_gpu(
            series, optimal_delay, max_dim=10
        )
        
        return {
            'embedding_dim': int(optimal_dim),
            'delay': int(optimal_delay),
            'estimated_from': 'mutual_info_and_fnn'
        }
    
    def _reconstruct_phase_space_gpu(self,
                                   series: cp.ndarray,
                                   embedding_dim: int,
                                   delay: int) -> cp.ndarray:
        """時系列から位相空間を再構成"""
        n = len(series)
        embed_length = n - (embedding_dim - 1) * delay
        
        if embed_length <= 0:
            raise ValueError(f"Series too short for embedding: {n} < {(embedding_dim - 1) * delay}")
        
        # GPU上で埋め込み
        phase_space = cp.zeros((embed_length, embedding_dim))
        
        for i in range(embedding_dim):
            phase_space[:, i] = series[i*delay:i*delay + embed_length]
        
        return phase_space
    
    def _analyze_attractor_gpu(self, phase_space: cp.ndarray) -> Dict:
        """アトラクタの特性を解析"""
        features = {}
        
        # 1. 次元推定（相関次元）
        features['correlation_dimension'] = float(
            self._correlation_dimension_gpu(phase_space)
        )
        
        # 2. Lyapunov指数（簡易推定）
        features['lyapunov_exponent'] = float(
            self._estimate_lyapunov_gpu(phase_space)
        )
        
        # 3. アトラクタの体積
        features['attractor_volume'] = float(
            self._compute_attractor_volume_gpu(phase_space)
        )
        
        # 4. フラクタル性
        features['fractal_measure'] = float(
            self._compute_fractal_measure_gpu(phase_space)
        )
        
        return features
    
    def _recurrence_analysis_gpu(self,
                                phase_space: cp.ndarray,
                                threshold: float) -> Dict:
        """リカレンスプロット解析"""
        n = len(phase_space)
        
        # 距離行列の計算（メモリ効率のため分割）
        max_size = 5000
        if n > max_size:
            # サンプリング
            indices = cp.random.choice(n, max_size, replace=False)
            phase_space_sample = phase_space[indices]
            n = max_size
        else:
            phase_space_sample = phase_space
        
        # リカレンス行列
        recurrence_matrix = self._compute_recurrence_matrix_gpu(
            phase_space_sample, threshold
        )
        
        # リカレンス定量化解析（RQA）
        features = {
            'recurrence_rate': float(cp.mean(recurrence_matrix)),
            'determinism': float(self._compute_determinism_gpu(recurrence_matrix)),
            'laminarity': float(self._compute_laminarity_gpu(recurrence_matrix)),
            'entropy': float(self._recurrence_entropy_gpu(recurrence_matrix))
        }
        
        return features
    
    def _detect_anomalous_trajectories_gpu(self,
                                         phase_space: cp.ndarray,
                                         n_neighbors: int) -> cp.ndarray:
        """異常軌道の検出"""
        n = len(phase_space)
        anomaly_scores = cp.zeros(n)
        
        # k-NN異常検出
        threads = 256
        blocks = (n + threads - 1) // threads
        
        self._knn_trajectory_anomaly_kernel[blocks, threads](
            phase_space, anomaly_scores, n_neighbors, n, phase_space.shape[1]
        )
        
        # 軌道の曲率異常
        curvature_anomaly = self._detect_curvature_anomaly_gpu(phase_space)
        
        # 速度異常
        velocity_anomaly = self._detect_velocity_anomaly_gpu(phase_space)
        
        # 統合
        anomaly_scores = 0.5 * anomaly_scores + 0.3 * curvature_anomaly + 0.2 * velocity_anomaly
        
        return anomaly_scores
    
    def _analyze_dynamics_gpu(self, phase_space: cp.ndarray) -> Dict:
        """ダイナミクス特性の解析"""
        features = {}
        
        # 1. 周期性の検出
        features['periodicity'] = float(
            self._detect_periodicity_phase_space_gpu(phase_space)
        )
        
        # 2. カオス性の指標
        features['chaos_measure'] = float(
            self._compute_chaos_measure_gpu(phase_space)
        )
        
        # 3. 安定性
        features['stability'] = float(
            self._compute_stability_gpu(phase_space)
        )
        
        # 4. 複雑性
        features['complexity'] = float(
            self._compute_complexity_gpu(phase_space)
        )
        
        return features
    
    def _compute_integrated_score_gpu(self,
                                    anomaly_scores: cp.ndarray,
                                    attractor_features: Dict,
                                    recurrence_features: Dict) -> cp.ndarray:
        """統合異常スコア"""
        # アトラクタ異常度
        attractor_anomaly = (
            abs(attractor_features['lyapunov_exponent']) * 0.3 +
            (1 - attractor_features['fractal_measure']) * 0.2 +
            attractor_features['correlation_dimension'] * 0.1
        )
        
        # リカレンス異常度
        recurrence_anomaly = (
            (1 - recurrence_features['determinism']) * 0.4 +
            recurrence_features['entropy'] * 0.3 +
            (1 - recurrence_features['laminarity']) * 0.3
        )
        
        # グローバル異常度を各時点に反映
        global_anomaly = (attractor_anomaly + recurrence_anomaly) / 2
        
        # 時系列全体に適用
        integrated = anomaly_scores + global_anomaly
        
        return integrated
    
    # === 補助GPU関数 ===
    
    def _estimate_delay_mutual_info_gpu(self, series: cp.ndarray) -> int:
        """相互情報量による遅延推定"""
        max_delay = min(100, len(series) // 10)
        mi_values = cp.zeros(max_delay)
        
        for delay in range(1, max_delay):
            # 簡易的な相互情報量計算
            x = series[:-delay]
            y = series[delay:]
            
            # ビニング
            n_bins = 20
            hist_2d = cp.histogram2d(x, y, bins=n_bins)[0]
            
            # 相互情報量
            pxy = hist_2d / cp.sum(hist_2d)
            px = cp.sum(pxy, axis=1)
            py = cp.sum(pxy, axis=0)
            
            # MI計算
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                        mi += pxy[i, j] * cp.log(pxy[i, j] / (px[i] * py[j]))
            
            mi_values[delay] = mi
        
        # 最初の極小値
        for i in range(2, len(mi_values)-1):
            if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
                return i
        
        return 10  # デフォルト
    
    def _estimate_dimension_fnn_gpu(self,
                                  series: cp.ndarray,
                                  delay: int,
                                  max_dim: int) -> int:
        """False Nearest Neighbors法による次元推定"""
        fnn_fractions = []
        
        for dim in range(1, max_dim):
            # 埋め込み
            try:
                phase_space = self._reconstruct_phase_space_gpu(series, dim, delay)
                
                # FNN計算（簡易版）
                n = min(1000, len(phase_space))  # サンプリング
                fnn_count = 0
                
                for i in range(n):
                    # 最近傍を見つける
                    distances = cp.sqrt(cp.sum((phase_space - phase_space[i])**2, axis=1))
                    distances[i] = cp.inf
                    nn_idx = cp.argmin(distances)
                    
                    # 次元を増やした時の距離
                    if dim < len(series) - delay * dim:
                        next_dim_dist = abs(
                            series[i + dim * delay] - series[nn_idx + dim * delay]
                        )
                        
                        if next_dim_dist / distances[nn_idx] > 10:
                            fnn_count += 1
                
                fnn_fraction = fnn_count / n
                fnn_fractions.append(float(fnn_fraction))
                
                # FNNが十分小さくなったら
                if fnn_fraction < 0.01:
                    return dim
            
            except:
                break
        
        return 3  # デフォルト
    
    def _correlation_dimension_gpu(self, phase_space: cp.ndarray) -> float:
        """相関次元の計算"""
        n = min(1000, len(phase_space))
        
        # サンプリング
        if len(phase_space) > n:
            indices = cp.random.choice(len(phase_space), n, replace=False)
            phase_space = phase_space[indices]
        
        # 距離計算
        radii = cp.logspace(-2, 1, 20)
        correlation_sums = []
        
        for r in radii:
            count = 0
            for i in range(n):
                distances = cp.sqrt(cp.sum((phase_space - phase_space[i])**2, axis=1))
                count += cp.sum(distances < r) - 1  # 自分自身を除く
            
            correlation_sum = count / (n * (n - 1))
            if correlation_sum > 0:
                correlation_sums.append((float(cp.log(r)), float(cp.log(correlation_sum))))
        
        # 線形フィット（簡易版）
        if len(correlation_sums) > 2:
            x = cp.array([c[0] for c in correlation_sums])
            y = cp.array([c[1] for c in correlation_sums])
            
            # 最小二乗法
            slope = cp.sum((x - cp.mean(x)) * (y - cp.mean(y))) / cp.sum((x - cp.mean(x))**2)
            
            return float(slope)
        
        return 2.0  # デフォルト
    
    def _estimate_lyapunov_gpu(self, phase_space: cp.ndarray) -> float:
        """Lyapunov指数の簡易推定"""
        n = len(phase_space)
        lyap_sum = 0.0
        count = 0
        
        # 近傍点の発散率を計算
        for i in range(min(100, n-10)):
            # 近傍点を探す
            distances = cp.sqrt(cp.sum((phase_space - phase_space[i])**2, axis=1))
            distances[i] = cp.inf
            
            # 小さな半径内の点
            radius = cp.std(distances) * 0.1
            neighbors = cp.where(distances < radius)[0]
            
            if len(neighbors) > 0:
                for j in neighbors[:5]:  # 最大5つ
                    # 時間発展後の距離
                    for dt in range(1, min(10, n-max(i,int(j)))):
                        d0 = float(distances[j])
                        d1 = float(cp.sqrt(cp.sum((phase_space[i+dt] - phase_space[j+dt])**2)))
                        
                        if d0 > 0 and d1 > 0:
                            lyap_sum += cp.log(d1 / d0) / dt
                            count += 1
        
        return float(lyap_sum / count) if count > 0 else 0.0
    
    def _compute_recurrence_matrix_gpu(self,
                                     phase_space: cp.ndarray,
                                     threshold: float) -> cp.ndarray:
        """リカレンス行列の計算"""
        n = len(phase_space)
        
        # 距離行列
        distances = distance_matrix(phase_space, phase_space)
        
        # 閾値処理
        recurrence_matrix = (distances < threshold).astype(cp.float32)
        
        return recurrence_matrix
    
    def _smooth_and_normalize_gpu(self, scores: cp.ndarray) -> cp.ndarray:
        """スムージングと正規化"""
        from cupyx.scipy.ndimage import gaussian_filter1d as gaussian_filter1d_gpu
        
        # スムージング
        smoothed = gaussian_filter1d_gpu(scores, sigma=10)
        
        # 正規化
        if cp.std(smoothed) > 1e-10:
            normalized = (smoothed - cp.mean(smoothed)) / cp.std(smoothed)
        else:
            normalized = smoothed
        
        return normalized
    
    def _print_summary(self, results: Dict):
        """解析結果のサマリー表示"""
        print("\n📊 Phase Space Analysis Summary:")
        print(f"   Embedding dimension: {results['optimal_parameters']['embedding_dim']}")
        print(f"   Delay: {results['optimal_parameters']['delay']}")
        
        if 'attractor_features' in results:
            att = results['attractor_features']
            print(f"\n   Attractor Features:")
            print(f"   - Correlation dimension: {att['correlation_dimension']:.3f}")
            print(f"   - Lyapunov exponent: {att['lyapunov_exponent']:.3f}")
            print(f"   - Fractal measure: {att['fractal_measure']:.3f}")
        
        if 'recurrence_features' in results:
            rec = results['recurrence_features']
            print(f"\n   Recurrence Features:")
            print(f"   - Recurrence rate: {rec['recurrence_rate']:.3f}")
            print(f"   - Determinism: {rec['determinism']:.3f}")
            print(f"   - Entropy: {rec['entropy']:.3f}")
    
    # === CUDAカーネル ===
    
    @cuda.jit
    def _knn_trajectory_anomaly_kernel(phase_space, anomaly_scores, k, n, dim):
        """k-NN軌道異常検出カーネル"""
        idx = cuda.grid(1)
        
        if idx < n:
            # 簡易的なk-NN（実際はもっと効率的な実装が必要）
            distances = cuda.local.array(100, dtype=cuda.float32)
            
            # 距離計算
            for j in range(min(100, n)):
                if j != idx:
                    dist = 0.0
                    for d in range(dim):
                        diff = phase_space[j, d] - phase_space[idx, d]
                        dist += diff * diff
                    distances[j] = cuda.sqrt(dist)
                else:
                    distances[j] = 1e10
            
            # k近傍の平均
            knn_sum = 0.0
            for _ in range(k):
                min_idx = 0
                min_dist = distances[0]
                for j in range(1, min(100, n)):
                    if distances[j] < min_dist:
                        min_dist = distances[j]
                        min_idx = j
                knn_sum += min_dist
                distances[min_idx] = 1e10
            
            anomaly_scores[idx] = knn_sum / k
    
    # 他の補助メソッド（簡略化）
    def _compute_attractor_volume_gpu(self, phase_space: cp.ndarray) -> float:
        """アトラクタ体積の推定"""
        return float(cp.prod(cp.max(phase_space, axis=0) - cp.min(phase_space, axis=0)))
    
    def compute_fractal_measure_gpu(self, phase_space: cp.ndarray) -> float:
        """
        相関次元による フラクタル測度（Grassberger-Procaccia法）
        """
        n_points = min(1000, len(phase_space))  # サンプリング
        if n_points < 100:
            return 1.5  # データ不足
        
        # ランダムサンプリング
        indices = cp.random.choice(len(phase_space), n_points, replace=False)
        sample = phase_space[indices]
        
        # 距離行列（上三角のみ）
        distances = []
        for i in range(n_points - 1):
            dists = cp.linalg.norm(sample[i+1:] - sample[i], axis=1)
            distances.append(dists)
        
        all_distances = cp.concatenate(distances)
        all_distances = all_distances[all_distances > 0]
        
        if len(all_distances) == 0:
            return 1.5
        
        # 相関積分C(r)を計算
        r_values = cp.logspace(
            cp.log10(cp.min(all_distances)),
            cp.log10(cp.max(all_distances)),
            20
        )
        
        correlation_sum = cp.zeros(len(r_values))
        for i, r in enumerate(r_values):
            correlation_sum[i] = cp.sum(all_distances < r)
        
        # log-log勾配から次元推定
        valid = correlation_sum > 0
        if cp.sum(valid) > 5:
            log_r = cp.log(r_values[valid])
            log_c = cp.log(correlation_sum[valid] / (n_points * (n_points - 1) / 2))
            
            # 中間領域で勾配計算
            mid_idx = len(log_r) // 2
            slope = cp.polyfit(
                log_r[mid_idx-2:mid_idx+3],
                log_c[mid_idx-2:mid_idx+3],
                1
            )[0]
            
            return float(cp.clip(slope, 0.5, 3.0))
        
        return 1.5  # それでもダメなら...
    
    def _compute_determinism_gpu(self, rec_matrix: cp.ndarray) -> float:
        """決定論性の計算"""
        # 対角線構造の割合
        return float(cp.mean(cp.diag(rec_matrix, k=1)))
    
    def _compute_laminarity_gpu(self, rec_matrix: cp.ndarray) -> float:
        """層流性の計算"""
        # 垂直/水平構造の割合
        return float(cp.mean(rec_matrix))
    
    def _recurrence_entropy_gpu(self, rec_matrix: cp.ndarray) -> float:
        """リカレンスエントロピー"""
        # シャノンエントロピーの簡易計算
        p = cp.mean(rec_matrix)
        if p > 0 and p < 1:
            return float(-p * cp.log(p) - (1-p) * cp.log(1-p))
        return 0.0
    
    def _extract_phase_features_gpu(self, phase_space: cp.ndarray) -> Dict:
        """位相空間の特徴抽出"""
        return {
            'center': cp.mean(phase_space, axis=0),
            'spread': cp.std(phase_space, axis=0),
            'volume': self._compute_attractor_volume_gpu(phase_space)
        }
    
    def _phase_space_distance_gpu(self, features1: Dict, features2: Dict) -> float:
        """位相空間特徴間の距離"""
        dist = cp.linalg.norm(features1['center'] - features2['center'])
        dist += cp.linalg.norm(features1['spread'] - features2['spread'])
        dist += abs(features1['volume'] - features2['volume']) / (features1['volume'] + 1e-10)
        return float(dist)
    
    def _detect_curvature_anomaly_gpu(self, phase_space: cp.ndarray) -> cp.ndarray:
        """曲率異常の検出"""
        n = len(phase_space)
        curvature = cp.zeros(n)
        
        for i in range(1, n-1):
            # 3点での曲率近似
            v1 = phase_space[i] - phase_space[i-1]
            v2 = phase_space[i+1] - phase_space[i]
            
            # 角度変化
            cos_angle = cp.dot(v1, v2) / (cp.linalg.norm(v1) * cp.linalg.norm(v2) + 1e-10)
            curvature[i] = 1 - cos_angle
        
        return curvature
    
    def _detect_velocity_anomaly_gpu(self, phase_space: cp.ndarray) -> cp.ndarray:
        """速度異常の検出"""
        velocity = cp.sqrt(cp.sum(cp.diff(phase_space, axis=0)**2, axis=1))
        velocity = cp.pad(velocity, (1, 0), mode='edge')
        
        # 正規化
        mean_vel = cp.mean(velocity)
        std_vel = cp.std(velocity)
        
        if std_vel > 1e-10:
            return cp.abs(velocity - mean_vel) / std_vel
        else:
            return cp.zeros_like(velocity)
    
    def _detect_periodicity_phase_space_gpu(self, phase_space: cp.ndarray) -> float:
        """位相空間での周期性検出"""
        # 簡易版：自己相関
        n = len(phase_space)
        max_lag = min(n // 2, 1000)
        
        autocorr = []
        for lag in range(1, max_lag, 10):
            if lag < n:
                corr = cp.mean(cp.sum(phase_space[:-lag] * phase_space[lag:], axis=1))
                autocorr.append(float(corr))
        
        # 周期性の強さ
        if autocorr:
            return float(cp.max(cp.array(autocorr)))
        return 0.0
    
    def _compute_chaos_measure_gpu(self, phase_space: cp.ndarray) -> float:
        """カオス性の測定"""
        # Lyapunov指数の絶対値を使用
        lyap = abs(self._estimate_lyapunov_gpu(phase_space))
        return float(1 / (1 + cp.exp(-lyap)))  # シグモイド変換
    
    def _compute_stability_gpu(self, phase_space: cp.ndarray) -> float:
        """安定性の測定"""
        # 軌道の分散
        variance = cp.var(phase_space, axis=0)
        return float(1 / (1 + cp.mean(variance)))
    
    def _compute_complexity_gpu(self, phase_space: cp.ndarray) -> float:
        """複雑性の測定"""
        # 相関次元を正規化
        corr_dim = self._correlation_dimension_gpu(phase_space)
        return float(corr_dim / phase_space.shape[1])
