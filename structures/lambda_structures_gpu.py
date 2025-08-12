"""
Lambda³ Structure Computation (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造の計算をGPUで超高速化！
NO TIME, NO PHYSICS, ONLY STRUCTURE... but FASTER! 🚀

⚡ 2025/01/16 完全リファクタリング版 by 環ちゃん
- 重複コード削除
- シンプルで確実なGPU検出
- A100完全対応
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass

# ===============================
# GPU Setup (シンプル版)
# ===============================
try:
    import cupy as cp
    from cupyx.scipy.signal import savgol_filter as cp_savgol_filter
    HAS_GPU = True
    print("✅ CuPy successfully imported")
except ImportError as e:
    cp = None
    cp_savgol_filter = None
    HAS_GPU = False
    print(f"❌ CuPy not available: {e}")

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager

# カーネルインポート（エラーを詳細表示）
tension_field_kernel = None
topological_charge_kernel = None

if HAS_GPU:
    try:
        from ..core import tension_field_kernel, topological_charge_kernel
        print("✅ Custom kernels imported")
    except ImportError as e:
        print(f"⚠️ Custom kernels not available: {e}")
        # カーネルなくても基本機能は動くようにする

logger = logging.getLogger(__name__)

# ===============================
# Configuration
# ===============================

@dataclass
class LambdaStructureConfig:
    """Lambda構造計算の設定"""
    use_mixed_precision: bool = True
    batch_size: Optional[int] = None
    cache_intermediates: bool = True
    profile: bool = False

# ===============================
# Lambda Structures GPU Class
# ===============================

class LambdaStructuresGPU(GPUBackend):
    """
    Lambda³構造計算のGPU実装クラス
    A100で爆速！！✨
    """
    
    def __init__(self, 
                 config: Optional[LambdaStructureConfig] = None,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 force_cpu: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        config : LambdaStructureConfig
            計算設定
        memory_manager : GPUMemoryManager
            メモリ管理インスタンス
        force_cpu : bool
            強制的にCPUモードにする
        """
        # 親クラス初期化
        super().__init__(
            force_cpu=force_cpu,
            mixed_precision=config.use_mixed_precision if config else False,
            profile=config.profile if config else False,
            **kwargs
        )
        
        self.config = config or LambdaStructureConfig()
        self.memory_manager = memory_manager or GPUMemoryManager()
        self._cache = {} if self.config.cache_intermediates else None
        
        # GPU/CPU状態をログ出力
        if self.is_gpu:
            logger.info(f"✅ GPU mode enabled: {self.device}")
            self.xp = cp
        else:
            logger.info("✅ CPU mode enabled")
            self.xp = np
            
    def compute_lambda_structures(self,
                                trajectory: np.ndarray,
                                md_features: Dict[str, np.ndarray],
                                window_steps: int) -> Dict[str, np.ndarray]:
        """
        Lambda³構造をGPUで計算
        
        Parameters
        ----------
        trajectory : np.ndarray
            MDトラジェクトリ
        md_features : Dict[str, np.ndarray]
            MD特徴量（com_positions, rmsd, radius_of_gyration等）
        window_steps : int
            ウィンドウサイズ
            
        Returns
        -------
        Dict[str, np.ndarray]
            Lambda構造辞書
        """
        try:
            with self.timer('compute_lambda_structures'):
                logger.info(f"🚀 Computing Lambda³ structures (window={window_steps}, mode={'GPU' if self.is_gpu else 'CPU'})")
                
                # 入力検証
                if 'com_positions' not in md_features:
                    raise ValueError("com_positions not found in md_features")
                
                # GPU転送
                positions_gpu = self.to_gpu(md_features['com_positions'])
                n_frames = positions_gpu.shape[0]
                
                # 1. ΛF - 構造フロー
                with self.timer('lambda_F'):
                    lambda_F, lambda_F_mag = self._compute_lambda_F(positions_gpu)
                
                # 2. ΛFF - 二次構造
                with self.timer('lambda_FF'):
                    lambda_FF, lambda_FF_mag = self._compute_lambda_FF(lambda_F)
                
                # 3. ρT - テンション場
                with self.timer('rho_T'):
                    rho_T = self._compute_rho_T(positions_gpu, window_steps)
                
                # 4. Q_Λ - トポロジカルチャージ
                with self.timer('Q_lambda'):
                    Q_lambda, Q_cumulative = self._compute_Q_lambda(lambda_F, lambda_F_mag)
                
                # 5. σₛ - 構造同期率
                with self.timer('sigma_s'):
                    sigma_s = self._compute_sigma_s(md_features, window_steps)
                
                # 6. 構造的コヒーレンス
                with self.timer('coherence'):
                    coherence = self._compute_coherence(lambda_F, window_steps)
                
                # 結果をCPUに転送
                results = {
                    'lambda_F': self.to_cpu(lambda_F),
                    'lambda_F_mag': self.to_cpu(lambda_F_mag),
                    'lambda_FF': self.to_cpu(lambda_FF),
                    'lambda_FF_mag': self.to_cpu(lambda_FF_mag),
                    'rho_T': self.to_cpu(rho_T),
                    'Q_lambda': self.to_cpu(Q_lambda),
                    'Q_cumulative': self.to_cpu(Q_cumulative),
                    'sigma_s': self.to_cpu(sigma_s),
                    'structural_coherence': self.to_cpu(coherence)
                }
                
                # 統計情報出力
                self._print_statistics(results)
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Error in compute_lambda_structures: {e}")
            if self.is_gpu and "out of memory" in str(e).lower():
                logger.info("💡 Try reducing batch_size or use force_cpu=True")
            raise
    
    def _compute_lambda_F(self, positions: NDArray) -> Tuple[NDArray, NDArray]:
        """ΛF - 構造フロー計算"""
        # フレーム間の差分ベクトル
        lambda_F = self.xp.diff(positions, axis=0)
        
        # 大きさ（ノルム）
        lambda_F_mag = self.xp.linalg.norm(lambda_F, axis=1)
        
        return lambda_F, lambda_F_mag
    
    def _compute_lambda_FF(self, lambda_F: NDArray) -> Tuple[NDArray, NDArray]:
        """ΛFF - 二次構造フロー計算"""
        # 二次差分（加速度的な量）
        lambda_FF = self.xp.diff(lambda_F, axis=0)
        
        # 大きさ
        lambda_FF_mag = self.xp.linalg.norm(lambda_FF, axis=1)
        
        return lambda_FF, lambda_FF_mag
    
    def _compute_rho_T(self, positions: NDArray, window_steps: int) -> NDArray:
        """
        ρT - テンション場計算
        """
        n_frames = len(positions)
        
        # カスタムカーネルが使える場合
        if self.is_gpu and tension_field_kernel is not None:
            try:
                return tension_field_kernel(positions, window_steps)
            except Exception as e:
                logger.warning(f"Custom kernel failed: {e}, using fallback")
        
        # フォールバック実装（GPU/CPU両対応）
        rho_T = self.xp.zeros(n_frames)
        
        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            local_positions = positions[start:end]
            
            if len(local_positions) > 1:
                # 共分散行列のトレース
                centered = local_positions - self.xp.mean(local_positions, axis=0)
                cov = self.xp.cov(centered.T)
                rho_T[step] = self.xp.trace(cov)
        
        return rho_T
    
    def _compute_Q_lambda(self, 
                         lambda_F: NDArray, 
                         lambda_F_mag: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Q_Λ - トポロジカルチャージ計算
        """
        n_steps = len(lambda_F_mag)
        
        # カスタムカーネルが使える場合
        if self.is_gpu and topological_charge_kernel is not None:
            try:
                Q_lambda = topological_charge_kernel(lambda_F, lambda_F_mag)
                Q_cumulative = self.xp.cumsum(Q_lambda)
                return Q_lambda, Q_cumulative
            except Exception as e:
                logger.warning(f"Custom kernel failed: {e}, using fallback")
        
        # フォールバック実装
        Q_lambda = self.xp.zeros(n_steps)
        
        for step in range(1, n_steps):
            if lambda_F_mag[step] > 1e-10 and lambda_F_mag[step-1] > 1e-10:
                # 正規化ベクトル
                v1 = lambda_F[step-1] / lambda_F_mag[step-1]
                v2 = lambda_F[step] / lambda_F_mag[step]
                
                # 角度計算
                cos_angle = self.xp.clip(self.xp.dot(v1, v2), -1, 1)
                angle = self.xp.arccos(cos_angle)
                
                # 2D回転方向（符号付き角度）
                if len(v1) >= 2:  # 2D以上
                    cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                    signed_angle = angle if cross_z >= 0 else -angle
                else:
                    signed_angle = angle
                
                Q_lambda[step] = signed_angle / (2 * self.xp.pi)
        
        Q_cumulative = self.xp.cumsum(Q_lambda)
        
        return Q_lambda, Q_cumulative
    
    def _compute_sigma_s(self, 
                        md_features: Dict[str, np.ndarray],
                        window_steps: int) -> NDArray:
        """σₛ - 構造同期率計算"""
        # 必要な特徴量がない場合
        if 'rmsd' not in md_features or 'radius_of_gyration' not in md_features:
            n_frames = len(md_features.get('com_positions', []))
            return self.xp.zeros(n_frames)
        
        # GPU転送
        rmsd = self.to_gpu(md_features['rmsd'])
        rg = self.to_gpu(md_features['radius_of_gyration'])
        n_frames = len(rmsd)
        
        sigma_s = self.xp.zeros(n_frames)
        
        # スライディングウィンドウで相関計算
        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            
            if end - start > 2:  # 相関計算に最低3点必要
                local_rmsd = rmsd[start:end]
                local_rg = rg[start:end]
                
                # 標準偏差チェック
                std_rmsd = self.xp.std(local_rmsd)
                std_rg = self.xp.std(local_rg)
                
                if std_rmsd > 1e-10 and std_rg > 1e-10:
                    # 相関係数
                    corr_matrix = self.xp.corrcoef(
                        self.xp.stack([local_rmsd, local_rg])
                    )
                    sigma_s[step] = self.xp.abs(corr_matrix[0, 1])
        
        return sigma_s
    
    def _compute_coherence(self, 
                          lambda_F: NDArray,
                          window: int) -> NDArray:
        """構造的コヒーレンス計算"""
        n_frames = len(lambda_F)
        coherence = self.xp.zeros(n_frames)
        
        for i in range(window, n_frames - window):
            local_F = lambda_F[i-window:i+window]
            
            # 平均方向ベクトル
            mean_dir = self.xp.mean(local_F, axis=0)
            mean_norm = self.xp.linalg.norm(mean_dir)
            
            if mean_norm > 1e-10:
                mean_dir /= mean_norm
                
                # 各ベクトルとの内積（コサイン類似度）
                norms = self.xp.linalg.norm(local_F, axis=1)
                valid_mask = norms > 1e-10
                
                if self.xp.any(valid_mask):
                    normalized_F = local_F[valid_mask] / norms[valid_mask, self.xp.newaxis]
                    coherences = self.xp.dot(normalized_F, mean_dir)
                    coherence[i] = self.xp.mean(coherences)
        
        return coherence
    
    def _print_statistics(self, results: Dict[str, np.ndarray]):
        """統計情報を出力"""
        try:
            logger.info("📊 Lambda Structure Statistics:")
            
            for key in ['lambda_F_mag', 'lambda_FF_mag', 'rho_T', 'Q_cumulative', 'sigma_s']:
                if key in results and len(results[key]) > 0:
                    data = results[key]
                    logger.info(f"   {key}: min={np.min(data):.3e}, max={np.max(data):.3e}, "
                               f"mean={np.mean(data):.3e}, std={np.std(data):.3e}")
                
        except Exception as e:
            logger.warning(f"Failed to print statistics: {e}")
    
    def compute_adaptive_window_size(self,
                                   md_features: Dict[str, np.ndarray],
                                   lambda_structures: Dict[str, np.ndarray],
                                   n_frames: int,
                                   config: any) -> Dict[str, Union[int, float, dict]]:
        """
        適応的ウィンドウサイズ計算
        """
        try:
            base_window = int(n_frames * config.window_scale)
            
            # 変動性メトリクス計算
            volatility_metrics = {}
            
            # RMSD変動性
            if 'rmsd' in md_features:
                rmsd = self.to_gpu(md_features['rmsd'])
                mean_val = float(self.xp.mean(rmsd))
                if abs(mean_val) > 1e-10:
                    volatility_metrics['rmsd'] = float(self.xp.std(rmsd) / mean_val)
                else:
                    volatility_metrics['rmsd'] = 0.0
            
            # Lambda F変動性
            if 'lambda_F_mag' in lambda_structures:
                lf_mag = self.to_gpu(lambda_structures['lambda_F_mag'])
                mean_val = float(self.xp.mean(lf_mag))
                if abs(mean_val) > 1e-10:
                    volatility_metrics['lambda_f'] = float(self.xp.std(lf_mag) / mean_val)
                else:
                    volatility_metrics['lambda_f'] = 0.0
            
            # ρT安定性
            if 'rho_T' in lambda_structures:
                rho_t = self.to_gpu(lambda_structures['rho_T'])
                mean_val = float(self.xp.mean(rho_t))
                if abs(mean_val) > 1e-10:
                    volatility_metrics['rho_t'] = float(self.xp.std(rho_t) / mean_val)
                else:
                    volatility_metrics['rho_t'] = 0.0
            
            # スケールファクター計算
            scale_factor = 1.0
            
            if volatility_metrics.get('rmsd', 0) > 0.5:
                scale_factor *= 0.7
            elif volatility_metrics.get('rmsd', 0) < 0.1:
                scale_factor *= 1.5
            
            if volatility_metrics.get('lambda_f', 0) > 1.0:
                scale_factor *= 0.8
            elif volatility_metrics.get('lambda_f', 0) < 0.2:
                scale_factor *= 1.3
            
            # ウィンドウサイズ決定
            adaptive_window = int(base_window * scale_factor)
            adaptive_window = np.clip(adaptive_window, config.min_window, config.max_window)
            
            windows = {
                'primary': adaptive_window,
                'fast': max(config.min_window, adaptive_window // 2),
                'slow': min(config.max_window, adaptive_window * 2),
                'boundary': max(10, adaptive_window // 3),
                'scale_factor': scale_factor,
                'volatility_metrics': volatility_metrics
            }
            
            logger.info(f"🎯 Adaptive window sizes: primary={windows['primary']}, "
                       f"scale_factor={scale_factor:.2f}")
            
            return windows
            
        except Exception as e:
            logger.error(f"Failed to compute adaptive window size: {e}")
            # フォールバック
            return {
                'primary': 50,
                'fast': 25,
                'slow': 100,
                'boundary': 10,
                'scale_factor': 1.0,
                'volatility_metrics': {}
            }

# ===============================
# スタンドアロン関数（重複なし！）
# ===============================

def compute_lambda_structures_gpu(trajectory: np.ndarray,
                                md_features: Dict[str, np.ndarray],
                                window_steps: int,
                                config: Optional[LambdaStructureConfig] = None,
                                memory_manager: Optional[GPUMemoryManager] = None,
                                force_cpu: bool = False) -> Dict[str, np.ndarray]:
    """
    Lambda³構造計算のメイン関数
    
    Parameters
    ----------
    trajectory : np.ndarray
        MDトラジェクトリ
    md_features : Dict[str, np.ndarray]
        MD特徴量
    window_steps : int
        ウィンドウサイズ
    config : LambdaStructureConfig, optional
        設定
    memory_manager : GPUMemoryManager, optional
        メモリ管理
    force_cpu : bool
        強制的にCPUを使用
        
    Returns
    -------
    Dict[str, np.ndarray]
        Lambda構造
    """
    calculator = LambdaStructuresGPU(config, memory_manager, force_cpu=force_cpu)
    return calculator.compute_lambda_structures(trajectory, md_features, window_steps)

def compute_adaptive_window_size_gpu(md_features: Dict[str, np.ndarray],
                                   lambda_structures: Dict[str, np.ndarray],
                                   n_frames: int,
                                   config: any,
                                   force_cpu: bool = False) -> Dict[str, Union[int, float, dict]]:
    """適応的ウィンドウサイズ計算"""
    calculator = LambdaStructuresGPU(force_cpu=force_cpu)
    return calculator.compute_adaptive_window_size(
        md_features, lambda_structures, n_frames, config
    )

# ===============================
# ヘルパー関数（他モジュール用）
# ===============================

def compute_structural_coherence_gpu(lambda_F: np.ndarray,
                                   window: int = 50,
                                   force_cpu: bool = False) -> np.ndarray:
    """構造的コヒーレンス計算"""
    calculator = LambdaStructuresGPU(force_cpu=force_cpu)
    lambda_F_gpu = calculator.to_gpu(lambda_F)
    coherence = calculator._compute_coherence(lambda_F_gpu, window)
    return calculator.to_cpu(coherence)

def compute_local_fractal_dimension_gpu(q_cumulative: np.ndarray,
                                       window: int = 50,
                                       force_cpu: bool = False) -> np.ndarray:
    """局所フラクタル次元計算"""
    # boundary_detection_gpuモジュールに委譲
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU(force_cpu=force_cpu)
    return detector._compute_fractal_dimensions_gpu(q_cumulative, window)

def compute_coupling_strength_gpu(q_cumulative: np.ndarray,
                                 window: int = 50,
                                 force_cpu: bool = False) -> np.ndarray:
    """結合強度計算"""
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU(force_cpu=force_cpu)
    return detector._compute_coupling_strength_gpu(q_cumulative, window)

def compute_structural_entropy_gpu(rho_t: np.ndarray,
                                  window: int = 50,
                                  force_cpu: bool = False) -> np.ndarray:
    """構造エントロピー計算"""
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU(force_cpu=force_cpu)
    return detector._compute_structural_entropy_gpu(rho_t, window)

# ===============================
# エクスポート（重複なし！）
# ===============================
__all__ = [
    'LambdaStructuresGPU',
    'LambdaStructureConfig',
    'compute_lambda_structures_gpu',
    'compute_adaptive_window_size_gpu',
    'compute_structural_coherence_gpu',
    'compute_local_fractal_dimension_gpu',
    'compute_coupling_strength_gpu',
    'compute_structural_entropy_gpu',
]
