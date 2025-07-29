"""
Lambda³ Structure Computation (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造の計算をGPUで超高速化！
NO TIME, NO PHYSICS, ONLY STRUCTURE... but FASTER! 🚀

by 環ちゃん
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.signal import savgol_filter as cp_savgol_filter
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..core import GPUBackend, GPUMemoryManager, GPUTimer
from ..core import tension_field_kernel, topological_charge_kernel
from ..types import ArrayType, NDArray

logger = logging.getLogger('lambda3_gpu.structures.lambda_structures')

# ===============================
# Configuration
# ===============================

@dataclass
class LambdaStructureConfig:
    """Lambda構造計算の設定"""
    use_mixed_precision: bool = False
    batch_size: Optional[int] = None
    cache_intermediates: bool = True
    profile: bool = False

# ===============================
# Lambda Structures GPU Class
# ===============================

class LambdaStructuresGPU(GPUBackend):
    """
    Lambda³構造計算のGPU実装クラス
    めちゃくちゃ速いよ〜！✨
    """
    
    def __init__(self, 
                 config: Optional[LambdaStructureConfig] = None,
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        config : LambdaStructureConfig
            計算設定
        memory_manager : GPUMemoryManager
            メモリ管理インスタンス
        """
        super().__init__(
            mixed_precision=config.use_mixed_precision if config else False,
            profile=config.profile if config else False,
            **kwargs
        )
        
        self.config = config or LambdaStructureConfig()
        self.memory_manager = memory_manager or GPUMemoryManager()
        self._cache = {} if self.config.cache_intermediates else None
        
    def compute_lambda_structures(self,
                                trajectory: np.ndarray,
                                md_features: Dict[str, np.ndarray],
                                window_steps: int) -> Dict[str, np.ndarray]:
        """
        Lambda³構造をGPUで計算
        
        Parameters
        ----------
        trajectory : np.ndarray
            トラジェクトリ (n_frames, n_atoms, 3)
        md_features : dict
            MD特徴量辞書
        window_steps : int
            ウィンドウサイズ
            
        Returns
        -------
        dict
            Lambda構造の辞書
        """
        with self.timer('compute_lambda_structures'):
            logger.info(f"🚀 GPU Computing Lambda³ structures (window={window_steps})")
            
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
    
    def _compute_lambda_F(self, positions: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """ΛF計算（GPU版）"""
        # フレーム間の差分
        lambda_F = self.xp.diff(positions, axis=0)
        
        # 大きさ
        lambda_F_mag = self.xp.linalg.norm(lambda_F, axis=1)
        
        return lambda_F, lambda_F_mag
    
    def _compute_lambda_FF(self, lambda_F: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """ΛFF計算（GPU版）"""
        # 二次差分
        lambda_FF = self.xp.diff(lambda_F, axis=0)
        
        # 大きさ
        lambda_FF_mag = self.xp.linalg.norm(lambda_FF, axis=1)
        
        return lambda_FF, lambda_FF_mag
    
    def _compute_rho_T(self, positions: cp.ndarray, window_steps: int) -> cp.ndarray:
        """ρT計算（カスタムカーネル使用）"""
        if self.is_gpu and HAS_GPU:
            # カスタムカーネル使用
            return tension_field_kernel(positions, window_steps)
        else:
            # CPU版フォールバック
            return self._compute_rho_T_cpu(positions, window_steps)
    
    def _compute_rho_T_cpu(self, positions: np.ndarray, window_steps: int) -> np.ndarray:
        """ρT計算（CPU版）"""
        n_frames = positions.shape[0]
        rho_T = np.zeros(n_frames)
        
        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            local_positions = positions[start:end]
            
            if len(local_positions) > 1:
                centered = local_positions - np.mean(local_positions, axis=0)
                cov = np.cov(centered.T)
                rho_T[step] = np.trace(cov)
        
        return rho_T
    
    def _compute_Q_lambda(self, 
                         lambda_F: cp.ndarray, 
                         lambda_F_mag: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Q_Λ計算（カスタムカーネル使用）"""
        if self.is_gpu and HAS_GPU:
            # カスタムカーネル使用
            Q_lambda = topological_charge_kernel(lambda_F, lambda_F_mag)
        else:
            # CPU版
            Q_lambda = self._compute_Q_lambda_cpu(lambda_F, lambda_F_mag)
        
        # 累積
        Q_cumulative = self.xp.cumsum(Q_lambda)
        
        return Q_lambda, Q_cumulative
    
    def _compute_Q_lambda_cpu(self, 
                             lambda_F: np.ndarray,
                             lambda_F_mag: np.ndarray) -> np.ndarray:
        """Q_Λ計算（CPU版）"""
        n_steps = len(lambda_F_mag)
        Q_lambda = np.zeros(n_steps + 1)
        
        for step in range(1, n_steps):
            if lambda_F_mag[step] > 1e-10 and lambda_F_mag[step-1] > 1e-10:
                v1 = lambda_F[step-1] / lambda_F_mag[step-1]
                v2 = lambda_F[step] / lambda_F_mag[step]
                
                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)
                
                # 2D回転方向
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                signed_angle = angle if cross_z >= 0 else -angle
                
                Q_lambda[step] = signed_angle / (2 * np.pi)
        
        return Q_lambda[:-1]
    
    def _compute_sigma_s(self, 
                        md_features: Dict[str, np.ndarray],
                        window_steps: int) -> cp.ndarray:
        """σₛ計算（GPU版）"""
        n_frames = len(md_features.get('rmsd', []))
        
        if 'rmsd' not in md_features or 'radius_of_gyration' not in md_features:
            return self.zeros(n_frames)
        
        # GPU転送
        rmsd_gpu = self.to_gpu(md_features['rmsd'])
        rg_gpu = self.to_gpu(md_features['radius_of_gyration'])
        
        sigma_s = self.zeros(n_frames)
        
        # スライディングウィンドウで相関計算
        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            
            if end - start > 1:
                local_rmsd = rmsd_gpu[start:end]
                local_rg = rg_gpu[start:end]
                
                std_rmsd = self.xp.std(local_rmsd)
                std_rg = self.xp.std(local_rg)
                
                if std_rmsd > 1e-10 and std_rg > 1e-10:
                    # GPU上で相関計算
                    corr_matrix = self.xp.corrcoef(
                        self.xp.stack([local_rmsd, local_rg])
                    )
                    sigma_s[step] = self.xp.abs(corr_matrix[0, 1])
        
        return sigma_s
    
    def _compute_coherence(self, 
                          lambda_F: cp.ndarray,
                          window: int) -> cp.ndarray:
        """構造的コヒーレンス計算（GPU版）"""
        n_frames = len(lambda_F)
        coherence = self.zeros(n_frames + 1)
        
        for i in range(window, n_frames - window):
            local_F = lambda_F[i-window:i+window]
            
            # 平均方向
            mean_dir = self.xp.mean(local_F, axis=0)
            mean_norm = self.xp.linalg.norm(mean_dir)
            
            if mean_norm > 1e-10:
                mean_dir /= mean_norm
                
                # 各ベクトルとの内積
                norms = self.xp.linalg.norm(local_F, axis=1)
                valid_mask = norms > 1e-10
                
                if self.xp.any(valid_mask):
                    normalized_F = local_F[valid_mask] / norms[valid_mask, self.xp.newaxis]
                    coherences = self.xp.dot(normalized_F, mean_dir)
                    coherence[i] = self.xp.mean(coherences)
        
        return coherence[:-1]
    
    def _print_statistics(self, results: Dict[str, np.ndarray]):
        """統計情報を出力"""
        logger.info(f"   ΛF magnitude range: {np.min(results['lambda_F_mag']):.3e} - "
                   f"{np.max(results['lambda_F_mag']):.3e}")
        logger.info(f"   ρT (tension) range: {np.min(results['rho_T']):.3f} - "
                   f"{np.max(results['rho_T']):.3f}")
        logger.info(f"   Q_Λ cumulative drift: {results['Q_cumulative'][-1]:.3f}")
        logger.info(f"   Average σₛ (sync): {np.mean(results['sigma_s']):.3f}")
    
    def compute_adaptive_window_size(self,
                                   md_features: Dict[str, np.ndarray],
                                   lambda_structures: Dict[str, np.ndarray],
                                   n_frames: int,
                                   config: any) -> Dict[str, Union[int, float, dict]]:
        """
        適応的ウィンドウサイズ計算（GPU版）
        """
        with self.timer('adaptive_window_size'):
            base_window = int(n_frames * config.window_scale)
            
            # GPU転送
            if 'rmsd' in md_features:
                rmsd_gpu = self.to_gpu(md_features['rmsd'])
                rmsd_volatility = float(self.xp.std(rmsd_gpu) / (self.xp.mean(rmsd_gpu) + 1e-10))
            else:
                rmsd_volatility = 0.0
            
            if 'lambda_F_mag' in lambda_structures:
                lf_mag_gpu = self.to_gpu(lambda_structures['lambda_F_mag'])
                lambda_f_volatility = float(self.xp.std(lf_mag_gpu) / (self.xp.mean(lf_mag_gpu) + 1e-10))
            else:
                lambda_f_volatility = 0.0
            
            if 'rho_T' in lambda_structures:
                rho_t_gpu = self.to_gpu(lambda_structures['rho_T'])
                
                # 局所分散計算
                test_window = min(50, n_frames // 20)
                local_vars = []
                
                for i in range(0, len(rho_t_gpu) - test_window, test_window // 2):
                    local_var = float(self.xp.var(rho_t_gpu[i:i+test_window]))
                    local_vars.append(local_var)
                
                if local_vars:
                    rho_t_stability = np.std(local_vars) / (np.mean(local_vars) + 1e-10)
                else:
                    rho_t_stability = 1.0
            else:
                rho_t_stability = 1.0
            
            # スケールファクター計算
            scale_factor = 1.0
            
            if rmsd_volatility > 0.5:
                scale_factor *= 0.7
            elif rmsd_volatility < 0.1:
                scale_factor *= 1.5
            
            if lambda_f_volatility > 1.0:
                scale_factor *= 0.8
            elif lambda_f_volatility < 0.2:
                scale_factor *= 1.3
            
            if rho_t_stability > 1.5:
                scale_factor *= 0.85
            
            # ウィンドウサイズ決定
            adaptive_window = int(base_window * scale_factor)
            adaptive_window = np.clip(adaptive_window, config.min_window, config.max_window)
            
            windows = {
                'primary': adaptive_window,
                'fast': max(config.min_window, adaptive_window // 2),
                'slow': min(config.max_window, adaptive_window * 2),
                'boundary': max(10, adaptive_window // 3),
                'scale_factor': scale_factor,
                'volatility_metrics': {
                    'rmsd': rmsd_volatility,
                    'lambda_f': lambda_f_volatility,
                    'rho_t': rho_t_stability
                }
            }
            
            logger.info(f"\n🎯 Adaptive window sizes (GPU computed):")
            logger.info(f"   Primary: {windows['primary']} frames")
            logger.info(f"   Scale factor: {scale_factor:.2f}")
            
            return windows

# ===============================
# Helper Functions (GPU版)
# ===============================

def compute_structural_coherence_gpu(lambda_F: Union[np.ndarray, cp.ndarray],
                                   window: int,
                                   backend: Optional[GPUBackend] = None) -> np.ndarray:
    """構造的コヒーレンス計算（スタンドアロン版）"""
    if backend is None:
        backend = GPUBackend()
    
    lambda_F_gpu = backend.to_gpu(lambda_F)
    n_frames = len(lambda_F_gpu)
    coherence = backend.zeros(n_frames)
    
    for i in range(window, n_frames - window):
        local_F = lambda_F_gpu[i-window:i+window]
        mean_dir = backend.xp.mean(local_F, axis=0)
        mean_norm = backend.xp.linalg.norm(mean_dir)
        
        if mean_norm > 1e-10:
            mean_dir /= mean_norm
            norms = backend.xp.linalg.norm(local_F, axis=1)
            valid_mask = norms > 1e-10
            
            if backend.xp.any(valid_mask):
                normalized_F = local_F[valid_mask] / norms[valid_mask, backend.xp.newaxis]
                coherences = backend.xp.dot(normalized_F, mean_dir)
                coherence[i] = backend.xp.mean(coherences)
    
    return backend.to_cpu(coherence)

def compute_local_fractal_dimension_gpu(series: Union[np.ndarray, cp.ndarray],
                                      window: int,
                                      backend: Optional[GPUBackend] = None) -> np.ndarray:
    """局所フラクタル次元計算（GPU版）"""
    if backend is None:
        backend = GPUBackend()
    
    series_gpu = backend.to_gpu(series)
    n = len(series_gpu)
    dims = backend.ones(n)
    
    scales = backend.xp.array([2, 4, 8, 16])
    
    for i in range(window, n - window):
        local = series_gpu[i-window:i+window]
        counts = backend.zeros(len(scales))
        
        for j, scale in enumerate(scales):
            boxes = 0
            for k in range(0, len(local)-scale, scale):
                segment = local[k:k+scale]
                if backend.xp.max(segment) - backend.xp.min(segment) > 1e-10:
                    boxes += 1
            counts[j] = max(boxes, 1)
        
        if backend.xp.max(counts) > backend.xp.min(counts):
            log_scales = backend.xp.log(scales)
            log_counts = backend.xp.log(counts)
            
            # 線形回帰
            A = backend.xp.vstack([log_scales, backend.xp.ones(len(log_scales))]).T
            slope, _ = backend.xp.linalg.lstsq(A, log_counts, rcond=None)[0]
            dims[i] = max(0.5, min(2.0, -slope))
    
    return backend.to_cpu(dims)

def compute_coupling_strength_gpu(Q_cumulative: Union[np.ndarray, cp.ndarray],
                                window: int,
                                backend: Optional[GPUBackend] = None) -> np.ndarray:
    """結合強度計算（GPU版）"""
    if backend is None:
        backend = GPUBackend()
    
    Q_gpu = backend.to_gpu(Q_cumulative)
    coupling = backend.ones_like(Q_gpu)
    
    for i in range(window, len(Q_gpu) - window):
        local_Q = Q_gpu[i-window:i+window]
        var = backend.xp.var(local_Q)
        if var > 1e-10:
            coupling[i] = 1.0 / (1.0 + var)
    
    return backend.to_cpu(coupling)

def compute_structural_entropy_gpu(rho_T: Union[np.ndarray, cp.ndarray],
                                 window: int,
                                 backend: Optional[GPUBackend] = None) -> np.ndarray:
    """構造エントロピー計算（GPU版）"""
    if backend is None:
        backend = GPUBackend()
    
    rho_T_gpu = backend.to_gpu(rho_T)
    entropy = backend.zeros_like(rho_T_gpu)
    
    for i in range(window, len(rho_T_gpu) - window):
        local_rho = rho_T_gpu[i-window:i+window]
        
        if backend.xp.sum(local_rho) > 0:
            # 正規化
            p = local_rho / backend.xp.sum(local_rho)
            # シャノンエントロピー
            entropy[i] = -backend.xp.sum(p * backend.xp.log(p + 1e-10))
    
    return backend.to_cpu(entropy)

# ===============================
# Main Function
# ===============================

def compute_lambda_structures_gpu(trajectory: np.ndarray,
                                md_features: Dict[str, np.ndarray],
                                window_steps: int,
                                config: Optional[LambdaStructureConfig] = None,
                                memory_manager: Optional[GPUMemoryManager] = None) -> Dict[str, np.ndarray]:
    """
    Lambda³構造計算のメイン関数（GPU版）
    
    既存のAPIと互換性を保つラッパー関数
    """
    calculator = LambdaStructuresGPU(config, memory_manager)
    return calculator.compute_lambda_structures(trajectory, md_features, window_steps)

def compute_adaptive_window_size_gpu(md_features: Dict[str, np.ndarray],
                                   lambda_structures: Dict[str, np.ndarray],
                                   n_frames: int,
                                   config: any) -> Dict[str, Union[int, float, dict]]:
    """適応的ウィンドウサイズ計算のラッパー"""
    calculator = LambdaStructuresGPU()
    return calculator.compute_adaptive_window_size(
        md_features, lambda_structures, n_frames, config
    )
