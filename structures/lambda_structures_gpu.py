"""
Lambda³ Structure Computation (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造の計算をGPUで超高速化！
NO TIME, NO PHYSICS, ONLY STRUCTURE... but FASTER! 🚀

⚡ 2025/01/15 完全リファクタリング版 by 環ちゃん
- GPU/CPU切り替えの安全性を完全保証
- エラーハンドリング強化
- ヘルパー関数で重複コード削減
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy.signal import savgol_filter as cp_savgol_filter
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cp_savgol_filter = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager
from ..core import tension_field_kernel, topological_charge_kernel

logger = logging.getLogger('lambda3_gpu.structures.lambda_structures')

# ===============================
# 安全性ヘルパー関数（新規追加）
# ===============================

def safe_is_gpu_array(arr: Any) -> bool:
    """
    安全にGPU配列かチェック
    
    環ちゃんの特製安全チェッカー！💪
    """
    return HAS_GPU and cp is not None and isinstance(arr, cp.ndarray)

def safe_to_cpu(arr: Any) -> np.ndarray:
    """
    安全にCPU配列に変換
    
    GPU配列でもNumPy配列でも、リストでも何でも来い！
    """
    if safe_is_gpu_array(arr):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def safe_to_gpu(arr: Any, dtype: Optional[np.dtype] = None) -> NDArray:
    """
    安全にGPU配列に変換（GPU利用可能な場合のみ）
    
    GPU使えない時は自動的にNumPy配列として返すよ！
    """
    if HAS_GPU and cp is not None:
        if safe_is_gpu_array(arr):
            return arr.astype(dtype) if dtype else arr
        return cp.asarray(arr, dtype=dtype)
    else:
        return np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)

def get_array_module(arr: Any) -> Any:
    """
    配列に適したモジュール（np or cp）を返す
    
    これで xp の混乱を防げる！
    """
    if safe_is_gpu_array(arr):
        return cp
    return np

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
    safe_mode: bool = True  # 新規追加：安全モード（デフォルトON）

# ===============================
# Lambda Structures GPU Class
# ===============================

class LambdaStructuresGPU(GPUBackend):
    """
    Lambda³構造計算のGPU実装クラス
    めちゃくちゃ速いし、今度は安全だよ〜！✨
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
        
        # ⚡ 重要: 初期化後に状態を完全検証
        self._validate_and_fix_gpu_state()
        
    def _validate_and_fix_gpu_state(self):
        """
        GPU/CPU状態の整合性を完全検証して修正
        
        環ちゃんの完璧な状態チェッカー！🔍
        """
        # GPU利用可能性の再確認
        if self.is_gpu:
            if not HAS_GPU or cp is None:
                logger.warning("⚠️ GPU mode requested but CuPy not available! Falling back to CPU mode.")
                self.is_gpu = False
                self.xp = np
            else:
                self.xp = cp
                logger.info("✅ GPU mode confirmed: Using CuPy")
        else:
            self.xp = np
            logger.info("✅ CPU mode confirmed: Using NumPy")
        
        # デバッグ情報
        logger.debug(f"State validation complete: is_gpu={self.is_gpu}, xp={self.xp.__name__}, HAS_GPU={HAS_GPU}")
    
    def to_gpu(self, array: Any, dtype: Optional[np.dtype] = None) -> NDArray:
        """
        安全なGPU転送（オーバーライド）
        """
        if self.config.safe_mode:
            return safe_to_gpu(array, dtype)
        return super().to_gpu(array, dtype)
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """
        安全なCPU転送（オーバーライド）
        """
        if self.config.safe_mode:
            return safe_to_cpu(array)
        return super().to_cpu(array)
    
    def compute_lambda_structures(self,
                                trajectory: np.ndarray,
                                md_features: Dict[str, np.ndarray],
                                window_steps: int) -> Dict[str, np.ndarray]:
        """
        Lambda³構造をGPUで計算（エラーハンドリング強化版）
        """
        try:
            with self.timer('compute_lambda_structures'):
                logger.info(f"🚀 Computing Lambda³ structures (window={window_steps}, mode={'GPU' if self.is_gpu else 'CPU'})")
                
                # 入力検証
                if trajectory is None or len(trajectory) == 0:
                    raise ValueError("Empty trajectory provided")
                
                # GPU転送（安全版）
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
                
                # 結果をCPUに転送（安全版）
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
                logger.info("💡 Hint: Try reducing batch_size or switch to CPU mode")
            raise
    
    def _compute_lambda_F(self, positions: NDArray) -> Tuple[NDArray, NDArray]:
        """ΛF計算（安全版）"""
        xp = get_array_module(positions)
        
        # フレーム間の差分
        lambda_F = xp.diff(positions, axis=0)
        
        # 大きさ
        lambda_F_mag = xp.linalg.norm(lambda_F, axis=1)
        
        return lambda_F, lambda_F_mag
    
    def _compute_lambda_FF(self, lambda_F: NDArray) -> Tuple[NDArray, NDArray]:
        """ΛFF計算（安全版）"""
        xp = get_array_module(lambda_F)
        
        # 二次差分
        lambda_FF = xp.diff(lambda_F, axis=0)
        
        # 大きさ
        lambda_FF_mag = xp.linalg.norm(lambda_FF, axis=1)
        
        return lambda_FF, lambda_FF_mag
    
    def _compute_rho_T(self, positions: NDArray, window_steps: int) -> NDArray:
        """
        ρT計算（完全安全版）
        
        GPU/CPUの切り替えを完全に安全に！
        """
        if self.is_gpu and HAS_GPU and cp is not None:
            try:
                # GPU版：カスタムカーネル使用
                if not safe_is_gpu_array(positions):
                    positions = cp.asarray(positions)
                
                rho_T = tension_field_kernel(positions, window_steps)
                
                # 返り値の型確認
                if not safe_is_gpu_array(rho_T):
                    logger.warning("tension_field_kernel returned CPU array, converting to GPU")
                    rho_T = cp.asarray(rho_T)
                
                return rho_T
                
            except Exception as e:
                logger.warning(f"GPU kernel failed: {e}, falling back to CPU")
                positions = safe_to_cpu(positions)
                return self._compute_rho_T_cpu(positions, window_steps)
        else:
            # CPU版フォールバック
            positions = safe_to_cpu(positions)
            return self._compute_rho_T_cpu(positions, window_steps)
    
    def _compute_rho_T_cpu(self, positions: np.ndarray, window_steps: int) -> np.ndarray:
        """ρT計算（CPU版・完全安全）"""
        # 入力を確実にNumPy配列に
        positions = safe_to_cpu(positions)
        
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
                         lambda_F: NDArray, 
                         lambda_F_mag: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Q_Λ計算（完全安全版）
        
        どんな入力が来ても大丈夫！💪
        """
        if self.is_gpu and HAS_GPU and cp is not None:
            try:
                # GPU版：入力を安全に変換
                if not safe_is_gpu_array(lambda_F):
                    lambda_F = cp.asarray(lambda_F)
                if not safe_is_gpu_array(lambda_F_mag):
                    lambda_F_mag = cp.asarray(lambda_F_mag)
                
                # カスタムカーネル使用
                Q_lambda = topological_charge_kernel(lambda_F, lambda_F_mag)
                
                # 返り値チェック
                if not safe_is_gpu_array(Q_lambda):
                    logger.warning("topological_charge_kernel returned CPU array")
                    Q_lambda = cp.asarray(Q_lambda)
                
                # GPU上で累積計算
                Q_cumulative = cp.cumsum(Q_lambda)
                
                return Q_lambda, Q_cumulative
                
            except Exception as e:
                logger.warning(f"GPU kernel failed: {e}, falling back to CPU")
                lambda_F = safe_to_cpu(lambda_F)
                lambda_F_mag = safe_to_cpu(lambda_F_mag)
                return self._compute_Q_lambda_cpu_safe(lambda_F, lambda_F_mag)
        else:
            # CPU版：入力を安全に変換
            lambda_F = safe_to_cpu(lambda_F)
            lambda_F_mag = safe_to_cpu(lambda_F_mag)
            return self._compute_Q_lambda_cpu_safe(lambda_F, lambda_F_mag)
    
    def _compute_Q_lambda_cpu_safe(self, 
                                   lambda_F: np.ndarray,
                                   lambda_F_mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Q_Λ計算（CPU版・完全安全）"""
        # 入力を確実にNumPy配列に
        lambda_F = safe_to_cpu(lambda_F)
        lambda_F_mag = safe_to_cpu(lambda_F_mag)
        
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
        
        Q_lambda_result = Q_lambda[:-1]
        Q_cumulative = np.cumsum(Q_lambda_result)
        
        return Q_lambda_result, Q_cumulative
    
    def _compute_sigma_s(self, 
                        md_features: Dict[str, np.ndarray],
                        window_steps: int) -> NDArray:
        """σₛ計算（安全版）"""
        n_frames = len(md_features.get('rmsd', []))
        
        if 'rmsd' not in md_features or 'radius_of_gyration' not in md_features:
            xp = cp if (self.is_gpu and HAS_GPU) else np
            return xp.zeros(n_frames)
        
        # 安全なGPU転送
        rmsd_gpu = self.to_gpu(md_features['rmsd'])
        rg_gpu = self.to_gpu(md_features['radius_of_gyration'])
        
        xp = get_array_module(rmsd_gpu)
        sigma_s = xp.zeros(n_frames)
        
        # スライディングウィンドウで相関計算
        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            
            if end - start > 1:
                local_rmsd = rmsd_gpu[start:end]
                local_rg = rg_gpu[start:end]
                
                std_rmsd = xp.std(local_rmsd)
                std_rg = xp.std(local_rg)
                
                if std_rmsd > 1e-10 and std_rg > 1e-10:
                    try:
                        # GPU上で相関計算
                        corr_matrix = xp.corrcoef(
                            xp.stack([local_rmsd, local_rg])
                        )
                        sigma_s[step] = xp.abs(corr_matrix[0, 1])
                    except Exception as e:
                        logger.debug(f"Correlation calculation failed at step {step}: {e}")
                        sigma_s[step] = 0.0
        
        return sigma_s
    
    def _compute_coherence(self, 
                          lambda_F: NDArray,
                          window: int) -> NDArray:
        """構造的コヒーレンス計算（安全版）"""
        xp = get_array_module(lambda_F)
        n_frames = len(lambda_F)
        coherence = xp.zeros(n_frames + 1)
        
        for i in range(window, n_frames - window):
            local_F = lambda_F[i-window:i+window]
            
            # 平均方向
            mean_dir = xp.mean(local_F, axis=0)
            mean_norm = xp.linalg.norm(mean_dir)
            
            if mean_norm > 1e-10:
                mean_dir /= mean_norm
                
                # 各ベクトルとの内積
                norms = xp.linalg.norm(local_F, axis=1)
                valid_mask = norms > 1e-10
                
                if xp.any(valid_mask):
                    normalized_F = local_F[valid_mask] / norms[valid_mask, xp.newaxis]
                    coherences = xp.dot(normalized_F, mean_dir)
                    coherence[i] = xp.mean(coherences)
        
        return coherence[:-1]
    
    def _print_statistics(self, results: Dict[str, np.ndarray]):
        """統計情報を出力（エラーハンドリング付き）"""
        try:
            logger.info("📊 Lambda Structure Statistics:")
            
            if 'lambda_F_mag' in results and len(results['lambda_F_mag']) > 0:
                logger.info(f"   ΛF magnitude range: {np.min(results['lambda_F_mag']):.3e} - "
                           f"{np.max(results['lambda_F_mag']):.3e}")
            
            if 'rho_T' in results and len(results['rho_T']) > 0:
                logger.info(f"   ρT (tension) range: {np.min(results['rho_T']):.3f} - "
                           f"{np.max(results['rho_T']):.3f}")
            
            if 'Q_cumulative' in results and len(results['Q_cumulative']) > 0:
                logger.info(f"   Q_Λ cumulative drift: {results['Q_cumulative'][-1]:.3f}")
            
            if 'sigma_s' in results and len(results['sigma_s']) > 0:
                logger.info(f"   Average σₛ (sync): {np.mean(results['sigma_s']):.3f}")
                
        except Exception as e:
            logger.warning(f"Failed to print statistics: {e}")
    
    def compute_adaptive_window_size(self,
                                   md_features: Dict[str, np.ndarray],
                                   lambda_structures: Dict[str, np.ndarray],
                                   n_frames: int,
                                   config: any) -> Dict[str, Union[int, float, dict]]:
        """
        適応的ウィンドウサイズ計算（完全安全版）
        """
        try:
            with self.timer('adaptive_window_size'):
                base_window = int(n_frames * config.window_scale)
                
                # RMSD変動性
                rmsd_volatility = 0.0
                if 'rmsd' in md_features:
                    rmsd_gpu = self.to_gpu(md_features['rmsd'])
                    xp = get_array_module(rmsd_gpu)
                    mean_val = xp.mean(rmsd_gpu)
                    if abs(mean_val) > 1e-10:
                        rmsd_volatility = float(xp.std(rmsd_gpu) / mean_val)
                
                # Lambda F変動性
                lambda_f_volatility = 0.0
                if 'lambda_F_mag' in lambda_structures:
                    lf_mag_gpu = self.to_gpu(lambda_structures['lambda_F_mag'])
                    xp = get_array_module(lf_mag_gpu)
                    mean_val = xp.mean(lf_mag_gpu)
                    if abs(mean_val) > 1e-10:
                        lambda_f_volatility = float(xp.std(lf_mag_gpu) / mean_val)
                
                # ρT安定性
                rho_t_stability = 1.0
                if 'rho_T' in lambda_structures:
                    rho_t_gpu = self.to_gpu(lambda_structures['rho_T'])
                    xp = get_array_module(rho_t_gpu)
                    
                    # 局所分散計算
                    test_window = min(50, n_frames // 20)
                    local_vars = []
                    
                    for i in range(0, len(rho_t_gpu) - test_window, test_window // 2):
                        local_var = float(xp.var(rho_t_gpu[i:i+test_window]))
                        local_vars.append(local_var)
                    
                    if local_vars:
                        mean_var = np.mean(local_vars)
                        if abs(mean_var) > 1e-10:
                            rho_t_stability = np.std(local_vars) / mean_var
                
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
                
                logger.info(f"\n🎯 Adaptive window sizes:")
                logger.info(f"   Primary: {windows['primary']} frames")
                logger.info(f"   Scale factor: {scale_factor:.2f}")
                
                return windows
                
        except Exception as e:
            logger.error(f"Failed to compute adaptive window size: {e}")
            # フォールバック値を返す
            return {
                'primary': config.min_window if hasattr(config, 'min_window') else 50,
                'fast': 25,
                'slow': 100,
                'boundary': 10,
                'scale_factor': 1.0,
                'volatility_metrics': {}
            }

# ===============================
# Helper Functions (安全版)
# ===============================

def compute_structural_coherence_gpu(lambda_F: NDArray,
                                   window: int,
                                   backend: Optional[GPUBackend] = None) -> np.ndarray:
    """構造的コヒーレンス計算（スタンドアロン安全版）"""
    if backend is None:
        backend = GPUBackend()
    
    lambda_F_gpu = backend.to_gpu(lambda_F) if backend.is_gpu else lambda_F
    xp = get_array_module(lambda_F_gpu)
    n_frames = len(lambda_F_gpu)
    coherence = xp.zeros(n_frames)
    
    for i in range(window, n_frames - window):
        local_F = lambda_F_gpu[i-window:i+window]
        mean_dir = xp.mean(local_F, axis=0)
        mean_norm = xp.linalg.norm(mean_dir)
        
        if mean_norm > 1e-10:
            mean_dir /= mean_norm
            norms = xp.linalg.norm(local_F, axis=1)
            valid_mask = norms > 1e-10
            
            if xp.any(valid_mask):
                normalized_F = local_F[valid_mask] / norms[valid_mask, xp.newaxis]
                coherences = xp.dot(normalized_F, mean_dir)
                coherence[i] = xp.mean(coherences)
    
    return safe_to_cpu(coherence)

def compute_local_fractal_dimension_gpu(series: NDArray,
                                      window: int,
                                      backend: Optional[GPUBackend] = None) -> np.ndarray:
    """局所フラクタル次元計算（安全版）"""
    if backend is None:
        backend = GPUBackend()
    
    series_gpu = backend.to_gpu(series) if backend.is_gpu else series
    xp = get_array_module(series_gpu)
    n = len(series_gpu)
    dims = xp.ones(n)
    
    scales = xp.array([2, 4, 8, 16])
    
    for i in range(window, n - window):
        local = series_gpu[i-window:i+window]
        counts = xp.zeros(len(scales))
        
        for j, scale in enumerate(scales):
            boxes = 0
            for k in range(0, len(local)-scale, scale):
                segment = local[k:k+scale]
                if xp.max(segment) - xp.min(segment) > 1e-10:
                    boxes += 1
            counts[j] = max(boxes, 1)
        
        if xp.max(counts) > xp.min(counts):
            log_scales = xp.log(scales)
            log_counts = xp.log(counts + 1e-10)  # ゼロ除算防止
            
            # 線形回帰
            A = xp.vstack([log_scales, xp.ones(len(log_scales))]).T
            try:
                slope, _ = xp.linalg.lstsq(A, log_counts, rcond=None)[0]
                dims[i] = max(0.5, min(2.0, -slope))
            except:
                dims[i] = 1.5  # デフォルト値
    
    return safe_to_cpu(dims)

# ===============================
# Main Function
# ===============================
def compute_lambda_structures_gpu(trajectory: np.ndarray,
                                md_features: Dict[str, np.ndarray],
                                window_steps: int,
                                config: Optional[LambdaStructureConfig] = None,
                                memory_manager: Optional[GPUMemoryManager] = None) -> Dict[str, np.ndarray]:
    """
    Lambda³構造計算のメイン関数（完全安全版）
    
    既存のAPIと互換性を保つラッパー関数
    """
    try:
        calculator = LambdaStructuresGPU(config, memory_manager)
        return calculator.compute_lambda_structures(trajectory, md_features, window_steps)
    except Exception as e:
        logger.error(f"Lambda structure computation failed: {e}")
        raise

def compute_adaptive_window_size_gpu(md_features: Dict[str, np.ndarray],
                                   lambda_structures: Dict[str, np.ndarray],
                                   n_frames: int,
                                   config: any) -> Dict[str, Union[int, float, dict]]:
    """適応的ウィンドウサイズ計算のラッパー（安全版）"""
    try:
        calculator = LambdaStructuresGPU()
        return calculator.compute_adaptive_window_size(
            md_features, lambda_structures, n_frames, config
        )
    except Exception as e:
        logger.error(f"Adaptive window size computation failed: {e}")
        # フォールバック
        return {
            'primary': 50,
            'fast': 25, 
            'slow': 100,
            'boundary': 10,
            'scale_factor': 1.0,
            'volatility_metrics': {}
        }

def compute_structural_coherence_gpu(lambda_F: np.ndarray,
                                    window: int = 50) -> np.ndarray:
    """構造的コヒーレンス計算のラッパー"""
    computer = LambdaStructuresGPU()
    lambda_F_gpu = computer.to_gpu(lambda_F)
    coherence = computer._compute_coherence(lambda_F_gpu, window)
    return computer.to_cpu(coherence)

def compute_local_fractal_dimension_gpu(q_cumulative: np.ndarray,
                                       window: int = 50) -> np.ndarray:
    """局所フラクタル次元計算のラッパー"""
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU()
    return detector._compute_fractal_dimensions_gpu(q_cumulative, window)

def compute_coupling_strength_gpu(q_cumulative: np.ndarray,
                                 window: int = 50) -> np.ndarray:
    """結合強度計算のラッパー"""
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU()
    return detector._compute_coupling_strength_gpu(q_cumulative, window)

def compute_structural_entropy_gpu(rho_t: np.ndarray,
                                  window: int = 50) -> np.ndarray:
    """構造エントロピー計算のラッパー"""
    from ..detection.boundary_detection_gpu import BoundaryDetectorGPU
    detector = BoundaryDetectorGPU()
    return detector._compute_structural_entropy_gpu(rho_t, window)

# エクスポート更新
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
