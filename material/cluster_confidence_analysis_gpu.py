#!/usr/bin/env python3
"""
Statistical Confidence Analysis for Material Clusters (GPU Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

材料クラスターの統計的信頼性をGPUで高速評価！
歪み分布、転位密度、損傷確率の信頼区間を計算！💎

残基版をベースに材料解析に特化

by 環ちゃん - Material Edition
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass

# GPU imports
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# Local imports
from ..types import ArrayType, NDArray
from ..core import GPUBackend, GPUMemoryManager, handle_gpu_errors, profile_gpu

logger = logging.getLogger('lambda3_gpu.material.confidence')

# ===============================
# Data Classes (Material版)
# ===============================

@dataclass
class MaterialConfidenceResult:
    """材料統計解析の結果"""
    statistic_name: str           # 統計量の名前
    point_estimate: float         # 点推定値
    ci_lower: float              # 信頼区間下限
    ci_upper: float              # 信頼区間上限
    confidence_level: float       # 信頼水準
    n_bootstrap: int             # ブートストラップ回数
    p_value: Optional[float] = None     # p値
    standard_error: Optional[float] = None  # 標準誤差
    bias: Optional[float] = None         # バイアス
    
    # 材料特有の追加フィールド
    material_property: Optional[str] = None  # 'strain', 'coordination', 'damage'
    failure_probability: Optional[float] = None  # 破壊確率
    weibull_modulus: Optional[float] = None  # ワイブル係数
    reliability_index: Optional[float] = None  # 信頼性指標
    
    @property
    def ci_width(self) -> float:
        """信頼区間の幅"""
        return self.ci_upper - self.ci_lower
    
    @property
    def is_significant(self) -> bool:
        """統計的有意性（0を含まない）"""
        return self.ci_lower > 0 or self.ci_upper < 0
    
    @property
    def is_critical(self) -> bool:
        """材料的に臨界状態か（歪み>1%など）"""
        if self.material_property == 'strain':
            return self.point_estimate > 0.01  # 1%歪み
        elif self.material_property == 'damage':
            return self.point_estimate > 0.5  # 50%損傷
        return False

# ===============================
# CUDA Kernels (Material特化版)
# ===============================

# ワイブル分布パラメータ推定カーネル
WEIBULL_PARAMETER_KERNEL = r'''
extern "C" __global__
void estimate_weibull_parameters_kernel(
    const float* __restrict__ failure_data,  // (n_samples,) 破壊強度データ
    float* __restrict__ shape_param,        // ワイブル形状パラメータ(m)
    float* __restrict__ scale_param,        // ワイブル尺度パラメータ(σ0)
    const int n_samples
) {
    // 簡易最尤推定（1スレッドで実行）
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // データをソート済みと仮定
    // ln(ln(1/(1-F))) vs ln(σ) の線形回帰
    
    float sum_x = 0.0f, sum_y = 0.0f;
    float sum_xx = 0.0f, sum_xy = 0.0f;
    
    for (int i = 0; i < n_samples; i++) {
        float F = (float)(i + 0.5) / n_samples;  // 中央ランク法
        float sigma = failure_data[i];
        
        if (sigma > 0 && F > 0 && F < 1) {
            float x = logf(sigma);
            float y = logf(-logf(1.0f - F));
            
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
    }
    
    // 回帰係数
    float n = (float)n_samples;
    float m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    float b = (sum_y - m * sum_x) / n;
    
    *shape_param = m;  // ワイブル係数
    *scale_param = expf(-b / m);  // 特性強度
}
'''

# 歪みエネルギー信頼区間カーネル
STRAIN_ENERGY_CONFIDENCE_KERNEL = r'''
extern "C" __global__
void compute_strain_energy_confidence_kernel(
    const float* __restrict__ strain_tensor,  // (n_samples, 9) フラット化
    const float* __restrict__ elastic_modulus,  // 弾性係数
    float* __restrict__ energy_samples,      // (n_bootstrap,)
    const int* __restrict__ bootstrap_indices,  // (n_bootstrap, n_samples)
    const int n_samples,
    const int n_bootstrap
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= n_bootstrap) return;
    
    const int* indices = &bootstrap_indices[b * n_samples];
    
    // 歪みエネルギー密度の計算: U = 0.5 * σ:ε = 0.5 * E * ε:ε
    float total_energy = 0.0f;
    
    for (int i = 0; i < n_samples; i++) {
        int idx = indices[i];
        const float* strain = &strain_tensor[idx * 9];
        
        // von Mises相当歪み
        float e11 = strain[0], e22 = strain[4], e33 = strain[8];
        float e12 = strain[1], e23 = strain[5], e13 = strain[2];
        
        float hydro = (e11 + e22 + e33) / 3.0f;
        float dev11 = e11 - hydro;
        float dev22 = e22 - hydro;
        float dev33 = e33 - hydro;
        
        float von_mises_sq = dev11*dev11 + dev22*dev22 + dev33*dev33 +
                            2.0f * (e12*e12 + e23*e23 + e13*e13);
        
        // 歪みエネルギー密度
        float energy = 0.5f * (*elastic_modulus) * von_mises_sq;
        total_energy += energy;
    }
    
    energy_samples[b] = total_energy / n_samples;
}
'''

# 損傷累積確率カーネル
DAMAGE_ACCUMULATION_KERNEL = r'''
extern "C" __global__
void compute_damage_accumulation_kernel(
    const float* __restrict__ stress_history,   // (n_frames,) 応力履歴
    const float* __restrict__ n_cycles,        // (n_frames,) サイクル数
    float* __restrict__ damage_samples,        // (n_bootstrap,)
    const int* __restrict__ bootstrap_indices,  // (n_bootstrap, n_frames)
    const float fatigue_strength,              // 疲労限度
    const float ultimate_strength,             // 極限強度
    const int n_frames,
    const int n_bootstrap
) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= n_bootstrap) return;
    
    const int* indices = &bootstrap_indices[b * n_frames];
    
    // Palmgren-Miner則による損傷累積
    float total_damage = 0.0f;
    
    for (int i = 0; i < n_frames; i++) {
        int idx = indices[i];
        float stress = stress_history[idx];
        float cycles = n_cycles[idx];
        
        if (stress > fatigue_strength) {
            // S-N曲線: N = A * S^(-b)
            // 簡易版: Basquin則
            float stress_ratio = stress / ultimate_strength;
            float N_f = 1e6f * powf(stress_ratio, -3.0f);  // 破壊までのサイクル数
            
            // 損傷増分
            float damage_increment = cycles / N_f;
            total_damage += damage_increment;
        }
    }
    
    damage_samples[b] = total_damage;
}
'''

# ===============================
# Material Confidence Analyzer GPU Class
# ===============================

class MaterialConfidenceAnalyzerGPU(GPUBackend):
    """
    材料クラスターの統計的信頼性解析のGPU実装
    
    歪み分布、転位密度、損傷確率の信頼区間を高速計算！
    ワイブル統計、疲労解析も対応！
    """
    
    def __init__(self,
                 n_bootstrap: int = 500,  # 材料は少なめでOK
                 confidence_level: float = 0.95,
                 random_seed: int = 42,
                 
                 # 材料パラメータ
                 elastic_modulus: float = 210.0,  # GPa (鋼鉄)
                 poisson_ratio: float = 0.3,
                 yield_strength: float = 1.5,  # GPa (SUJ2)
                 ultimate_strength: float = 2.0,  # GPa
                 fatigue_strength: float = 0.7,  # GPa
                 fracture_toughness: float = 30.0,  # MPa√m
                 
                 memory_manager: Optional[GPUMemoryManager] = None,
                 **kwargs):
        """
        Parameters
        ----------
        n_bootstrap : int
            ブートストラップサンプル数
        confidence_level : float
            信頼水準（0.95 = 95%）
        random_seed : int
            乱数シード
        elastic_modulus : float
            弾性係数 (GPa)
        poisson_ratio : float
            ポアソン比
        yield_strength : float
            降伏強度 (GPa)
        ultimate_strength : float
            極限強度 (GPa)
        fatigue_strength : float
            疲労限度 (GPa)
        fracture_toughness : float
            破壊靭性 (MPa√m)
        """
        super().__init__(**kwargs)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        # 材料定数
        self.elastic_modulus = elastic_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_strength = yield_strength
        self.ultimate_strength = ultimate_strength
        self.fatigue_strength = fatigue_strength
        self.fracture_toughness = fracture_toughness
        
        # 派生定数
        self.shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
        self.bulk_modulus = elastic_modulus / (3 * (1 - 2 * poisson_ratio))
        
        self.memory_manager = memory_manager or GPUMemoryManager()
        
        # 乱数生成器
        if self.is_gpu:
            self.rng = cp.random.RandomState(seed=random_seed)
        else:
            self.rng = np.random.RandomState(seed=random_seed)
        
        # カーネルコンパイル
        if self.is_gpu and HAS_GPU:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """カスタムカーネルをコンパイル"""
        try:
            self.weibull_kernel = cp.RawKernel(
                WEIBULL_PARAMETER_KERNEL, 'estimate_weibull_parameters_kernel'
            )
            self.strain_energy_kernel = cp.RawKernel(
                STRAIN_ENERGY_CONFIDENCE_KERNEL, 'compute_strain_energy_confidence_kernel'
            )
            self.damage_kernel = cp.RawKernel(
                DAMAGE_ACCUMULATION_KERNEL, 'compute_damage_accumulation_kernel'
            )
            logger.debug("Material confidence kernels compiled successfully")
        except Exception as e:
            logger.warning(f"Failed to compile custom kernels: {e}")
            self.weibull_kernel = None
            self.strain_energy_kernel = None
            self.damage_kernel = None
    
    @handle_gpu_errors
    @profile_gpu
    def analyze_material_reliability(self,
                                    causality_chains: List[Tuple[int, int, float]],
                                    cluster_data: Dict[int, Dict[str, np.ndarray]],
                                    analysis_type: str = 'strain',
                                    n_bootstrap: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        材料因果連鎖の信頼性を解析（メインエントリーポイント）
        
        Parameters
        ----------
        causality_chains : List[Tuple[int, int, float]]
            因果連鎖のリスト [(from_cluster, to_cluster, strength), ...]
        cluster_data : Dict[int, Dict[str, np.ndarray]]
            クラスターID -> {'strain': array, 'coordination': array, 'damage': array}
        analysis_type : str
            解析タイプ ('strain', 'coordination', 'damage')
        n_bootstrap : int, optional
            ブートストラップ回数
            
        Returns
        -------
        List[Dict[str, Any]]
            各因果ペアの材料信頼性解析結果
        """
        if n_bootstrap is None:
            n_bootstrap = min(self.n_bootstrap, 100)  # 高速化
        
        results = []
        
        logger.info(f"⚙️ Analyzing material reliability for {len(causality_chains)} pairs")
        
        # 各因果ペアを解析
        for pair_idx, (from_cluster, to_cluster, strength) in enumerate(causality_chains):
            if from_cluster not in cluster_data or to_cluster not in cluster_data:
                continue
            
            try:
                # 解析タイプに応じて処理
                if analysis_type == 'strain':
                    result = self._analyze_strain_reliability(
                        cluster_data[from_cluster].get('strain'),
                        cluster_data[to_cluster].get('strain'),
                        strength,
                        n_bootstrap
                    )
                elif analysis_type == 'coordination':
                    result = self._analyze_coordination_reliability(
                        cluster_data[from_cluster].get('coordination'),
                        cluster_data[to_cluster].get('coordination'),
                        strength,
                        n_bootstrap
                    )
                elif analysis_type == 'damage':
                    result = self._analyze_damage_reliability(
                        cluster_data[from_cluster].get('damage'),
                        cluster_data[to_cluster].get('damage'),
                        strength,
                        n_bootstrap
                    )
                else:
                    # デフォルト：異常スコアの相関
                    result = self._analyze_general_reliability(
                        cluster_data[from_cluster].get('anomaly', np.zeros(10)),
                        cluster_data[to_cluster].get('anomaly', np.zeros(10)),
                        strength,
                        n_bootstrap
                    )
                
                # 結果を構築
                result_dict = {
                    'pair_index': pair_idx,
                    'from_cluster': from_cluster,
                    'to_cluster': to_cluster,
                    'original_strength': float(strength),
                    'analysis_type': analysis_type,
                    **result
                }
                
                results.append(result_dict)
                
            except Exception as e:
                logger.warning(f"Analysis failed for pair ({from_cluster}, {to_cluster}): {e}")
                continue
        
        # サマリー
        if results:
            n_critical = sum(1 for r in results if r.get('is_critical', False))
            logger.info(f"   {len(results)} pairs analyzed, {n_critical} in critical state")
        
        return results
    
    def _analyze_strain_reliability(self,
                                   strain_i: Optional[np.ndarray],
                                   strain_j: Optional[np.ndarray],
                                   strength: float,
                                   n_bootstrap: int) -> Dict[str, Any]:
        """歪み信頼性解析"""
        if strain_i is None or strain_j is None:
            return {}
        
        # von Mises相当歪み計算
        von_mises_i = self._compute_von_mises_strain(strain_i)
        von_mises_j = self._compute_von_mises_strain(strain_j)
        
        # 歪み相関の信頼区間
        conf_result = self.bootstrap_strain_confidence(
            von_mises_i, von_mises_j, n_bootstrap
        )
        
        # 降伏確率
        yield_prob_i = self._compute_yield_probability(von_mises_i)
        yield_prob_j = self._compute_yield_probability(von_mises_j)
        
        return {
            'von_mises_strain_i': float(self.xp.mean(von_mises_i)),
            'von_mises_strain_j': float(self.xp.mean(von_mises_j)),
            'strain_correlation': float(conf_result.point_estimate),
            'ci_lower': float(conf_result.ci_lower),
            'ci_upper': float(conf_result.ci_upper),
            'yield_probability_i': float(yield_prob_i),
            'yield_probability_j': float(yield_prob_j),
            'is_significant': bool(conf_result.is_significant),
            'is_critical': bool(max(yield_prob_i, yield_prob_j) > 0.5),
            'material_property': 'strain'
        }
    
    def _analyze_coordination_reliability(self,
                                         coord_i: Optional[np.ndarray],
                                         coord_j: Optional[np.ndarray],
                                         strength: float,
                                         n_bootstrap: int) -> Dict[str, Any]:
        """配位数信頼性解析"""
        if coord_i is None or coord_j is None:
            return {}
        
        # 配位数欠陥
        defect_i = self.xp.abs(coord_i - 12.0)  # FCC理想値
        defect_j = self.xp.abs(coord_j - 12.0)
        
        # 転位密度の推定
        dislocation_density_i = self._estimate_dislocation_density(defect_i)
        dislocation_density_j = self._estimate_dislocation_density(defect_j)
        
        # 相関の信頼区間
        conf_result = self.bootstrap_correlation_confidence(
            defect_i, defect_j, n_bootstrap
        )
        
        return {
            'mean_defect_i': float(self.xp.mean(defect_i)),
            'mean_defect_j': float(self.xp.mean(defect_j)),
            'dislocation_density_i': float(dislocation_density_i),
            'dislocation_density_j': float(dislocation_density_j),
            'defect_correlation': float(conf_result.point_estimate),
            'ci_lower': float(conf_result.ci_lower),
            'ci_upper': float(conf_result.ci_upper),
            'is_significant': bool(conf_result.is_significant),
            'is_critical': bool(max(dislocation_density_i, dislocation_density_j) > 1e12),  # 10^12/cm^2
            'material_property': 'coordination'
        }
    
    def _analyze_damage_reliability(self,
                                   damage_i: Optional[np.ndarray],
                                   damage_j: Optional[np.ndarray],
                                   strength: float,
                                   n_bootstrap: int) -> Dict[str, Any]:
        """損傷信頼性解析"""
        if damage_i is None or damage_j is None:
            return {}
        
        # ワイブル解析
        weibull_params_i = self._fit_weibull_distribution(damage_i)
        weibull_params_j = self._fit_weibull_distribution(damage_j)
        
        # 破壊確率
        failure_prob_i = self._compute_failure_probability(damage_i)
        failure_prob_j = self._compute_failure_probability(damage_j)
        
        # 相関の信頼区間
        conf_result = self.bootstrap_correlation_confidence(
            damage_i, damage_j, n_bootstrap
        )
        
        return {
            'mean_damage_i': float(self.xp.mean(damage_i)),
            'mean_damage_j': float(self.xp.mean(damage_j)),
            'weibull_modulus_i': float(weibull_params_i['shape']),
            'weibull_modulus_j': float(weibull_params_j['shape']),
            'failure_probability_i': float(failure_prob_i),
            'failure_probability_j': float(failure_prob_j),
            'damage_correlation': float(conf_result.point_estimate),
            'ci_lower': float(conf_result.ci_lower),
            'ci_upper': float(conf_result.ci_upper),
            'is_significant': bool(conf_result.is_significant),
            'is_critical': bool(max(failure_prob_i, failure_prob_j) > 0.1),  # 10%破壊確率
            'material_property': 'damage'
        }
    
    def _analyze_general_reliability(self,
                                    data_i: np.ndarray,
                                    data_j: np.ndarray,
                                    strength: float,
                                    n_bootstrap: int) -> Dict[str, Any]:
        """一般的な信頼性解析"""
        conf_result = self.bootstrap_correlation_confidence(
            data_i, data_j, n_bootstrap
        )
        
        return {
            'correlation': float(conf_result.point_estimate),
            'ci_lower': float(conf_result.ci_lower),
            'ci_upper': float(conf_result.ci_upper),
            'is_significant': bool(conf_result.is_significant),
            'material_property': 'general'
        }
    
    @handle_gpu_errors
    @profile_gpu
    def bootstrap_strain_confidence(self,
                                   strain_x: np.ndarray,
                                   strain_y: np.ndarray,
                                   n_bootstrap: Optional[int] = None) -> MaterialConfidenceResult:
        """
        歪み相関のブートストラップ信頼区間（材料版）
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        
        # GPU転送
        strain_x_gpu = self.to_gpu(strain_x)
        strain_y_gpu = self.to_gpu(strain_y)
        
        # 元の相関
        original_corr = float(self.xp.corrcoef(strain_x_gpu, strain_y_gpu)[0, 1])
        
        # ブートストラップ
        bootstrap_corrs = self._bootstrap_statistic(
            strain_x_gpu, strain_y_gpu,
            lambda x, y: self.xp.corrcoef(x, y)[0, 1],
            n_bootstrap
        )
        
        # 信頼区間
        ci_lower, ci_upper = self._compute_confidence_interval(
            bootstrap_corrs, self.confidence_level
        )
        
        # 信頼性指標（構造信頼性理論）
        reliability_index = self._compute_reliability_index(
            strain_x_gpu, strain_y_gpu
        )
        
        result = MaterialConfidenceResult(
            statistic_name='strain_correlation',
            point_estimate=original_corr,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            confidence_level=self.confidence_level,
            n_bootstrap=n_bootstrap,
            standard_error=float(self.xp.std(bootstrap_corrs)),
            bias=float(self.xp.mean(bootstrap_corrs)) - original_corr,
            material_property='strain',
            reliability_index=float(reliability_index)
        )
        
        return result
    
    @handle_gpu_errors
    def estimate_weibull_parameters(self,
                                   failure_data: np.ndarray) -> Dict[str, float]:
        """
        ワイブル分布パラメータ推定
        
        Parameters
        ----------
        failure_data : np.ndarray
            破壊強度データ
            
        Returns
        -------
        dict
            {'shape': m, 'scale': σ0}
        """
        # GPU転送
        data_gpu = self.to_gpu(failure_data)
        
        # ソート
        sorted_data = self.xp.sort(data_gpu)
        
        if self.is_gpu and self.weibull_kernel is not None:
            # カスタムカーネル
            shape = cp.zeros(1, dtype=cp.float32)
            scale = cp.zeros(1, dtype=cp.float32)
            
            self.weibull_kernel(
                (1,), (1,),
                (sorted_data, shape, scale, len(sorted_data))
            )
            
            return {
                'shape': float(shape[0]),
                'scale': float(scale[0])
            }
        else:
            # 汎用版：最小二乗法
            n = len(sorted_data)
            F = self.xp.arange(1, n + 1) / (n + 1)  # 経験分布関数
            
            # ln(ln(1/(1-F))) vs ln(σ)
            y = self.xp.log(-self.xp.log(1 - F))
            x = self.xp.log(sorted_data)
            
            # 線形回帰
            A = self.xp.vstack([x, self.xp.ones(n)]).T
            m, c = self.xp.linalg.lstsq(A, y, rcond=None)[0]
            
            return {
                'shape': float(m),  # ワイブル係数
                'scale': float(self.xp.exp(-c / m))  # 特性強度
            }
    
    # ========================================
    # ヘルパーメソッド（材料特有）
    # ========================================
    
    def _compute_von_mises_strain(self, strain_tensor: np.ndarray) -> np.ndarray:
        """von Mises相当歪み計算"""
        strain_gpu = self.to_gpu(strain_tensor)
        
        if strain_gpu.ndim == 3:  # (n_samples, 3, 3)
            # 主歪み
            e11 = strain_gpu[:, 0, 0]
            e22 = strain_gpu[:, 1, 1]
            e33 = strain_gpu[:, 2, 2]
            e12 = strain_gpu[:, 0, 1]
            e23 = strain_gpu[:, 1, 2]
            e13 = strain_gpu[:, 0, 2]
        else:  # フラット化済み
            e11 = strain_gpu[0]
            e22 = strain_gpu[4]
            e33 = strain_gpu[8]
            e12 = strain_gpu[1]
            e23 = strain_gpu[5]
            e13 = strain_gpu[2]
        
        # 体積歪み
        hydro = (e11 + e22 + e33) / 3.0
        
        # 偏差歪み
        dev11 = e11 - hydro
        dev22 = e22 - hydro
        dev33 = e33 - hydro
        
        # von Mises相当歪み
        von_mises = self.xp.sqrt(
            2.0/3.0 * (dev11**2 + dev22**2 + dev33**2 + 
                      2.0 * (e12**2 + e23**2 + e13**2))
        )
        
        return self.to_cpu(von_mises)
    
    def _compute_yield_probability(self, von_mises_strain: np.ndarray) -> float:
        """降伏確率計算"""
        strain_gpu = self.to_gpu(von_mises_strain)
        
        # von Mises応力
        von_mises_stress = self.elastic_modulus * strain_gpu
        
        # 降伏判定
        yield_mask = von_mises_stress > self.yield_strength
        
        return float(self.xp.mean(yield_mask))
    
    def _estimate_dislocation_density(self, coord_defect: np.ndarray) -> float:
        """転位密度推定（簡易版）"""
        defect_gpu = self.to_gpu(coord_defect)
        
        # 経験式：ρ = α * (Δn)^2
        # α ≈ 10^14 /cm^2 per unit defect
        alpha = 1e14
        mean_defect = self.xp.mean(defect_gpu)
        
        return float(alpha * mean_defect ** 2)
    
    def _fit_weibull_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """ワイブル分布フィッティング"""
        return self.estimate_weibull_parameters(data)
    
    def _compute_failure_probability(self, damage: np.ndarray) -> float:
        """破壊確率計算（損傷則ベース）"""
        damage_gpu = self.to_gpu(damage)
        
        # Palmgren-Miner則：D > 1で破壊
        failure_mask = damage_gpu > 1.0
        
        return float(self.xp.mean(failure_mask))
    
    def _compute_reliability_index(self,
                                  strain_x: Union[np.ndarray, cp.ndarray],
                                  strain_y: Union[np.ndarray, cp.ndarray]) -> float:
        """構造信頼性指標β計算"""
        # FORM/SORM簡易版
        # β = (μ_R - μ_S) / sqrt(σ_R^2 + σ_S^2)
        
        mean_x = self.xp.mean(strain_x)
        mean_y = self.xp.mean(strain_y)
        std_x = self.xp.std(strain_x)
        std_y = self.xp.std(strain_y)
        
        # 限界状態関数 g = R - S
        # R: 抵抗（降伏歪み）, S: 荷重効果（実歪み）
        R = self.yield_strength / self.elastic_modulus  # 降伏歪み
        S = (mean_x + mean_y) / 2  # 平均歪み
        
        beta = (R - S) / self.xp.sqrt(std_x**2 + std_y**2 + 1e-10)
        
        return float(beta)
    
    # ========================================
    # 継承メソッド（基本機能）
    # ========================================
    
    def bootstrap_correlation_confidence(self,
                                        series_x: np.ndarray,
                                        series_y: np.ndarray,
                                        n_bootstrap: Optional[int] = None) -> MaterialConfidenceResult:
        """相関係数の信頼区間（材料版拡張）"""
        result = self.bootstrap_strain_confidence(series_x, series_y, n_bootstrap)
        return result
    
    def _bootstrap_statistic(self,
                           x_gpu: Union[np.ndarray, cp.ndarray],
                           y_gpu: Union[np.ndarray, cp.ndarray],
                           statistic_func: Callable,
                           n_bootstrap: int) -> Union[np.ndarray, cp.ndarray]:
        """汎用ブートストラップ"""
        n_samples = len(x_gpu)
        bootstrap_stats = self.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            indices = self.rng.randint(0, n_samples, size=n_samples)
            x_resampled = x_gpu[indices]
            y_resampled = y_gpu[indices]
            
            try:
                stat_value = statistic_func(x_resampled, y_resampled)
                if not self.xp.isnan(stat_value):
                    bootstrap_stats[i] = stat_value
            except:
                bootstrap_stats[i] = 0.0
        
        return bootstrap_stats
    
    def _compute_confidence_interval(self,
                                   bootstrap_stats: Union[np.ndarray, cp.ndarray],
                                   confidence_level: float) -> Tuple[float, float]:
        """信頼区間計算"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        valid_stats = bootstrap_stats[~self.xp.isnan(bootstrap_stats)]
        
        if len(valid_stats) == 0:
            return 0.0, 0.0
        
        ci_lower = self.xp.percentile(valid_stats, lower_percentile)
        ci_upper = self.xp.percentile(valid_stats, upper_percentile)
        
        return float(ci_lower), float(ci_upper)

# ===============================
# Standalone Functions (Material版)
# ===============================

def analyze_material_reliability_gpu(causality_chains: List[Tuple[int, int, float]],
                                    cluster_data: Dict[int, Dict[str, np.ndarray]],
                                    analysis_type: str = 'strain',
                                    n_bootstrap: int = 100,
                                    **kwargs) -> List[Dict[str, Any]]:
    """材料信頼性解析のスタンドアロン関数"""
    analyzer = MaterialConfidenceAnalyzerGPU(n_bootstrap=n_bootstrap, **kwargs)
    return analyzer.analyze_material_reliability(
        causality_chains, cluster_data, analysis_type, n_bootstrap
    )

def bootstrap_strain_confidence_gpu(strain_x: np.ndarray,
                                   strain_y: np.ndarray,
                                   n_bootstrap: int = 500,
                                   **kwargs) -> MaterialConfidenceResult:
    """歪み相関の信頼区間計算"""
    analyzer = MaterialConfidenceAnalyzerGPU(n_bootstrap=n_bootstrap, **kwargs)
    return analyzer.bootstrap_strain_confidence(strain_x, strain_y)

def estimate_weibull_parameters_gpu(failure_data: np.ndarray,
                                   **kwargs) -> Dict[str, float]:
    """ワイブルパラメータ推定"""
    analyzer = MaterialConfidenceAnalyzerGPU(**kwargs)
    return analyzer.estimate_weibull_parameters(failure_data)

def compute_material_reliability_index(strain_data: np.ndarray,
                                      stress_data: Optional[np.ndarray] = None,
                                      material_properties: Optional[Dict] = None,
                                      **kwargs) -> float:
    """構造信頼性指標の計算"""
    if material_properties:
        analyzer = MaterialConfidenceAnalyzerGPU(**material_properties, **kwargs)
    else:
        analyzer = MaterialConfidenceAnalyzerGPU(**kwargs)
    
    if stress_data is not None:
        # 応力-歪み関係から計算
        strain_gpu = analyzer.to_gpu(strain_data)
        stress_gpu = analyzer.to_gpu(stress_data)
        return analyzer._compute_reliability_index(strain_gpu, stress_gpu)
    else:
        # 歪みのみから推定
        strain_gpu = analyzer.to_gpu(strain_data)
        return analyzer._compute_reliability_index(strain_gpu, strain_gpu)
