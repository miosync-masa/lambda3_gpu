"""
Lambda³ GPU Residue-Level Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

残基レベルの詳細解析をGPUで超高速化！💕
因果関係とか、ネットワーク解析とか、全部速いよ〜！

Components:
    - ResidueStructuresGPU: 残基レベル構造計算
    - ResidueNetworkGPU: ネットワーク解析
    - CausalityAnalyzerGPU: 因果関係解析
    - ConfidenceAnalyzerGPU: 信頼区間解析
"""

from .residue_structures_gpu import (
    ResidueStructuresGPU,
    compute_residue_structures_gpu,
    compute_residue_lambda_f_gpu,
    compute_residue_rho_t_gpu,
    compute_residue_coupling_gpu,
    ResidueStructureResult
)

from .residue_network_gpu import (
    ResidueNetworkGPU,
    NetworkAnalysisResult,
    analyze_residue_network_gpu,
    compute_spatial_constraints_gpu,
    filter_causal_network_gpu,
    build_propagation_paths_gpu
)

from .causality_analysis_gpu import (
    CausalityAnalyzerGPU,
    CausalityResult,
    calculate_structural_causality_gpu,
    compute_lagged_correlation_gpu,
    detect_causal_pairs_gpu,
    compute_transfer_entropy_gpu
)

from .confidence_analysis_gpu import (
    ConfidenceAnalyzerGPU,
    ConfidenceResult,
    bootstrap_correlation_confidence_gpu,
    permutation_test_gpu,
    compute_confidence_intervals_gpu,
    evaluate_statistical_significance_gpu
)

__all__ = [
    # Residue Structures
    'ResidueStructuresGPU',
    'compute_residue_structures_gpu',
    'compute_residue_lambda_f_gpu',
    'compute_residue_rho_t_gpu',
    'compute_residue_coupling_gpu',
    'ResidueStructureResult',
    
    # Network Analysis
    'ResidueNetworkGPU',
    'NetworkAnalysisResult',
    'analyze_residue_network_gpu',
    'compute_spatial_constraints_gpu',
    'filter_causal_network_gpu',
    'build_propagation_paths_gpu',
    
    # Causality Analysis
    'CausalityAnalyzerGPU',
    'CausalityResult',
    'calculate_structural_causality_gpu',
    'compute_lagged_correlation_gpu',
    'detect_causal_pairs_gpu',
    'compute_transfer_entropy_gpu',
    
    # Confidence Analysis
    'ConfidenceAnalyzerGPU',
    'ConfidenceResult',
    'bootstrap_correlation_confidence_gpu',
    'permutation_test_gpu',
    'compute_confidence_intervals_gpu',
    'evaluate_statistical_significance_gpu'
]

# 初期化メッセージ
import logging
logger = logging.getLogger('lambda3_gpu.residue')
logger.debug("Lambda³ GPU Residue module initialized")
