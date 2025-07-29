"""
Lambda³ GPU版解析モジュール
MD軌道の完全GPU化解析パイプライン
"""

from .md_lambda3_detector_gpu import (
    MDLambda3DetectorGPU,
    MDConfig,
    MDLambda3Result
)
from .two_stage_analyzer_gpu import (
    TwoStageAnalyzerGPU,
    ResidueAnalysisConfig,
    TwoStageLambda3Result,
    ResidueEvent,
    ResidueLevelAnalysis,
    perform_two_stage_analysis_gpu
)
from .evaluation_gpu import (
    PerformanceEvaluatorGPU,
    PerformanceMetrics,
    EventDetectionResult,
    evaluate_two_stage_performance
)

__all__ = [
    # メイン検出器
    'MDLambda3DetectorGPU',
    'MDConfig',
    'MDLambda3Result',
    
    # 2段階解析
    'TwoStageAnalyzerGPU',
    'ResidueAnalysisConfig',
    'TwoStageLambda3Result',
    'ResidueEvent',
    'ResidueLevelAnalysis',
    'perform_two_stage_analysis_gpu',
    
    # 評価
    'PerformanceEvaluatorGPU',
    'PerformanceMetrics',
    'EventDetectionResult',
    'evaluate_two_stage_performance'
]

__version__ = '1.0.0'
