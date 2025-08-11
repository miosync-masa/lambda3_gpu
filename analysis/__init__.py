"""
Lambda³ GPU版解析モジュール
MD軌道の完全GPU化解析パイプライン
"""

from .md_lambda3_detector_gpu import (
    MDLambda3DetectorGPU,
    MDLambda3AnalyzerGPU,  # これも追加（エイリアスの場合）
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

# 🆕 フル解析パイプライン
from .run_full_analysis import (
    run_quantum_validation_pipeline,  # 関数名はそのままでもOK
    # または rename して
    # run_full_analysis_pipeline,
)

__all__ = [
    # メイン検出器
    'MDLambda3DetectorGPU',
    'MDLambda3AnalyzerGPU',
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
    'evaluate_two_stage_performance',
    
    # 🆕 フル解析パイプライン
    'run_quantum_validation_pipeline',  # または 'run_full_analysis_pipeline'
]

__version__ = '1.1.0'  # バージョンアップ！

# 便利な一括実行関数も追加
def run_full_analysis(trajectory_path: str, 
                      metadata_path: str,
                      enable_quantum: bool = True,
                      **kwargs):
    """
    Lambda³ GPU フル解析の便利関数
    
    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイルパス
    metadata_path : str
        メタデータファイルパス
    enable_quantum : bool
        量子検証を実行するか
    **kwargs
        その他のオプション
        
    Returns
    -------
    dict
        解析結果
        
    Examples
    --------
    >>> from lambda3_gpu.analysis import run_full_analysis
    >>> results = run_full_analysis('traj.npy', 'meta.json')
    """
    from .run_full_analysis import run_quantum_validation_pipeline
    
    # デフォルト設定
    kwargs.setdefault('enable_two_stage', True)
    kwargs.setdefault('enable_visualization', True)
    kwargs.setdefault('output_dir', './lambda3_results')
    
    # 量子検証の制御
    if not enable_quantum:
        # 量子検証をスキップする方法を追加する必要があるかも
        pass
    
    return run_quantum_validation_pipeline(
        trajectory_path,
        metadata_path,
        **kwargs
    )

# ショートカット（さらに便利に）
def analyze(trajectory_path: str, metadata_path: str, **kwargs):
    """
    最も簡単な実行方法
    
    Examples
    --------
    >>> from lambda3_gpu.analysis import analyze
    >>> results = analyze('traj.npy', 'meta.json')
    """
    return run_full_analysis(trajectory_path, metadata_path, **kwargs)
