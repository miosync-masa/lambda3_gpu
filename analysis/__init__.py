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

# 🆕 フル解析パイプライン
from .run_full_analysis import (
    run_quantum_validation_pipeline,
)

# 🆕🆕 最強レポート生成機能！
from .maximum_report_generator import (
    generate_maximum_report_from_results,
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
    'evaluate_two_stage_performance',
    
    # フル解析パイプライン
    'run_quantum_validation_pipeline',
    
    # 🆕 最強レポート生成
    'generate_maximum_report_from_results',
]

__version__ = '1.2.0'  # バージョンアップ！

# ========================================
# 便利な一括実行関数
# ========================================

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

# ========================================
# 最強レポート生成の便利関数
# ========================================

def generate_max_report(results_or_path, **kwargs):
    """
    最強レポートを生成する超便利関数！
    
    Parameters
    ----------
    results_or_path : dict or str
        解析結果の辞書、またはトラジェクトリパス
    
    Examples
    --------
    # パターン1: 既存の結果から
    >>> results = run_full_analysis('traj.npy', 'meta.json')
    >>> report = generate_max_report(results)
    
    # パターン2: ファイルから一気に
    >>> report = generate_max_report('traj.npy', metadata_path='meta.json')
    """
    if isinstance(results_or_path, dict):
        # 既存の結果から
        return generate_maximum_report_from_results(
            lambda_result=results_or_path.get('lambda_result'),
            two_stage_result=results_or_path.get('two_stage_result'),
            quantum_events=results_or_path.get('quantum_events'),
            **kwargs
        )
    else:
        # ファイルから解析して最強レポート
        results = run_full_analysis(
            results_or_path,
            kwargs.pop('metadata_path', None),
            **kwargs
        )
        return generate_maximum_report_from_results(
            lambda_result=results['lambda_result'],
            two_stage_result=results.get('two_stage_result'),
            quantum_events=results.get('quantum_events'),
            **kwargs
        )

# ========================================
# ショートカット（さらに便利に）
# ========================================

def analyze(trajectory_path: str, metadata_path: str, **kwargs):
    """
    最も簡単な実行方法
    
    Examples
    --------
    >>> from lambda3_gpu.analysis import analyze
    >>> results = analyze('traj.npy', 'meta.json')
    """
    return run_full_analysis(trajectory_path, metadata_path, **kwargs)

def max_report(results):
    """
    超簡単な最強レポート生成！
    
    Examples
    --------
    >>> from lambda3_gpu.analysis import analyze, max_report
    >>> results = analyze('traj.npy', 'meta.json')
    >>> report = max_report(results)  # これだけ！
    """
    return generate_maximum_report_from_results(
        lambda_result=results.get('lambda_result'),
        two_stage_result=results.get('two_stage_result'),
        quantum_events=results.get('quantum_events')
    )
