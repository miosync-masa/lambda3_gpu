"""
Lambda³ GPU版可視化モジュール
GPU解析結果の高度な可視化ツール
"""

# 遅延インポートに変更
def __getattr__(name):
    """遅延インポートで循環参照を回避"""
    if name == 'Lambda3VisualizerGPU':
        from .plot_results_gpu import Lambda3VisualizerGPU
        return Lambda3VisualizerGPU
    elif name == 'visualize_residue_analysis':
        from .plot_results_gpu import visualize_residue_analysis
        return visualize_residue_analysis
    elif name == 'CausalityVisualizerGPU':
        from .causality_viz_gpu import CausalityVisualizerGPU
        return CausalityVisualizerGPU
    elif name == 'create_network_comparison':
        from .causality_viz_gpu import create_network_comparison
        return create_network_comparison
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'Lambda3VisualizerGPU',
    'visualize_residue_analysis',
    'CausalityVisualizerGPU',
    'create_network_comparison'
]

__version__ = '1.0.0'
