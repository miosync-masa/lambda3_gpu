"""
Quantum Validation Module for Lambda³ GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

量子検証機能のメインパッケージ
- CHSH不等式検証（async strong bonds ベース）
- エンタングルメント証人
- 3種類のトンネリング検出
- 量子コヒーレンス解析

by 環ちゃん & 紅莉栖さん & ご主人さま 💕
"""

from .quantum_validation_gpu import (
    # メインクラス
    QuantumValidationGPU,
    
    # データクラス
    QuantumMetrics,
    QuantumCascadeEvent,
    
    # 型定義（必要なら）
    ArrayType,
)

# バージョン情報
__version__ = '2.0.0'

# 公開するAPI
__all__ = [
    'QuantumValidationGPU',
    'QuantumMetrics', 
    'QuantumCascadeEvent',
]

# 依存関係チェック（オプション）
def check_dependencies():
    """依存関係のチェックとステータス表示"""
    import_status = {}
    
    # CuPy（GPU計算）
    try:
        import cupy as cp
        if cp.cuda.is_available():
            import_status['cupy'] = f"✅ Available (CUDA {cp.cuda.runtime.runtimeGetVersion()})"
        else:
            import_status['cupy'] = "⚠️ Installed but no GPU detected"
    except ImportError:
        import_status['cupy'] = "❌ Not installed (will use CPU mode)"
    
    # NumPy（必須）
    try:
        import numpy as np
        import_status['numpy'] = f"✅ {np.__version__}"
    except ImportError:
        import_status['numpy'] = "❌ Not installed (REQUIRED)"
    
    # Matplotlib（可視化用）
    try:
        import matplotlib
        import_status['matplotlib'] = f"✅ {matplotlib.__version__}"
    except ImportError:
        import_status['matplotlib'] = "⚠️ Not installed (optional, for visualization)"
    
    # Lambda³ GPU本体
    try:
        from ..analysis import MDLambda3AnalyzerGPU
        import_status['lambda3_gpu'] = "✅ Available"
    except ImportError:
        import_status['lambda3_gpu'] = "❌ Lambda³ GPU not properly installed"
    
    return import_status

# 初期化時の情報表示（デバッグモード）
def _print_init_info():
    """初期化情報の表示（デバッグ用）"""
    print("🌌 Quantum Validation Module Loaded")
    print(f"   Version: {__version__}")
    
    status = check_dependencies()
    for lib, stat in status.items():
        print(f"   {lib}: {stat}")

# 環境変数でデバッグモード制御
import os
if os.environ.get('QUANTUM_DEBUG', '').lower() == 'true':
    _print_init_info()

# 便利な統合関数
def validate_with_lambda3(trajectory, metadata, lambda_result=None, **kwargs):
    """
    Lambda³結果に対する量子検証の便利関数
    
    Parameters
    ----------
    trajectory : np.ndarray
        トラジェクトリデータ
    metadata : dict
        メタデータ
    lambda_result : MDLambda3Result, optional
        既存のLambda³結果（なければ新規実行）
    **kwargs
        追加オプション
        
    Returns
    -------
    dict
        量子検証結果
        
    Examples
    --------
    >>> from lambda3_gpu.quantum import validate_with_lambda3
    >>> results = validate_with_lambda3(traj, meta, lambda_result)
    """
    # Lambda³結果がなければ実行
    if lambda_result is None:
        from ..analysis import MDLambda3AnalyzerGPU
        analyzer = MDLambda3AnalyzerGPU()
        lambda_result = analyzer.analyze(trajectory)
    
    # 量子検証
    validator = QuantumValidationGPU(trajectory, metadata, **kwargs)
    quantum_events = validator.analyze_quantum_cascade(lambda_result)
    
    # サマリー表示
    validator.print_validation_summary(quantum_events)
    
    return {
        'quantum_events': quantum_events,
        'n_bell_violations': sum(1 for e in quantum_events 
                                if e.quantum_metrics.bell_violated),
        'n_critical': sum(1 for e in quantum_events if e.is_critical),
        'validator': validator
    }

# クイックテスト関数
def test_quantum_module():
    """モジュールの簡易テスト"""
    import numpy as np
    
    print("\n🧪 Testing Quantum Validation Module...")
    
    # ダミーデータ
    trajectory = np.random.randn(100, 900, 3)  # 100 frames, 900 atoms, 3D
    metadata = {
        'temperature': 310.0,
        'time_step_ps': 1.0,
        'n_molecules': 10,
        'n_atoms_per_molecule': 90
    }
    
    try:
        # 初期化テスト
        validator = QuantumValidationGPU(trajectory, metadata, force_cpu=True)
        print("   ✅ Module initialization successful")
        
        # 基本機能テスト
        print(f"   Thermal de Broglie wavelength: {validator.lambda_th_A:.3e} Å")
        print(f"   Thermal decoherence time: {validator.thermal_decoherence_ps:.3e} ps")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

# メイン実行（テスト用）
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Quantum Validation Module for Lambda³ GPU")
    print("="*60)
    
    # 依存関係チェック
    print("\n📋 Checking dependencies...")
    status = check_dependencies()
    for lib, stat in status.items():
        print(f"   {lib}: {stat}")
    
    # テスト実行
    if test_quantum_module():
        print("\n✨ Module is ready to use!")
        print("\nUsage:")
        print("  from lambda3_gpu.quantum import QuantumValidationGPU")
        print("  validator = QuantumValidationGPU(trajectory, metadata)")
    else:
        print("\n⚠️ Module test failed. Please check installation.")
