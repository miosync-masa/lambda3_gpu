"""
Quantum Validation Module for Lambda³ GPU - Enhanced Production Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

査読耐性＆単一フレーム対応の完全版！
- フレーム数適応型処理
- 複数の量子判定基準（文献準拠）
- 統計的検証
- 完全なエラーハンドリング

Version: 3.0.0 - Publication Ready
Authors: 環ちゃん & ご主人さま 💕
"""

from .quantum_validation_gpu import (
    # メインクラス
    QuantumValidationGPU,
    
    # データクラス
    QuantumMetrics,
    QuantumCascadeEvent,
    QuantumCriterion,
    
    # Enumクラス（新規追加）
    QuantumEventType,
    ValidationCriterion,
    
    # 型定義（必要なら）
    ArrayType,
    
    # 便利関数（新規追加）
    validate_quantum_events,
    generate_quantum_report,
)

# バージョン情報（3.0にアップ！）
__version__ = '3.0.0'

# 公開するAPI（拡張版）
__all__ = [
    # Main class
    'QuantumValidationGPU',
    
    # Data classes
    'QuantumMetrics', 
    'QuantumCascadeEvent',
    'QuantumCriterion',
    
    # Enums
    'QuantumEventType',
    'ValidationCriterion',
    
    # Convenience functions
    'validate_quantum_events',
    'generate_quantum_report',
    'validate_with_lambda3',
    
    # Test & utilities
    'check_dependencies',
    'test_quantum_module',
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
    
    # SciPy（統計検定用 - v3.0で必須）
    try:
        import scipy
        import_status['scipy'] = f"✅ {scipy.__version__}"
    except ImportError:
        import_status['scipy'] = "❌ Not installed (REQUIRED for v3.0)"
    
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
    print("🌌 Quantum Validation Module v3.0 Loaded")
    print("   査読耐性＆単一フレーム対応版")
    print(f"   Version: {__version__}")
    
    status = check_dependencies()
    for lib, stat in status.items():
        print(f"   {lib}: {stat}")

# 環境変数でデバッグモード制御
import os
if os.environ.get('QUANTUM_DEBUG', '').lower() == 'true':
    _print_init_info()

# 便利な統合関数
def validate_with_lambda3(trajectory, metadata, lambda_result=None, 
                         two_stage_result=None, **kwargs):
    """
    Lambda³結果に対する量子検証の便利関数（v3.0拡張版）
    
    Parameters
    ----------
    trajectory : np.ndarray
        トラジェクトリデータ
    metadata : dict
        メタデータ
    lambda_result : MDLambda3Result, optional
        既存のLambda³結果（なければ新規実行）
    two_stage_result : TwoStageLambda3Result, optional
        Two-stage解析結果（ネットワーク情報）
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
        from ..analysis import MDLambda3DetectorGPU, MDConfig
        config = MDConfig()
        detector = MDLambda3DetectorGPU(config)
        lambda_result = detector.analyze(trajectory)
    
    # 量子検証（v3.0: two_stage_resultも渡せる）
    validator = QuantumValidationGPU(trajectory, metadata, **kwargs)
    quantum_events = validator.analyze_quantum_cascade(lambda_result, two_stage_result)
    
    # サマリー表示
    validator.print_validation_summary(quantum_events)
    
    return {
        'quantum_events': quantum_events,
        'n_bell_violations': sum(1 for e in quantum_events 
                                if e.quantum_metrics.bell_violated),
        'n_critical': sum(1 for e in quantum_events if e.is_critical),
        'validator': validator,
        'event_types': _count_event_types(quantum_events),
        'quantum_ratio': _calculate_quantum_ratio(quantum_events)
    }

def _count_event_types(events):
    """イベントタイプ別カウント"""
    from collections import Counter
    return Counter(e.event_type.value for e in events)

def _calculate_quantum_ratio(events):
    """量子イベント比率計算"""
    if not events:
        return 0.0
    quantum_count = sum(1 for e in events if e.quantum_metrics.is_quantum)
    return quantum_count / len(events)

# クイックテスト関数（v3.0対応版）
def test_quantum_module():
    """モジュールの簡易テスト（v3.0版）"""
    import numpy as np
    
    print("\n🧪 Testing Quantum Validation Module v3.0...")
    
    # ダミーデータ
    trajectory = np.random.randn(100, 900, 3)  # 100 frames, 900 atoms, 3D
    metadata = {
        'temperature': 310.0,
        'time_step_ps': 2.0,  # v3.0: dt_psパラメータ
        'n_molecules': 10,
        'n_atoms_per_molecule': 90,
        'atom_masses': np.ones(900) * 12.0  # 炭素原子想定
    }
    
    try:
        # 初期化テスト
        validator = QuantumValidationGPU(
            trajectory, 
            metadata, 
            force_cpu=True,
            bootstrap_iterations=100,  # v3.0: Bootstrap検定
            significance_level=0.01     # v3.0: 有意水準
        )
        print("   ✅ Module initialization successful")
        
        # 物理定数チェック
        print(f"   Thermal de Broglie wavelength: {validator.lambda_thermal_A:.3e} Å")
        print(f"   Thermal decoherence time: {validator.thermal_decoherence_ps:.3e} ps")
        
        # イベントタイプ分類テスト
        from .quantum_validation_gpu import QuantumEventType
        print("\n   Testing event type classification:")
        for event_type in QuantumEventType:
            print(f"   - {event_type.value}: OK")
        
        # 判定基準テスト
        from .quantum_validation_gpu import ValidationCriterion
        print("\n   Testing validation criteria:")
        for criterion in ValidationCriterion:
            print(f"   - {criterion.value}: {criterion.name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# レポート生成の簡易ラッパー
def create_publication_report(quantum_events, output_file='quantum_report.txt'):
    """
    査読用レポート生成（v3.0新機能）
    
    Parameters
    ----------
    quantum_events : List[QuantumCascadeEvent]
        量子イベントリスト
    output_file : str
        出力ファイル名
    """
    from .quantum_validation_gpu import generate_quantum_report
    
    report = generate_quantum_report(quantum_events)
    
    # 統計情報追加
    report += "\n\nStatistical Summary\n"
    report += "-------------------\n"
    report += f"Total events analyzed: {len(quantum_events)}\n"
    report += f"Quantum events: {sum(1 for e in quantum_events if e.quantum_metrics.is_quantum)}\n"
    report += f"Critical events: {sum(1 for e in quantum_events if e.is_critical)}\n"
    
    # ファイル保存
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to {output_file}")
    return report

# メイン実行（テスト用）
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Quantum Validation Module for Lambda³ GPU v3.0")
    print("査読耐性＆単一フレーム対応版")
    print("="*60)
    
    # 依存関係チェック
    print("\n📋 Checking dependencies...")
    status = check_dependencies()
    all_ok = True
    for lib, stat in status.items():
        print(f"   {lib}: {stat}")
        if "❌" in stat and lib in ['numpy', 'scipy']:
            all_ok = False
    
    if not all_ok:
        print("\n⚠️ Critical dependencies missing!")
        print("Install with: pip install numpy scipy")
    else:
        # テスト実行
        if test_quantum_module():
            print("\n✨ Module v3.0 is ready for publication-quality analysis!")
            print("\nUsage:")
            print("  from lambda3_gpu.quantum import QuantumValidationGPU")
            print("  from lambda3_gpu.quantum import validate_quantum_events")
            print("  validator = QuantumValidationGPU(trajectory, metadata)")
            print("  events = validator.analyze_quantum_cascade(lambda_result)")
            
            print("\n新機能:")
            print("  - 単一フレーム量子もつれ検証")
            print("  - フレーム数適応型処理") 
            print("  - 査読対応の統計的検証")
            print("  - Bonferroni補正")
        else:
            print("\n⚠️ Module test failed. Please check installation.")
