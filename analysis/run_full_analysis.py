#!/usr/bin/env python3
"""
Lambda³ GPU Quantum Validation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

完全な量子検証パイプライン - 汎用版
データから推測せず、メタデータを信頼する設計

Author: Lambda³ Team
"""

import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Lambda³ GPU imports
try:
    from lambda3_gpu.analysis.md_lambda3_detector_gpu import (
        MDLambda3DetectorGPU, 
        MDLambda3Result,
        MDConfig
    )
    from lambda3_gpu.analysis.two_stage_analyzer_gpu import (
        TwoStageAnalyzerGPU,
        TwoStageLambda3Result,
        ResidueAnalysisConfig
    )
    from lambda3_gpu.quantum import QuantumValidationGPU
    from lambda3_gpu.visualization import Lambda3VisualizerGPU
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure lambda3_gpu is properly installed")
    raise

# Logger設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_validation')

# ============================================
# メイン実行関数
# ============================================

def run_quantum_validation_pipeline(
    trajectory_path: str,
    metadata_path: str,
    backbone_indices_path: Optional[str] = None,
    enable_two_stage: bool = True,
    enable_visualization: bool = True,
    output_dir: str = './quantum_results',
    verbose: bool = False
) -> Dict:
    """
    完全な量子検証パイプライン
    
    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイルのパス (.npy)
    metadata_path : str
        メタデータファイルのパス (.json)
    backbone_indices_path : str, optional
        バックボーン原子インデックス (.npy)
    enable_two_stage : bool
        2段階解析（残基レベル）を実行するか
    enable_visualization : bool
        可視化を実行するか
    output_dir : str
        出力ディレクトリ
    verbose : bool
        詳細出力
        
    Returns
    -------
    Dict
        解析結果の辞書
    """
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 LAMBDA³ GPU QUANTUM VALIDATION PIPELINE")
    logger.info("="*70)
    
    # ========================================
    # Step 1: データ読み込み
    # ========================================
    logger.info("\n📁 Loading data...")
    
    try:
        # トラジェクトリ読み込み
        trajectory = np.load(trajectory_path)
        logger.info(f"   Trajectory shape: {trajectory.shape}")
        n_frames, n_atoms, n_dims = trajectory.shape
        
        if n_dims != 3:
            raise ValueError(f"Trajectory must be 3D, got {n_dims}D")
        
        # メタデータ読み込み
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info("   Metadata loaded")
        
        # メタデータ検証（必須項目のチェック）
        required_fields = ['n_atoms', 'n_frames']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"   Metadata missing '{field}', using trajectory shape")
                if field == 'n_atoms':
                    metadata['n_atoms'] = n_atoms
                elif field == 'n_frames':
                    metadata['n_frames'] = n_frames
        
        # バックボーンインデックス
        if backbone_indices_path and Path(backbone_indices_path).exists():
            backbone_indices = np.load(backbone_indices_path)
            logger.info(f"   Backbone indices loaded: {len(backbone_indices)} atoms")
        else:
            # バックボーンインデックスがない場合はメタデータから取得
            if 'protein_indices' in metadata:
                backbone_indices = np.array(metadata['protein_indices'])
                logger.info(f"   Using protein indices from metadata: {len(backbone_indices)} atoms")
            else:
                # それもない場合はエラー
                logger.error("   No backbone/protein indices provided!")
                logger.error("   Please provide either:")
                logger.error("     - backbone_indices_path parameter")
                logger.error("     - 'protein_indices' in metadata")
                raise ValueError("Backbone indices required for Lambda³ analysis")
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # ========================================
    # Step 2: Lambda³ GPU解析
    # ========================================
    logger.info("\n🔬 Running Lambda³ GPU Analysis...")
    
    try:
        # MDConfig設定（正しい形式！）
        config = MDConfig()  # 引数なし！
        config.use_extended_detection = True
        config.use_phase_space = True
        config.use_periodic = True
        config.use_gradual = True
        config.use_drift = True
        config.adaptive_window = True
        config.window_scale = 0.005
        config.gpu_batch_size = 5000
        config.verbose = verbose
        
        logger.info("   Config initialized with advanced detection modes")
        
        # Lambda³検出器初期化（正しいクラス名！）
        detector = MDLambda3DetectorGPU(config)
        logger.info("   Detector initialized")
        
        # 解析実行
        lambda_result = detector.analyze(trajectory, backbone_indices)
        
        logger.info(f"   ✅ Lambda³ analysis complete")
        logger.info(f"   Critical events detected: {len(lambda_result.critical_events)}")
        
        # 結果の保存
        result_dict = {
            'n_frames': lambda_result.n_frames,
            'n_atoms': lambda_result.n_atoms,
            'n_critical_events': len(lambda_result.critical_events),
            'computation_time': lambda_result.computation_time,
            'gpu_info': lambda_result.gpu_info if hasattr(lambda_result, 'gpu_info') else {}
        }
        
        with open(output_path / 'lambda_result_summary.json', 'w') as f:
            json.dump(result_dict, f, indent=2)
            
    except Exception as e:
        logger.error(f"Lambda³ analysis failed: {e}")
        raise
    
    # ========================================
    # Step 3: 2段階解析（オプション）
    # ========================================
    two_stage_result = None
    
    if enable_two_stage and len(lambda_result.critical_events) > 0:
        logger.info("\n🔬 Running Two-Stage Analysis...")
        
        try:
            # 残基数の取得（メタデータから）
            if 'n_residues' not in metadata:
                logger.error("   'n_residues' not found in metadata!")
                logger.error("   Skipping two-stage analysis")
            else:
                n_residues = metadata['n_residues']
                logger.info(f"   Number of residues: {n_residues}")
                
                # 残基解析設定
                residue_config = ResidueAnalysisConfig()
                residue_config.n_residues = n_residues
                residue_config.min_persistence = 5
                residue_config.use_confidence = True
                
                # 2段階アナライザー初期化
                two_stage_analyzer = TwoStageAnalyzerGPU(residue_config)
                
                # イベント窓の定義
                events = []
                for i, event in enumerate(lambda_result.critical_events[:10]):  # 最大10イベント
                    if hasattr(event, 'frame'):
                        frame = event.frame
                    elif isinstance(event, dict) and 'frame' in event:
                        frame = event['frame']
                    else:
                        continue
                    
                    # イベント前後の窓
                    start = max(0, frame - 100)
                    end = min(n_frames, frame + 100)
                    events.append((start, end, f'event_{i}'))
                
                if events:
                    # 2段階解析実行
                    two_stage_result = two_stage_analyzer.analyze(
                        trajectory,
                        lambda_result,
                        events,
                        n_residues
                    )
                    
                    logger.info("   ✅ Two-stage analysis complete")
                    
                    # ネットワーク統計を表示
                    if hasattr(two_stage_result, 'global_network_stats'):
                        stats = two_stage_result.global_network_stats
                        logger.info(f"   Total causal links: {stats.get('total_causal_links', 0)}")
                        logger.info(f"   Total async bonds: {stats.get('total_async_bonds', 0)}")
                        
        except Exception as e:
            logger.warning(f"Two-stage analysis failed: {e}")
            logger.warning("Continuing without residue-level analysis")
    
    # ========================================
    # Step 4: 量子検証
    # ========================================
    logger.info("\n⚛️ Running Quantum Validation...")
    
    quantum_events = []
    
    try:
        # 量子検証器初期化
        quantum_validator = QuantumValidationGPU(
            trajectory=trajectory,
            metadata=metadata
        )
        
        # Lambda³結果に対する量子検証
        quantum_events = quantum_validator.analyze_quantum_cascade(lambda_result)
        
        # 2段階解析結果があれば追加情報を付与
        if two_stage_result and hasattr(two_stage_result, 'residue_analyses'):
            for qevent in quantum_events:
                # 対応する残基解析を探す
                for analysis_name, analysis in two_stage_result.residue_analyses.items():
                    if hasattr(analysis, 'async_strong_bonds'):
                        qevent.async_bonds_used = analysis.async_strong_bonds[:3]
                        break
        
        logger.info(f"   ✅ Quantum validation complete")
        logger.info(f"   Quantum events detected: {len(quantum_events)}")
        
        # Bell違反の統計
        n_bell_violations = sum(1 for e in quantum_events 
                               if hasattr(e, 'quantum_metrics') 
                               and e.quantum_metrics.bell_violated)
        if len(quantum_events) > 0:
            logger.info(f"   Bell violations: {n_bell_violations}/{len(quantum_events)} "
                       f"({100*n_bell_violations/len(quantum_events):.1f}%)")
        
        # 量子イベントの保存
        quantum_data = []
        for event in quantum_events:
            event_dict = {
                'frame': event.frame if hasattr(event, 'frame') else 0,
                'type': event.event_type if hasattr(event, 'event_type') else 'unknown',
                'is_critical': event.is_critical if hasattr(event, 'is_critical') else False
            }
            
            if hasattr(event, 'quantum_metrics'):
                qm = event.quantum_metrics
                event_dict['quantum_metrics'] = {
                    'bell_violated': qm.bell_violated if hasattr(qm, 'bell_violated') else False,
                    'chsh_value': qm.chsh_value if hasattr(qm, 'chsh_value') else 0,
                    'quantum_score': qm.quantum_score if hasattr(qm, 'quantum_score') else 0
                }
            
            quantum_data.append(event_dict)
        
        with open(output_path / 'quantum_events.json', 'w') as f:
            json.dump(quantum_data, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Quantum validation failed: {e}")
        logger.warning("Continuing without quantum analysis")
    
    # ========================================
    # Step 5: 可視化（オプション）
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating visualizations...")
        
        try:
            from lambda3_gpu.visualization import Lambda3VisualizerGPU
            
            visualizer = Lambda3VisualizerGPU()
            
            # Lambda³結果の可視化
            fig = visualizer.visualize_results(
                lambda_result,
                save_path=str(output_path / 'lambda_results.png')
            )
            logger.info("   ✅ Visualizations saved")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # ========================================
    # Step 6: レポート生成
    # ========================================
    logger.info("\n📝 Generating report...")
    
    report = generate_analysis_report(
        lambda_result,
        quantum_events,
        two_stage_result,
        metadata
    )
    
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"   Report saved to {output_path / 'analysis_report.md'}")
    
    # ========================================
    # 完了
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info(f"   Results saved to: {output_path}")
    logger.info("="*70)
    
    return {
        'lambda_result': lambda_result,
        'quantum_events': quantum_events,
        'two_stage_result': two_stage_result,
        'output_dir': output_path,
        'success': True
    }


# ============================================
# レポート生成
# ============================================

def generate_analysis_report(
    lambda_result: Any,
    quantum_events: List,
    two_stage_result: Optional[Any],
    metadata: Dict
) -> str:
    """解析レポートの生成"""
    
    report = f"""# Lambda³ GPU Analysis Report

## System Information
- **System**: {metadata.get('system_name', 'Unknown')}
- **Temperature**: {metadata.get('temperature', 'N/A')} K
- **Frames analyzed**: {lambda_result.n_frames}
- **Atoms**: {lambda_result.n_atoms}
- **Computation time**: {lambda_result.computation_time:.2f} seconds

## Lambda³ Analysis Results
- **Critical events**: {len(lambda_result.critical_events)}
- **Structural boundaries**: {len(lambda_result.structural_boundaries)}
- **Topological breaks**: {len(lambda_result.topological_breaks)}

## Quantum Validation Results
- **Total quantum events**: {len(quantum_events)}
"""
    
    if quantum_events:
        n_bell = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
        report += f"- **Bell violations**: {n_bell} ({100*n_bell/len(quantum_events):.1f}%)\n"
        
        # CHSH統計
        chsh_values = [e.quantum_metrics.chsh_value 
                      for e in quantum_events 
                      if hasattr(e, 'quantum_metrics')]
        if chsh_values:
            report += f"- **Average CHSH value**: {np.mean(chsh_values):.3f}\n"
            report += f"- **Max CHSH value**: {np.max(chsh_values):.3f}\n"
    
    if two_stage_result:
        report += f"""
## Two-Stage Analysis Results
- **Residue analyses**: {len(two_stage_result.residue_analyses) if hasattr(two_stage_result, 'residue_analyses') else 0}
"""
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            report += f"- **Total causal links**: {stats.get('total_causal_links', 0)}\n"
            report += f"- **Total async bonds**: {stats.get('total_async_bonds', 0)}\n"
    
    report += f"""
## Status
✅ Analysis pipeline completed successfully

---
*Generated by Lambda³ GPU Quantum Validation Pipeline*
"""
    
    return report


# ============================================
# CLI Interface  
# ============================================

def main():
    """コマンドラインインターフェース"""
    
    parser = argparse.ArgumentParser(
        description='Lambda³ GPU Quantum Validation Pipeline'
    )
    
    # 必須引数
    parser.add_argument('trajectory', 
                       help='Path to trajectory file (.npy)')
    parser.add_argument('metadata', 
                       help='Path to metadata file (.json)')
    
    # オプション引数
    parser.add_argument('--backbone', '-b',
                       help='Path to backbone indices file (.npy)')
    parser.add_argument('--output', '-o', 
                       default='./quantum_results',
                       help='Output directory (default: ./quantum_results)')
    parser.add_argument('--no-two-stage', 
                       action='store_true',
                       help='Skip two-stage analysis')
    parser.add_argument('--no-viz', 
                       action='store_true',
                       help='Skip visualization')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # パイプライン実行
    try:
        results = run_quantum_validation_pipeline(
            trajectory_path=args.trajectory,
            metadata_path=args.metadata,
            backbone_indices_path=args.backbone,
            enable_two_stage=not args.no_two_stage,
            enable_visualization=not args.no_viz,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        if results and results.get('success'):
            print(f"\n✅ Success! Results saved to: {results['output_dir']}")
            return 0
        else:
            print("\n❌ Pipeline failed. Check logs for details.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
