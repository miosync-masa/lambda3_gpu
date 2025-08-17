#!/usr/bin/env python3
"""
Lambda³ GPU Quantum Validation Pipeline - 完全修正版
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

完全な量子検証パイプライン
TwoStageAnalyzerGPUとQuantumValidationGPUを正しく呼び出す

Author: Lambda³ Team
Modified by: 環ちゃん & ご主人さま 💕
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    logger.info("🚀 LAMBDA³ GPU QUANTUM VALIDATION PIPELINE v2.0")
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
        
        # バックボーンインデックス
        if backbone_indices_path and Path(backbone_indices_path).exists():
            backbone_indices = np.load(backbone_indices_path)
            logger.info(f"   Backbone indices loaded: {len(backbone_indices)} atoms")
        else:
            # メタデータから取得
            if 'protein_indices' in metadata:
                backbone_indices = np.array(metadata['protein_indices'])
                logger.info(f"   Using protein indices from metadata: {len(backbone_indices)} atoms")
            else:
                logger.error("   No backbone/protein indices provided!")
                raise ValueError("Backbone indices required for Lambda³ analysis")
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # ========================================
    # Step 2: Lambda³ GPU解析
    # ========================================
    logger.info("\n🔬 Running Lambda³ GPU Analysis...")
    
    try:
        # MDConfig設定
        config = MDConfig()
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
        
        # Lambda³検出器初期化
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
    # Step 3: 2段階解析（修正版）
    # ========================================
    two_stage_result = None
    
    if enable_two_stage and len(lambda_result.critical_events) > 0:
        logger.info("\n🔬 Running Two-Stage Analysis...")
        
        try:
            # 残基数の取得（デフォルト値付き）
            n_residues = metadata.get('n_residues', 833)  # TDP-43のデフォルト
            logger.info(f"   Number of residues: {n_residues}")
            
            # critical_eventsからイベント窓を作成
            events = []
            for i, event in enumerate(lambda_result.critical_events[:10]):  # 最大10イベント
                # eventはタプル (start, end) の場合が多い
                if isinstance(event, (tuple, list)) and len(event) >= 2:
                    start = int(event[0])
                    end = int(event[1])
                elif hasattr(event, 'frame'):
                    # フレーム番号を持つオブジェクトの場合
                    frame = event.frame
                    start = max(0, frame - 100)
                    end = min(n_frames, frame + 100)
                else:
                    continue
                
                events.append((start, end, f'critical_{i}'))
            
            if events:
                logger.info(f"   Processing {len(events)} events")
                
                # TwoStageAnalyzerGPU初期化
                analyzer = TwoStageAnalyzerGPU()
                
                # ✅ 正しいメソッド名と引数順序で実行！
                two_stage_result = analyzer.analyze_trajectory(
                    trajectory,      # 1番目: トラジェクトリ
                    lambda_result,   # 2番目: マクロ結果
                    events,          # 3番目: イベントリスト
                    n_residues       # 4番目: 残基数
                )
                
                logger.info("   ✅ Two-stage analysis complete")
                
                # ネットワーク統計を表示
                if hasattr(two_stage_result, 'global_network_stats'):
                    stats = two_stage_result.global_network_stats
                    logger.info(f"   Total causal links: {stats.get('total_causal_links', 0)}")
                    logger.info(f"   Total async bonds: {stats.get('total_async_bonds', 0)}")
                    logger.info(f"   Async/Causal ratio: {stats.get('async_to_causal_ratio', 0):.2%}")
            else:
                logger.warning("   No events to analyze")
                
        except Exception as e:
            logger.warning(f"Two-stage analysis failed: {e}")
            logger.warning("Continuing without residue-level analysis")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # ========================================
    # Step 4: 量子検証
    # ========================================
    logger.info("\n⚛️ Running Quantum Validation...")
    
    quantum_events = []
    
    try:
        # 量子検証器初期化
        quantum_validator = QuantumValidationGPU(
            trajectory=trajectory,
            metadata=metadata,
            validation_offset=10,
            min_samples_for_chsh=10
        )
        
        # ✅ 正しいメソッド名で実行！
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
        
        # サマリー表示
        quantum_validator.print_validation_summary(quantum_events)
        
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
                    'bell_violated': qm.bell_violated,
                    'chsh_value': qm.chsh_value,
                    'chsh_raw_value': qm.chsh_raw_value,
                    'chsh_confidence': qm.chsh_confidence,
                    'quantum_score': qm.quantum_score,
                    'n_samples': qm.n_samples_used
                }
            
            # async bondsの情報
            if hasattr(event, 'async_bonds_used') and event.async_bonds_used:
                event_dict['async_bonds'] = []
                for bond in event.async_bonds_used[:3]:
                    if isinstance(bond, dict):
                        event_dict['async_bonds'].append({
                            'pair': bond.get('residue_pair', []),
                            'causality': bond.get('causality', 0),
                            'sync_rate': bond.get('sync_rate', 0)
                        })
            
            quantum_data.append(event_dict)
        
        with open(output_path / 'quantum_events.json', 'w') as f:
            json.dump(quantum_data, f, indent=2)
            
    except Exception as e:
        logger.warning(f"Quantum validation failed: {e}")
        logger.warning("Continuing without quantum analysis")
        if verbose:
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Step 5: 可視化（オプション）
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating visualizations...")
        
        try:
            visualizer = Lambda3VisualizerGPU()
            
            # Lambda³結果の可視化
            fig = visualizer.visualize_results(
                lambda_result,
                save_path=str(output_path / 'lambda_results.png')
            )
            
            # 量子イベントの可視化（あれば）
            if quantum_events:
                fig_quantum = visualize_quantum_results(
                    quantum_events,
                    save_path=str(output_path / 'quantum_events.png')
                )
            
            logger.info("   ✅ Visualizations saved")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
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
# 量子結果の可視化関数
# ============================================

def visualize_quantum_results(quantum_events: List, 
                             save_path: Optional[str] = None) -> plt.Figure:
    """量子検証結果の可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if not quantum_events:
        fig.text(0.5, 0.5, 'No Quantum Events Detected', 
                ha='center', va='center', fontsize=20)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    # 1. CHSH値の時系列
    ax1 = axes[0, 0]
    frames = [e.frame for e in quantum_events]
    chsh_values = [e.quantum_metrics.chsh_value for e in quantum_events]
    
    if frames and chsh_values:
        ax1.scatter(frames, chsh_values, c='blue', alpha=0.6, s=50)
        ax1.axhline(y=2.0, color='red', linestyle='--', label='Classical Bound')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', 
                    label=f'Tsirelson Bound ({2*np.sqrt(2):.3f})')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('CHSH Value')
        ax1.set_title('CHSH Inequality Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Bell違反の分布
    ax2 = axes[0, 1]
    n_violated = sum(1 for e in quantum_events if e.quantum_metrics.bell_violated)
    n_classical = len(quantum_events) - n_violated
    
    if n_violated > 0 or n_classical > 0:
        ax2.pie([n_violated, n_classical], 
               labels=[f'Violated ({n_violated})', f'Classical ({n_classical})'],
               colors=['red', 'blue'],
               autopct='%1.1f%%')
    ax2.set_title('Bell Violation Distribution')
    
    # 3. 量子スコア分布
    ax3 = axes[1, 0]
    quantum_scores = [e.quantum_metrics.quantum_score for e in quantum_events]
    
    if quantum_scores:
        ax3.hist(quantum_scores, bins=20, alpha=0.7, color='purple')
    ax3.set_xlabel('Quantum Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Quantum Score Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. 信頼度vs CHSH値
    ax4 = axes[1, 1]
    confidences = [e.quantum_metrics.chsh_confidence for e in quantum_events]
    
    if confidences and chsh_values and quantum_scores:
        scatter = ax4.scatter(confidences, chsh_values, 
                             c=quantum_scores, cmap='viridis',
                             s=50, alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label='Quantum Score')
    
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('CHSH Value')
    ax4.set_title('Confidence vs CHSH Value')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Quantum Validation Results', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


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
- **System**: {metadata.get('system_name', 'TDP-43 LLPS')}
- **Temperature**: {metadata.get('temperature', 310)} K
- **Frames analyzed**: {lambda_result.n_frames}
- **Atoms**: {lambda_result.n_atoms}
- **Computation time**: {lambda_result.computation_time:.2f} seconds

## Lambda³ Analysis Results
- **Critical events**: {len(lambda_result.critical_events)}
"""
    
    if hasattr(lambda_result, 'structural_boundaries'):
        report += f"- **Structural boundaries**: {len(lambda_result.structural_boundaries)}\n"
    
    if hasattr(lambda_result, 'topological_breaks'):
        report += f"- **Topological breaks**: {len(lambda_result.topological_breaks)}\n"
    
    report += f"""
## Quantum Validation Results
- **Total quantum events**: {len(quantum_events)}
"""
    
    if quantum_events:
        n_bell = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
        n_critical = sum(1 for e in quantum_events if e.is_critical)
        
        report += f"""- **Bell violations**: {n_bell} ({100*n_bell/len(quantum_events):.1f}%)
- **Critical quantum events**: {n_critical}
"""
        
        # CHSH統計
        chsh_values = [e.quantum_metrics.chsh_value 
                      for e in quantum_events 
                      if hasattr(e, 'quantum_metrics')]
        if chsh_values:
            report += f"""
### CHSH Statistics
- **Average CHSH value**: {np.mean(chsh_values):.3f}
- **Max CHSH value**: {np.max(chsh_values):.3f}
- **Min CHSH value**: {np.min(chsh_values):.3f}
- **Classical bound**: 2.000
- **Tsirelson bound**: {2*np.sqrt(2):.3f}
"""
    
    if two_stage_result:
        report += f"""
## Two-Stage Analysis Results
"""
        if hasattr(two_stage_result, 'residue_analyses'):
            report += f"- **Residue analyses completed**: {len(two_stage_result.residue_analyses)}\n"
        
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            report += f"""- **Total causal links**: {stats.get('total_causal_links', 0)}
- **Total sync links**: {stats.get('total_sync_links', 0)}
- **Total async bonds**: {stats.get('total_async_bonds', 0)}
- **Async/Causal ratio**: {stats.get('async_to_causal_ratio', 0):.2%}
"""
        
        if hasattr(two_stage_result, 'suggested_intervention_points'):
            points = two_stage_result.suggested_intervention_points[:5]
            if points:
                report += f"- **Suggested intervention points**: {points}\n"
    
    report += f"""
## Conclusions

The Lambda³ GPU analysis successfully completed:

1. **Structural anomalies**: {len(lambda_result.critical_events)} critical events detected
2. **Quantum signatures**: {len(quantum_events)} quantum cascade events validated
"""
    
    if quantum_events:
        n_bell = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
        if n_bell > 0:
            report += f"3. **Bell violations**: Confirmed quantum correlations beyond classical limits ({n_bell} violations)\n"
    
    report += """
---
*Generated by Lambda³ GPU Quantum Validation Pipeline v2.0*
*NO TIME, NO PHYSICS, ONLY STRUCTURE!*
"""
    
    return report


# ============================================
# CLI Interface  
# ============================================

def main():
    """コマンドラインインターフェース"""
    
    parser = argparse.ArgumentParser(
        description='Lambda³ GPU Quantum Validation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trajectory.npy metadata.json
  %(prog)s trajectory.npy metadata.json --backbone backbone.npy
  %(prog)s trajectory.npy metadata.json --output ./results --verbose
  %(prog)s trajectory.npy metadata.json --no-two-stage --no-viz
        """
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
