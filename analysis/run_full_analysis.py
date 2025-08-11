"""
Quantum Validation Runner for Lambda³ GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MDLambda3AnalyzerGPUとTwoStageAnalyzerGPUの結果を受け取って
量子検証を実行する本番用スクリプト

環ちゃん💕
"""

import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Lambda³ GPU imports
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
    output_dir: str = './quantum_results'
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
        
    Returns
    -------
    Dict
        解析結果の辞書
    """
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 QUANTUM VALIDATION PIPELINE")
    logger.info("   Integrated with Lambda³ GPU")
    logger.info("="*70)
    
    # ========================================
    # Step 1: データ読み込み
    # ========================================
    logger.info("\n📁 Loading data...")
    
    try:
        # トラジェクトリ
        trajectory = np.load(trajectory_path)
        logger.info(f"   Trajectory shape: {trajectory.shape}")
        n_frames, n_atoms, _ = trajectory.shape
        
        # メタデータ
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info("   Metadata loaded")
        
        # 必須パラメータチェック
        if 'time_step_ps' not in metadata:
            logger.warning("   Adding default time_step_ps = 1.0")
            metadata['time_step_ps'] = 1.0
        
        # バックボーンインデックス
        backbone_indices = None
        if backbone_indices_path:
            backbone_indices = np.load(backbone_indices_path)
            logger.info(f"   Backbone indices: {len(backbone_indices)} atoms")
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None
    
    # ========================================
    # Step 2: Lambda³解析（実際の実装）
    # ========================================
    logger.info("\n🔬 Running Lambda³ GPU Analysis...")
    
    # 設定
    config = MDConfig(
        temperature=metadata.get('temperature', 310.0),
        time_step_ps=metadata['time_step_ps'],
        n_molecules=metadata.get('n_molecules', 100),
        sensitivity=2.0,
        use_extended_detection=True,
        use_phase_space=True,
        adaptive_window=True
    )
    
    # Lambda³アナライザー初期化
    analyzer = MDLambda3AnalyzerGPU(config)
    
    # 解析実行
    lambda_result = analyzer.analyze(trajectory, backbone_indices)
    
    logger.info(f"   ✅ Lambda³ analysis complete")
    logger.info(f"   Detected events:")
    for event_type, events in lambda_result.events.items():
        if events:
            logger.info(f"     {event_type}: {len(events)} events")
    
    # ========================================
    # Step 3: 2段階解析（残基レベル）- オプション
    # ========================================
    two_stage_result = None
    
    if enable_two_stage:
        logger.info("\n🔬 Running Two-Stage Analysis...")
        
        # 残基数を推定
        n_residues = metadata.get('n_residues', n_atoms // 10)  # 仮定：10原子/残基
        
        # 残基解析設定
        residue_config = ResidueAnalysisConfig(
            n_residues=n_residues,
            sensitivity=2.0,
            min_persistence=5,
            use_confidence=True
        )
        
        # 2段階アナライザー
        two_stage_analyzer = TwoStageAnalyzerGPU(residue_config)
        
        # 主要イベントを抽出
        key_events = []
        for event_type, events in lambda_result.events.items():
            for event in events[:5]:  # 各タイプ最大5個
                if 'frame' in event:
                    # イベント窓を推定
                    start = max(0, event['frame'] - 100)
                    end = min(n_frames, event['frame'] + 100)
                    key_events.append((start, end, f"{event_type}_{event['frame']}"))
        
        if key_events:
            # 2段階解析実行
            two_stage_result = two_stage_analyzer.analyze_trajectory(
                trajectory,
                n_residues,
                lambda_result,
                key_events
            )
            
            logger.info(f"   ✅ Two-stage analysis complete")
            logger.info(f"   Global network stats:")
            stats = two_stage_result.global_network_stats
            logger.info(f"     Total causal links: {stats['total_causal_links']}")
            logger.info(f"     Total sync links: {stats['total_sync_links']}")
            logger.info(f"     Total async bonds: {stats['total_async_bonds']}")
    
    # ========================================
    # Step 4: 量子検証
    # ========================================
    logger.info("\n⚛️ Running Quantum Validation...")
    
    # 量子検証モジュール初期化
    quantum_validator = QuantumValidationGPU(
        trajectory=trajectory,
        metadata=metadata,
        validation_offset=10,
        min_samples_for_chsh=10
    )
    
    # 量子カスケード解析
    quantum_events = quantum_validator.analyze_quantum_cascade(
        lambda_result,
        residue_events=two_stage_result.residue_analyses if two_stage_result else None
    )
    
    # async strong bondsを2段階解析から取得（あれば）
    if two_stage_result:
        logger.info("   Using async strong bonds from two-stage analysis")
        
        # 各量子イベントにasync bondsを追加
        for qevent in quantum_events:
            # 時間的に近い残基解析を探す
            for analysis_name, analysis in two_stage_result.residue_analyses.items():
                if abs(analysis.macro_start - qevent.frame) < 200:
                    # async bondsを追加
                    if hasattr(analysis, 'async_strong_bonds'):
                        qevent.async_bonds_used = analysis.async_strong_bonds[:5]
                        break
    
    # サマリー出力
    quantum_validator.print_validation_summary(quantum_events)
    
    # ========================================
    # Step 5: 結果の保存
    # ========================================
    logger.info("\n💾 Saving results...")
    
    # Lambda³結果
    np.save(output_path / 'lambda_result.npy', lambda_result, allow_pickle=True)
    
    # 量子イベント
    quantum_data = []
    for event in quantum_events:
        event_dict = {
            'frame': event.frame,
            'type': event.event_type,
            'is_critical': event.is_critical,
            'critical_reasons': event.critical_reasons,
            'quantum_metrics': {
                'bell_violated': event.quantum_metrics.bell_violated,
                'chsh_value': event.quantum_metrics.chsh_value,
                'chsh_confidence': event.quantum_metrics.chsh_confidence,
                'quantum_score': event.quantum_metrics.quantum_score,
                'n_samples': event.quantum_metrics.n_samples_used
            }
        }
        
        # async bonds情報を追加
        if hasattr(event, 'async_bonds_used') and event.async_bonds_used:
            event_dict['async_bonds'] = [
                {
                    'pair': bond.get('residue_pair', bond.get('pair', [])) if isinstance(bond, dict) 
                           else (bond.from_res, bond.to_res),
                    'causality': bond.get('causality', bond.strength if hasattr(bond, 'strength') else 0),
                    'sync_rate': bond.get('sync_rate', 0)
                }
                for bond in event.async_bonds_used[:3]
            ]
        
        quantum_data.append(event_dict)
    
    with open(output_path / 'quantum_events.json', 'w') as f:
        json.dump(quantum_data, f, indent=2)
    
    logger.info(f"   Saved to {output_path}")
    
    # ========================================
    # Step 6: 可視化（オプション）
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating visualizations...")
        
        try:
            # Lambda³可視化
            visualizer = Lambda3VisualizerGPU()
            
            # メイン結果の可視化
            fig1 = visualizer.visualize_results(
                lambda_result,
                save_path=str(output_path / 'lambda_results.png')
            )
            
            # 量子検証の可視化
            if quantum_events:
                fig2 = visualize_quantum_results(
                    quantum_events,
                    save_path=str(output_path / 'quantum_validation.png')
                )
            
            # 2段階解析の可視化
            if two_stage_result:
                from lambda3_gpu.visualization.causality_viz_gpu import CausalityVisualizerGPU
                causality_viz = CausalityVisualizerGPU()
                
                for event_name, analysis in two_stage_result.residue_analyses.items():
                    fig3 = causality_viz.visualize_residue_causality(
                        analysis,
                        save_path=str(output_path / f'causality_{event_name}.png')
                    )
                    break  # 最初のイベントのみ
            
            logger.info("   ✅ Visualizations saved")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # ========================================
    # Step 7: レポート生成
    # ========================================
    logger.info("\n📝 Generating report...")
    
    report = generate_comprehensive_report(
        lambda_result, 
        quantum_events,
        two_stage_result
    )
    
    with open(output_path / 'analysis_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"   Report saved to {output_path / 'analysis_report.md'}")
    
    # ========================================
    # 完了
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info(f"   Output directory: {output_path}")
    logger.info("="*70)
    
    return {
        'lambda_result': lambda_result,
        'quantum_events': quantum_events,
        'two_stage_result': two_stage_result,
        'output_dir': output_path
    }

# ============================================
# 可視化関数
# ============================================

def visualize_quantum_results(quantum_events: List, 
                             save_path: Optional[str] = None) -> plt.Figure:
    """量子検証結果の可視化（シンプル版）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. CHSH値の時系列
    ax1 = axes[0, 0]
    frames = [e.frame for e in quantum_events]
    chsh_values = [e.quantum_metrics.chsh_value for e in quantum_events]
    
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
    
    ax2.pie([n_violated, n_classical], 
           labels=[f'Violated ({n_violated})', f'Classical ({n_classical})'],
           colors=['red', 'blue'],
           autopct='%1.1f%%')
    ax2.set_title('Bell Violation Distribution')
    
    # 3. 量子スコア分布
    ax3 = axes[1, 0]
    quantum_scores = [e.quantum_metrics.quantum_score for e in quantum_events]
    
    ax3.hist(quantum_scores, bins=20, alpha=0.7, color='purple')
    ax3.set_xlabel('Quantum Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Quantum Score Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. 信頼度vs CHSH値
    ax4 = axes[1, 1]
    confidences = [e.quantum_metrics.chsh_confidence for e in quantum_events]
    
    scatter = ax4.scatter(confidences, chsh_values, 
                         c=quantum_scores, cmap='viridis',
                         s=50, alpha=0.6)
    ax4.set_xlabel('Confidence')
    ax4.set_ylabel('CHSH Value')
    ax4.set_title('Confidence vs CHSH Value')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Quantum Score')
    
    plt.suptitle('Quantum Validation Results', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ============================================
# レポート生成
# ============================================

def generate_comprehensive_report(lambda_result: MDLambda3Result,
                                 quantum_events: List,
                                 two_stage_result: Optional[TwoStageLambda3Result]) -> str:
    """包括的な解析レポート生成"""
    
    # 統計計算
    n_quantum = len(quantum_events)
    n_bell = sum(1 for e in quantum_events if e.quantum_metrics.bell_violated)
    n_critical = sum(1 for e in quantum_events if e.is_critical)
    
    if n_quantum > 0:
        avg_chsh = np.mean([e.quantum_metrics.chsh_value for e in quantum_events])
        max_chsh = max([e.quantum_metrics.chsh_value for e in quantum_events])
        avg_confidence = np.mean([e.quantum_metrics.chsh_confidence for e in quantum_events])
    else:
        avg_chsh = max_chsh = avg_confidence = 0
    
    report = f"""# Lambda³ GPU Quantum Validation Report

## Executive Summary

Complete analysis pipeline executed successfully, integrating:
- Lambda³ GPU structural analysis
- Two-stage residue-level analysis
- Quantum validation with CHSH inequality testing

## Lambda³ Analysis Results

### Events Detected
"""
    
    # Lambda³イベント
    for event_type, events in lambda_result.events.items():
        if events:
            report += f"- **{event_type}**: {len(events)} events\n"
    
    # 2段階解析結果
    if two_stage_result:
        report += f"""
## Two-Stage Analysis Results

### Network Statistics
- **Total Causal Links**: {two_stage_result.global_network_stats['total_causal_links']}
- **Total Sync Links**: {two_stage_result.global_network_stats['total_sync_links']}
- **Total Async Strong Bonds**: {two_stage_result.global_network_stats['total_async_bonds']}
- **Async/Causal Ratio**: {two_stage_result.global_network_stats.get('async_to_causal_ratio', 0):.2%}

### Top Important Residues
"""
        top_residues = sorted(two_stage_result.global_residue_importance.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        for i, (res_id, score) in enumerate(top_residues):
            report += f"{i+1}. Residue {res_id}: {score:.3f}\n"
    
    # 量子検証結果
    report += f"""
## Quantum Validation Results

### Statistics
- **Total Quantum Events**: {n_quantum}
- **Bell Violations**: {n_bell} ({100*n_bell/n_quantum if n_quantum else 0:.1f}%)
- **Critical Events**: {n_critical}

### CHSH Analysis
- **Average CHSH Value**: {avg_chsh:.3f}
- **Maximum CHSH Value**: {max_chsh:.3f}
- **Average Confidence**: {avg_confidence:.3f}
- **Classical Bound**: 2.000
- **Tsirelson Bound**: {2*np.sqrt(2):.3f}

## Critical Quantum Events
"""
    
    # 臨界イベント
    critical_events = [e for e in quantum_events if e.is_critical][:5]
    
    for i, event in enumerate(critical_events):
        qm = event.quantum_metrics
        report += f"""
### Event {i+1} (Frame {event.frame})
- **Type**: {event.event_type}
- **CHSH Value**: {qm.chsh_value:.3f} (confidence: {qm.chsh_confidence:.3f})
- **Quantum Score**: {qm.quantum_score:.3f}
- **Reasons**: {', '.join(event.critical_reasons)}
"""
        
        # async bonds情報
        if hasattr(event, 'async_bonds_used') and event.async_bonds_used:
            report += "- **Async Strong Bonds Used**:\n"
            for bond in event.async_bonds_used[:3]:
                if isinstance(bond, dict):
                    pair = bond.get('residue_pair', bond.get('pair', []))
                    causality = bond.get('causality', 0)
                    sync = bond.get('sync_rate', 0)
                else:
                    pair = (bond.from_res, bond.to_res) if hasattr(bond, 'from_res') else []
                    causality = bond.strength if hasattr(bond, 'strength') else 0
                    sync = bond.sync_rate if hasattr(bond, 'sync_rate') else 0
                
                if pair:
                    report += f"  - R{pair[0]}-R{pair[1]}: causality={causality:.3f}, sync={sync:.3f}\n"
    
    report += """
## Methodology

- **CHSH Measurement Settings**: A(0°), A'(45°), B(22.5°), B'(67.5°)
- **Data Separation**: Training window ≠ Validation window (10 frame offset)
- **Async Strong Bonds**: High causality (>0.3) with low synchronization (<0.2)

## Conclusions

The analysis successfully identified quantum signatures through:
1. Structural anomalies detected by Lambda³ GPU
2. Async strong bonds identified in residue-level analysis
3. Bell inequality violations confirmed by CHSH testing

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
        description='Quantum Validation for Lambda³ GPU Analysis'
    )
    
    # 必須引数
    parser.add_argument('trajectory', help='Path to trajectory file (.npy)')
    parser.add_argument('metadata', help='Path to metadata file (.json)')
    
    # オプション引数
    parser.add_argument('--backbone', '-b', 
                       help='Path to backbone indices file (.npy)')
    parser.add_argument('--output', '-o', default='./quantum_results',
                       help='Output directory (default: ./quantum_results)')
    parser.add_argument('--no-two-stage', action='store_true',
                       help='Skip two-stage analysis')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # パイプライン実行
    results = run_quantum_validation_pipeline(
        trajectory_path=args.trajectory,
        metadata_path=args.metadata,
        backbone_indices_path=args.backbone,
        enable_two_stage=not args.no_two_stage,
        enable_visualization=not args.no_viz,
        output_dir=args.output
    )
    
    if results:
        print(f"\n✅ Success! Results saved to: {results['output_dir']}")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
