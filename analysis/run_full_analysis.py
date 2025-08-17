#!/usr/bin/env python3
"""
Lambda³ GPU Quantum Validation Pipeline - Two-Stage最適化版
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

タンパク質全原子を使った詳細な残基レベル解析に最適化
正確なメタデータ構造に対応

Author: Lambda³ Team
Modified by: 環ちゃん & ご主人さま 💕
Date: 2025-08-18
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
    protein_indices_path: str,  # タンパク質全原子インデックス（必須）
    enable_two_stage: bool = True,
    enable_visualization: bool = True,
    output_dir: str = './quantum_results',
    verbose: bool = False
) -> Dict:
    """
    完全な量子検証パイプライン（Two-Stage最適化版）
    
    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイルのパス (.npy)
    metadata_path : str
        メタデータファイルのパス (.json)
    protein_indices_path : str
        タンパク質原子インデックス (.npy) - Two-Stageでは全原子推奨
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
    logger.info("   Two-Stage Optimized Version")
    logger.info("="*70)
    
    # ========================================
    # Step 1: データ読み込み（シンプル版）
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
        logger.info("   Metadata loaded successfully")
        
        # タンパク質インデックス読み込み（必須）
        if not Path(protein_indices_path).exists():
            raise FileNotFoundError(f"Protein indices file not found: {protein_indices_path}")
            
        protein_indices = np.load(protein_indices_path)
        logger.info(f"   Protein indices loaded: {len(protein_indices)} atoms")
        
        # タンパク質情報の表示
        if 'protein' in metadata:
            protein_info = metadata['protein']
            logger.info(f"   Protein: {protein_info.get('n_residues', 'N/A')} residues")
            logger.info(f"   Sequence length: {len(protein_info.get('sequence', ''))}")
        
        # 妥当性チェック
        max_idx = np.max(protein_indices)
        if max_idx >= n_atoms:
            raise ValueError(f"Protein index {max_idx} exceeds atom count {n_atoms}")
        
        logger.info(f"   ✅ Data validation passed")
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # ========================================
    # Step 2: Lambda³ GPU解析（高速初期探索）
    # ========================================
    logger.info("\n🔬 Running Lambda³ GPU Analysis...")
    logger.info(f"   Using {'full protein' if len(protein_indices) > 100 else 'backbone'} atoms")
    
    try:
        # MDConfig設定（Two-Stage用に最適化）
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
        
        # 感度を上げて詳細なイベントを検出
        config.sensitivity = 2.0
        config.use_hierarchical = True  # 階層的検出
        
        logger.info("   Config optimized for Two-Stage analysis")
        
        # Lambda³検出器初期化
        detector = MDLambda3DetectorGPU(config)
        logger.info("   Detector initialized on GPU")
        
        # 解析実行（タンパク質原子のみ使用）
        lambda_result = detector.analyze(trajectory, protein_indices)
        
        logger.info(f"   ✅ Lambda³ analysis complete")
        logger.info(f"   Critical events detected: {len(lambda_result.critical_events)}")
        
        # イベントの詳細表示
        if verbose and lambda_result.critical_events:
            for i, event in enumerate(lambda_result.critical_events[:5]):
                logger.info(f"     Event {i}: frames {event[0]}-{event[1]}")
        
        # 結果の保存
        result_summary = {
            'n_frames': lambda_result.n_frames,
            'n_atoms': lambda_result.n_atoms,
            'n_protein_atoms': len(protein_indices),
            'n_critical_events': len(lambda_result.critical_events),
            'computation_time': lambda_result.computation_time,
            'gpu_info': getattr(lambda_result, 'gpu_info', {})
        }
        
        with open(output_path / 'lambda_result_summary.json', 'w') as f:
            json.dump(result_summary, f, indent=2)
            
    except Exception as e:
        logger.error(f"Lambda³ analysis failed: {e}")
        raise
    
    # ========================================
    # Step 3: Two-Stage詳細解析（メイン）
    # ========================================
    two_stage_result = None
    
    if enable_two_stage and len(lambda_result.critical_events) > 0:
        logger.info("\n🔬 Running Two-Stage Residue-Level Analysis...")
        logger.info("   This is the main analysis for protein dynamics")
        
        try:
            # タンパク質残基数の取得
            if 'protein' in metadata and 'n_residues' in metadata['protein']:
                n_protein_residues = metadata['protein']['n_residues']
            elif 'n_protein_residues' in metadata:
                n_protein_residues = metadata['n_protein_residues']
            else:
                # デフォルト値（TrpCageの場合）
                n_protein_residues = 20
                logger.warning(f"   Using default n_residues: {n_protein_residues}")
            
            logger.info(f"   Protein residues: {n_protein_residues}")
            
            # タンパク質部分のトラジェクトリを抽出
            protein_trajectory = trajectory[:, protein_indices, :]
            logger.info(f"   Protein trajectory: {protein_trajectory.shape}")
            
            # イベント窓の作成（critical eventsから）
            events = []
            for i, event in enumerate(lambda_result.critical_events[:10]):  # 最大10イベント
                if isinstance(event, (tuple, list)) and len(event) >= 2:
                    start = int(event[0])
                    end = int(event[1])
                elif hasattr(event, 'frame'):
                    frame = event.frame
                    start = max(0, frame - 100)
                    end = min(n_frames, frame + 100)
                else:
                    continue
                
                # イベント名を付与
                event_name = f'critical_{i}'
                events.append((start, end, event_name))
                logger.info(f"     Event {event_name}: frames {start}-{end}")
            
            if events:
                logger.info(f"   Processing {len(events)} events for residue analysis")
                
                # 残基解析設定
                residue_config = ResidueAnalysisConfig()
                residue_config.sensitivity = 2.0
                residue_config.correlation_threshold = 0.15
                residue_config.use_confidence = True
                residue_config.n_bootstrap = 100
                residue_config.parallel_events = True
                
                # TwoStageAnalyzerGPU初期化
                analyzer = TwoStageAnalyzerGPU(residue_config)
                logger.info("   Two-stage analyzer initialized")
                
                # タンパク質のみのトラジェクトリで解析
                two_stage_result = analyzer.analyze_trajectory(
                    protein_trajectory,      # タンパク質のみ
                    lambda_result,          # マクロ結果
                    events,                 # イベントリスト
                    n_protein_residues      # タンパク質残基数
                )
                
                logger.info("   ✅ Two-stage analysis complete")
                
                # ネットワーク統計を表示
                if hasattr(two_stage_result, 'global_network_stats'):
                    stats = two_stage_result.global_network_stats
                    logger.info("\n   🌐 Global Network Statistics:")
                    logger.info(f"     Total causal links: {stats.get('total_causal_links', 0)}")
                    logger.info(f"     Total sync links: {stats.get('total_sync_links', 0)}")
                    logger.info(f"     Total async bonds: {stats.get('total_async_bonds', 0)}")
                    logger.info(f"     Async/Causal ratio: {stats.get('async_to_causal_ratio', 0):.2%}")
                    logger.info(f"     Mean adaptive window: {stats.get('mean_adaptive_window', 0):.1f} frames")
                
                # 重要残基の表示
                if hasattr(two_stage_result, 'global_residue_importance'):
                    top_residues = sorted(
                        two_stage_result.global_residue_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    logger.info("\n   🎯 Top Important Residues:")
                    for res_id, score in top_residues:
                        if 'protein' in metadata and 'residue_mapping' in metadata['protein']:
                            res_name = metadata['protein']['residue_mapping'].get(
                                str(res_id), {}
                            ).get('name', f'RES{res_id+1}')
                        else:
                            res_name = f'RES{res_id+1}'
                        logger.info(f"     {res_name}: {score:.3f}")
                
                # 介入ポイントの提案
                if hasattr(two_stage_result, 'suggested_intervention_points'):
                    points = two_stage_result.suggested_intervention_points[:3]
                    logger.info(f"\n   💡 Suggested intervention points: {points}")
                    
            else:
                logger.warning("   No events suitable for Two-Stage analysis")
                
        except Exception as e:
            logger.error(f"Two-stage analysis failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            # 続行（量子検証は可能）
    
    # ========================================
    # Step 4: 量子検証（詳細版）
    # ========================================
    logger.info("\n⚛️ Running Quantum Validation...")
    
    quantum_events = []
    
    try:
        # 量子検証器初期化（タンパク質用に最適化）
        quantum_validator = QuantumValidationGPU(
            trajectory=trajectory[:, protein_indices, :],  # タンパク質のみ
            metadata=metadata,
            validation_offset=10,
            min_samples_for_chsh=10
        )
        
        logger.info("   Quantum validator initialized for protein atoms")
        
        # 量子カスケード解析
        quantum_events = quantum_validator.analyze_quantum_cascade(
            lambda_result,
            residue_events=two_stage_result.residue_analyses if two_stage_result else None
        )
        
        logger.info(f"   ✅ Quantum validation complete")
        logger.info(f"   Quantum events detected: {len(quantum_events)}")
        
        # Two-Stage結果との統合
        if two_stage_result and hasattr(two_stage_result, 'residue_analyses'):
            logger.info("   Integrating with Two-Stage results...")
            
            for qevent in quantum_events:
                # 対応する残基解析を探す
                for analysis_name, analysis in two_stage_result.residue_analyses.items():
                    if hasattr(analysis, 'async_strong_bonds'):
                        # async bondsを量子イベントに追加
                        qevent.async_bonds_used = analysis.async_strong_bonds[:5]
                        qevent.residue_context = {
                            'event_name': analysis_name,
                            'n_residues_involved': len(analysis.residue_events),
                            'initiators': analysis.initiator_residues[:3]
                        }
                        break
        
        # Bell違反の統計
        n_bell_violations = sum(1 for e in quantum_events 
                               if hasattr(e, 'quantum_metrics') 
                               and e.quantum_metrics.bell_violated)
        
        if len(quantum_events) > 0:
            violation_rate = 100 * n_bell_violations / len(quantum_events)
            logger.info(f"   Bell violations: {n_bell_violations}/{len(quantum_events)} "
                       f"({violation_rate:.1f}%)")
            
            # CHSH値の統計
            chsh_values = [e.quantum_metrics.chsh_value 
                          for e in quantum_events 
                          if hasattr(e, 'quantum_metrics')]
            if chsh_values:
                logger.info(f"   CHSH values: mean={np.mean(chsh_values):.3f}, "
                           f"max={np.max(chsh_values):.3f}")
        
        # サマリー表示
        quantum_validator.print_validation_summary(quantum_events)
        
        # 量子イベントの詳細保存
        save_quantum_events(quantum_events, output_path)
        
    except Exception as e:
        logger.error(f"Quantum validation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Step 5: 可視化（詳細版）
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating detailed visualizations...")
        
        try:
            visualizer = Lambda3VisualizerGPU()
            
            # Lambda³結果の可視化
            fig_lambda = visualizer.visualize_results(
                lambda_result,
                save_path=str(output_path / 'lambda_results.png')
            )
            logger.info("   Lambda³ results visualized")
            
            # 量子イベントの可視化
            if quantum_events:
                fig_quantum = visualize_quantum_results(
                    quantum_events,
                    save_path=str(output_path / 'quantum_events.png')
                )
                logger.info("   Quantum events visualized")
            
            # Two-Stage結果の可視化
            if two_stage_result:
                fig_network = visualize_residue_network(
                    two_stage_result,
                    save_path=str(output_path / 'residue_network.png')
                )
                logger.info("   Residue network visualized")
            
            logger.info("   ✅ All visualizations saved")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # ========================================
    # Step 6: 包括的レポート生成
    # ========================================
    logger.info("\n📝 Generating comprehensive report...")
    
    report = generate_comprehensive_report(
        lambda_result,
        quantum_events,
        two_stage_result,
        metadata,
        protein_indices
    )
    
    report_path = output_path / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"   Report saved to {report_path}")
    
    # ========================================
    # 完了
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info(f"   Results directory: {output_path}")
    logger.info(f"   Key findings:")
    
    if lambda_result:
        logger.info(f"     - {len(lambda_result.critical_events)} critical events")
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        logger.info(f"     - {two_stage_result.global_network_stats.get('total_causal_links', 0)} causal links")
    if quantum_events:
        logger.info(f"     - {len(quantum_events)} quantum signatures")
    
    logger.info("="*70)
    
    return {
        'lambda_result': lambda_result,
        'quantum_events': quantum_events,
        'two_stage_result': two_stage_result,
        'output_dir': output_path,
        'success': True
    }


# ============================================
# 補助関数群
# ============================================

def save_quantum_events(quantum_events: List, output_path: Path):
    """量子イベントの詳細保存"""
    quantum_data = []
    
    for event in quantum_events:
        event_dict = {
            'frame': getattr(event, 'frame', 0),
            'type': getattr(event, 'event_type', 'unknown'),
            'is_critical': getattr(event, 'is_critical', False)
        }
        
        # 量子メトリクス
        if hasattr(event, 'quantum_metrics'):
            qm = event.quantum_metrics
            event_dict['quantum_metrics'] = {
                'bell_violated': qm.bell_violated,
                'chsh_value': float(qm.chsh_value),
                'chsh_raw_value': float(qm.chsh_raw_value),
                'chsh_confidence': float(qm.chsh_confidence),
                'quantum_score': float(qm.quantum_score),
                'n_samples': qm.n_samples_used
            }
        
        # async bonds情報
        if hasattr(event, 'async_bonds_used') and event.async_bonds_used:
            event_dict['async_bonds'] = []
            for bond in event.async_bonds_used[:5]:
                if isinstance(bond, dict):
                    event_dict['async_bonds'].append({
                        'pair': bond.get('residue_pair', []),
                        'causality': float(bond.get('causality', 0)),
                        'sync_rate': float(bond.get('sync_rate', 0))
                    })
        
        # 残基コンテキスト
        if hasattr(event, 'residue_context'):
            event_dict['residue_context'] = event.residue_context
        
        quantum_data.append(event_dict)
    
    with open(output_path / 'quantum_events_detailed.json', 'w') as f:
        json.dump(quantum_data, f, indent=2)


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
    
    # データ抽出
    frames = [e.frame for e in quantum_events if hasattr(e, 'frame')]
    chsh_values = [e.quantum_metrics.chsh_value 
                  for e in quantum_events 
                  if hasattr(e, 'quantum_metrics')]
    quantum_scores = [e.quantum_metrics.quantum_score 
                     for e in quantum_events 
                     if hasattr(e, 'quantum_metrics')]
    confidences = [e.quantum_metrics.chsh_confidence 
                  for e in quantum_events 
                  if hasattr(e, 'quantum_metrics')]
    
    # 1. CHSH値の時系列
    ax1 = axes[0, 0]
    if frames and chsh_values:
        ax1.scatter(frames, chsh_values[:len(frames)], c='blue', alpha=0.6, s=50)
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
    n_violated = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
    n_classical = len(quantum_events) - n_violated
    
    ax2.pie([n_violated, n_classical], 
           labels=[f'Violated ({n_violated})', f'Classical ({n_classical})'],
           colors=['red', 'blue'],
           autopct='%1.1f%%',
           startangle=90)
    ax2.set_title('Bell Violation Distribution')
    
    # 3. 量子スコア分布
    ax3 = axes[1, 0]
    if quantum_scores:
        ax3.hist(quantum_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Quantum Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Quantum Score Distribution')
        ax3.grid(True, alpha=0.3)
    
    # 4. 信頼度vs CHSH値
    ax4 = axes[1, 1]
    if confidences and chsh_values and quantum_scores:
        scatter = ax4.scatter(confidences[:len(chsh_values)], 
                            chsh_values[:len(confidences)], 
                            c=quantum_scores[:min(len(confidences), len(chsh_values))], 
                            cmap='viridis',
                            s=50, alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label='Quantum Score')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('CHSH Value')
        ax4.set_title('Confidence vs CHSH Value')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Quantum Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_residue_network(two_stage_result: Any,
                             save_path: Optional[str] = None) -> plt.Figure:
    """残基ネットワークの可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 残基重要度ランキング
    ax1 = axes[0, 0]
    if hasattr(two_stage_result, 'global_residue_importance'):
        top_residues = sorted(
            two_stage_result.global_residue_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        if top_residues:
            residue_ids = [f"R{r[0]+1}" for r in top_residues]
            scores = [r[1] for r in top_residues]
            
            ax1.barh(residue_ids, scores, alpha=0.7, color='steelblue')
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Top 20 Important Residues')
            ax1.invert_yaxis()
    
    # 2. ネットワーク統計
    ax2 = axes[0, 1]
    if hasattr(two_stage_result, 'global_network_stats'):
        stats = two_stage_result.global_network_stats
        labels = ['Causal', 'Sync', 'Async']
        values = [
            stats.get('total_causal_links', 0),
            stats.get('total_sync_links', 0),
            stats.get('total_async_bonds', 0)
        ]
        
        if sum(values) > 0:
            ax2.pie(values, labels=labels, autopct='%1.1f%%',
                   startangle=90, colors=['red', 'green', 'blue'])
            ax2.set_title('Network Link Distribution')
    
    # 3. イベント別統計
    ax3 = axes[1, 0]
    if hasattr(two_stage_result, 'residue_analyses'):
        event_names = list(two_stage_result.residue_analyses.keys())
        n_residues = [len(a.residue_events) 
                      for a in two_stage_result.residue_analyses.values()]
        
        if event_names and n_residues:
            ax3.bar(range(len(event_names)), n_residues, alpha=0.7, color='orange')
            ax3.set_xticks(range(len(event_names)))
            ax3.set_xticklabels(event_names, rotation=45, ha='right')
            ax3.set_ylabel('Number of Residues Involved')
            ax3.set_title('Residues per Event')
    
    # 4. GPU性能
    ax4 = axes[1, 1]
    if hasattr(two_stage_result, 'residue_analyses'):
        gpu_times = [a.gpu_time for a in two_stage_result.residue_analyses.values()]
        event_names = list(two_stage_result.residue_analyses.keys())
        
        if gpu_times and event_names:
            ax4.plot(range(len(event_names)), gpu_times, 'go-', 
                    linewidth=2, markersize=8)
            ax4.set_xticks(range(len(event_names)))
            ax4.set_xticklabels(event_names, rotation=45, ha='right')
            ax4.set_ylabel('GPU Time (seconds)')
            ax4.set_title('GPU Processing Time per Event')
            ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Residue-Level Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_comprehensive_report(
    lambda_result: Any,
    quantum_events: List,
    two_stage_result: Optional[Any],
    metadata: Dict,
    protein_indices: np.ndarray
) -> str:
    """包括的な解析レポートの生成"""
    
    # システム名の取得
    system_name = metadata.get('system_name', 'Unknown System')
    
    report = f"""# Lambda³ GPU Analysis Report

## System Information
- **System**: {system_name}
- **Temperature**: {metadata.get('temperature', metadata.get('simulation', {}).get('temperature_K', 310))} K
- **Frames analyzed**: {lambda_result.n_frames}
- **Total atoms**: {lambda_result.n_atoms}
- **Protein atoms**: {len(protein_indices)}
"""
    
    # タンパク質情報
    if 'protein' in metadata:
        protein_info = metadata['protein']
        report += f"""- **Protein residues**: {protein_info.get('n_residues', 'N/A')}
- **Sequence length**: {len(protein_info.get('sequence', ''))}
"""
    
    report += f"""- **Computation time**: {lambda_result.computation_time:.2f} seconds

## Lambda³ Analysis Results
- **Critical events**: {len(lambda_result.critical_events)}
"""
    
    # 構造境界とトポロジカル破れ
    if hasattr(lambda_result, 'structural_boundaries'):
        report += f"- **Structural boundaries**: {len(lambda_result.structural_boundaries)}\n"
    
    if hasattr(lambda_result, 'topological_breaks'):
        report += f"- **Topological breaks**: {len(lambda_result.topological_breaks)}\n"
    
    # Two-Stage解析結果
    if two_stage_result:
        report += f"""
## Two-Stage Residue-Level Analysis

### Overview
"""
        if hasattr(two_stage_result, 'residue_analyses'):
            report += f"- **Events analyzed**: {len(two_stage_result.residue_analyses)}\n"
        
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            report += f"""
### Network Statistics
- **Total causal links**: {stats.get('total_causal_links', 0)}
- **Total sync links**: {stats.get('total_sync_links', 0)}
- **Total async bonds**: {stats.get('total_async_bonds', 0)}
- **Async/Causal ratio**: {stats.get('async_to_causal_ratio', 0):.2%}
- **Mean adaptive window**: {stats.get('mean_adaptive_window', 0):.1f} frames
"""
        
        # 重要残基
        if hasattr(two_stage_result, 'global_residue_importance'):
            top_residues = sorted(
                two_stage_result.global_residue_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            if top_residues:
                report += """
### Top Important Residues
| Rank | Residue | Importance Score |
|------|---------|-----------------|
"""
                for i, (res_id, score) in enumerate(top_residues, 1):
                    # 残基名の取得
                    if 'protein' in metadata and 'residue_mapping' in metadata['protein']:
                        res_name = metadata['protein']['residue_mapping'].get(
                            str(res_id), {}
                        ).get('name', f'RES{res_id+1}')
                    else:
                        res_name = f'RES{res_id+1}'
                    report += f"| {i} | {res_name} | {score:.3f} |\n"
        
        # 介入ポイント
        if hasattr(two_stage_result, 'suggested_intervention_points'):
            points = two_stage_result.suggested_intervention_points[:5]
            if points:
                report += f"\n### Suggested Intervention Points\n"
                report += f"Residues: {points}\n"
    
    # 量子検証結果
    report += f"""
## Quantum Validation Results
- **Total quantum events**: {len(quantum_events)}
"""
    
    if quantum_events:
        n_bell = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
        n_critical = sum(1 for e in quantum_events 
                        if hasattr(e, 'is_critical') and e.is_critical)
        
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
- **Standard deviation**: {np.std(chsh_values):.3f}
- **Classical bound**: 2.000
- **Tsirelson bound**: {2*np.sqrt(2):.3f}

### Interpretation
"""
            if np.max(chsh_values) > 2.0:
                report += "⚛️ **Quantum correlations detected**: CHSH values exceed classical bound\n"
            if np.max(chsh_values) > 2.4:
                report += "🌟 **Strong quantum signatures**: Significant Bell inequality violations\n"
    
    # 結論
    report += f"""
## Conclusions

The Lambda³ GPU analysis of {system_name} successfully completed with the following key findings:

1. **Structural dynamics**: {len(lambda_result.critical_events)} critical events identified
"""
    
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        stats = two_stage_result.global_network_stats
        report += f"""2. **Residue interactions**: {stats.get('total_causal_links', 0)} causal relationships detected
3. **Async coupling**: {stats.get('total_async_bonds', 0)} async strong bonds identified
"""
    
    if quantum_events:
        n_bell = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
        if n_bell > 0:
            report += f"""4. **Quantum signatures**: {n_bell} Bell violations confirm non-classical correlations
"""
    
    report += """
## Recommendations

Based on the analysis results:
"""
    
    # 推奨事項
    if two_stage_result and hasattr(two_stage_result, 'suggested_intervention_points'):
        points = two_stage_result.suggested_intervention_points[:3]
        if points:
            report += f"""
1. **Target residues for intervention**: Focus on residues {points} for potential drug targeting or mutation studies
"""
    
    if quantum_events and len(quantum_events) > 0:
        report += """
2. **Quantum effects**: Consider quantum mechanical effects in protein dynamics modeling
"""
    
    report += """
---
*Generated by Lambda³ GPU Quantum Validation Pipeline*
*Two-Stage Optimized Version*
*NO TIME, NO PHYSICS, ONLY STRUCTURE!*
"""
    
    return report


# ============================================
# CLI Interface
# ============================================

def main():
    """コマンドラインインターフェース"""
    
    parser = argparse.ArgumentParser(
        description='Lambda³ GPU Quantum Validation Pipeline - Two-Stage Optimized',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with protein atoms
  %(prog)s trajectory.npy metadata.json protein.npy
  
  # With custom output directory
  %(prog)s trajectory.npy metadata.json protein.npy --output ./my_results
  
  # Skip visualizations for faster processing
  %(prog)s trajectory.npy metadata.json protein.npy --no-viz
  
  # Verbose mode for debugging
  %(prog)s trajectory.npy metadata.json protein.npy --verbose
        """
    )
    
    # 必須引数
    parser.add_argument('trajectory', 
                       help='Path to trajectory file (.npy)')
    parser.add_argument('metadata', 
                       help='Path to metadata file (.json)')
    parser.add_argument('protein',
                       help='Path to protein indices file (.npy) - use protein.npy for detailed analysis')
    
    # オプション引数
    parser.add_argument('--output', '-o', 
                       default='./quantum_results',
                       help='Output directory (default: ./quantum_results)')
    parser.add_argument('--no-two-stage', 
                       action='store_true',
                       help='Skip two-stage residue analysis')
    parser.add_argument('--no-viz', 
                       action='store_true',
                       help='Skip visualization')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Verbose output for debugging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # パイプライン実行
    try:
        results = run_quantum_validation_pipeline(
            trajectory_path=args.trajectory,
            metadata_path=args.metadata,
            protein_indices_path=args.protein,
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
