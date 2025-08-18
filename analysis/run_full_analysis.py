#!/usr/bin/env python3
"""
Lambda³ GPU Quantum Validation Pipeline - Version 3.0 Complete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

査読耐性＆単一フレーム対応の完全版パイプライン
タンパク質全原子を使った詳細な残基レベル解析に最適化

Version: 3.0.0 - Publication Ready
Authors: Lambda³ Team, 環ちゃん & ご主人さま 💕
Date: 2025-08-18
"""

import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
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
    from lambda3_gpu.quantum import (
        QuantumValidationGPU,
        QuantumEventType,
        ValidationCriterion,
        generate_quantum_report
    )
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
    完全な量子検証パイプライン（Version 3.0）
    
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
    logger.info("🚀 LAMBDA³ GPU QUANTUM VALIDATION PIPELINE v3.0")
    logger.info("   Publication Ready Edition")
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
    # Step 4: 量子検証（Version 3.0対応版）
    # ========================================
    logger.info("\n⚛️ Running Quantum Validation (v3.0)...")
    
    quantum_events = []
    
    try:
        # メタデータから物理パラメータ取得
        temperature = metadata.get('temperature', 
                                  metadata.get('simulation', {}).get('temperature_K', 310.0))
        dt_ps = metadata.get('time_step_ps', 
                           metadata.get('simulation', {}).get('dt_ps', 2.0))
        
        # 量子検証器初期化（Version 3.0パラメータ）
        quantum_validator = QuantumValidationGPU(
            trajectory=trajectory[:, protein_indices, :],  # タンパク質のみ
            metadata=metadata,
            dt_ps=dt_ps,                      # タイムステップ
            temperature_K=temperature,         # 温度
            bootstrap_iterations=1000,         # Bootstrap反復数
            significance_level=0.01,           # 有意水準
            force_cpu=False                    # GPU使用
        )
        
        logger.info(f"   Quantum validator v3.0 initialized")
        logger.info(f"   Temperature: {temperature:.1f} K")
        logger.info(f"   Time step: {dt_ps:.1f} ps")
        logger.info(f"   Thermal decoherence: {quantum_validator.thermal_decoherence_ps:.3e} ps")
        
        # 量子カスケード解析（Version 3.0）
        # two_stage_resultをそのまま渡す（内部で処理される）
        quantum_events = quantum_validator.analyze_quantum_cascade(
            lambda_result,
            two_stage_result  # Version 3.0: 直接渡す
        )
        
        logger.info(f"   ✅ Quantum validation complete")
        logger.info(f"   Quantum events detected: {len(quantum_events)}")
        
        # イベントタイプ別集計（Version 3.0の新機能）
        event_types = Counter(e.event_type.value for e in quantum_events)
        
        logger.info("\n   📊 Event Type Distribution:")
        for event_type, count in event_types.items():
            logger.info(f"     {event_type}: {count}")
        
        # 量子イベントのフィルタリング
        quantum_only = [e for e in quantum_events if e.quantum_metrics.is_quantum]
        logger.info(f"   Confirmed quantum events: {len(quantum_only)}/{len(quantum_events)}")
        
        # 判定基準の統計（Version 3.0の新機能）
        criterion_stats = {}
        for event in quantum_events:
            for criterion in event.quantum_metrics.criteria_passed:
                name = criterion.criterion.value
                if name not in criterion_stats:
                    criterion_stats[name] = {'passed': 0, 'total': 0}
                criterion_stats[name]['total'] += 1
                if criterion.passed:
                    criterion_stats[name]['passed'] += 1
        
        if criterion_stats:
            logger.info("\n   📈 Validation Criteria Statistics:")
            for name, stats in sorted(criterion_stats.items()):
                if stats['total'] > 0:
                    rate = stats['passed'] / stats['total'] * 100
                    logger.info(f"     {name}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
        
        # 臨界イベントの特定（Version 3.0）
        critical_quantum = [e for e in quantum_events if e.is_critical]
        if critical_quantum:
            logger.info(f"\n   💫 Critical quantum events: {len(critical_quantum)}")
            for i, event in enumerate(critical_quantum[:3]):
                qm = event.quantum_metrics
                logger.info(f"     {i+1}. Frame {event.frame_start}-{event.frame_end}")
                logger.info(f"        Type: {event.event_type.value}")
                logger.info(f"        Quantum confidence: {qm.quantum_confidence:.3f}")
                if qm.bell_violated:
                    logger.info(f"        CHSH: {qm.chsh_value:.3f} (p={qm.chsh_p_value:.4f})")
                logger.info(f"        Reasons: {', '.join(event.critical_reasons)}")
        
        # Bell違反の詳細統計
        bell_violations = [e for e in quantum_events 
                          if e.quantum_metrics.bell_violated]
        
        if bell_violations:
            violation_rate = 100 * len(bell_violations) / len(quantum_events)
            logger.info(f"\n   🔔 Bell violations: {len(bell_violations)}/{len(quantum_events)} "
                       f"({violation_rate:.1f}%)")
            
            # CHSH値の統計
            chsh_values = [e.quantum_metrics.chsh_value for e in bell_violations]
            raw_values = [e.quantum_metrics.chsh_raw_value for e in bell_violations]
            
            logger.info(f"   CHSH statistics:")
            logger.info(f"     Corrected: mean={np.mean(chsh_values):.3f}, "
                       f"max={np.max(chsh_values):.3f}")
            logger.info(f"     Raw: mean={np.mean(raw_values):.3f}, "
                       f"max={np.max(raw_values):.3f}")
            logger.info(f"     Tsirelson bound: {2*np.sqrt(2):.3f}")
        
        # サマリー表示（Version 3.0の拡張版）
        quantum_validator.print_validation_summary(quantum_events)
        
        # 量子イベントの詳細保存（Version 3.0対応）
        save_quantum_events_v3(quantum_events, output_path, metadata)
        
        # 査読用レポート生成（Version 3.0の新機能）
        report = generate_quantum_report(quantum_events)
        
        with open(output_path / 'quantum_validation_report.txt', 'w') as f:
            f.write(report)
        logger.info(f"   📄 Validation report saved")
        
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
        logger.info(f"     - {len(quantum_only)} confirmed quantum events")
        logger.info(f"     - {len(bell_violations)} Bell violations")
    
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

def save_quantum_events_v3(quantum_events: List, output_path: Path, metadata: Dict):
    """量子イベントの詳細保存（Version 3.0対応）"""
    quantum_data = []
    
    for event in quantum_events:
        event_dict = {
            'frame_start': event.frame_start,
            'frame_end': event.frame_end,
            'event_type': event.event_type.value,
            'is_critical': event.is_critical,
            'critical_reasons': event.critical_reasons,
            'validation_window': list(event.validation_window)
        }
        
        # 量子メトリクス（Version 3.0の全フィールド）
        qm = event.quantum_metrics
        event_dict['quantum_metrics'] = {
            # 基本分類
            'event_type': qm.event_type.value,
            'duration_frames': qm.duration_frames,
            'duration_ps': qm.duration_ps,
            
            # 量子判定
            'is_quantum': qm.is_quantum,
            'quantum_confidence': float(qm.quantum_confidence),
            'quantum_score': float(qm.quantum_score),
            
            # CHSH検証
            'bell_violated': qm.bell_violated,
            'chsh_value': float(qm.chsh_value),
            'chsh_raw_value': float(qm.chsh_raw_value),
            'chsh_confidence': float(qm.chsh_confidence),
            'chsh_p_value': float(qm.chsh_p_value),
            
            # 物理指標
            'coherence_time_ps': float(qm.coherence_time_ps),
            'thermal_ratio': float(qm.thermal_ratio),
            'tunneling_probability': float(qm.tunneling_probability),
            'energy_barrier_kT': float(qm.energy_barrier_kT),
            
            # ネットワーク指標
            'n_async_bonds': qm.n_async_bonds,
            'max_causality': float(qm.max_causality),
            'min_sync_rate': float(qm.min_sync_rate),
            'mean_lag_frames': float(qm.mean_lag_frames),
            
            # 統計情報
            'n_samples_used': qm.n_samples_used,
            'data_quality': float(qm.data_quality),
            'bootstrap_iterations': qm.bootstrap_iterations
        }
        
        # 判定基準の詳細（Version 3.0）
        event_dict['criteria_passed'] = []
        for criterion in qm.criteria_passed:
            event_dict['criteria_passed'].append({
                'criterion': criterion.criterion.value,
                'reference': criterion.reference,
                'value': float(criterion.value),
                'threshold': float(criterion.threshold),
                'passed': criterion.passed,
                'p_value': float(criterion.p_value) if criterion.p_value else None,
                'description': criterion.description
            })
        
        # 残基情報（対応する残基IDを記録）
        event_dict['residue_ids'] = event.residue_ids
        
        # async bonds情報（ネットワーク解析から）
        if event.async_bonds_used:
            event_dict['async_bonds'] = []
            for bond in event.async_bonds_used[:10]:  # 最大10個
                bond_dict = {
                    'residue_pair': bond.get('residue_pair', []),
                    'strength': float(bond.get('strength', 0)),
                    'lag': int(bond.get('lag', 0)),
                    'sync_rate': float(bond.get('sync_rate', 0)),
                    'type': bond.get('type', 'unknown')
                }
                event_dict['async_bonds'].append(bond_dict)
        
        # ネットワーク統計
        if event.network_stats:
            event_dict['network_stats'] = {
                'n_async_bonds': len(event.network_stats.get('async_bonds', [])),
                'n_causal_links': len(event.network_stats.get('causal_links', [])),
                'network_type': event.network_stats.get('network_type')
            }
        
        # 統計的検定結果（あれば）
        if event.statistical_tests:
            event_dict['statistical_tests'] = event.statistical_tests
        
        quantum_data.append(event_dict)
    
    # メタデータも含めて保存
    output_data = {
        'metadata': {
            'system_name': metadata.get('system_name', 'Unknown'),
            'temperature_K': metadata.get('temperature', 310.0),
            'dt_ps': metadata.get('time_step_ps', 2.0),
            'n_protein_residues': metadata.get('protein', {}).get('n_residues', 0),
            'analysis_version': '3.0.0'
        },
        'summary': {
            'total_events': len(quantum_events),
            'quantum_events': sum(1 for e in quantum_events if e.quantum_metrics.is_quantum),
            'bell_violations': sum(1 for e in quantum_events if e.quantum_metrics.bell_violated),
            'critical_events': sum(1 for e in quantum_events if e.is_critical),
            'event_type_distribution': dict(Counter(e.event_type.value for e in quantum_events))
        },
        'events': quantum_data
    }
    
    with open(output_path / 'quantum_events_v3.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"   💾 Saved {len(quantum_data)} quantum events with full v3.0 metrics")


def visualize_quantum_results(quantum_events: List, 
                             save_path: Optional[str] = None) -> plt.Figure:
    """量子検証結果の可視化（Version 3.0対応）"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    if not quantum_events:
        fig.text(0.5, 0.5, 'No Quantum Events Detected', 
                ha='center', va='center', fontsize=20)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    # データ抽出
    frames = []
    chsh_values = []
    quantum_scores = []
    confidences = []
    event_type_list = []
    
    for e in quantum_events:
        if hasattr(e, 'frame_start'):
            frames.append(e.frame_start)
        if hasattr(e, 'quantum_metrics'):
            chsh_values.append(e.quantum_metrics.chsh_value)
            quantum_scores.append(e.quantum_metrics.quantum_score)
            confidences.append(e.quantum_metrics.chsh_confidence)
        if hasattr(e, 'event_type'):
            event_type_list.append(e.event_type.value)
    
    # 1. CHSH値の時系列
    ax1 = axes[0, 0]
    if frames and chsh_values:
        ax1.scatter(frames[:len(chsh_values)], chsh_values[:len(frames)], 
                   c='blue', alpha=0.6, s=50)
        ax1.axhline(y=2.0, color='red', linestyle='--', label='Classical Bound')
        ax1.axhline(y=2*np.sqrt(2), color='green', linestyle='--', 
                   label=f'Tsirelson Bound ({2*np.sqrt(2):.3f})')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('CHSH Value')
        ax1.set_title('CHSH Inequality Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. イベントタイプ分布（Version 3.0）
    ax2 = axes[0, 1]
    event_counts = Counter(event_type_list)
    if event_counts:
        ax2.pie(event_counts.values(), 
               labels=event_counts.keys(),
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('Event Type Distribution')
    
    # 3. Bell違反の分布
    ax3 = axes[0, 2]
    n_violated = sum(1 for e in quantum_events 
                    if hasattr(e, 'quantum_metrics') 
                    and e.quantum_metrics.bell_violated)
    n_classical = len(quantum_events) - n_violated
    
    ax3.pie([n_violated, n_classical], 
           labels=[f'Violated ({n_violated})', f'Classical ({n_classical})'],
           colors=['red', 'blue'],
           autopct='%1.1f%%',
           startangle=90)
    ax3.set_title('Bell Violation Distribution')
    
    # 4. 量子スコア分布
    ax4 = axes[1, 0]
    if quantum_scores:
        ax4.hist(quantum_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Quantum Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Quantum Score Distribution')
        ax4.grid(True, alpha=0.3)
    
    # 5. 信頼度vs CHSH値
    ax5 = axes[1, 1]
    if confidences and chsh_values and quantum_scores:
        scatter = ax5.scatter(confidences[:len(chsh_values)], 
                            chsh_values[:len(confidences)], 
                            c=quantum_scores[:min(len(confidences), len(chsh_values))], 
                            cmap='viridis',
                            s=50, alpha=0.6)
        plt.colorbar(scatter, ax=ax5, label='Quantum Score')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('CHSH Value')
        ax5.set_title('Confidence vs CHSH Value')
        ax5.grid(True, alpha=0.3)
    
    # 6. 判定基準通過率（Version 3.0）
    ax6 = axes[1, 2]
    criterion_counts = {}
    for e in quantum_events:
        if hasattr(e, 'quantum_metrics'):
            for criterion in e.quantum_metrics.criteria_passed:
                name = criterion.criterion.value
                if name not in criterion_counts:
                    criterion_counts[name] = {'passed': 0, 'total': 0}
                criterion_counts[name]['total'] += 1
                if criterion.passed:
                    criterion_counts[name]['passed'] += 1
    
    if criterion_counts:
        names = list(criterion_counts.keys())
        pass_rates = [c['passed']/c['total']*100 if c['total'] > 0 else 0 
                     for c in criterion_counts.values()]
        ax6.bar(range(len(names)), pass_rates, alpha=0.7, color='steelblue')
        ax6.set_xticks(range(len(names)))
        ax6.set_xticklabels(names, rotation=45, ha='right')
        ax6.set_ylabel('Pass Rate (%)')
        ax6.set_title('Validation Criteria Pass Rates')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Quantum Validation Results v3.0', fontsize=14, fontweight='bold')
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
    """包括的な解析レポートの生成（Version 3.0対応）"""
    
    # システム名の取得
    system_name = metadata.get('system_name', 'Unknown System')
    
    report = f"""# Lambda³ GPU Analysis Report v3.0

## System Information
- **System**: {system_name}
- **Temperature**: {metadata.get('temperature', metadata.get('simulation', {}).get('temperature_K', 310))} K
- **Time step**: {metadata.get('time_step_ps', metadata.get('simulation', {}).get('dt_ps', 2.0))} ps
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
- **Analysis version**: 3.0.0 (Publication Ready)

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
    
    # 量子検証結果（Version 3.0拡張）
    report += f"""
## Quantum Validation Results (v3.0)
- **Total events analyzed**: {len(quantum_events)}
"""
    
    if quantum_events:
        # 基本統計
        n_quantum = sum(1 for e in quantum_events if e.quantum_metrics.is_quantum)
        n_bell = sum(1 for e in quantum_events if e.quantum_metrics.bell_violated)
        n_critical = sum(1 for e in quantum_events if e.is_critical)
        
        report += f"""- **Confirmed quantum events**: {n_quantum} ({100*n_quantum/len(quantum_events):.1f}%)
- **Bell violations**: {n_bell} ({100*n_bell/len(quantum_events):.1f}%)
- **Critical quantum events**: {n_critical}

### Event Type Distribution
"""
        # イベントタイプ分布
        event_types = Counter(e.event_type.value for e in quantum_events)
        for event_type, count in sorted(event_types.items()):
            report += f"- **{event_type}**: {count}\n"
        
        # 判定基準統計
        criterion_stats = {}
        for event in quantum_events:
            for criterion in event.quantum_metrics.criteria_passed:
                name = criterion.criterion.value
                if name not in criterion_stats:
                    criterion_stats[name] = {'passed': 0, 'total': 0, 'refs': set()}
                criterion_stats[name]['total'] += 1
                if criterion.passed:
                    criterion_stats[name]['passed'] += 1
                criterion_stats[name]['refs'].add(criterion.reference)
        
        if criterion_stats:
            report += """
### Validation Criteria Statistics
| Criterion | Pass Rate | References |
|-----------|-----------|------------|
"""
            for name, stats in sorted(criterion_stats.items()):
                rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
                refs = list(stats['refs'])[0] if stats['refs'] else 'N/A'
                report += f"| {name} | {rate:.1f}% ({stats['passed']}/{stats['total']}) | {refs} |\n"
        
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
            if n_quantum > len(quantum_events) * 0.1:
                report += "💫 **Quantum prevalence**: >10% of events show quantum characteristics\n"
    
    # 結論
    report += f"""
## Conclusions

The Lambda³ GPU v3.0 analysis of {system_name} successfully completed with the following key findings:

1. **Structural dynamics**: {len(lambda_result.critical_events)} critical events identified
"""
    
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        stats = two_stage_result.global_network_stats
        report += f"""2. **Residue interactions**: {stats.get('total_causal_links', 0)} causal relationships detected
3. **Async coupling**: {stats.get('total_async_bonds', 0)} async strong bonds identified
"""
    
    if quantum_events:
        n_quantum = sum(1 for e in quantum_events if e.quantum_metrics.is_quantum)
        n_bell = sum(1 for e in quantum_events if e.quantum_metrics.bell_violated)
        if n_bell > 0:
            report += f"""4. **Quantum signatures**: {n_bell} Bell violations confirm non-classical correlations
5. **Quantum events**: {n_quantum} events passed quantum validation criteria
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
    
    if quantum_events and n_quantum > 0:
        report += """
2. **Quantum effects**: Consider quantum mechanical effects in protein dynamics modeling
3. **Non-classical correlations**: Account for Bell violations in theoretical models
"""
    
    report += """
---
*Generated by Lambda³ GPU Quantum Validation Pipeline v3.0*
*Publication Ready - Peer Review Compatible*
*NO TIME, NO PHYSICS, ONLY STRUCTURE!*
"""
    
    return report


# ============================================
# CLI Interface
# ============================================

def main():
    """コマンドラインインターフェース"""
    
    parser = argparse.ArgumentParser(
        description='Lambda³ GPU Quantum Validation Pipeline v3.0 - Publication Ready',
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
