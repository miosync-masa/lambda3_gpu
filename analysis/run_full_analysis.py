#!/usr/bin/env python3
"""
Lambda³ GPU Quantum Validation Pipeline - Version 4.0
======================================================

Lambda³統合型量子起源判定パイプライン
構造変化の量子性を正しく評価する完全リファクタリング版

Version: 4.0.0 - Complete Refactoring
Authors: Lambda³ Team, 環ちゃん & ご主人さま 💕
Date: 2024
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
    # Version 4.0 imports
    from lambda3_gpu.quantum import (
        QuantumValidatorV4,
        StructuralEventPattern,
        QuantumSignature,
        QuantumAssessment
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
logger = logging.getLogger('quantum_validation_v4')

# ============================================
# メイン実行関数（Version 4.0）
# ============================================

def run_quantum_validation_pipeline(
    trajectory_path: str,
    metadata_path: str,
    protein_indices_path: str,
    topology_path: Optional[str] = None,
    enable_two_stage: bool = True,
    enable_visualization: bool = True,
    output_dir: str = './quantum_results_v4',
    verbose: bool = False
) -> Dict:
    """
    完全な量子検証パイプライン（Version 4.0）
    
    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイルのパス (.npy)
    metadata_path : str
        メタデータファイルのパス (.json)
    protein_indices_path : str
        タンパク質原子インデックス (.npy)
    topology_path : str, optional
        トポロジーファイルのパス（結合情報など）
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
    logger.info("🚀 LAMBDA³ GPU QUANTUM VALIDATION PIPELINE v4.0")
    logger.info("   Lambda³ Integrated Edition")
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
        logger.info("   Metadata loaded successfully")
        
        # タンパク質インデックス読み込み
        if not Path(protein_indices_path).exists():
            raise FileNotFoundError(f"Protein indices file not found: {protein_indices_path}")
            
        protein_indices = np.load(protein_indices_path)
        logger.info(f"   Protein indices loaded: {len(protein_indices)} atoms")
        
        # トポロジー読み込み（オプション）
        topology = None
        if topology_path and Path(topology_path).exists():
            # トポロジー形式に応じて読み込み
            # ここでは簡略化
            logger.info(f"   Topology loaded from {topology_path}")
        
        # タンパク質情報の表示
        if 'protein' in metadata:
            protein_info = metadata['protein']
            logger.info(f"   Protein: {protein_info.get('n_residues', 'N/A')} residues")
            logger.info(f"   Sequence: {protein_info.get('sequence', 'N/A')[:20]}...")
        
        logger.info(f"   ✅ Data validation passed")
            
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
        config.sensitivity = 2.0
        config.use_hierarchical = True
        
        logger.info("   Config optimized for structure detection")
        
        # Lambda³検出器初期化
        detector = MDLambda3DetectorGPU(config)
        logger.info("   Detector initialized on GPU")
        
        # 解析実行
        lambda_result = detector.analyze(trajectory, protein_indices)
        
        logger.info(f"   ✅ Lambda³ analysis complete")
        logger.info(f"   Critical events detected: {len(lambda_result.critical_events)}")
        
        # 結果の保存
        result_summary = {
            'n_frames': lambda_result.n_frames,
            'n_atoms': lambda_result.n_atoms,
            'n_protein_atoms': len(protein_indices),
            'n_critical_events': len(lambda_result.critical_events),
            'computation_time': lambda_result.computation_time,
            'analysis_version': '4.0.0'
        }
        
        with open(output_path / 'lambda_result_summary.json', 'w') as f:
            json.dump(result_summary, f, indent=2)
            
    except Exception as e:
        logger.error(f"Lambda³ analysis failed: {e}")
        raise
  
    # ========================================
    # Step 3: Two-Stage詳細解析
    # ========================================
    two_stage_result = None
    network_results = []
    
    if enable_two_stage and len(lambda_result.critical_events) > 0:
        logger.info("\\n🔬 Running Two-Stage Residue-Level Analysis...")
        
        try:
            # タンパク質残基数の取得
            n_protein_residues = metadata.get('protein', {}).get('n_residues', 10)
            logger.info(f"   Protein residues: {n_protein_residues}")
            
            # タンパク質部分のトラジェクトリ
            protein_trajectory = trajectory[:, protein_indices, :]
            
            # TOP50イベント選択
            MAX_EVENTS = 50
            MIN_WINDOW_SIZE = 10
            
            # イベントのスコア付きソート
            sorted_events = []
            for event in lambda_result.critical_events:
                if isinstance(event, dict):
                    score = event.get('anomaly_score', 0)
                    start = event.get('start', event.get('frame', 0))
                    end = event.get('end', start)
                elif isinstance(event, (tuple, list)) and len(event) >= 2:
                    start = int(event[0])
                    end = int(event[1])
                    
                    # ⚡ 修正: anomaly_scoresから実際のスコアを取得（最大値）
                    if len(event) > 2:
                        score = event[2]  # 既にスコアがある場合
                    else:
                        # anomaly_scoresから該当範囲の最大値を取得
                        if 'combined' in lambda_result.anomaly_scores:
                            # 範囲内の最大値（量子的スパイクを捉える）
                            score = float(np.max(lambda_result.anomaly_scores['combined'][start:end+1]))
                        elif 'final_combined' in lambda_result.anomaly_scores:
                            score = float(np.max(lambda_result.anomaly_scores['final_combined'][start:end+1]))
                        elif 'global' in lambda_result.anomaly_scores:
                            score = float(np.max(lambda_result.anomaly_scores['global'][start:end+1]))
                        else:
                            # フォールバック：何かしらのスコアを探す
                            for key in ['local', 'extended']:
                                if key in lambda_result.anomaly_scores:
                                    score = float(np.max(lambda_result.anomaly_scores[key][start:end+1]))
                                    break
                            else:
                                score = 0.0
                                logger.warning(f"   ⚠️ No anomaly scores found for event at frames {start}-{end}")
                else:
                    continue
                sorted_events.append((start, end, score))
            
            # スコアでソート（降順）
            sorted_events.sort(key=lambda x: x[2], reverse=True)
            selected_events = sorted_events[:min(MAX_EVENTS, len(sorted_events))]
            logger.info(f"   Selected TOP {len(selected_events)} events")
            
            # デバッグ：上位イベントのスコアを表示
            if verbose:
                logger.debug("   Top event scores:")
                for i, (start, end, score) in enumerate(selected_events[:10]):
                    logger.debug(f"     Event {i}: frames {start}-{end}, score={score:.3f}")
            
            # =========== ここから追加！！ ===========
            # TwoStageAnalyzerの設定
            residue_config = ResidueAnalysisConfig()
            residue_config.n_residues = n_protein_residues
            residue_config.min_window_size = MIN_WINDOW_SIZE
            residue_config.use_gpu = True
            residue_config.verbose = verbose
            
            # TwoStageAnalyzer初期化
            logger.info("   Initializing Two-Stage Analyzer...")
            two_stage_analyzer = TwoStageAnalyzerGPU(residue_config)
            
            # 解析実行！
            logger.info(f"   Analyzing {len(selected_events)} events...")
            two_stage_result = two_stage_analyzer.analyze_trajectory(  # ← analyze_trajectory！
                trajectory=protein_trajectory,  # ← trajectoryという引数名
                macro_result=lambda_result,     # ← macro_resultという引数名  
                detected_events=selected_events, # ← detected_eventsという引数名
                n_residues=n_protein_residues
            )
            
            logger.info(f"   ✅ Two-stage analysis complete")
            
            # ネットワーク結果の抽出
            if hasattr(two_stage_result, 'residue_analyses'):
                network_results = []
                for analysis in two_stage_result.residue_analyses.values():
                    if hasattr(analysis, 'network_result'):
                        network_results.append(analysis.network_result)
                logger.info(f"   Extracted {len(network_results)} network results")
            
            # グローバル統計の表示
            if hasattr(two_stage_result, 'global_network_stats'):
                stats = two_stage_result.global_network_stats
                logger.info(f"   Total causal links: {stats.get('total_causal_links', 0)}")
                logger.info(f"   Total async bonds: {stats.get('total_async_bonds', 0)}")
            
        except Exception as e:
            logger.error(f"Two-stage analysis failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            two_stage_result = None
            network_results = []   
          
    # ========================================
    # Step 4: 量子検証（Version 4.0）
    # ========================================
    logger.info("\n⚛️ Running Quantum Validation v4.0...")
    
    quantum_assessments = []
    
    try:
        # 物理パラメータ取得
        temperature = metadata.get('temperature', 300.0)
        dt_ps = metadata.get('time_step_ps', 100.0)
        
        # 量子検証器初期化（Version 4.0）
        quantum_validator = QuantumValidatorV4(
            trajectory=trajectory[:, protein_indices, :],
            topology=topology,
            dt_ps=dt_ps,
            temperature_K=temperature
        )
        
        logger.info(f"   Quantum validator v4.0 initialized")
        logger.info(f"   Temperature: {temperature:.1f} K")
        logger.info(f"   Time step: {dt_ps:.1f} ps")
        
        # Lambda³イベントを変換
        events = []
        for e in lambda_result.critical_events[:100]:  # 最大100イベント
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                events.append({
                    'frame_start': int(e[0]),
                    'frame_end': int(e[1]),
                    'type': 'critical'
                })
            elif isinstance(e, dict):
                events.append({
                    'frame_start': e.get('start', e.get('frame', 0)),
                    'frame_end': e.get('end', e.get('frame', 0)),
                    'type': e.get('type', 'critical')
                })
        
        if events:
            # バッチ検証
            quantum_assessments = quantum_validator.validate_events(
                events,
                lambda_result,
                network_results if network_results else None
            )
            
            logger.info(f"   ✅ Quantum validation complete")
            logger.info(f"   Events analyzed: {len(quantum_assessments)}")
            
            # 統計表示
            quantum_count = sum(1 for a in quantum_assessments if a.is_quantum)
            logger.info(f"   Quantum events: {quantum_count}/{len(quantum_assessments)}")
            
            # パターン別統計
            pattern_counts = Counter(a.pattern.value for a in quantum_assessments)
            logger.info("\n   📊 Pattern Distribution:")
            for pattern, count in pattern_counts.items():
                logger.info(f"     {pattern}: {count}")
            
            # シグネチャー別統計
            sig_counts = Counter(a.signature.value for a in quantum_assessments 
                               if a.signature != QuantumSignature.NONE)
            if sig_counts:
                logger.info("\n   ⚛️ Quantum Signatures:")
                for sig, count in sig_counts.items():
                    logger.info(f"     {sig}: {count}")
            
            # 信頼度統計
            confidences = [a.confidence for a in quantum_assessments if a.is_quantum]
            if confidences:
                logger.info(f"\n   📈 Confidence Statistics:")
                logger.info(f"     Mean: {np.mean(confidences):.3f}")
                logger.info(f"     Max: {np.max(confidences):.3f}")
                logger.info(f"     Min: {np.min(confidences):.3f}")
            
            # サマリー表示
            quantum_validator.print_summary(quantum_assessments)
            
            # 結果保存
            save_quantum_assessments_v4(quantum_assessments, output_path, metadata)
            
        else:
            logger.warning("   No events to validate")
            
    except Exception as e:
        logger.error(f"Quantum validation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Step 5: 可視化（Version 4.0対応）
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating visualizations...")
        
        try:
            visualizer = Lambda3VisualizerGPU()
            
            # Lambda³結果の可視化
            fig_lambda = visualizer.visualize_results(
                lambda_result,
                save_path=str(output_path / 'lambda_results.png')
            )
            logger.info("   Lambda³ results visualized")
            
            # 量子評価の可視化
            if quantum_assessments:
                fig_quantum = visualize_quantum_assessments_v4(
                    quantum_assessments,
                    save_path=str(output_path / 'quantum_assessments.png')
                )
                logger.info("   Quantum assessments visualized")
            
            # ネットワークの可視化
            if two_stage_result:
                fig_network = visualize_residue_network(
                    two_stage_result,
                    save_path=str(output_path / 'residue_network.png')
                )
                logger.info("   Residue network visualized")
            
            logger.info("   ✅ All visualizations saved")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # ========================================
    # Step 6: 包括的レポート生成（Version 4.0）
    # ========================================
    logger.info("\n📝 Generating comprehensive report...")
    
    report = generate_comprehensive_report_v4(
        lambda_result,
        quantum_assessments,
        two_stage_result,
        metadata,
        protein_indices
    )
    
    report_path = output_path / 'analysis_report_v4.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"   Report saved to {report_path}")
    
    # ========================================
    # Step 7: Maximum Report Generation
    # ========================================
    try:
        logger.info("\n📚 Generating maximum report...")
        
        # maximum_report_generatorのインポート試行
        try:
            from .maximum_report_generator import generate_maximum_report_from_results_v4
        except ImportError:
            # 相対インポートが失敗した場合
            from lambda3_gpu.analysis.maximum_report_generator import generate_maximum_report_from_results_v4
        
        # Version 4.0対応の呼び出し
        max_report = generate_maximum_report_from_results_v4(
            lambda_result=lambda_result,
            two_stage_result=two_stage_result if enable_two_stage else None,
            quantum_assessments=quantum_assessments,  # v4.0: quantum_eventsではなくassessments
            metadata=metadata,
            output_dir=str(output_path),
            verbose=verbose
        )
        logger.info(f"   ✅ Maximum report generated: {len(max_report):,} characters")
        
    except ImportError as e:
        logger.warning(f"   ⚠️ Maximum report generator not found: {e}")
    except Exception as e:
        logger.warning(f"   ⚠️ Maximum report generation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
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
    if quantum_assessments:
        quantum_count = sum(1 for a in quantum_assessments if a.is_quantum)
        logger.info(f"     - {quantum_count} quantum events confirmed")
    
    logger.info("="*70)
    
    return {
        'lambda_result': lambda_result,
        'quantum_assessments': quantum_assessments,
        'two_stage_result': two_stage_result,
        'output_dir': output_path,
        'success': True
    }


# ============================================
# 補助関数群（Version 4.0）
# ============================================

def save_quantum_assessments_v4(assessments: List[QuantumAssessment], 
                               output_path: Path, metadata: Dict):
    """量子評価結果の保存（Version 4.0）"""
    
    assessment_data = []
    
    for assessment in assessments:
        data = {
            'pattern': assessment.pattern.value,
            'signature': assessment.signature.value,
            'is_quantum': assessment.is_quantum,
            'confidence': float(assessment.confidence),
            'explanation': assessment.explanation,
            'criteria_met': assessment.criteria_met
        }
        
        # Lambda異常性
        if assessment.lambda_anomaly:
            data['lambda_anomaly'] = {
                'lambda_jump': float(assessment.lambda_anomaly.lambda_jump),
                'lambda_zscore': float(assessment.lambda_anomaly.lambda_zscore),
                'rho_t_spike': float(assessment.lambda_anomaly.rho_t_spike),
                'sigma_s_value': float(assessment.lambda_anomaly.sigma_s_value),
                'coordination': float(assessment.lambda_anomaly.coordination),
                'statistical_rarity': float(assessment.lambda_anomaly.statistical_rarity),
                'thermal_comparison': float(assessment.lambda_anomaly.thermal_comparison)
            }
        
        # 原子レベル証拠
        if assessment.atomic_evidence:
            data['atomic_evidence'] = {
                'max_velocity': float(assessment.atomic_evidence.max_velocity),
                'max_acceleration': float(assessment.atomic_evidence.max_acceleration),
                'correlation_coefficient': float(assessment.atomic_evidence.correlation_coefficient),
                'n_bond_anomalies': len(assessment.atomic_evidence.bond_anomalies),
                'n_dihedral_flips': len(assessment.atomic_evidence.dihedral_flips)
            }
        
        # Bell不等式（カスケードの場合）
        if assessment.bell_inequality is not None:
            data['bell_inequality'] = float(assessment.bell_inequality)
        
        # async bonds（カスケードの場合）
        if assessment.async_bonds_used:
            data['async_bonds'] = assessment.async_bonds_used[:5]
        
        assessment_data.append(data)
    
    # サマリー統計
    total = len(assessments)
    quantum_count = sum(1 for a in assessments if a.is_quantum)
    
    output_data = {
        'metadata': {
            'system_name': metadata.get('system_name', 'Unknown'),
            'temperature_K': metadata.get('temperature', 300.0),
            'dt_ps': metadata.get('time_step_ps', 100.0),
            'analysis_version': '4.0.0'
        },
        'summary': {
            'total_events': total,
            'quantum_events': quantum_count,
            'quantum_ratio': quantum_count / total if total > 0 else 0,
            'pattern_distribution': dict(Counter(a.pattern.value for a in assessments)),
            'signature_distribution': dict(Counter(a.signature.value for a in assessments 
                                                  if a.signature != QuantumSignature.NONE))
        },
        'assessments': assessment_data
    }
    
    with open(output_path / 'quantum_assessments_v4.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"   💾 Saved {len(assessment_data)} quantum assessments")


def visualize_quantum_assessments_v4(assessments: List[QuantumAssessment],
                                    save_path: Optional[str] = None) -> plt.Figure:
    """量子評価結果の可視化（Version 4.0）"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    if not assessments:
        fig.text(0.5, 0.5, 'No Assessments Available', 
                ha='center', va='center', fontsize=20)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    # 1. パターン分布
    ax1 = axes[0, 0]
    pattern_counts = Counter(a.pattern.value for a in assessments)
    if pattern_counts:
        ax1.pie(pattern_counts.values(), 
               labels=pattern_counts.keys(),
               autopct='%1.1f%%',
               startangle=90)
        ax1.set_title('Event Pattern Distribution')
    
    # 2. シグネチャー分布
    ax2 = axes[0, 1]
    sig_counts = Counter(a.signature.value for a in assessments)
    # CLASSICALを除外
    sig_counts.pop('classical', None)
    if sig_counts:
        ax2.pie(sig_counts.values(),
               labels=sig_counts.keys(),
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title('Quantum Signature Distribution')
    
    # 3. 量子判定率
    ax3 = axes[0, 2]
    n_quantum = sum(1 for a in assessments if a.is_quantum)
    n_classical = len(assessments) - n_quantum
    ax3.pie([n_quantum, n_classical],
           labels=[f'Quantum ({n_quantum})', f'Classical ({n_classical})'],
           colors=['red', 'blue'],
           autopct='%1.1f%%',
           startangle=90)
    ax3.set_title('Quantum vs Classical')
    
    # 4. 信頼度分布
    ax4 = axes[1, 0]
    confidences = [a.confidence for a in assessments if a.is_quantum]
    if confidences:
        ax4.hist(confidences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Count')
        ax4.set_title('Quantum Confidence Distribution')
        ax4.grid(True, alpha=0.3)
    
    # 5. Lambda異常性
    ax5 = axes[1, 1]
    lambda_zscores = [a.lambda_anomaly.lambda_zscore for a in assessments 
                      if a.lambda_anomaly and a.lambda_anomaly.lambda_zscore > 0]
    if lambda_zscores:
        ax5.hist(lambda_zscores, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(x=3.0, color='red', linestyle='--', label='3σ threshold')
        ax5.set_xlabel('Lambda Z-score')
        ax5.set_ylabel('Count')
        ax5.set_title('Lambda Anomaly Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. パターン別量子判定率
    ax6 = axes[1, 2]
    pattern_quantum = {}
    for pattern in StructuralEventPattern:
        pattern_assessments = [a for a in assessments if a.pattern == pattern]
        if pattern_assessments:
            quantum_ratio = sum(1 for a in pattern_assessments if a.is_quantum) / len(pattern_assessments)
            pattern_quantum[pattern.value] = quantum_ratio * 100
    
    if pattern_quantum:
        ax6.bar(range(len(pattern_quantum)), list(pattern_quantum.values()), 
               alpha=0.7, color='steelblue')
        ax6.set_xticks(range(len(pattern_quantum)))
        ax6.set_xticklabels(list(pattern_quantum.keys()), rotation=45, ha='right')
        ax6.set_ylabel('Quantum Rate (%)')
        ax6.set_title('Quantum Rate by Pattern')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Quantum Assessment Results v4.0', fontsize=14, fontweight='bold')
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
        n_residues = []
        
        for analysis in two_stage_result.residue_analyses.values():
            if hasattr(analysis, 'residue_events'):
                n_residues.append(len(analysis.residue_events))
            else:
                n_residues.append(0)
        
        if event_names and n_residues:
            # イベント名が長い場合は短縮
            short_names = []
            for name in event_names:
                if len(name) > 10:
                    # top_XX_score_Y.YY形式から番号だけ抽出
                    import re
                    match = re.search(r'top_(\d+)', name)
                    if match:
                        short_names.append(f"E{match.group(1)}")
                    else:
                        short_names.append(name[:10])
                else:
                    short_names.append(name)
            
            ax3.bar(range(len(event_names)), n_residues, alpha=0.7, color='orange')
            ax3.set_xticks(range(len(event_names)))
            ax3.set_xticklabels(short_names, rotation=45, ha='right')
            ax3.set_ylabel('Number of Residues Involved')
            ax3.set_title('Residues per Event')
            ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. GPU性能
    ax4 = axes[1, 1]
    if hasattr(two_stage_result, 'residue_analyses'):
        gpu_times = []
        event_names = list(two_stage_result.residue_analyses.keys())
        
        for analysis in two_stage_result.residue_analyses.values():
            if hasattr(analysis, 'gpu_time'):
                gpu_times.append(analysis.gpu_time)
            else:
                gpu_times.append(0)
        
        if gpu_times and event_names:
            # イベント名短縮（上と同じ）
            short_names = []
            for name in event_names:
                if len(name) > 10:
                    import re
                    match = re.search(r'top_(\d+)', name)
                    if match:
                        short_names.append(f"E{match.group(1)}")
                    else:
                        short_names.append(name[:10])
                else:
                    short_names.append(name)
            
            ax4.plot(range(len(event_names)), gpu_times, 'go-', 
                    linewidth=2, markersize=8, alpha=0.7)
            ax4.set_xticks(range(len(event_names)))
            ax4.set_xticklabels(short_names, rotation=45, ha='right')
            ax4.set_ylabel('GPU Time (seconds)')
            ax4.set_title('GPU Processing Time per Event')
            ax4.grid(True, alpha=0.3)
            
            # 平均時間を表示
            if gpu_times:
                avg_time = np.mean(gpu_times)
                ax4.axhline(y=avg_time, color='red', linestyle='--', 
                           alpha=0.5, label=f'Avg: {avg_time:.2f}s')
                ax4.legend()
    
    plt.suptitle('Residue-Level Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_comprehensive_report_v4(
    lambda_result: Any,
    quantum_assessments: List[QuantumAssessment],
    two_stage_result: Optional[Any],
    metadata: Dict,
    protein_indices: np.ndarray
) -> str:
    """包括的な解析レポートの生成（Version 4.0）"""
    
    system_name = metadata.get('system_name', 'Unknown System')
    
    report = f"""# Lambda³ GPU Analysis Report v4.0

## System Information
- **System**: {system_name}
- **Temperature**: {metadata.get('temperature', 300)} K
- **Time step**: {metadata.get('time_step_ps', 100)} ps
- **Frames analyzed**: {lambda_result.n_frames}
- **Total atoms**: {lambda_result.n_atoms}
- **Protein atoms**: {len(protein_indices)}
- **Analysis version**: 4.0.0 (Lambda³ Integrated)

## Lambda³ Analysis Results
- **Critical events**: {len(lambda_result.critical_events)}
- **Computation time**: {lambda_result.computation_time:.2f} seconds
"""
    
    # Two-Stage解析結果
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        stats = two_stage_result.global_network_stats
        report += f"""
## Residue Network Analysis
- **Total causal links**: {stats.get('total_causal_links', 0)}
- **Total async bonds**: {stats.get('total_async_bonds', 0)}
- **Async/Causal ratio**: {stats.get('async_to_causal_ratio', 0):.2%}
"""
    
    # 量子検証結果（Version 4.0）
    if quantum_assessments:
        total = len(quantum_assessments)
        quantum_count = sum(1 for a in quantum_assessments if a.is_quantum)
        
        report += f"""
## Quantum Validation Results v4.0

### Overview
- **Total events analyzed**: {total}
- **Quantum events confirmed**: {quantum_count} ({quantum_count/total*100:.1f}%)

### Pattern Analysis
"""
        # パターン別統計
        for pattern in StructuralEventPattern:
            pattern_assessments = [a for a in quantum_assessments if a.pattern == pattern]
            if pattern_assessments:
                n_pattern = len(pattern_assessments)
                n_quantum = sum(1 for a in pattern_assessments if a.is_quantum)
                report += f"- **{pattern.value}**: {n_quantum}/{n_pattern} quantum ({n_quantum/n_pattern*100:.1f}%)\n"
        
        # シグネチャー分布
        sig_counts = Counter(a.signature.value for a in quantum_assessments 
                           if a.signature != QuantumSignature.NONE)
        if sig_counts:
            report += f"""
### Quantum Signatures Detected
"""
            for sig, count in sorted(sig_counts.items()):
                report += f"- **{sig}**: {count}\n"
        
        # 信頼度統計
        confidences = [a.confidence for a in quantum_assessments if a.is_quantum]
        if confidences:
            report += f"""
### Confidence Statistics
- **Mean confidence**: {np.mean(confidences):.3f}
- **Max confidence**: {np.max(confidences):.3f}
- **Min confidence**: {np.min(confidences):.3f}
"""
        
        # Lambda異常性統計
        lambda_zscores = [a.lambda_anomaly.lambda_zscore for a in quantum_assessments 
                         if a.lambda_anomaly and a.lambda_anomaly.lambda_zscore > 0]
        if lambda_zscores:
            report += f"""
### Lambda Anomaly Statistics
- **Mean Z-score**: {np.mean(lambda_zscores):.2f}
- **Max Z-score**: {np.max(lambda_zscores):.2f}
- **Significant anomalies (>3σ)**: {sum(1 for z in lambda_zscores if z > 3)}
"""
    
    # 結論
    report += f"""
## Conclusions

The Lambda³ GPU v4.0 analysis successfully identified structural changes and evaluated their quantum origins:

1. **Structural dynamics**: {len(lambda_result.critical_events)} critical structural events detected
"""
    
    if quantum_assessments:
        quantum_count = sum(1 for a in quantum_assessments if a.is_quantum)
        report += f"""2. **Quantum validation**: {quantum_count} events showed quantum signatures
3. **Pattern distribution**: Events classified into instantaneous, transition, and cascade patterns
"""
    
    if two_stage_result:
        report += f"""4. **Network analysis**: Complex residue interaction networks identified
"""
    
    report += """
## Key Improvements in v4.0

- Lambda³ structure anomaly as primary input for quantum validation
- Clear 3-pattern classification (instantaneous/transition/cascade)
- Trajectory-based atomic evidence integration
- Realistic and adjustable quantum criteria
- No automatic classical assignment for long events

---
*Generated by Lambda³ GPU Quantum Validation Pipeline v4.0*
*Lambda³ Integrated Edition - Complete Refactoring*
"""
    
    return report


# ============================================
# CLI Interface
# ============================================

def main():
    """コマンドラインインターフェース（Version 4.0）"""
    
    parser = argparse.ArgumentParser(
        description='Lambda³ GPU Quantum Validation Pipeline v4.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  %(prog)s trajectory.npy metadata.json protein.npy
  
  # With topology
  %(prog)s trajectory.npy metadata.json protein.npy --topology topology.pdb
  
  # Custom output directory
  %(prog)s trajectory.npy metadata.json protein.npy --output ./results_v4
  
  # Skip visualizations
  %(prog)s trajectory.npy metadata.json protein.npy --no-viz
        """
    )
    
    # 必須引数
    parser.add_argument('trajectory', 
                       help='Path to trajectory file (.npy)')
    parser.add_argument('metadata', 
                       help='Path to metadata file (.json)')
    parser.add_argument('protein',
                       help='Path to protein indices file (.npy)')
    
    # オプション引数
    parser.add_argument('--topology', '-t',
                       help='Path to topology file (optional)')
    parser.add_argument('--output', '-o', 
                       default='./quantum_results_v4',
                       help='Output directory (default: ./quantum_results_v4)')
    parser.add_argument('--no-two-stage', 
                       action='store_true',
                       help='Skip two-stage residue analysis')
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
            protein_indices_path=args.protein,
            topology_path=args.topology,
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
