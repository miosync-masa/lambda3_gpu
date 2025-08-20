#!/usr/bin/env python3
"""
Maximum Report Generator - Complete Fix
========================================
quantum_validation_gpu.pyとの不整合を修正した完全版
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import logging
import traceback

logger = logging.getLogger('maximum_report_generator')

def generate_maximum_report_from_results_v4(
    lambda_result: Any,
    two_stage_result: Optional[Any] = None,
    quantum_assessments: Optional[List[Any]] = None,  # v4から変更: quantum_events → quantum_assessments
    metadata: Optional[Dict] = None,
    output_dir: str = './results',
    verbose: bool = False
) -> str:
    """
    解析結果から詳細レポートを生成
    
    Parameters
    ----------
    lambda_result : Lambda3Result
        Lambda³解析結果
    two_stage_result : TwoStageLambda3Result, optional
        残基レベル解析結果
    quantum_assessments : List[QuantumAssessment], optional
        構造異常評価結果（v4/v5互換）
    metadata : Dict, optional
        メタデータ
    output_dir : str
        出力ディレクトリ
    verbose : bool
        詳細表示
    
    Returns
    -------
    str
        生成されたレポート
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # レポート開始
    report = []
    report.append("="*80)
    report.append("# LAMBDA³ MAXIMUM ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # ========================================
    # 1. メタデータセクション
    # ========================================
    if metadata:
        report.append("## System Information")
        report.append(f"- System: {metadata.get('system_name', 'Unknown')}")
        report.append(f"- Temperature: {metadata.get('temperature', 300)} K")
        report.append(f"- Time step: {metadata.get('time_step_ps', 100)} ps")
        
        if 'protein' in metadata:
            protein_info = metadata['protein']
            report.append(f"- Residues: {protein_info.get('n_residues', 'N/A')}")
            seq = protein_info.get('sequence', '')
            if seq:
                report.append(f"- Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        report.append("")
    
    # ========================================
    # 2. Lambda³解析結果
    # ========================================
    if lambda_result:
        report.append("## Lambda³ Structure Analysis")
        report.append(f"- Frames analyzed: {getattr(lambda_result, 'n_frames', 'N/A')}")
        report.append(f"- Atoms: {getattr(lambda_result, 'n_atoms', 'N/A')}")
        report.append(f"- Critical events: {len(getattr(lambda_result, 'critical_events', []))}")
        report.append(f"- Computation time: {getattr(lambda_result, 'computation_time', 0):.2f} s")
        report.append("")
        
        # 構造パラメータの統計
        if hasattr(lambda_result, 'lambda_structures'):
            structures = lambda_result.lambda_structures
            report.append("### Structure Parameters")
            
            if 'lambda_F_mag' in structures:
                lambda_vals = structures['lambda_F_mag']
                report.append(f"- Lambda magnitude: mean={np.mean(lambda_vals):.3f}, std={np.std(lambda_vals):.3f}")
            
            if 'rho_T' in structures:
                rho_vals = structures['rho_T']
                report.append(f"- Tension density: mean={np.mean(rho_vals):.3f}, max={np.max(rho_vals):.3f}")
            
            if 'sigma_s' in structures:
                sigma_vals = structures['sigma_s']
                report.append(f"- Sync rate: mean={np.mean(sigma_vals):.3f}, max={np.max(sigma_vals):.3f}")
            
            report.append("")
        
        # イベント詳細
        report.append("### Critical Events Analysis")
        events = getattr(lambda_result, 'critical_events', [])
        
        # anomaly_scoresから実際のスコアを取得する関数
        def get_event_score(event_data, start, end):
            """イベントのanomalyスコアを取得"""
            score = 0.0
            
            # イベント自体にスコアがある場合
            if isinstance(event_data, dict) and 'anomaly_score' in event_data:
                return event_data['anomaly_score']
            elif isinstance(event_data, (tuple, list)) and len(event_data) > 2:
                return event_data[2]
            
            # anomaly_scoresから取得
            if hasattr(lambda_result, 'anomaly_scores'):
                for key in ['combined', 'final_combined', 'global', 'local', 'extended']:
                    if key in lambda_result.anomaly_scores:
                        scores = lambda_result.anomaly_scores[key]
                        if start < len(scores):
                            score = float(np.max(scores[start:min(end+1, len(scores))]))
                            break
            
            return score
        
        # 上位20イベントを表示
        for i, event in enumerate(events[:20], 1):
            if isinstance(event, dict):
                start = event.get('start', event.get('frame', 0))
                end = event.get('end', start)
                event_type = event.get('type', 'unknown')
            elif isinstance(event, (tuple, list)) and len(event) >= 2:
                start = int(event[0])
                end = int(event[1])
                event_type = 'critical'
            else:
                continue
            
            score = get_event_score(event, start, end)
            duration = end - start + 1
            
            report.append(f"#### Event {i} (frames {start}-{end})")
            report.append(f"- Duration: {duration} frames ({duration * metadata.get('time_step_ps', 100):.1f} ps)")
            report.append(f"- Anomaly score: {score:.3f}")
            report.append(f"- Type: {event_type}")
            
            # 残基解析結果を探す
            residue_analysis_found = False
            if two_stage_result and hasattr(two_stage_result, 'residue_analyses'):
                # 複数のキー形式を試す
                possible_keys = [
                    f"top_{i-1:02d}_score_{score:.2f}",  # 0-indexed
                    f"top_{i-1:02d}_score_{score:.1f}",  # 異なる精度
                    f"event_{i}",  # 1-indexed legacy
                    f"event_{i-1}",  # 0-indexed legacy
                ]
                
                # キーを順番に試す
                for key in possible_keys:
                    if key in two_stage_result.residue_analyses:
                        analysis = two_stage_result.residue_analyses[key]
                        residue_analysis_found = True
                        report.append("- **Residue-level analysis:**")
                        
                        if hasattr(analysis, 'residue_events'):
                            n_residues = len(analysis.residue_events)
                            report.append(f"  - Residues involved: {n_residues}")
                            
                            # 上位残基を表示
                            if hasattr(analysis, 'residue_anomaly_scores'):
                                top_residues = sorted(
                                    analysis.residue_anomaly_scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )[:5]
                                if top_residues:
                                    report.append("  - Top anomalous residues:")
                                    for res_id, res_score in top_residues:
                                        report.append(f"    - R{res_id}: {res_score:.3f}")
                        
                        if hasattr(analysis, 'network_stats'):
                            stats = analysis.network_stats
                            report.append(f"  - Causal links: {stats.get('causal_links', 0)}")
                            report.append(f"  - Sync links: {stats.get('sync_links', 0)}")
                        
                        if hasattr(analysis, 'gpu_time'):
                            report.append(f"  - GPU computation: {analysis.gpu_time:.3f} s")
                        
                        break
                
                # パターンマッチングも試す
                if not residue_analysis_found:
                    for key, analysis in two_stage_result.residue_analyses.items():
                        if key.startswith(f"top_{i-1:02d}_"):
                            residue_analysis_found = True
                            report.append("- **Residue-level analysis found**")
                            break
            
            if not residue_analysis_found:
                report.append("- *Residue analysis not available*")
            
            report.append("")
    
    # ========================================
    # 3. 残基ネットワーク解析
    # ========================================
    if two_stage_result:
        report.append("## Residue Network Analysis")
        
        # グローバル統計
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            report.append("### Global Network Statistics")
            report.append(f"- Total causal links: {stats.get('total_causal_links', 0)}")
            report.append(f"- Total sync links: {stats.get('total_sync_links', 0)}")
            report.append(f"- Total async bonds: {stats.get('total_async_bonds', 0)}")
            
            async_ratio = stats.get('async_to_causal_ratio', 0)
            if async_ratio:
                report.append(f"- Async/Causal ratio: {async_ratio:.2%}")
            
            mean_window = stats.get('mean_adaptive_window', 0)
            if mean_window:
                report.append(f"- Mean adaptive window: {mean_window:.1f} frames")
            
            report.append("")
        
        # 残基重要度
        if hasattr(two_stage_result, 'global_residue_importance'):
            report.append("### Residue Importance Ranking")
            top_residues = sorted(
                two_stage_result.global_residue_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            if top_residues:
                report.append("| Rank | Residue | Score | Category |")
                report.append("|------|---------|-------|----------|")
                
                for rank, (res_id, score) in enumerate(top_residues, 1):
                    category = "Critical" if score > 120 else "Important" if score > 100 else "Notable"
                    report.append(f"| {rank} | R{res_id} | {score:.2f} | {category} |")
                
                report.append("")
        
        # イベント別詳細統計
        if hasattr(two_stage_result, 'residue_analyses'):
            report.append("### Event-Specific Analysis Summary")
            
            total_gpu_time = 0
            total_pairs = 0
            significant_pairs = 0
            
            for event_key, analysis in two_stage_result.residue_analyses.items():
                if hasattr(analysis, 'gpu_time'):
                    total_gpu_time += analysis.gpu_time
                
                if hasattr(analysis, 'confidence_results'):
                    conf = analysis.confidence_results
                    total_pairs += conf.get('total_pairs', 0)
                    significant_pairs += conf.get('significant_pairs', 0)
            
            report.append(f"- Total events analyzed: {len(two_stage_result.residue_analyses)}")
            report.append(f"- Total GPU time: {total_gpu_time:.2f} s")
            
            if total_pairs > 0:
                report.append(f"- Total residue pairs: {total_pairs}")
                report.append(f"- Significant pairs: {significant_pairs} ({significant_pairs/total_pairs*100:.1f}%)")
            
            report.append("")
    
    # ========================================
    # 4. 構造異常評価（量子→異常に変更）
    # ========================================
    if quantum_assessments:
        report.append("## Structural Anomaly Assessment")
        
        total = len(quantum_assessments)
        anomalous = sum(1 for a in quantum_assessments if getattr(a, 'is_quantum', False))
        
        report.append(f"- Total events assessed: {total}")
        report.append(f"- Anomalous events: {anomalous} ({anomalous/total*100:.1f}%)")
        report.append("")
        
        # パターン統計
        if total > 0 and hasattr(quantum_assessments[0], 'pattern'):
            report.append("### Event Pattern Distribution")
            
            pattern_stats = {}
            for assessment in quantum_assessments:
                pattern = getattr(assessment, 'pattern')
                pattern_name = pattern.value if hasattr(pattern, 'value') else str(pattern)
                
                if pattern_name not in pattern_stats:
                    pattern_stats[pattern_name] = {'total': 0, 'anomalous': 0}
                
                pattern_stats[pattern_name]['total'] += 1
                if getattr(assessment, 'is_quantum', False):
                    pattern_stats[pattern_name]['anomalous'] += 1
            
            for pattern, stats in pattern_stats.items():
                total_pattern = stats['total']
                anomalous_pattern = stats['anomalous']
                report.append(f"- {pattern}: {anomalous_pattern}/{total_pattern} anomalous ({anomalous_pattern/total_pattern*100:.1f}%)")
            
            report.append("")
        
        # シグネチャー分布
        if total > 0 and hasattr(quantum_assessments[0], 'signature'):
            report.append("### Anomaly Signatures")
            
            signatures = []
            for assessment in quantum_assessments:
                if getattr(assessment, 'is_quantum', False):
                    sig = getattr(assessment, 'signature')
                    sig_name = sig.value if hasattr(sig, 'value') else str(sig)
                    if sig_name != 'classical' and sig_name != 'NONE':
                        signatures.append(sig_name)
            
            if signatures:
                sig_counts = Counter(signatures)
                for sig, count in sig_counts.most_common():
                    report.append(f"- {sig}: {count}")
            else:
                report.append("- No specific signatures detected")
            
            report.append("")
        
        # 信頼度統計
        confidences = [getattr(a, 'confidence', 0) for a in quantum_assessments 
                      if getattr(a, 'is_quantum', False)]
        
        if confidences:
            report.append("### Anomaly Confidence Statistics")
            report.append(f"- Mean confidence: {np.mean(confidences):.3f}")
            report.append(f"- Std deviation: {np.std(confidences):.3f}")
            report.append(f"- Max confidence: {np.max(confidences):.3f}")
            report.append(f"- Min confidence: {np.min(confidences):.3f}")
            
            # 信頼度分布
            high_conf = sum(1 for c in confidences if c > 0.7)
            med_conf = sum(1 for c in confidences if 0.3 <= c <= 0.7)
            low_conf = sum(1 for c in confidences if c < 0.3)
            
            report.append("")
            report.append("### Confidence Distribution")
            report.append(f"- High (>0.7): {high_conf}")
            report.append(f"- Medium (0.3-0.7): {med_conf}")
            report.append(f"- Low (<0.3): {low_conf}")
            report.append("")
        
        # Lambda異常統計
        lambda_zscores = []
        for assessment in quantum_assessments:
            if hasattr(assessment, 'lambda_anomaly'):
                anomaly = assessment.lambda_anomaly
                if hasattr(anomaly, 'lambda_zscore'):
                    zscore = anomaly.lambda_zscore
                    if zscore > 0:
                        lambda_zscores.append(zscore)
        
        if lambda_zscores:
            report.append("### Lambda Structure Anomalies")
            report.append(f"- Mean Z-score: {np.mean(lambda_zscores):.2f}")
            report.append(f"- Max Z-score: {np.max(lambda_zscores):.2f}")
            report.append(f"- Events with Z>3σ: {sum(1 for z in lambda_zscores if z > 3)}")
            report.append("")
    
    # ========================================
    # 5. 総合サマリー
    # ========================================
    report.append("## Summary and Conclusions")
    report.append("")
    
    # 主要な発見
    report.append("### Key Findings")
    
    if lambda_result and hasattr(lambda_result, 'critical_events'):
        n_events = len(lambda_result.critical_events)
        report.append(f"1. **Structure dynamics**: {n_events} critical structural events detected")
        
        # イベントの時間分布
        if n_events > 0:
            event_frames = []
            for event in lambda_result.critical_events:
                if isinstance(event, dict):
                    frame = event.get('frame', event.get('start', 0))
                elif isinstance(event, (tuple, list)):
                    frame = event[0]
                else:
                    continue
                event_frames.append(frame)
            
            if event_frames and hasattr(lambda_result, 'n_frames'):
                early = sum(1 for f in event_frames if f < lambda_result.n_frames * 0.33)
                middle = sum(1 for f in event_frames if lambda_result.n_frames * 0.33 <= f < lambda_result.n_frames * 0.67)
                late = sum(1 for f in event_frames if f >= lambda_result.n_frames * 0.67)
                
                report.append(f"   - Early phase: {early} events")
                report.append(f"   - Middle phase: {middle} events")
                report.append(f"   - Late phase: {late} events")
    
    if two_stage_result and hasattr(two_stage_result, 'global_residue_importance'):
        top_res = sorted(two_stage_result.global_residue_importance.items(),
                        key=lambda x: x[1], reverse=True)[:3]
        if top_res:
            report.append(f"2. **Key residues**: R{top_res[0][0]}, R{top_res[1][0]}, R{top_res[2][0]} show highest importance")
    
    if quantum_assessments and anomalous > 0:
        report.append(f"3. **Anomalous behavior**: {anomalous} events show non-classical characteristics")
    
    report.append("")
    
    # 推奨事項
    report.append("### Recommendations")
    
    if two_stage_result and hasattr(two_stage_result, 'global_residue_importance'):
        top_res = sorted(two_stage_result.global_residue_importance.items(),
                        key=lambda x: x[1], reverse=True)[:5]
        if top_res:
            res_list = ", ".join([f"R{r[0]}" for r in top_res])
            report.append(f"- **Target residues for intervention**: Focus on residues {res_list}")
    
    if quantum_assessments and anomalous > 0:
        report.append("- **Structural anomalies**: Consider non-classical effects in dynamics modeling")
    
    report.append("")
    
    # ========================================
    # 6. 技術情報
    # ========================================
    report.append("## Technical Information")
    report.append("")
    report.append("### Analysis Pipeline")
    report.append("- Lambda³ GPU structural analysis")
    if two_stage_result:
        report.append("- Two-stage residue-level network analysis")
    if quantum_assessments:
        report.append("- Structural anomaly assessment (v4/v5)")
    
    report.append("")
    report.append("### Performance Metrics")
    
    total_time = 0
    if lambda_result and hasattr(lambda_result, 'computation_time'):
        total_time += lambda_result.computation_time
        report.append(f"- Lambda³ analysis: {lambda_result.computation_time:.2f} s")
    
    if two_stage_result and hasattr(two_stage_result, 'residue_analyses'):
        gpu_time = sum(getattr(a, 'gpu_time', 0) for a in two_stage_result.residue_analyses.values())
        total_time += gpu_time
        report.append(f"- Residue analysis: {gpu_time:.2f} s")
    
    if total_time > 0:
        report.append(f"- **Total computation**: {total_time:.2f} s")
    
    report.append("")
    
    # ========================================
    # 終了
    # ========================================
    report.append("="*80)
    report.append("Generated by Lambda³ GPU Analysis Pipeline")
    report.append("Structure-focused analysis without quantum assumptions")
    report.append("="*80)
    
    # 文字列に結合
    report_text = "\n".join(report)
    
    # ファイル保存
    report_path = output_path / 'maximum_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    if verbose:
        logger.info(f"Report saved to {report_path}")
        logger.info(f"Report length: {len(report_text):,} characters")
    
    return report_text


# ========================================
# 補助関数
# ========================================

def format_event_key(event_index: int, score: float = 0.0) -> str:
    """イベントキーをフォーマット"""
    return f"top_{event_index:02d}_score_{score:.2f}"


def find_residue_analysis(two_stage_result: Any, event_index: int, score: float = 0.0) -> Optional[Any]:
    """
    イベントインデックスから対応する残基解析を検索
    
    より柔軟なマッチングを行う：
    - top_XX_score_Y.YY形式
    - event_X形式
    - パターンマッチング
    """
    if not hasattr(two_stage_result, 'residue_analyses'):
        return None
    
    # 完全一致を試す
    exact_key = f"top_{event_index:02d}_score_{score:.2f}"
    if exact_key in two_stage_result.residue_analyses:
        return two_stage_result.residue_analyses[exact_key]
    
    # スコアの精度を変えて試す
    approx_key = f"top_{event_index:02d}_score_{score:.1f}"
    if approx_key in two_stage_result.residue_analyses:
        return two_stage_result.residue_analyses[approx_key]
    
    # top_XXで始まるキーを探す
    prefix = f"top_{event_index:02d}_"
    for key in two_stage_result.residue_analyses.keys():
        if key.startswith(prefix):
            return two_stage_result.residue_analyses[key]
    
    # レガシー形式も試す
    legacy_keys = [
        f"event_{event_index}",
        f"event_{event_index + 1}",  # 1-indexedの場合
        f"Event {event_index + 1}",  # 大文字版
    ]
    
    for key in legacy_keys:
        if key in two_stage_result.residue_analyses:
            return two_stage_result.residue_analyses[key]
    
    return None


def extract_event_info(event: Any) -> Tuple[int, int, float, str]:
    """
    イベントから情報を抽出
    
    Returns
    -------
    Tuple[int, int, float, str]
        (start_frame, end_frame, score, event_type)
    """
    if isinstance(event, dict):
        start = event.get('start', event.get('frame', 0))
        end = event.get('end', start)
        score = event.get('anomaly_score', 0.0)
        event_type = event.get('type', 'unknown')
    elif isinstance(event, (tuple, list)) and len(event) >= 2:
        start = int(event[0])
        end = int(event[1])
        score = event[2] if len(event) > 2 else 0.0
        event_type = 'critical'
    else:
        start = end = 0
        score = 0.0
        event_type = 'unknown'
    
    return start, end, score, event_type


# ========================================
# CLI対応（オプション）
# ========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate maximum analysis report from Lambda³ results'
    )
    parser.add_argument('--lambda-result', required=True,
                       help='Path to Lambda³ result pickle/json')
    parser.add_argument('--two-stage-result',
                       help='Path to two-stage result pickle/json')
    parser.add_argument('--quantum-assessments',
                       help='Path to quantum assessments json')
    parser.add_argument('--metadata',
                       help='Path to metadata json')
    parser.add_argument('--output', default='./results',
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # ファイル読み込み処理を追加する場合はここに実装
    print("CLI interface not fully implemented. Use as module.")
