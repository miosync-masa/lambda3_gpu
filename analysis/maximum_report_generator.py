#!/usr/bin/env python3
"""
Maximum Report Generator from Lambda³ GPU Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
既存の解析結果から最大限の情報を抽出してレポート生成！
環ちゃんが全部の情報を絞り出すよ〜💕
"""

import numpy as np
from pathlib import Path
from collections import Counter
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Any
import json


def generate_maximum_report_from_results(
    lambda_result,
    two_stage_result=None,
    quantum_events=None,
    metadata=None,
    output_dir='./maximum_report',
    verbose=True
) -> str:
    """
    既に解析済みの結果から最強レポートを生成！
    
    Parameters
    ----------
    lambda_result : MDLambda3Result
        Lambda³マクロ解析結果
    two_stage_result : TwoStageLambda3Result, optional
        残基レベル解析結果
    quantum_events : List[QuantumCascadeEvent], optional
        量子検証結果
    metadata : dict, optional
        システムメタデータ
    output_dir : str
        出力ディレクトリ
    verbose : bool
        詳細出力
        
    Returns
    -------
    str
        生成されたレポート（Markdown形式）
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if verbose:
        print("\n" + "="*80)
        print("🌟 GENERATING MAXIMUM REPORT FROM EXISTING RESULTS")
        print("="*80)
    
    report = """# 🌟 Lambda³ GPU Complete Analysis Report - MAXIMUM VERSION

## Executive Summary
"""
    
    # システム情報
    if metadata:
        report += f"""
- **System**: {metadata.get('system_name', 'Unknown')}
- **Temperature**: {metadata.get('temperature', 310)} K
- **Time step**: {metadata.get('time_step_ps', 2.0)} ps
"""
    
    report += f"""
- **Frames analyzed**: {lambda_result.n_frames}
- **Atoms**: {lambda_result.n_atoms}
- **Computation time**: {lambda_result.computation_time:.2f}s
"""
    
    # GPU情報
    if hasattr(lambda_result, 'gpu_info') and lambda_result.gpu_info:
        report += f"- **GPU**: {lambda_result.gpu_info.get('device_name', 'Unknown')}\n"
        if 'memory_used' in lambda_result.gpu_info:
            report += f"- **Memory used**: {lambda_result.gpu_info['memory_used']:.2f} GB\n"
        if 'speedup' in lambda_result.gpu_info:
            report += f"- **GPU speedup**: {lambda_result.gpu_info['speedup']:.1f}x\n"
    
    # ========================================
    # 1. Lambda³結果の完全解析
    # ========================================
    if verbose:
        print("\n📊 Extracting all Lambda³ details...")
    
    report += "\n## 📊 Lambda³ Macro Analysis (Complete)\n"
    
    all_events = []
    
    # 構造境界の詳細
    if hasattr(lambda_result, 'structural_boundaries'):
        boundaries = lambda_result.structural_boundaries.get('boundary_locations', [])
        if boundaries:
            report += f"\n### Structural Boundaries ({len(boundaries)} detected)\n"
            for i, loc in enumerate(boundaries):
                all_events.append({
                    'frame': loc,
                    'type': 'structural_boundary',
                    'score': 5.0
                })
                if i < 20:  # 最初の20個を表示
                    report += f"- Boundary {i+1}: Frame {loc}\n"
            if len(boundaries) > 20:
                report += f"- ... and {len(boundaries)-20} more\n"
    
    # トポロジカル破れ
    if hasattr(lambda_result, 'topological_breaks'):
        breaks = lambda_result.topological_breaks.get('break_points', [])
        if breaks:
            report += f"\n### Topological Breaks ({len(breaks)} detected)\n"
            for i, point in enumerate(breaks[:10]):
                report += f"- Break {i+1}: Frame {point}\n"
                all_events.append({
                    'frame': point,
                    'type': 'topological_break',
                    'score': 4.0
                })
    
    # 異常スコアの完全解析
    if hasattr(lambda_result, 'anomaly_scores'):
        report += "\n### Anomaly Score Analysis (All Types)\n"
        
        score_stats = {}
        for score_type in ['global', 'structural', 'topological', 'combined', 'final_combined']:
            if score_type in lambda_result.anomaly_scores:
                scores = lambda_result.anomaly_scores[score_type]
                
                # 統計
                score_stats[score_type] = {
                    'mean': np.mean(scores),
                    'max': np.max(scores),
                    'min': np.min(scores),
                    'std': np.std(scores),
                    'median': np.median(scores)
                }
                
                # ピーク検出（異なる閾値で）
                for threshold in [2.0, 2.5, 3.0]:
                    peaks, properties = find_peaks(scores, height=threshold, distance=50)
                    if len(peaks) > 0:
                        score_stats[score_type][f'peaks_{threshold}'] = len(peaks)
                        
                        # イベントとして追加
                        for peak in peaks[:10]:  # 各閾値で最大10個
                            all_events.append({
                                'frame': peak,
                                'type': f'{score_type}_peak_{threshold}',
                                'score': float(scores[peak])
                            })
        
        # 統計表示
        report += "\n| Score Type | Mean | Max | Std | Median | Peaks(2.0) | Peaks(3.0) |\n"
        report += "|------------|------|-----|-----|--------|------------|------------|\n"
        for stype, stats in score_stats.items():
            report += f"| {stype} | {stats['mean']:.3f} | {stats['max']:.3f} | "
            report += f"{stats['std']:.3f} | {stats['median']:.3f} | "
            report += f"{stats.get('peaks_2.0', 0)} | {stats.get('peaks_3.0', 0)} |\n"
    
    # クリティカルイベントの詳細
    if lambda_result.critical_events:
        report += f"\n### Critical Events ({len(lambda_result.critical_events)} detected)\n"
        for i, event in enumerate(lambda_result.critical_events):
            if isinstance(event, tuple) and len(event) >= 2:
                report += f"- Event {i+1}: Frames {event[0]}-{event[1]} "
                report += f"(duration: {event[1]-event[0]} frames)\n"
                all_events.append({
                    'frame': (event[0] + event[1]) // 2,
                    'type': 'critical',
                    'score': 10.0,
                    'duration': event[1] - event[0]
                })
    
    # 位相空間解析結果（あれば）
    if hasattr(lambda_result, 'phase_space_analysis') and lambda_result.phase_space_analysis:
        report += "\n### Phase Space Analysis\n"
        phase_data = lambda_result.phase_space_analysis
        if 'singularities' in phase_data:
            report += f"- Singularities detected: {len(phase_data['singularities'])}\n"
        if 'coherence' in phase_data:
            report += f"- Phase coherence: {phase_data['coherence']:.3f}\n"
    
    # イベント総計
    report += f"\n### Total Events Extracted: {len(all_events)}\n"
    event_types = Counter(e['type'] for e in all_events)
    for etype, count in event_types.most_common():
        report += f"- {etype}: {count}\n"
    
    # ========================================
    # 2. Two-Stage結果の完全解析
    # ========================================
    if two_stage_result:
        if verbose:
            print("\n🧬 Extracting all residue-level details...")
        
        report += "\n## 🧬 Residue-Level Analysis (Complete)\n"
        
        # グローバルネットワーク統計
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            report += f"""
### Global Network Statistics
- **Total causal links**: {stats.get('total_causal_links', 0)}
- **Total sync links**: {stats.get('total_sync_links', 0)}
- **Total async bonds**: {stats.get('total_async_bonds', 0)}
- **Async/Causal ratio**: {stats.get('async_to_causal_ratio', 0):.2%}
- **Mean adaptive window**: {stats.get('mean_adaptive_window', 0):.1f} frames
- **Network density**: {stats.get('network_density', 0):.3f}
"""
            
            # 追加統計（あれば）
            for key, value in stats.items():
                if key not in ['total_causal_links', 'total_sync_links', 'total_async_bonds',
                              'async_to_causal_ratio', 'mean_adaptive_window', 'network_density']:
                    report += f"- **{key}**: {value}\n"
        
        # 全残基の重要度スコア
        if hasattr(two_stage_result, 'global_residue_importance'):
            all_residues = sorted(
                two_stage_result.global_residue_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            if all_residues:
                report += "\n### Complete Residue Importance Ranking\n"
                report += "| Rank | Residue | Score | Category |\n"
                report += "|------|---------|-------|----------|\n"
                
                for rank, (res_id, score) in enumerate(all_residues, 1):
                    # カテゴリ分類
                    if rank <= 5:
                        category = "🔥 Critical"
                    elif rank <= 20:
                        category = "⭐ Important"
                    elif rank <= len(all_residues) * 0.3:
                        category = "📍 Notable"
                    else:
                        category = "Normal"
                    
                    report += f"| {rank} | R{res_id+1} | {score:.4f} | {category} |\n"
        
        # 各イベントの超詳細解析
        if hasattr(two_stage_result, 'residue_analyses'):
            report += "\n### Event-by-Event Detailed Analysis\n"
            
            for event_idx, (event_name, analysis) in enumerate(two_stage_result.residue_analyses.items()):
                report += f"\n#### Event: {event_name}\n"
                
                # 基本情報
                if hasattr(analysis, 'frame_range'):
                    report += f"- Frame range: {analysis.frame_range[0]}-{analysis.frame_range[1]}\n"
                if hasattr(analysis, 'gpu_time'):
                    report += f"- GPU computation time: {analysis.gpu_time:.3f}s\n"
                
                # 残基イベント詳細
                if hasattr(analysis, 'residue_events'):
                    report += f"- **Residues involved**: {len(analysis.residue_events)}\n"
                    
                    # 全残基のスコア
                    all_scores = [(re.residue_id, re.anomaly_score) 
                                 for re in analysis.residue_events]
                    all_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    report += "  - Top 10 anomalous residues:\n"
                    for res_id, score in all_scores[:10]:
                        report += f"    - R{res_id+1}: {score:.3f}\n"
                    
                    # スコア分布
                    scores = [s for _, s in all_scores]
                    if scores:
                        report += f"  - Score distribution: mean={np.mean(scores):.3f}, "
                        report += f"max={np.max(scores):.3f}, std={np.std(scores):.3f}\n"
                
                # 因果ネットワーク詳細
                if hasattr(analysis, 'causality_chain'):
                    report += f"- **Causality chains**: {len(analysis.causality_chain)}\n"
                    
                    if analysis.causality_chain:
                        # 因果強度の統計
                        strengths = [c[2] for c in analysis.causality_chain if len(c) > 2]
                        if strengths:
                            report += f"  - Strength: mean={np.mean(strengths):.3f}, "
                            report += f"max={np.max(strengths):.3f}\n"
                        
                        # トップ因果ペア
                        report += "  - Strongest causal pairs:\n"
                        sorted_chains = sorted(analysis.causality_chain, 
                                             key=lambda x: x[2] if len(x) > 2 else 0,
                                             reverse=True)
                        for chain in sorted_chains[:5]:
                            if len(chain) >= 3:
                                report += f"    - R{chain[0]+1} → R{chain[1]+1}: {chain[2]:.3f}\n"
                
                # イニシエーター残基
                if hasattr(analysis, 'initiator_residues'):
                    report += f"- **Initiator residues**: {analysis.initiator_residues[:10]}\n"
                
                # 伝播パス
                if hasattr(analysis, 'key_propagation_paths'):
                    report += f"- **Propagation paths**: {len(analysis.key_propagation_paths)}\n"
                    for i, path in enumerate(analysis.key_propagation_paths[:3], 1):
                        path_str = " → ".join([f"R{r+1}" for r in path])
                        report += f"  - Path {i}: {path_str}\n"
                
                # ネットワーク結果
                if hasattr(analysis, 'network_result'):
                    network = analysis.network_result
                    if hasattr(network, 'async_strong_bonds'):
                        report += f"- **Async bonds**: {len(network.async_strong_bonds)}\n"
                        # トップAsync結合
                        if network.async_strong_bonds:
                            report += "  - Strongest async bonds:\n"
                            for bond in network.async_strong_bonds[:5]:
                                report += f"    - R{bond[0]+1} ⟷ R{bond[1]+1}: "
                                report += f"lag={bond[2] if len(bond) > 2 else 'N/A'}\n"
        
        # ハブ残基の完全解析
        all_hub_residues = []
        for analysis in two_stage_result.residue_analyses.values():
            if hasattr(analysis, 'initiator_residues'):
                all_hub_residues.extend(analysis.initiator_residues)
        
        if all_hub_residues:
            hub_counts = Counter(all_hub_residues)
            report += "\n### Hub Residues Analysis (Drug Targets)\n"
            report += "| Rank | Residue | Frequency | Events Involved |\n"
            report += "|------|---------|-----------|----------------|\n"
            
            for rank, (res_id, count) in enumerate(hub_counts.most_common(), 1):
                # どのイベントに関与しているか
                events_involved = []
                for event_name, analysis in two_stage_result.residue_analyses.items():
                    if hasattr(analysis, 'initiator_residues'):
                        if res_id in analysis.initiator_residues:
                            events_involved.append(event_name)
                
                events_str = ", ".join(events_involved[:3])
                if len(events_involved) > 3:
                    events_str += f" +{len(events_involved)-3}"
                
                report += f"| {rank} | R{res_id+1} | {count} | {events_str} |\n"
        
        # 既存のブートストラップ結果を探す
        if hasattr(two_stage_result, 'bootstrap_results'):
            report += "\n### Bootstrap Statistical Validation (Found)\n"
            bootstrap = two_stage_result.bootstrap_results
            
            if isinstance(bootstrap, dict):
                for event_name, stats in bootstrap.items():
                    report += f"\n#### {event_name}\n"
                    report += f"- Pairs tested: {stats.get('n_pairs_tested', 'N/A')}\n"
                    report += f"- Significant pairs: {stats.get('n_significant', 'N/A')}\n"
                    report += f"- Significance rate: {stats.get('significance_rate', 0):.1%}\n"
        
        # 信頼区間情報（あれば）
        if hasattr(two_stage_result, 'confidence_intervals'):
            ci = two_stage_result.confidence_intervals
            report += "\n### Statistical Confidence\n"
            report += f"- Mean CI width: {ci.get('mean_ci_width', 'N/A')}\n"
            report += f"- Correlation range: [{ci.get('min_correlation', 0):.3f}, "
            report += f"{ci.get('max_correlation', 0):.3f}]\n"
    
    # ========================================
    # 3. 量子イベントの完全解析
    # ========================================
    if quantum_events:
        if verbose:
            print("\n⚛️ Extracting all quantum details...")
        
        report += "\n## ⚛️ Quantum Events Analysis (Complete)\n"
        
        report += f"### Overview\n"
        report += f"- **Total events**: {len(quantum_events)}\n"
        
        # 量子イベントの分類
        n_quantum = sum(1 for e in quantum_events if e.quantum_metrics.is_quantum)
        n_bell = sum(1 for e in quantum_events if e.quantum_metrics.bell_violated)
        n_critical = sum(1 for e in quantum_events if e.is_critical)
        
        report += f"- **Confirmed quantum**: {n_quantum} ({100*n_quantum/max(1,len(quantum_events)):.1f}%)\n"
        report += f"- **Bell violations**: {n_bell} ({100*n_bell/max(1,len(quantum_events)):.1f}%)\n"
        report += f"- **Critical events**: {n_critical}\n"
        
        # イベントタイプ分布
        event_types = Counter(e.event_type.value for e in quantum_events)
        report += "\n### Event Type Distribution\n"
        for etype, count in event_types.most_common():
            report += f"- {etype}: {count}\n"
        
        # CHSH統計
        chsh_values = [e.quantum_metrics.chsh_value for e in quantum_events]
        if chsh_values:
            report += f"\n### CHSH Statistics\n"
            report += f"- Mean: {np.mean(chsh_values):.3f}\n"
            report += f"- Max: {np.max(chsh_values):.3f}\n"
            report += f"- Min: {np.min(chsh_values):.3f}\n"
            report += f"- Std: {np.std(chsh_values):.3f}\n"
            report += f"- Classical bound: 2.000\n"
            report += f"- Tsirelson bound: {2*np.sqrt(2):.3f}\n"
        
        # 全量子イベントの詳細
        report += "\n### All Quantum Events (Detailed)\n"
        
        for i, event in enumerate(quantum_events, 1):
            qm = event.quantum_metrics
            report += f"\n#### Event {i}\n"
            report += f"- Frames: {event.frame_start}-{event.frame_end}\n"
            report += f"- Type: {event.event_type.value}\n"
            report += f"- Duration: {qm.duration_frames} frames ({qm.duration_ps:.2f} ps)\n"
            report += f"- Quantum: {'✅' if qm.is_quantum else '❌'}\n"
            report += f"- Bell violated: {'✅' if qm.bell_violated else '❌'}\n"
            report += f"- CHSH: {qm.chsh_value:.3f} (raw: {qm.chsh_raw_value:.3f})\n"
            report += f"- Confidence: {qm.quantum_confidence:.3f}\n"
            report += f"- Score: {qm.quantum_score:.3f}\n"
            
            # 物理指標
            report += f"- Coherence time: {qm.coherence_time_ps:.3e} ps\n"
            report += f"- Thermal ratio: {qm.thermal_ratio:.3f}\n"
            report += f"- Tunneling prob: {qm.tunneling_probability:.3e}\n"
            
            # クリティカル理由
            if event.is_critical:
                report += f"- **CRITICAL**: {', '.join(event.critical_reasons)}\n"
            
            # 判定基準
            if qm.criteria_passed:
                report += "- Criteria passed:\n"
                for criterion in qm.criteria_passed:
                    if criterion.passed:
                        report += f"  - ✅ {criterion.criterion.value}\n"
    
    # ========================================
    # 4. 統合的洞察とサマリー
    # ========================================
    if verbose:
        print("\n💡 Generating integrated insights...")
    
    report += "\n## 💡 Integrated Insights & Summary\n"
    
    insights = []
    
    # イベント総数
    total_unique_events = len(set(e['frame'] for e in all_events))
    insights.append(f"✓ {total_unique_events} unique frames with events detected")
    insights.append(f"✓ {len(all_events)} total events across all detection methods")
    
    # ネットワーク密度
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        stats = two_stage_result.global_network_stats
        total_links = (stats.get('total_causal_links', 0) + 
                      stats.get('total_sync_links', 0) +
                      stats.get('total_async_bonds', 0))
        if total_links > 0:
            insights.append(f"✓ {total_links} total network connections identified")
            
            # ネットワークの特徴
            if stats.get('async_to_causal_ratio', 0) > 0.5:
                insights.append(f"✓ High async/causal ratio ({stats['async_to_causal_ratio']:.1%}) suggests long-range coupling")
    
    # 量子特性
    if quantum_events:
        if n_quantum > 0:
            quantum_rate = 100 * n_quantum / len(quantum_events)
            insights.append(f"✓ {quantum_rate:.1f}% events show quantum signatures")
        if n_bell > 0:
            bell_rate = 100 * n_bell / len(quantum_events)
            insights.append(f"✓ {bell_rate:.1f}% events violate Bell inequalities (non-classical)")
    
    # GPU性能
    if hasattr(lambda_result, 'gpu_info') and lambda_result.gpu_info:
        if 'speedup' in lambda_result.gpu_info:
            speedup = lambda_result.gpu_info['speedup']
            if speedup > 1:
                insights.append(f"✓ {speedup:.1f}x GPU acceleration achieved")
        if 'device_name' in lambda_result.gpu_info:
            insights.append(f"✓ Computed on {lambda_result.gpu_info['device_name']}")
    
    for insight in insights:
        report += f"\n{insight}"
    
    # ========================================
    # 5. 創薬ターゲット提案
    # ========================================
    if two_stage_result and all_hub_residues:
        report += "\n\n## 💊 Drug Design Recommendations\n"
        
        hub_counts = Counter(all_hub_residues)
        top_targets = hub_counts.most_common(15)
        
        report += "\n### Primary Targets (Top Hub Residues)\n"
        
        for i, (res_id, count) in enumerate(top_targets, 1):
            report += f"\n{i}. **Residue {res_id+1}**\n"
            report += f"   - Hub frequency: {count} events\n"
            
            # 重要度スコア
            if hasattr(two_stage_result, 'global_residue_importance'):
                importance = two_stage_result.global_residue_importance.get(res_id, 0)
                report += f"   - Importance score: {importance:.3f}\n"
            
            # ネットワーク中心性
            if count >= 3:
                report += f"   - Network centrality: HIGH\n"
            elif count >= 2:
                report += f"   - Network centrality: MEDIUM\n"
            
            # カテゴリ
            if i <= 3:
                report += f"   - **Priority: CRITICAL TARGET**\n"
            elif i <= 10:
                report += f"   - Priority: Secondary target\n"
    
    # ========================================
    # 6. 推奨事項
    # ========================================
    report += "\n## 📋 Recommendations\n"
    
    recommendations = []
    
    # ハブ残基ベース
    if two_stage_result and hub_counts:
        top3 = [f"R{r+1}" for r, _ in hub_counts.most_common(3)]
        recommendations.append(f"Focus on residues {', '.join(top3)} for drug targeting")
    
    # 量子イベントベース
    if quantum_events and n_bell > 0:
        recommendations.append("Consider quantum mechanical effects in modeling")
        recommendations.append("Bell violations suggest non-classical correlations requiring QM treatment")
    
    # ネットワークベース
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        stats = two_stage_result.global_network_stats
        if stats.get('total_async_bonds', 0) > 100:
            recommendations.append("Strong async bonds indicate allosteric communication pathways")
    
    for i, rec in enumerate(recommendations, 1):
        report += f"\n{i}. {rec}"
    
    # フッター
    report += f"""

---
*Analysis Complete!*
*Total report length: {len(report):,} characters*
*NO TIME, NO PHYSICS, ONLY STRUCTURE!*
*Generated by 環ちゃん with love 💕*
"""
    
    # ========================================
    # 保存とエクスポート
    # ========================================
    
    # Markdownレポート保存
    report_path = output_path / 'maximum_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # JSON形式でも保存（データ解析用）
    json_data = {
        'summary': {
            'n_frames': lambda_result.n_frames,
            'n_atoms': lambda_result.n_atoms,
            'computation_time': lambda_result.computation_time,
            'total_events': len(all_events),
            'event_types': dict(Counter(e['type'] for e in all_events))
        },
        'events': all_events[:100],  # 最初の100イベント
        'metadata': metadata if metadata else {}
    }
    
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        json_data['network_stats'] = two_stage_result.global_network_stats
    
    if quantum_events:
        json_data['quantum_summary'] = {
            'total': len(quantum_events),
            'quantum': n_quantum,
            'bell_violations': n_bell,
            'critical': n_critical
        }
    
    json_path = output_path / 'analysis_data.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    if verbose:
        print(f"\n✨ COMPLETE!")
        print(f"   📄 Report saved to: {report_path}")
        print(f"   📊 Data saved to: {json_path}")
        print(f"   📏 Report length: {len(report):,} characters")
        print(f"   🎯 Events extracted: {len(all_events)}")
        if two_stage_result and hub_counts:
            print(f"   💊 Drug targets identified: {len(hub_counts)}")
        print(f"\n   All available information has been extracted!")
    
    return report
