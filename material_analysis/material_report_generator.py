#!/usr/bin/env python3
"""
Material Report Generator from Lambda³ GPU Results - Version 1.0
==================================================================

材料解析結果から最大限の情報を抽出してレポート生成！
転位・亀裂・相変態の完全解析対応版

Version: 1.0.0 - Material Edition
Authors: 環ちゃん
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from scipy.signal import find_peaks
import logging

logger = logging.getLogger('material_report_generator')

def generate_material_report_from_results(
    macro_result,
    two_stage_result=None,
    impact_results=None,
    sorted_events=None,
    metadata=None,
    material_type='SUJ2',
    output_dir='./material_report',
    verbose=True
) -> str:
    """
    材料解析の統合レポート生成
    
    Parameters
    ----------
    macro_result : MaterialLambda3Result
        マクロレベルLambda³結果
    two_stage_result : MaterialTwoStageResult
        2段階解析結果
    impact_results : Dict[str, MaterialImpactResult]
        原子レベル欠陥解析結果
    sorted_events : List[Tuple[int, int, float]]
        スコア順イベントリスト
    metadata : dict
        シミュレーションメタデータ
    material_type : str
        材料タイプ（SUJ2, AL7075等）
    output_dir : str
        出力ディレクトリ
    verbose : bool
        詳細出力
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if verbose:
        print("\n" + "="*80)
        print("💎 GENERATING MATERIAL ANALYSIS REPORT v1.0")
        print("="*80)
    
    # ========================================
    # レポートヘッダー
    # ========================================
    report = f"""# 💎 Material Lambda³ GPU Analysis Report - {material_type}

## Executive Summary
"""
    
    # システム情報
    if metadata:
        report += f"""
- **Material**: {material_type}
- **System**: {metadata.get('system_name', 'Unknown')}
- **Temperature**: {metadata.get('temperature', 300)} K
- **Strain rate**: {metadata.get('strain_rate', '1e-3')} /ps
- **Loading**: {metadata.get('loading_type', 'Tensile')}
"""
    
    # 基本統計
    report += f"""
- **Frames analyzed**: {macro_result.n_frames}
- **Atoms**: {macro_result.n_atoms}
- **Computation time**: {macro_result.computation_time:.2f}s
- **Analysis version**: Material Lambda³ v1.0
"""
    
    # GPU情報
    if hasattr(macro_result, 'gpu_info') and macro_result.gpu_info:
        report += f"- **GPU**: {macro_result.gpu_info.get('device_name', 'Unknown')}\n"
        if 'speedup' in macro_result.gpu_info:
            report += f"- **GPU speedup**: {macro_result.gpu_info['speedup']:.1f}x\n"
    
    # ========================================
    # 1. マクロ材料解析
    # ========================================
    if verbose:
        print("\n📊 Extracting macro material analysis...")
    
    report += "\n## 📊 Macro Material Analysis\n"
    
    # 検出イベント
    all_events = []
    
    if hasattr(macro_result, 'material_events'):
        events = macro_result.material_events
        report += f"\n### Detected Material Events ({len(events)})\n"
        
        event_types = Counter()
        for event in events:
            if isinstance(event, tuple) and len(event) >= 3:
                start, end, event_type = event
                event_types[event_type] += 1
                all_events.append({
                    'start': start,
                    'end': end,
                    'type': event_type,
                    'duration': end - start
                })
        
        # イベントタイプ別統計
        report += "\n| Event Type | Count | Avg Duration |\n"
        report += "|------------|-------|-------------|\n"
        
        for etype, count in event_types.most_common():
            durations = [e['duration'] for e in all_events if e['type'] == etype]
            avg_dur = np.mean(durations) if durations else 0
            report += f"| {etype} | {count} | {avg_dur:.1f} frames |\n"
    
    # 応力-歪み解析
    if hasattr(macro_result, 'stress_strain'):
        report += "\n### Stress-Strain Analysis\n"
        ss = macro_result.stress_strain
        
        if 'max_stress' in ss:
            report += f"- **Ultimate strength**: {ss['max_stress']:.2f} GPa\n"
        if 'yield_stress' in ss:
            report += f"- **Yield strength**: {ss['yield_stress']:.2f} GPa\n"
        if 'elastic_modulus' in ss:
            report += f"- **Elastic modulus**: {ss['elastic_modulus']:.1f} GPa\n"
        if 'fracture_strain' in ss:
            report += f"- **Fracture strain**: {ss['fracture_strain']:.3f}\n"
    
    # 異常スコア分析
    if hasattr(macro_result, 'anomaly_scores'):
        report += "\n### Anomaly Score Analysis\n"
        
        for score_type in ['strain', 'coordination', 'damage']:
            if score_type in macro_result.anomaly_scores:
                scores = macro_result.anomaly_scores[score_type]
                
                # ピーク検出
                peaks, properties = find_peaks(scores, height=np.mean(scores) + 2*np.std(scores))
                
                report += f"\n**{score_type.capitalize()} anomalies**:\n"
                report += f"- Mean: {np.mean(scores):.3f}\n"
                report += f"- Max: {np.max(scores):.3f}\n"
                report += f"- Peaks detected: {len(peaks)}\n"
                
                if len(peaks) > 0:
                    report += f"- Peak frames: {peaks[:5].tolist()}\n"
    
    # ========================================
    # 2. クラスターレベル解析
    # ========================================
    if two_stage_result:
        if verbose:
            print("\n🔬 Extracting cluster-level analysis...")
        
        report += "\n## 🔬 Cluster-Level Analysis\n"
        
        # 材料状態
        if hasattr(two_stage_result, 'material_state'):
            state = two_stage_result.material_state
            report += f"""
### Material State Assessment
- **Overall state**: {state.get('state', 'Unknown')}
- **Health index**: {state.get('health_index', 1.0):.1%}
- **Max damage**: {state.get('max_damage', 0.0):.1%}
- **Mean strain**: {state.get('mean_strain', 0.0):.3f}
- **Critical clusters**: {state.get('n_critical_clusters', 0)}
"""
        
        # グローバルネットワーク統計
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            report += f"""
### Global Network Statistics
- **Strain network links**: {stats.get('total_strain_links', 0)}
- **Dislocation links**: {stats.get('total_dislocation_links', 0)}
- **Damage links**: {stats.get('total_damage_links', 0)}
- **Max failure probability**: {stats.get('max_failure_probability', 0):.1%}
- **Min reliability index**: {stats.get('min_reliability_index', 5.0):.2f}
"""
        
        # 臨界クラスター
        if hasattr(two_stage_result, 'critical_clusters'):
            critical = two_stage_result.critical_clusters
            if critical:
                report += f"\n### Critical Clusters (High Risk)\n"
                for i, cluster_id in enumerate(critical[:10], 1):
                    report += f"{i}. **Cluster {cluster_id}**"
                    
                    # 重要度スコア
                    if hasattr(two_stage_result, 'global_cluster_importance'):
                        importance = two_stage_result.global_cluster_importance.get(cluster_id, 0)
                        report += f" (importance: {importance:.3f})"
                    report += "\n"
        
        # 補強推奨箇所
        if hasattr(two_stage_result, 'suggested_reinforcement_points'):
            reinforce = two_stage_result.suggested_reinforcement_points
            if reinforce:
                report += f"\n### Reinforcement Recommendations\n"
                report += f"Priority clusters for reinforcement: {reinforce[:5]}\n"
        
        # 各イベントの詳細
        if hasattr(two_stage_result, 'cluster_analyses'):
            report += "\n### Event-Specific Analysis\n"
            
            for event_name, analysis in two_stage_result.cluster_analyses.items():
                report += f"\n#### {event_name}\n"
                
                # 基本統計
                if hasattr(analysis, 'cluster_events'):
                    n_clusters = len(analysis.cluster_events)
                    report += f"- Clusters involved: {n_clusters}\n"
                    
                    # 最大値抽出
                    max_strain = max((e.peak_strain for e in analysis.cluster_events), default=0)
                    max_damage = max((e.peak_damage for e in analysis.cluster_events), default=0)
                    
                    report += f"- Max strain: {max_strain:.3f}\n"
                    report += f"- Max damage: {max_damage:.1%}\n"
                    
                    # 転位密度
                    disl_densities = [e.dislocation_density for e in analysis.cluster_events 
                                    if e.dislocation_density]
                    if disl_densities:
                        report += f"- Mean dislocation density: {np.mean(disl_densities):.2e} /cm²\n"
                
                # ネットワーク統計
                if hasattr(analysis, 'network_stats'):
                    ns = analysis.network_stats
                    report += f"- Network links: strain={ns.get('n_strain_links', 0)}, "
                    report += f"dislocation={ns.get('n_dislocation_links', 0)}, "
                    report += f"damage={ns.get('n_damage_links', 0)}\n"
                
                # 破壊確率
                if hasattr(analysis, 'failure_probability'):
                    report += f"- Failure probability: {analysis.failure_probability:.1%}\n"
                
                # 信頼性指標
                if hasattr(analysis, 'reliability_index'):
                    report += f"- Reliability index β: {analysis.reliability_index:.2f}\n"
                
                # 伝播経路
                if hasattr(analysis, 'propagation_paths') and analysis.propagation_paths:
                    report += f"- Propagation paths:\n"
                    for i, path in enumerate(analysis.propagation_paths[:3], 1):
                        path_str = ' → '.join([f"C{c}" for c in path])
                        report += f"  {i}. {path_str}\n"
    
    # ========================================
    # 3. 原子レベル欠陥解析
    # ========================================
    if impact_results:
        if verbose:
            print("\n⚛️ Extracting atomic defect analysis...")
        
        report += "\n## ⚛️ Atomic-Level Defect Analysis\n"
        
        # 統計サマリー
        total_defects = sum(r.n_defect_atoms for r in impact_results.values())
        total_links = sum(r.n_network_links for r in impact_results.values())
        total_bridges = sum(r.n_cluster_bridges for r in impact_results.values())
        
        report += f"""
### Overview
- **Events analyzed**: {len(impact_results)}
- **Total defect atoms**: {total_defects}
- **Network links**: {total_links}
- **Cluster bridges**: {total_bridges}
"""
        
        # 欠陥タイプ分布
        defect_types = Counter()
        for result in impact_results.values():
            if result.dominant_defect:
                defect_types[result.dominant_defect] += 1
        
        if defect_types:
            report += "\n### Defect Type Distribution\n"
            for dtype, count in defect_types.most_common():
                report += f"- **{dtype}**: {count} events\n"
        
        # 各イベントの詳細
        report += "\n### Detailed Defect Analysis\n"
        
        for event_key, result in list(impact_results.items())[:10]:  # Top 10
            report += f"\n#### {event_key}\n"
            report += f"- **Event type**: {result.event_type}\n"
            report += f"- **Dominant defect**: {result.dominant_defect}\n"
            report += f"- **Max stress concentration**: {result.max_stress_concentration:.2f} GPa\n"
            
            # 核生成原子
            if result.origin.nucleation_atoms:
                report += f"- **Nucleation atoms**: {result.origin.nucleation_atoms[:5]}\n"
            
            # ネットワークパターン
            if result.defect_network:
                network = result.defect_network
                report += f"- **Network pattern**: {network.network_pattern}\n"
                
                # 転位コア
                if network.dislocation_cores:
                    report += f"- **Dislocation cores**: {network.dislocation_cores[:3]}\n"
                
                # 亀裂先端
                if network.crack_tips:
                    report += f"- **Crack tips**: {network.crack_tips[:3]}\n"
                
                # クラスター間ブリッジ
                if network.cluster_bridges:
                    bridge = network.cluster_bridges[0]
                    report += f"- **Main bridge**: C{bridge.from_cluster}→C{bridge.to_cluster} "
                    report += f"({bridge.bridge_type})\n"
            
            # 材料パラメータ
            if result.plastic_zone_size:
                report += f"- **Plastic zone size**: {result.plastic_zone_size:.2f} Å\n"
            
            if result.estimated_k_ic:
                report += f"- **Estimated K_IC**: {result.estimated_k_ic:.1f} MPa√m\n"
            
            # 補強ポイント
            if result.reinforcement_points:
                report += f"- **Reinforcement points**: {result.reinforcement_points[:5]}\n"
    
    # ========================================
    # 4. イベント時系列解析
    # ========================================
    if sorted_events:
        if verbose:
            print("\n📅 Extracting event timeline...")
        
        report += "\n## 📅 Event Timeline Analysis\n"
        
        # スコア順TOP20
        report += "\n### Top 20 Events by Score\n"
        report += "| Rank | Frames | Duration | Score | Type |\n"
        report += "|------|--------|----------|-------|------|\n"
        
        for i, (start, end, score) in enumerate(sorted_events[:20], 1):
            duration = end - start
            
            # イベントタイプ推定
            event_type = "unknown"
            for event in all_events:
                if event['start'] == start and event['end'] == end:
                    event_type = event['type']
                    break
            
            report += f"| {i} | {start:6d}-{end:6d} | {duration:5d} | {score:.3f} | {event_type} |\n"
        
        # 高スコアイベント
        high_score = [(s, e, sc) for s, e, sc in sorted_events if sc >= 10.0]
        if high_score:
            report += f"\n### Critical Events (Score ≥ 10.0)\n"
            report += f"Found {len(high_score)} critical events requiring immediate attention.\n"
    
    # ========================================
    # 5. 統合的材料評価
    # ========================================
    if verbose:
        print("\n💡 Generating integrated material insights...")
    
    report += "\n## 💡 Integrated Material Assessment\n"
    
    insights = []
    
    # マクロレベル
    if all_events:
        event_types = Counter(e['type'] for e in all_events)
        
        if event_types.get('plastic_deformation', 0) > 5:
            insights.append("✓ Multiple plastic deformation events - significant irreversible damage")
        
        if event_types.get('crack_initiation', 0) > 0:
            insights.append("⚠️ Crack initiation detected - structural integrity compromised")
        
        if event_types.get('dislocation_nucleation', 0) > 10:
            insights.append("✓ High dislocation activity - strain hardening expected")
    
    # クラスターレベル
    if two_stage_result:
        if hasattr(two_stage_result, 'material_state'):
            state = two_stage_result.material_state
            if state.get('state') == 'critical_damage':
                insights.append("🔴 CRITICAL: Material in failure state")
            elif state.get('state') == 'moderate_damage':
                insights.append("🟡 WARNING: Moderate damage detected")
        
        if hasattr(two_stage_result, 'global_network_stats'):
            stats = two_stage_result.global_network_stats
            if stats.get('max_failure_probability', 0) > 0.5:
                insights.append(f"⚠️ High failure probability: {stats['max_failure_probability']:.1%}")
    
    # 原子レベル
    if impact_results:
        # 転位密度
        has_dislocations = any(r.dominant_defect == 'dislocation_core' for r in impact_results.values())
        if has_dislocations:
            insights.append("✓ Dislocation cores identified at atomic level")
        
        # 亀裂
        has_cracks = any(r.dominant_defect == 'crack_tip' for r in impact_results.values())
        if has_cracks:
            insights.append("⚠️ Crack tips detected - fracture mechanics analysis recommended")
        
        # 応力集中
        max_stress = max((r.max_stress_concentration for r in impact_results.values()), default=0)
        if max_stress > 5.0:  # 5 GPa
            insights.append(f"⚠️ Extreme stress concentration: {max_stress:.1f} GPa")
    
    for insight in insights:
        report += f"\n{insight}"
    
    # ========================================
    # 6. 材料設計推奨事項
    # ========================================
    report += "\n\n## 🛠️ Material Design Recommendations\n"
    
    recommendations = []
    
    # 補強推奨
    if two_stage_result and hasattr(two_stage_result, 'suggested_reinforcement_points'):
        reinforce = two_stage_result.suggested_reinforcement_points
        if reinforce:
            recommendations.append(f"Reinforce clusters {reinforce[:3]} to prevent failure")
    
    # 転位制御
    if impact_results:
        disl_events = sum(1 for r in impact_results.values() 
                         if r.dominant_defect == 'dislocation_core')
        if disl_events > 5:
            recommendations.append("Consider grain refinement to control dislocation motion")
    
    # 亀裂抑制
    if impact_results:
        crack_events = sum(1 for r in impact_results.values() 
                          if r.dominant_defect == 'crack_tip')
        if crack_events > 0:
            recommendations.append("Add crack arresters or increase fracture toughness")
    
    # 応力緩和
    if impact_results:
        max_stress = max((r.max_stress_concentration for r in impact_results.values()), default=0)
        if max_stress > 3.0:
            recommendations.append("Design geometry modifications to reduce stress concentration")
    
    # 材料選択
    if material_type == 'SUJ2' and two_stage_result:
        if hasattr(two_stage_result, 'material_state'):
            if two_stage_result.material_state.get('max_damage', 0) > 0.3:
                recommendations.append("Consider higher toughness steel alloy")
    
    for i, rec in enumerate(recommendations, 1):
        report += f"\n{i}. {rec}"
    
    # ========================================
    # 7. 技術仕様
    # ========================================
    report += "\n\n## 📋 Technical Specifications\n"
    
    # 材料プロパティ
    report += f"\n### Material Properties ({material_type})\n"
    
    material_props = {
        'SUJ2': {
            'E': 210.0,
            'nu': 0.3,
            'yield': 1.5,
            'ultimate': 2.0,
            'K_IC': 30.0
        },
        'AL7075': {
            'E': 71.7,
            'nu': 0.33,
            'yield': 0.503,
            'ultimate': 0.572,
            'K_IC': 23.0
        }
    }
    
    if material_type in material_props:
        props = material_props[material_type]
        report += f"- Elastic modulus: {props['E']} GPa\n"
        report += f"- Poisson's ratio: {props['nu']}\n"
        report += f"- Yield strength: {props['yield']} GPa\n"
        report += f"- Ultimate strength: {props['ultimate']} GPa\n"
        report += f"- Fracture toughness: {props['K_IC']} MPa√m\n"
    
    # 解析パラメータ
    report += "\n### Analysis Parameters\n"
    report += "- Defect detection: Adaptive thresholding\n"
    report += "- Network analysis: GPU-accelerated\n"
    report += "- Reliability: 95% confidence level\n"
    report += "- Failure criteria: von Mises stress\n"
    
    # フッター
    report += f"""

---
*Material Analysis Complete!*
*Version: 1.0 - Material Lambda³ GPU Edition*
*Material: {material_type}*
*Total report length: {len(report):,} characters*
"""
    
    # ========================================
    # 保存とエクスポート
    # ========================================
    
    # Markdownレポート保存
    report_path = output_path / 'material_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    # JSON形式でも保存
    json_data = {
        'version': '1.0',
        'material_type': material_type,
        'summary': {
            'n_frames': macro_result.n_frames,
            'n_atoms': macro_result.n_atoms,
            'computation_time': macro_result.computation_time,
            'total_events': len(all_events)
        },
        'events': all_events[:100],
        'metadata': metadata if metadata else {}
    }
    
    # 材料状態
    if two_stage_result and hasattr(two_stage_result, 'material_state'):
        json_data['material_state'] = two_stage_result.material_state
    
    # ネットワーク統計
    if two_stage_result and hasattr(two_stage_result, 'global_network_stats'):
        json_data['network_stats'] = two_stage_result.global_network_stats
    
    # 欠陥統計
    if impact_results:
        json_data['defect_statistics'] = {
            'total_defect_atoms': sum(r.n_defect_atoms for r in impact_results.values()),
            'defect_types': dict(defect_types),
            'max_stress_concentration': max((r.max_stress_concentration for r in impact_results.values()), default=0)
        }
    
    json_path = output_path / 'material_analysis_data.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=float)
    
    # VTKファイル生成用データ
    vtk_data = prepare_vtk_export_data(
        macro_result, two_stage_result, impact_results
    )
    if vtk_data:
        vtk_path = output_path / 'visualization_data.json'
        with open(vtk_path, 'w') as f:
            json.dump(vtk_data, f, indent=2, default=float)
    
    if verbose:
        print(f"\n✨ COMPLETE! (Material Report v1.0)")
        print(f"   📄 Report saved to: {report_path}")
        print(f"   📊 Data saved to: {json_path}")
        print(f"   📏 Report length: {len(report):,} characters")
        print(f"   💎 Material events: {len(all_events)}")
        if two_stage_result:
            print(f"   🔬 Critical clusters: {len(getattr(two_stage_result, 'critical_clusters', []))}")
        if impact_results:
            print(f"   ⚛️ Defect analyses: {len(impact_results)}")
        print(f"\n   Material analysis complete for {material_type}!")
    
    return report

# ========================================
# ヘルパー関数
# ========================================

def prepare_vtk_export_data(macro_result, two_stage_result, impact_results):
    """
    VTK可視化用データ準備
    
    ParaViewやOVITOで可視化できる形式に整形
    """
    vtk_data = {
        'format': 'lambda3_material',
        'version': '1.0',
        'frames': []
    }
    
    # フレームごとのデータ
    if hasattr(macro_result, 'anomaly_scores'):
        n_frames = len(macro_result.anomaly_scores.get('strain', []))
        
        for frame in range(min(n_frames, 100)):  # 最初の100フレーム
            frame_data = {
                'frame_id': frame,
                'scalars': {},
                'vectors': {},
                'tensors': {}
            }
            
            # スカラー場
            for score_type in ['strain', 'coordination', 'damage']:
                if score_type in macro_result.anomaly_scores:
                    scores = macro_result.anomaly_scores[score_type]
                    if frame < len(scores):
                        frame_data['scalars'][score_type] = float(scores[frame])
            
            vtk_data['frames'].append(frame_data)
    
    # クラスター情報
    if two_stage_result and hasattr(two_stage_result, 'critical_clusters'):
        vtk_data['clusters'] = {
            'critical': two_stage_result.critical_clusters,
            'reinforcement': getattr(two_stage_result, 'suggested_reinforcement_points', [])
        }
    
    # 欠陥位置
    if impact_results:
        defects = []
        for result in impact_results.values():
            if result.origin.nucleation_atoms:
                defects.extend(result.origin.nucleation_atoms[:10])
        vtk_data['defect_atoms'] = list(set(defects))
    
    return vtk_data if vtk_data['frames'] else None

def generate_material_comparison_report(
    results_dict: Dict[str, Any],
    material_types: List[str],
    output_dir: str = './comparison_report'
) -> str:
    """
    複数材料の比較レポート生成
    
    Parameters
    ----------
    results_dict : Dict[str, Any]
        材料タイプごとの解析結果
    material_types : List[str]
        比較する材料タイプのリスト
    """
    report = "# Material Comparison Report\n\n"
    
    # 各材料の統計比較
    report += "## Performance Comparison\n\n"
    report += "| Material | Max Stress | Fracture Strain | Reliability | State |\n"
    report += "|----------|------------|-----------------|-------------|-------|\n"
    
    for material in material_types:
        if material in results_dict:
            result = results_dict[material]
            # 統計抽出と表示
            report += f"| {material} | ... | ... | ... | ... |\n"
    
    return report
