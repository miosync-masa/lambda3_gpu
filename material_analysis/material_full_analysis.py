#!/usr/bin/env python3
"""
Material Full Analysis Pipeline - Lambda³ GPU Material Edition
================================================================

材料解析の完全統合パイプライン
転位・亀裂・相変態を階層的に解析

Version: 1.0.0 - Material Edition
Authors: 環ちゃん
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

# Lambda³ Material imports
try:
    from material_lambda3_detector import (
        MaterialLambda3DetectorGPU,
        MaterialLambda3Result,
        MaterialConfig
    )
    from material_two_stage_analyzer import (
        MaterialTwoStageAnalyzerGPU,
        MaterialTwoStageResult,
        ClusterAnalysisConfig
    )
    from material_impact_analytics import (
        run_material_impact_analysis,
        MaterialImpactAnalyzer,
        MaterialImpactResult
    )
    from material_report_generator import (
        generate_material_report_from_results
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all material analysis modules are in the same directory")
    raise

# Logger設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('material_full_analysis')

# ============================================
# メイン実行関数
# ============================================

def run_material_analysis_pipeline(
    trajectory_path: str,
    metadata_path: str,
    atom_types_path: str,
    material_type: str = 'SUJ2',
    cluster_definition_path: Optional[str] = None,
    strain_field_path: Optional[str] = None,
    enable_two_stage: bool = True,
    enable_impact: bool = True,
    enable_visualization: bool = True,
    output_dir: str = './material_results',
    loading_type: str = 'tensile',
    strain_rate: float = 1e-3,
    temperature: float = 300.0,
    verbose: bool = False,
    **kwargs
) -> Dict:
    """
    材料解析の完全パイプライン
    
    Parameters
    ----------
    trajectory_path : str
        トラジェクトリファイル (.npy)
    metadata_path : str
        メタデータファイル (.json)
    atom_types_path : str
        原子タイプ配列 (.npy)
    material_type : str
        材料タイプ (SUJ2, AL7075等)
    cluster_definition_path : str, optional
        クラスター定義ファイル
    strain_field_path : str, optional
        歪み場データ
    enable_two_stage : bool
        2段階解析を実行
    enable_impact : bool
        原子レベル解析を実行
    enable_visualization : bool
        可視化を実行
    output_dir : str
        出力ディレクトリ
    loading_type : str
        負荷タイプ (tensile, compression, shear)
    strain_rate : float
        歪み速度 (/ps)
    temperature : float
        温度 (K)
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
    logger.info(f"💎 MATERIAL ANALYSIS PIPELINE - {material_type}")
    logger.info("   Lambda³ GPU Material Edition v1.0")
    logger.info("="*70)
    
    # ========================================
    # Step 1: データ読み込み
    # ========================================
    logger.info("\n📁 Loading material data...")
    
    try:
        # トラジェクトリ読み込み
        trajectory = np.load(trajectory_path)
        logger.info(f"   Trajectory shape: {trajectory.shape}")
        n_frames, n_atoms, n_dims = trajectory.shape
        
        if n_dims != 3:
            raise ValueError(f"Trajectory must be 3D, got {n_dims}D")
        
        # メタデータ読み込み（既存または新規作成）
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'system_name': f'{material_type}_simulation',
                'material_type': material_type,
                'temperature': temperature,
                'strain_rate': strain_rate,
                'loading_type': loading_type
            }
        
        # メタデータ更新
        metadata.update({
            'material_type': material_type,
            'temperature': temperature,
            'strain_rate': strain_rate,
            'loading_type': loading_type
        })
        
        logger.info(f"   Material: {material_type}")
        logger.info(f"   Temperature: {temperature} K")
        logger.info(f"   Loading: {loading_type}")
        logger.info(f"   Strain rate: {strain_rate} /ps")
        
        # 原子タイプ読み込み
        atom_types = np.load(atom_types_path)
        logger.info(f"   Atom types loaded: {len(np.unique(atom_types))} types")
        
        # クラスター定義読み込み
        if cluster_definition_path and Path(cluster_definition_path).exists():
            if cluster_definition_path.endswith('.json'):
                with open(cluster_definition_path, 'r') as f:
                    cluster_atoms_raw = json.load(f)
                    cluster_atoms = {int(k): v for k, v in cluster_atoms_raw.items()}
            else:
                cluster_atoms = np.load(cluster_definition_path, allow_pickle=True).item()
            logger.info(f"   Clusters defined: {len(cluster_atoms)}")
        else:
            # デフォルト：空間分割でクラスター作成
            n_clusters = min(100, n_atoms // 50)
            cluster_atoms = create_spatial_clusters(trajectory[0], n_clusters)
            logger.info(f"   Auto-generated {n_clusters} clusters")
        
        # 歪み場読み込み
        strain_field = None
        if strain_field_path and Path(strain_field_path).exists():
            strain_field = np.load(strain_field_path)
            logger.info(f"   Strain field loaded: shape {strain_field.shape}")
        
        logger.info("   ✅ Data validation passed")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # ========================================
    # Step 2: マクロ材料解析
    # ========================================
    logger.info(f"\n🔬 Running Macro Material Analysis ({material_type})...")
    
    try:
        # MaterialConfig設定
        config = MaterialConfig()
        config.material_type = material_type
        config.loading_type = loading_type
        config.detect_dislocations = True
        config.detect_cracks = True
        config.detect_phase_transitions = True
        config.sensitivity = 2.0
        config.gpu_batch_size = 50000
        config.verbose = verbose
        
        # 材料パラメータ設定
        material_params = get_material_parameters(material_type)
        config.elastic_modulus = material_params['E']
        config.yield_strength = material_params['yield']
        config.fracture_toughness = material_params['K_IC']
        
        logger.info(f"   E = {config.elastic_modulus} GPa")
        logger.info(f"   σ_y = {config.yield_strength} GPa")
        logger.info(f"   K_IC = {config.fracture_toughness} MPa√m")
        
        # 検出器初期化
        detector = MaterialLambda3DetectorGPU(config)
        logger.info("   Detector initialized on GPU")
        
        # 解析実行
        macro_result = detector.analyze(
            trajectory=trajectory,
            atom_types=atom_types,
            strain_field=strain_field
        )
        
        logger.info(f"   ✅ Macro analysis complete")
        logger.info(f"   Material events detected: {len(macro_result.material_events)}")
        
        # イベントタイプ別統計
        event_types = Counter()
        for event in macro_result.material_events:
            if isinstance(event, tuple) and len(event) >= 3:
                event_types[event[2]] += 1
        
        for etype, count in event_types.most_common():
            logger.info(f"     - {etype}: {count}")
        
        # 結果保存
        result_summary = {
            'material_type': material_type,
            'n_frames': macro_result.n_frames,
            'n_atoms': macro_result.n_atoms,
            'n_events': len(macro_result.material_events),
            'event_types': dict(event_types),
            'computation_time': macro_result.computation_time
        }
        
        with open(output_path / 'macro_result_summary.json', 'w') as f:
            json.dump(result_summary, f, indent=2)
            
    except Exception as e:
        logger.error(f"Macro analysis failed: {e}")
        raise
    
    # ========================================
    # Step 3: 2段階クラスター解析
    # ========================================
    two_stage_result = None
    sorted_events = []
    
    if enable_two_stage and len(macro_result.material_events) > 0:
        logger.info("\n🔬 Running Two-Stage Cluster Analysis...")
        
        try:
            # イベントのスコア付けとソート
            for event in macro_result.material_events:
                if isinstance(event, tuple) and len(event) >= 3:
                    start, end, event_type = event[:3]
                    
                    # スコア計算（イベントタイプ別の重要度）
                    type_scores = {
                        'crack_initiation': 10.0,
                        'crack_propagation': 9.0,
                        'plastic_deformation': 7.0,
                        'dislocation_nucleation': 6.0,
                        'phase_transition': 8.0,
                        'elastic_deformation': 3.0,
                        'fatigue_damage': 5.0
                    }
                    base_score = type_scores.get(event_type, 1.0)
                    
                    # 異常スコアがあれば追加
                    if hasattr(macro_result, 'anomaly_scores'):
                        if 'damage' in macro_result.anomaly_scores:
                            damage_score = np.max(
                                macro_result.anomaly_scores['damage'][start:end+1]
                            )
                            score = base_score * (1 + damage_score)
                        else:
                            score = base_score
                    else:
                        score = base_score
                    
                    sorted_events.append((start, end, score))
            
            # スコア順ソート
            sorted_events.sort(key=lambda x: x[2], reverse=True)
            
            # TOP50選択
            MAX_EVENTS = min(50, len(sorted_events))
            selected_events = sorted_events[:MAX_EVENTS]
            
            logger.info(f"   Selected TOP {len(selected_events)} events")
            
            # イベントリスト作成（名前付き）
            detected_events = []
            for i, (start, end, score) in enumerate(selected_events):
                # イベントタイプも含める
                event_type = 'unknown'
                for event in macro_result.material_events:
                    if isinstance(event, tuple) and len(event) >= 3:
                        if event[0] == start and event[1] == end:
                            event_type = event[2]
                            break
                
                event_name = f"{event_type}_{i:02d}_score_{score:.2f}"
                detected_events.append((start, end, event_name))
            
            # ClusterAnalysisConfig設定
            cluster_config = ClusterAnalysisConfig()
            cluster_config.detect_dislocations = True
            cluster_config.detect_cracks = True
            cluster_config.detect_phase_transitions = True
            cluster_config.use_confidence = True
            cluster_config.gpu_batch_clusters = 30
            cluster_config.verbose = verbose
            
            # TwoStageAnalyzer初期化
            logger.info("   Initializing Two-Stage Analyzer...")
            two_stage_analyzer = MaterialTwoStageAnalyzerGPU(
                config=cluster_config,
                material_type=material_type
            )
            
            # 解析実行
            logger.info(f"   Analyzing {len(detected_events)} events...")
            two_stage_result = two_stage_analyzer.analyze_trajectory(
                trajectory=trajectory,
                macro_result=macro_result,
                detected_events=detected_events,
                cluster_atoms=cluster_atoms,
                atom_types=atom_types
            )
            
            logger.info(f"   ✅ Two-stage analysis complete")
            
            # 材料状態表示
            if hasattr(two_stage_result, 'material_state'):
                state = two_stage_result.material_state
                logger.info(f"   Material state: {state['state']}")
                logger.info(f"   Health index: {state['health_index']:.1%}")
                logger.info(f"   Max damage: {state['max_damage']:.1%}")
            
            # 臨界クラスター
            if hasattr(two_stage_result, 'critical_clusters'):
                logger.info(f"   Critical clusters: {len(two_stage_result.critical_clusters)}")
                if two_stage_result.critical_clusters:
                    logger.info(f"     Top 5: {two_stage_result.critical_clusters[:5]}")
            
        except Exception as e:
            logger.error(f"Two-stage analysis failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            two_stage_result = None
    
    # ========================================
    # Step 4: 原子レベル欠陥解析
    # ========================================
    impact_results = None
    
    if enable_impact and two_stage_result is not None:
        logger.info("\n⚛️ Running Atomic-Level Defect Analysis...")
        
        try:
            # MaterialImpactAnalyzer初期化
            impact_analyzer = MaterialImpactAnalyzer(
                cluster_mapping=cluster_atoms,
                sigma_threshold=2.5,
                use_network_analysis=True,
                use_gpu=True,
                material_type=material_type
            )
            
            # 解析実行
            impact_results = impact_analyzer.analyze_critical_clusters(
                macro_result=macro_result,
                two_stage_result=two_stage_result,
                trajectory=trajectory,
                atom_types=atom_types,
                strain_field=strain_field,
                top_n=5  # Top 5クラスター
            )
            
            if impact_results:
                # 統計表示
                total_defects = sum(r.n_defect_atoms for r in impact_results.values())
                total_links = sum(r.n_network_links for r in impact_results.values())
                
                logger.info(f"   ✅ Defect analysis complete")
                logger.info(f"   Total defect atoms: {total_defects}")
                logger.info(f"   Network links: {total_links}")
                
                # 欠陥タイプ分布
                defect_types = Counter()
                for result in impact_results.values():
                    if result.dominant_defect:
                        defect_types[result.dominant_defect] += 1
                
                for dtype, count in defect_types.most_common():
                    logger.info(f"     - {dtype}: {count}")
                
                # 最大応力集中
                max_stress = max(
                    (r.max_stress_concentration for r in impact_results.values()),
                    default=0
                )
                if max_stress > 0:
                    logger.info(f"   Max stress concentration: {max_stress:.2f} GPa")
            
        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            impact_results = None
    
    # ========================================
    # Step 5: 統合レポート生成
    # ========================================
    logger.info("\n📝 Generating comprehensive material report...")
    
    try:
        report = generate_material_report_from_results(
            macro_result=macro_result,
            two_stage_result=two_stage_result,
            impact_results=impact_results,
            sorted_events=sorted_events,
            metadata=metadata,
            material_type=material_type,
            output_dir=str(output_path),
            verbose=verbose
        )
        
        logger.info(f"   ✅ Report generated: {len(report):,} characters")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Step 6: 可視化
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating visualizations...")
        
        try:
            # 応力-歪み曲線
            if hasattr(macro_result, 'stress_strain_curve'):
                visualize_stress_strain(
                    macro_result.stress_strain_curve,
                    material_type,
                    save_path=str(output_path / 'stress_strain.png')
                )
                logger.info("   Stress-strain curve visualized")
            
            # イベントタイムライン
            visualize_event_timeline(
                macro_result.material_events,
                save_path=str(output_path / 'event_timeline.png')
            )
            logger.info("   Event timeline visualized")
            
            # クラスターダメージマップ
            if two_stage_result:
                visualize_cluster_damage(
                    two_stage_result,
                    save_path=str(output_path / 'cluster_damage.png')
                )
                logger.info("   Cluster damage map visualized")
            
            # 欠陥ネットワーク
            if impact_results:
                visualize_defect_network(
                    impact_results,
                    save_path=str(output_path / 'defect_network.png')
                )
                logger.info("   Defect network visualized")
            
            logger.info("   ✅ All visualizations saved")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # ========================================
    # 完了
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("✅ MATERIAL ANALYSIS COMPLETE!")
    logger.info(f"   Material: {material_type}")
    logger.info(f"   Results directory: {output_path}")
    logger.info(f"   Key findings:")
    
    if macro_result:
        logger.info(f"     - {len(macro_result.material_events)} material events")
    if two_stage_result:
        if hasattr(two_stage_result, 'material_state'):
            logger.info(f"     - Material state: {two_stage_result.material_state['state']}")
        if hasattr(two_stage_result, 'critical_clusters'):
            logger.info(f"     - {len(two_stage_result.critical_clusters)} critical clusters")
    if impact_results:
        logger.info(f"     - {len(impact_results)} defect analyses")
    
    logger.info("="*70)
    
    return {
        'macro_result': macro_result,
        'two_stage_result': two_stage_result,
        'impact_results': impact_results,
        'output_dir': output_path,
        'material_type': material_type,
        'success': True
    }

# ============================================
# ヘルパー関数
# ============================================

def get_material_parameters(material_type: str) -> Dict:
    """材料パラメータ取得"""
    
    materials = {
        'SUJ2': {
            'E': 210.0,      # GPa
            'nu': 0.3,       # Poisson's ratio
            'yield': 1.5,    # GPa
            'ultimate': 2.0, # GPa
            'K_IC': 30.0,    # MPa√m
            'density': 7.85  # g/cm³
        },
        'AL7075': {
            'E': 71.7,
            'nu': 0.33,
            'yield': 0.503,
            'ultimate': 0.572,
            'K_IC': 23.0,
            'density': 2.81
        },
        'Ti6Al4V': {
            'E': 113.8,
            'nu': 0.342,
            'yield': 0.88,
            'ultimate': 0.95,
            'K_IC': 75.0,
            'density': 4.43
        },
        'SS316L': {
            'E': 193.0,
            'nu': 0.3,
            'yield': 0.205,
            'ultimate': 0.515,
            'K_IC': 112.0,
            'density': 8.0
        }
    }
    
    if material_type in materials:
        return materials[material_type]
    else:
        # デフォルト（鋼鉄）
        return materials['SUJ2']

def create_spatial_clusters(positions: np.ndarray, n_clusters: int) -> Dict[int, List[int]]:
    """空間分割によるクラスター作成"""
    
    from sklearn.cluster import KMeans
    
    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(positions)
    
    # クラスター辞書作成
    cluster_atoms = {}
    for i in range(n_clusters):
        cluster_atoms[i] = np.where(labels == i)[0].tolist()
    
    return cluster_atoms

# ============================================
# 可視化関数
# ============================================

def visualize_stress_strain(curve_data: Dict, material_type: str,
                          save_path: Optional[str] = None) -> plt.Figure:
    """応力-歪み曲線の可視化"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'strain' in curve_data and 'stress' in curve_data:
        strain = curve_data['strain']
        stress = curve_data['stress']
        
        ax.plot(strain, stress, 'b-', linewidth=2, label=material_type)
        
        # 降伏点マーク
        if 'yield_point' in curve_data:
            yield_idx = curve_data['yield_point']
            ax.plot(strain[yield_idx], stress[yield_idx], 'ro', 
                   markersize=10, label='Yield Point')
        
        # 破断点マーク
        if 'fracture_point' in curve_data:
            frac_idx = curve_data['fracture_point']
            ax.plot(strain[frac_idx], stress[frac_idx], 'rx',
                   markersize=12, label='Fracture')
        
        ax.set_xlabel('Strain', fontsize=12)
        ax.set_ylabel('Stress (GPa)', fontsize=12)
        ax.set_title(f'Stress-Strain Curve - {material_type}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_event_timeline(events: List, save_path: Optional[str] = None) -> plt.Figure:
    """イベントタイムラインの可視化"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # イベントタイプ別の色
    colors = {
        'elastic_deformation': 'green',
        'plastic_deformation': 'yellow',
        'dislocation_nucleation': 'orange',
        'crack_initiation': 'red',
        'crack_propagation': 'darkred',
        'phase_transition': 'purple',
        'fatigue_damage': 'brown'
    }
    
    y_positions = {}
    y_counter = 0
    
    for event in events:
        if isinstance(event, tuple) and len(event) >= 3:
            start, end, event_type = event[:3]
            
            if event_type not in y_positions:
                y_positions[event_type] = y_counter
                y_counter += 1
            
            y = y_positions[event_type]
            color = colors.get(event_type, 'gray')
            
            ax.barh(y, end - start, left=start, height=0.8,
                   color=color, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_title('Material Event Timeline', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_cluster_damage(two_stage_result: Any,
                            save_path: Optional[str] = None) -> plt.Figure:
    """クラスターダメージマップの可視化"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 臨界クラスター
    ax1 = axes[0]
    if hasattr(two_stage_result, 'critical_clusters'):
        critical = two_stage_result.critical_clusters[:20]
        if critical:
            ax1.bar(range(len(critical)), [1]*len(critical), color='red', alpha=0.7)
            ax1.set_xticks(range(len(critical)))
            ax1.set_xticklabels([f"C{c}" for c in critical], rotation=45)
            ax1.set_ylabel('Criticality', fontsize=12)
            ax1.set_title('Critical Clusters', fontsize=12)
    
    # クラスター重要度
    ax2 = axes[1]
    if hasattr(two_stage_result, 'global_cluster_importance'):
        top_clusters = sorted(
            two_stage_result.global_cluster_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        if top_clusters:
            cluster_ids = [f"C{c[0]}" for c in top_clusters]
            scores = [c[1] for c in top_clusters]
            
            ax2.barh(cluster_ids, scores, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Importance Score', fontsize=12)
            ax2.set_title('Cluster Importance Ranking', fontsize=12)
            ax2.invert_yaxis()
    
    plt.suptitle('Cluster Damage Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_defect_network(impact_results: Dict,
                            save_path: Optional[str] = None) -> plt.Figure:
    """欠陥ネットワークの可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 欠陥タイプ分布
    ax1 = axes[0, 0]
    defect_types = Counter()
    for result in impact_results.values():
        if result.dominant_defect:
            defect_types[result.dominant_defect] += 1
    
    if defect_types:
        ax1.pie(defect_types.values(), labels=defect_types.keys(),
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Defect Type Distribution')
    
    # 応力集中分布
    ax2 = axes[0, 1]
    stress_concs = [r.max_stress_concentration for r in impact_results.values()]
    if stress_concs:
        ax2.hist(stress_concs, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Stress Concentration (GPa)')
        ax2.set_ylabel('Count')
        ax2.set_title('Stress Concentration Distribution')
        ax2.grid(True, alpha=0.3)
    
    # ネットワークパターン
    ax3 = axes[1, 0]
    patterns = []
    for result in impact_results.values():
        if result.defect_network:
            patterns.append(result.defect_network.network_pattern)
    
    if patterns:
        pattern_counts = Counter(patterns)
        ax3.bar(pattern_counts.keys(), pattern_counts.values(),
               color='green', alpha=0.7)
        ax3.set_xlabel('Network Pattern')
        ax3.set_ylabel('Count')
        ax3.set_title('Defect Network Patterns')
    
    # 塑性域サイズ
    ax4 = axes[1, 1]
    plastic_zones = [r.plastic_zone_size for r in impact_results.values()
                    if r.plastic_zone_size]
    if plastic_zones:
        ax4.hist(plastic_zones, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Plastic Zone Size (Å)')
        ax4.set_ylabel('Count')
        ax4.set_title('Plastic Zone Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Atomic Defect Network Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# ============================================
# CLI Interface
# ============================================

def main():
    """コマンドラインインターフェース"""
    
    parser = argparse.ArgumentParser(
        description='Material Full Analysis Pipeline - Lambda³ GPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis for SUJ2 steel
  %(prog)s trajectory.npy atom_types.npy --material SUJ2
  
  # With metadata and clusters
  %(prog)s trajectory.npy atom_types.npy --material AL7075 \\
    --metadata metadata.json --clusters clusters.json
  
  # Full analysis with strain field
  %(prog)s trajectory.npy atom_types.npy --material Ti6Al4V \\
    --strain strain_field.npy --temperature 600 --output ./Ti_results
  
  # Compression test
  %(prog)s trajectory.npy atom_types.npy --material SS316L \\
    --loading compression --strain-rate 1e-2
        """
    )
    
    # 必須引数
    parser.add_argument('trajectory',
                       help='Path to trajectory file (.npy)')
    parser.add_argument('atom_types',
                       help='Path to atom types file (.npy)')
    
    # 材料設定
    parser.add_argument('--material', '-m',
                       default='SUJ2',
                       choices=['SUJ2', 'AL7075', 'Ti6Al4V', 'SS316L', 'custom'],
                       help='Material type (default: SUJ2)')
    parser.add_argument('--metadata',
                       help='Path to metadata file (.json)')
    parser.add_argument('--clusters',
                       help='Path to cluster definition file')
    parser.add_argument('--strain',
                       help='Path to strain field data (.npy)')
    
    # 解析設定
    parser.add_argument('--loading', '-l',
                       default='tensile',
                       choices=['tensile', 'compression', 'shear', 'fatigue'],
                       help='Loading type (default: tensile)')
    parser.add_argument('--strain-rate',
                       type=float, default=1e-3,
                       help='Strain rate (/ps, default: 1e-3)')
    parser.add_argument('--temperature', '-T',
                       type=float, default=300.0,
                       help='Temperature in K (default: 300)')
    
    # 出力設定
    parser.add_argument('--output', '-o',
                       default='./material_results',
                       help='Output directory')
    parser.add_argument('--no-two-stage',
                       action='store_true',
                       help='Skip two-stage cluster analysis')
    parser.add_argument('--no-impact',
                       action='store_true',
                       help='Skip atomic impact analysis')
    parser.add_argument('--no-viz',
                       action='store_true',
                       help='Skip visualization')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # メタデータパス設定
    metadata_path = args.metadata if args.metadata else 'metadata_auto.json'
    
    # パイプライン実行
    try:
        results = run_material_analysis_pipeline(
            trajectory_path=args.trajectory,
            metadata_path=metadata_path,
            atom_types_path=args.atom_types,
            material_type=args.material,
            cluster_definition_path=args.clusters,
            strain_field_path=args.strain,
            enable_two_stage=not args.no_two_stage,
            enable_impact=not args.no_impact,
            enable_visualization=not args.no_viz,
            output_dir=args.output,
            loading_type=args.loading,
            strain_rate=args.strain_rate,
            temperature=args.temperature,
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
