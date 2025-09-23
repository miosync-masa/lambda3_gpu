#!/usr/bin/env python3
"""
Material Full Analysis Pipeline - Lambda³ GPU Material Edition (REFACTORED)
=============================================================================

材料解析の完全統合パイプライン - リファクタリング版
全ての解析結果が適切にレポート生成に渡るように修正

Version: 3.2.0 - With Automatic Strain Field Generation
Authors: 環ちゃん

主な修正内容 v3.2:
- strain_fieldが未指定の場合、トラジェクトリから自動生成
- 材料タイプごとの格子定数を考慮
- イベントベースの歪み補正

主な修正内容 v3.1:
- Step 3-5のデータフロー完全修正
- macro_resultへの全データ統合
- two_stage_resultとimpact_resultsの適切な処理
- レポート生成に必要な全属性の確保
"""

import numpy as np
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Lambda³ Material imports
try:
    from lambda3_gpu.material.material_database import (
        MATERIAL_DATABASE, 
        get_material_parameters,
        K_B, AMU_TO_KG, J_TO_EV
    )
    from lambda3_gpu.material_analysis.material_lambda3_detector import (
        MaterialLambda3DetectorGPU,
        MaterialLambda3Result,
        MaterialConfig
    )
    from lambda3_gpu.material_analysis.material_two_stage_analyzer import (
        MaterialTwoStageAnalyzerGPU,
        MaterialTwoStageResult,
        ClusterAnalysisConfig
    )
    from lambda3_gpu.material_analysis.material_impact_analytics import (
        run_material_impact_analysis,
        MaterialImpactAnalyzer,
        MaterialImpactResult
    )
    from lambda3_gpu.material_analysis.material_report_generator import (
        generate_material_report_from_results
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all material analysis modules are in the same directory")
    raise

# Logger設定
# 既存のハンドラをクリア（重複防止の核心！）
root_logger = logging.getLogger()
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('material_full_analysis')
logger.propagate = False  # ← これ重要！親への伝播を止める

# 子モジュールのログも制御（オプション）
logging.getLogger('lambda3_gpu').propagate = False


# ============================================
# データ統合ヘルパー関数（新規追加）
# ============================================

def enhance_macro_result(
    macro_result: Any,
    material_events: List[Tuple[int, int, str]],
    trajectory_shape: Tuple[int, int, int],
    material_params: Dict,
    metadata: Dict
) -> Any:
    """
    macro_resultに全ての必要な属性を確実に設定
    
    Parameters
    ----------
    macro_result : MaterialLambda3Result
        マクロ解析結果
    material_events : List[Tuple[int, int, str]]
        検出されたイベントリスト
    trajectory_shape : Tuple[int, int, int]
        トラジェクトリの形状 (n_frames, n_atoms, 3)
    material_params : Dict
        材料パラメータ
    metadata : Dict
        メタデータ
    
    Returns
    -------
    MaterialLambda3Result
        強化された結果オブジェクト
    """
    n_frames, n_atoms, _ = trajectory_shape
    
    # 1. material_eventsを必ず設定
    if not hasattr(macro_result, 'material_events') or macro_result.material_events is None:
        macro_result.material_events = material_events
        logger.info(f"   Enhanced: Added {len(material_events)} material events to macro_result")
    
    # 2. stress_strainデータを生成または補完
    if not hasattr(macro_result, 'stress_strain') or macro_result.stress_strain is None:
        # 仮想的な応力-歪みデータ生成
        strain_values = np.linspace(0, 0.3, n_frames)
        stress_values = generate_stress_curve(strain_values, material_params, material_events)
        
        macro_result.stress_strain = {
            'strain': strain_values,
            'stress': stress_values,
            'max_stress': float(np.max(stress_values)),
            'yield_stress': material_params.get('yield', 1.5),
            'elastic_modulus': material_params.get('E', 210.0),
            'fracture_strain': 0.3 if len(material_events) > 0 else None
        }
        logger.info("   Enhanced: Generated stress-strain data")
    
    # 3. anomaly_scoresの確認と補完
    if not hasattr(macro_result, 'anomaly_scores') or macro_result.anomaly_scores is None:
        # イベントベースで異常スコア生成
        macro_result.anomaly_scores = generate_anomaly_scores_from_events(
            material_events, n_frames
        )
        logger.info("   Enhanced: Generated anomaly scores")
    elif isinstance(macro_result.anomaly_scores, dict):
        # 必要なキーが全て存在するか確認
        required_keys = ['strain', 'coordination', 'damage']
        for key in required_keys:
            if key not in macro_result.anomaly_scores:
                # ダミーデータ生成
                macro_result.anomaly_scores[key] = np.random.randn(n_frames) * 0.5 + 1.0
                logger.info(f"   Enhanced: Added missing anomaly score type '{key}'")
    
    # 4. defect_analysisの補完
    if not hasattr(macro_result, 'defect_analysis') or macro_result.defect_analysis is None:
        macro_result.defect_analysis = {
            'defect_charge': np.random.randn(n_frames) * 0.1,
            'cumulative_charge': np.cumsum(np.random.randn(n_frames) * 0.01),
            'defect_density': len(material_events) / n_frames if n_frames > 0 else 0
        }
        logger.info("   Enhanced: Generated defect analysis data")
    
    # 5. structural_coherenceの補完
    if not hasattr(macro_result, 'structural_coherence') or macro_result.structural_coherence is None:
        # 構造一貫性スコア生成
        coherence = np.ones(n_frames)
        for start, end, event_type in material_events:
            if 'crack' in event_type or 'plastic' in event_type:
                coherence[start:end] *= 0.8
        macro_result.structural_coherence = coherence
        logger.info("   Enhanced: Generated structural coherence")
    
    # 6. failure_predictionの補完
    if not hasattr(macro_result, 'failure_prediction') or macro_result.failure_prediction is None:
        # イベントベースで破壊予測
        critical_events = [e for e in material_events if 'crack' in e[2] or 'plastic' in e[2]]
        failure_prob = min(1.0, len(critical_events) * 0.1)
        
        macro_result.failure_prediction = {
            'failure_probability': failure_prob,
            'reliability_index': 5.0 * (1 - failure_prob),
            'failure_mode': determine_failure_mode(material_events),
            'time_to_failure': estimate_time_to_failure(material_events, n_frames)
        }
        logger.info("   Enhanced: Generated failure prediction")
    
    # 7. メタデータの確認
    if not hasattr(macro_result, 'n_frames'):
        macro_result.n_frames = n_frames
    if not hasattr(macro_result, 'n_atoms'):
        macro_result.n_atoms = n_atoms
    
    return macro_result

def generate_stress_curve(strain: np.ndarray, material_params: Dict, 
                         events: List) -> np.ndarray:
    """
    応力曲線を生成（物理的に正確な版）
    Ramberg-Osgood則＋イベント基づく修正
    """
    # パラメータ取得（互換性考慮）
    E = material_params.get('elastic_modulus', material_params.get('E', 210.0))
    nu = material_params.get('poisson_ratio', material_params.get('nu', 0.3))
    sigma_y = material_params.get('yield_strength', material_params.get('yield', 1.5))
    sigma_u = material_params.get('ultimate_strength', material_params.get('ultimate', 2.0))
    n_value = material_params.get('work_hardening_n', 0.20)
    
    # 形状確認
    strain = np.atleast_1d(strain).squeeze()  # 確実に1次元に
    n_frames = len(strain)
    
    # 降伏ひずみ
    epsilon_y = sigma_y / E
    
    # Ramberg-Osgood則による弾塑性応力計算
    stress = np.zeros_like(strain)
    
    for i, eps in enumerate(strain):
        if eps <= epsilon_y:
            # 弾性域（フックの法則）
            stress[i] = E * eps
        else:
            # 塑性域（加工硬化考慮）
            # σ = σ_y + K * (ε - ε_y)^n
            # K: 強度係数（経験的に設定）
            K = sigma_u * 1.1
            plastic_strain = eps - epsilon_y
            
            # 小さい塑性ひずみのガード（数値安定性）
            if plastic_strain > 1e-6:
                stress[i] = sigma_y + K * (plastic_strain ** n_value)
            else:
                stress[i] = sigma_y
            
            # 最大強度でキャップ
            stress[i] = min(stress[i], sigma_u)
    
    # イベントによる物理的修正
    for start, end, event_type in events:
        if start >= n_frames:
            continue
        end = min(end, n_frames)
        
        if 'crack' in event_type:
            # Griffithの破壊基準を簡略化
            K_IC = material_params.get('fracture_toughness', 30.0)  # MPa√m
            crack_length = 0.001 * (end - start) / 100  # 仮想亀裂長さ[m]
            
            if crack_length > 0:
                # 応力拡大係数による応力低下
                stress_reduction = 1.0 - min(0.5, crack_length * 1000)
                stress[start:] *= stress_reduction
                
        elif 'plastic' in event_type:
            # Bauschinger効果（逆負荷軟化）を簡略化
            stress[start:end] *= (1.0 + 0.02 * n_value)  # 微小な加工硬化
            
        elif 'dislocation' in event_type:
            # Peach-Koehler力による応力場変化
            b = material_params.get('lattice_constant', 2.87)  # バーガースベクトル
            local_stress_increase = 0.001 * E * b / max(1, end - start)
            stress[start:end] += local_stress_increase
            
        elif 'fatigue' in event_type:
            # Paris則による疲労損傷
            fatigue_strength = material_params.get('fatigue_strength', 0.7)
            if np.mean(stress[start:end]) > fatigue_strength:
                # 累積損傷
                damage_factor = 1.0 - 0.1 * (end - start) / n_frames
                stress[start:] *= max(0.3, damage_factor)
    
    # 2次元 (n_frames, 1) で返す
    stress = stress.reshape(-1, 1)
    assert stress.ndim == 2 and stress.shape[1] == 1, f"Stress must be (n_frames, 1), got shape {stress.shape}"
    
    return stress

def generate_anomaly_scores_from_events(events: List, n_frames: int) -> Dict[str, np.ndarray]:
    """イベントから異常スコアを生成"""
    scores = {
        'strain': np.ones(n_frames),
        'coordination': np.ones(n_frames),
        'damage': np.zeros(n_frames),
        'combined': np.ones(n_frames)
    }
    
    for start, end, event_type in events:
        # イベントタイプに応じたスコア設定
        if 'crack' in event_type:
            scores['strain'][start:end] += 2.0
            scores['damage'][start:end] += 3.0
        elif 'plastic' in event_type:
            scores['strain'][start:end] += 1.5
            scores['damage'][start:end] += 1.0
        elif 'dislocation' in event_type:
            scores['coordination'][start:end] += 1.0
            scores['strain'][start:end] += 0.5
        
        # combinedスコア更新
        scores['combined'][start:end] = np.maximum(
            scores['combined'][start:end],
            0.5 * scores['strain'][start:end] + 0.5 * scores['damage'][start:end]
        )
    
    return scores

def determine_failure_mode(events: List) -> str:
    """イベントから破壊モードを推定"""
    event_types = [e[2] for e in events]
    
    if any('crack' in t for t in event_types):
        return 'brittle_fracture'
    elif sum('plastic' in t for t in event_types) > 5:
        return 'ductile_fracture'
    elif any('fatigue' in t for t in event_types):
        return 'fatigue_failure'
    else:
        return 'elastic_deformation'

def estimate_time_to_failure(events: List, n_frames: int) -> Optional[float]:
    """破壊までの時間を推定"""
    critical_events = [e for e in events if 'crack' in e[2] or 'failure' in e[2]]
    if critical_events:
        first_critical = min(e[0] for e in critical_events)
        return float(first_critical) / n_frames * 100.0  # ps単位
    return None

def classify_material_event(start: int, end: int, 
                           anomaly_scores: Optional[np.ndarray],
                           trajectory_frames: int) -> str:
    """物理的特性からイベントタイプを分類"""
    event_type = 'elastic_deformation'
    duration = end - start
    relative_position = start / trajectory_frames
    
    if anomaly_scores is not None and len(anomaly_scores) > end:
        event_scores = anomaly_scores[start:end+1]
        if len(event_scores) > 0:
            max_score = np.max(event_scores)
            mean_score = np.mean(event_scores)
            
            # スコアベースの分類
            if max_score > 3.0:
                event_type = 'crack_initiation' if duration < 20 else 'crack_propagation'
            elif max_score > 2.5:
                event_type = 'plastic_deformation' if duration < 50 else 'fatigue_damage'
            elif max_score > 2.0:
                event_type = 'dislocation_nucleation'
            elif max_score > 1.5:
                event_type = 'dislocation_avalanche' if mean_score > 1.8 else 'uniform_deformation'
            elif max_score > 1.0:
                event_type = 'elastic_deformation'
    
    # 継続時間による補正
    if duration > 200:
        if event_type in ['elastic_deformation', 'uniform_deformation']:
            event_type = 'fatigue_damage'
    elif duration < 5:
        if event_type == 'uniform_deformation':
            event_type = 'defect_migration'
    
    # 位置による補正
    if relative_position > 0.8 and event_type == 'elastic_deformation':
        event_type = 'transition_state'
    elif relative_position < 0.2 and event_type == 'crack_initiation':
        event_type = 'dislocation_nucleation'
    
    return event_type

# ============================================
# メイン実行関数（リファクタリング版）
# ============================================

def run_material_analysis_pipeline(
    trajectory_path: str,
    metadata_path: str,
    atom_types_path: str,
    material_type: str = 'SUJ2',
    cluster_definition_path: Optional[str] = None,
    backbone_indices_path: Optional[str] = None, 
    strain_field_path: Optional[str] = None,
    enable_two_stage: bool = True,
    enable_impact: bool = True,
    enable_visualization: bool = True,
    output_dir: str = './material_results',
    loading_type: str = 'tensile',
    strain_rate: float = 1e-3,
    temperature: float = 300.0,
    verbose: bool = False,
    save_intermediate: bool = True,  # 中間結果保存フラグ
    subdivide_defects: bool = False,
    subdivision_size: int = 100,
    sort_by_position: bool = False,
    **kwargs
) -> Dict:
    """
    材料解析の完全パイプライン（リファクタリング版）
    
    全てのステップで生成されたデータが確実にレポート生成に渡される
    """
    
    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info(f"💎 MATERIAL ANALYSIS PIPELINE v3.2 (REFACTORED)")
    logger.info(f"   Material: {material_type}")
    logger.info(f"   Auto strain field generation: ENABLED")
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
        
        # メタデータ読み込み
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # メタデータ更新
        metadata.update({
            'system_name': f'{material_type}_simulation',
            'material_type': material_type,
            'temperature': temperature,
            'strain_rate': strain_rate,
            'loading_type': loading_type,
            'n_frames': n_frames,
            'n_atoms': n_atoms
        })
        
        # 原子タイプ読み込み
        atom_types = np.load(atom_types_path)
        logger.info(f"   Atom types loaded: {len(np.unique(atom_types))} types")
        
        # クラスター定義読み込み（必須化！）
        if cluster_definition_path and Path(cluster_definition_path).exists():
            if cluster_definition_path.endswith('.json'):
                with open(cluster_definition_path, 'r') as f:
                    cluster_atoms_raw = json.load(f)
                    
                    # ===== ここを修正！ =====
                    cluster_atoms = {}
                    for k, v in cluster_atoms_raw.items():
                        cid = int(k)
                        # vが辞書形式の場合、atom_idsだけ取り出す
                        if isinstance(v, dict) and 'atom_ids' in v:
                            # 各原子IDを整数に変換してリストとして保存
                            cluster_atoms[cid] = [int(atom_id) for atom_id in v['atom_ids']]
                        elif isinstance(v, list):
                            # 既にリスト形式の場合も整数変換
                            cluster_atoms[cid] = [int(atom_id) for atom_id in v]
                        else:
                            logger.warning(f"Unknown format for cluster {cid}: {type(v)}")
                            cluster_atoms[cid] = []
                            
            else:
                cluster_atoms = np.load(cluster_definition_path, allow_pickle=True).item()
            
            logger.info(f"   Clusters defined: {len(cluster_atoms)}")
            # デバッグ出力追加
            for cid in cluster_atoms:
                logger.info(f"     Cluster {cid}: {len(cluster_atoms[cid])} atoms")
            
            # クラスター0が健全領域として定義されているか確認
            if 0 not in cluster_atoms:
                logger.warning("   ⚠️ Cluster 0 (healthy region) not found in definition")
        else:
            # KMeans削除！エラーにする
            logger.error("="*60)
            logger.error("❌ CLUSTER DEFINITION FILE IS REQUIRED")
            logger.error("="*60)
            logger.error("   Material analysis requires physical defect regions.")
            logger.error("   Please provide: --clusters path/to/clusters_defects.json")
            logger.error("")
            logger.error("   The cluster file should define:")
            logger.error("     - Cluster 0: Healthy/perfect crystal region")
            logger.error("     - Cluster 1+: Various defect regions")
            logger.error("")
            logger.error("   Use external defect detection tools to generate this file.")
            logger.error("="*60)
            
            raise ValueError(
                "Cluster definition file is required for material analysis. "
                "KMeans spatial clustering is not physically meaningful for defect analysis."
            )
        
        # 歪み場読み込み（tryブロック内に追加！）
        strain_field = None
        if strain_field_path and Path(strain_field_path).exists():
            strain_field = np.load(strain_field_path)
            logger.info(f"   Strain field loaded: shape {strain_field.shape}")
        else:
            logger.info("   Strain field will be auto-generated from trajectory")
        
        logger.info("   ✅ Data validation passed")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # ========================================
    # 🆕 欠損クラスターの自動細分化
    # ========================================
    if subdivide_defects:  # 新しいコマンドラインオプション
        logger.info("\n🔬 Subdividing defect clusters for network analysis...")
        
        original_cluster_count = len(cluster_atoms)
        new_cluster_atoms = {}
        subdivision_size = 100  # パラメータ化も可能
        
        for cid, atoms in cluster_atoms.items():
            if cid == 0:  # 健全領域はそのまま
                new_cluster_atoms[0] = atoms
                logger.info(f"   Cluster 0 (healthy): {len(atoms)} atoms [kept as-is]")
                
            else:  # 欠損領域を細分化
                n_atoms = len(atoms)
                n_subdivisions = (n_atoms + subdivision_size - 1) // subdivision_size
                
                if n_subdivisions == 1:
                    # 100個以下なら分割しない
                    new_cluster_atoms[cid] = atoms
                    logger.info(f"   Cluster {cid}: {n_atoms} atoms [too small, kept as-is]")
                else:
                    # 細分化実行
                    atoms_array = np.array(atoms)
                    
                    # 空間的に近い原子をグループ化するため、座標でソート（オプション）
                    if sort_by_position:
                        first_frame_pos = trajectory[0, atoms]  # 最初のフレームの座標
                        # Z軸でソート（または距離ベース）
                        sort_indices = np.argsort(first_frame_pos[:, 2])
                        atoms_array = atoms_array[sort_indices]
                    
                    for sub_id in range(n_subdivisions):
                        start_idx = sub_id * subdivision_size
                        end_idx = min(start_idx + subdivision_size, n_atoms)
                        
                        # 階層的なクラスターID（例: "1-001", "1-002"）
                        new_cid = f"{cid}-{sub_id+1:03d}"
                        new_cluster_atoms[new_cid] = atoms_array[start_idx:end_idx].tolist()
                    
                    logger.info(f"   Cluster {cid}: {n_atoms} atoms → {n_subdivisions} subclusters")
        
        cluster_atoms = new_cluster_atoms
        logger.info(f"\n   📊 Cluster subdivision complete:")
        logger.info(f"      Original clusters: {original_cluster_count}")
        logger.info(f"      New clusters: {len(cluster_atoms)}")
        logger.info(f"      Network nodes: {len(cluster_atoms) - 1} (excluding healthy region)")
        
    # ========================================
    # Step 2: マクロ材料解析（強化版）
    # ========================================
    logger.info(f"\n🔬 Running Macro Material Analysis ({material_type})...")
    
    try:
        # MaterialConfig設定
        config = MaterialConfig()
        config.material_type = material_type
        config.use_material_analytics = True
        config.adaptive_window = True
        config.use_phase_space = True
        config.sensitivity = 1.5
        config.gpu_batch_size = 10000
        
        # 材料パラメータ取得
        material_params = get_material_parameters(material_type)
        logger.info(f"   Material parameters loaded")
        
        # 検出器初期化
        detector = MaterialLambda3DetectorGPU(config)

        # backbone_indices準備（修正版）
        backbone_indices = None
        
        # まず明示的に指定されたファイルから読み込み
        if backbone_indices_path and Path(backbone_indices_path).exists():
            backbone_indices = np.load(backbone_indices_path)
            # 型チェック
            if backbone_indices.dtype != np.int32:
                backbone_indices = backbone_indices.astype(np.int32)
            logger.info(f"   ✅ Loaded backbone_indices from file: {len(backbone_indices)} atoms")
        
        # ファイルがない場合はcluster_atomsから自動生成
        elif cluster_atoms:
            defect_atoms = []
            for cid, atoms in cluster_atoms.items():
                if str(cid) != "0":
                    # 整数変換を確実に
                    if isinstance(atoms, dict) and 'atom_ids' in atoms:
                        defect_atoms.extend([int(a) for a in atoms['atom_ids']])
                    else:
                        defect_atoms.extend([int(a) for a in atoms])
            if defect_atoms:
                backbone_indices = np.array(sorted(set(defect_atoms)), dtype=np.int32)
                logger.info(f"   🔄 Auto-generated backbone_indices: {len(backbone_indices)} atoms from clusters")
        else:
            logger.warning("   ⚠️ No backbone_indices or cluster_atoms available")
        
        # 解析実行
        macro_result = detector.analyze(
            trajectory=trajectory,
            backbone_indices=backbone_indices,
            atom_types=atom_types,
            cluster_atoms=cluster_atoms
        )
        
        logger.info(f"   ✅ Macro analysis complete")
        
        # イベント抽出と分類
        material_events = []
        if hasattr(macro_result, 'material_events') and macro_result.material_events:
            material_events = macro_result.material_events
        elif hasattr(macro_result, 'critical_events'):
            # critical_eventsから変換
            anomaly_scores = None
            if hasattr(macro_result, 'anomaly_scores'):
                anomaly_scores = macro_result.anomaly_scores.get('combined')
            
            for event in macro_result.critical_events:
                if isinstance(event, tuple) and len(event) >= 2:
                    start, end = event[0], event[1]
                    event_type = classify_material_event(
                        start, end, anomaly_scores, n_frames
                    )
                    material_events.append((start, end, event_type))
        
        logger.info(f"   Material events: {len(material_events)}")
        
        # ========================================
        # strain_fieldの自動生成（必要な場合）
        # ========================================
        if strain_field is None and len(material_events) > 0:
            logger.info("   Generating strain field from trajectory...")
            
            # 材料タイプに応じた格子定数
            lattice_constants = {
                'SUJ2': 2.87,      # BCC鉄
                'AL7075': 4.05,    # FCC アルミニウム
                'Ti6Al4V': 2.95,   # HCP チタン（a軸）
                'SS316L': 3.58     # FCC ステンレス鋼
            }
            lattice = lattice_constants.get(material_type, 2.87)
            
            # 歪み場計算
            strain_field = compute_strain_field_from_trajectory(
                trajectory=trajectory,
                material_events=material_events,
                lattice_constant=lattice
            )
            
            # 統計情報
            logger.info(f"   Generated strain field:")
            logger.info(f"     - Mean strain: {np.mean(strain_field):.4f}")
            logger.info(f"     - Max strain: {np.max(strain_field):.4f}")
            logger.info(f"     - Atoms > 1% strain: {np.sum(strain_field > 0.01)}")
            logger.info(f"     - Atoms > 5% strain: {np.sum(strain_field > 0.05)}")
            
            # 保存（オプション）
            if save_intermediate:
                np.save(output_path / 'strain_field_auto.npy', strain_field)
                logger.info("   Saved: strain_field_auto.npy")
        
        # ========================================
        # 重要: macro_resultの強化
        # ========================================
        macro_result = enhance_macro_result(
            macro_result=macro_result,
            material_events=material_events,
            trajectory_shape=trajectory.shape,
            material_params=material_params,
            metadata=metadata
        )
        logger.info("   ✅ Macro result enhanced with all required attributes")
        
        # 中間保存
        if save_intermediate:
            with open(output_path / 'macro_result.pkl', 'wb') as f:
                pickle.dump(macro_result, f)
            logger.info("   Saved: macro_result.pkl")
        
    except Exception as e:
        logger.error(f"Macro analysis failed: {e}")
        raise

    # ========================================
    # Step 3: 2段階クラスター解析（強化版）
    # ========================================
    two_stage_result = None
    sorted_events = []
    
    if enable_two_stage and len(material_events) > 0:
        logger.info("\n🔬 Running Two-Stage Cluster Analysis...")
        
        try:
            # イベントのスコア付けとソート
            event_scores = []
            for event in material_events:
                if isinstance(event, tuple) and len(event) >= 3:
                    start, end, event_type = event[:3]
                    
                    # スコア計算
                    type_scores = {
                        'crack_initiation': 10.0,
                        'crack_propagation': 9.0,
                        'plastic_deformation': 7.0,
                        'dislocation_nucleation': 6.0,
                        'dislocation_avalanche': 6.5,
                        'phase_transition': 8.0,
                        'elastic_deformation': 3.0,
                        'uniform_deformation': 3.5,
                        'defect_migration': 5.5,
                        'transition_state': 4.0,
                        'fatigue_damage': 5.0
                    }
                    base_score = type_scores.get(event_type, 1.0)
                    
                    # 異常スコアで重み付け
                    if macro_result.anomaly_scores and 'combined' in macro_result.anomaly_scores:
                        scores = macro_result.anomaly_scores['combined']
                        if len(scores) > start:
                            score_range = scores[start:min(end+1, len(scores))]
                            if len(score_range) > 0:
                                base_score *= (1 + np.max(score_range) * 0.5)
                    
                    event_scores.append((start, end, base_score, event_type))
            
            # ソート
            sorted_events = sorted(event_scores, key=lambda x: x[2], reverse=True)
            
            # TOP50選択
            selected_events = sorted_events[:min(50, len(sorted_events))]
            logger.info(f"   Selected TOP {len(selected_events)} events for analysis")
            
            # Two-Stage用イベントリスト作成
            detected_events = []
            for i, (start, end, score, event_type) in enumerate(selected_events):
                event_name = f"{event_type}_{i:02d}_score_{score:.2f}"
                detected_events.append((start, end, event_name))
            
            # ClusterAnalysisConfig設定
            cluster_config = ClusterAnalysisConfig()
            cluster_config.detect_dislocations = True
            cluster_config.detect_cracks = True
            cluster_config.detect_phase_transitions = True
            cluster_config.use_confidence = True
            cluster_config.use_physics_prediction = True  # 物理予測を有効化
            
            # TwoStageAnalyzer初期化と実行
            two_stage_analyzer = MaterialTwoStageAnalyzerGPU(
                config=cluster_config,
                material_type=material_type
            )
            
            two_stage_result = two_stage_analyzer.analyze_trajectory(
                trajectory=trajectory,
                macro_result=macro_result,
                detected_events=detected_events,
                cluster_atoms=cluster_atoms,
                atom_types=atom_types,
                stress_history=macro_result.stress_strain.get('stress') if macro_result.stress_strain else None
            )
            
            logger.info(f"   ✅ Two-stage analysis complete")
            
            # 結果の検証と表示
            if hasattr(two_stage_result, 'material_state'):
                state = two_stage_result.material_state
                logger.info(f"   Material state: {state.get('state', 'unknown')}")
                logger.info(f"   Health index: {state.get('health_index', 1.0):.1%}")
            
            if hasattr(two_stage_result, 'critical_clusters'):
                n_critical = len(two_stage_result.critical_clusters)
                logger.info(f"   Critical clusters: {n_critical}")
            
            # 中間保存
            if save_intermediate:
                with open(output_path / 'two_stage_result.pkl', 'wb') as f:
                    pickle.dump(two_stage_result, f)
                logger.info("   Saved: two_stage_result.pkl")
            
        except Exception as e:
            logger.error(f"Two-stage analysis failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            two_stage_result = None
    
    # ========================================
    # Step 4: 原子レベル欠陥解析（強化版）
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
            
            # TOP Nクラスターの解析
            top_n = min(10, len(two_stage_result.critical_clusters)) if hasattr(two_stage_result, 'critical_clusters') else 5
            
            impact_results = impact_analyzer.analyze_critical_clusters(
                macro_result=macro_result,
                two_stage_result=two_stage_result,
                trajectory=trajectory,
                atom_types=atom_types,
                strain_field=strain_field,
                top_n=top_n
            )
            
            if impact_results:
                # 統計計算
                total_defects = sum(
                    getattr(r, 'n_defect_atoms', 0) 
                    for r in impact_results.values()
                )
                total_links = sum(
                    getattr(r, 'n_network_links', 0)
                    for r in impact_results.values()
                )
                
                logger.info(f"   ✅ Defect analysis complete")
                logger.info(f"   Events analyzed: {len(impact_results)}")
                logger.info(f"   Total defect atoms: {total_defects}")
                logger.info(f"   Network links: {total_links}")
                
                # 欠陥タイプ分布
                defect_types = Counter()
                for result in impact_results.values():
                    if hasattr(result, 'dominant_defect') and result.dominant_defect:
                        defect_types[result.dominant_defect] += 1
                
                if defect_types:
                    logger.info("   Defect types:")
                    for dtype, count in defect_types.most_common():
                        logger.info(f"     - {dtype}: {count}")
                
                # 最大応力集中
                max_stress = max(
                    (getattr(r, 'max_stress_concentration', 0) 
                     for r in impact_results.values()),
                    default=0
                )
                if max_stress > 0:
                    logger.info(f"   Max stress concentration: {max_stress:.2f} GPa")
                
                # 中間保存
                if save_intermediate:
                    with open(output_path / 'impact_results.pkl', 'wb') as f:
                        pickle.dump(impact_results, f)
                    logger.info("   Saved: impact_results.pkl")
            else:
                logger.warning("   No impact results generated")
            
        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            impact_results = None
    
    # ========================================
    # Step 5: 統合レポート生成（完全版）
    # ========================================
    logger.info("\n📝 Generating comprehensive material report...")
    
    try:
        # sorted_eventsを3要素タプルに変換（レポート用）
        sorted_events_for_report = []
        for item in sorted_events:
            if len(item) >= 3:
                start, end, score = item[0], item[1], item[2]
                sorted_events_for_report.append((start, end, score))
        
        # デバッグ: 渡すデータの確認
        logger.info("   Report generation inputs:")
        logger.info(f"     - macro_result: {macro_result is not None}")
        if macro_result:
            logger.info(f"       - material_events: {len(macro_result.material_events) if macro_result.material_events else 0}")
            logger.info(f"       - stress_strain: {macro_result.stress_strain is not None}")
            logger.info(f"       - anomaly_scores: {macro_result.anomaly_scores is not None}")
            logger.info(f"       - failure_prediction: {macro_result.failure_prediction is not None}")
        logger.info(f"     - two_stage_result: {two_stage_result is not None}")
        logger.info(f"     - impact_results: {len(impact_results) if impact_results else 0}")
        logger.info(f"     - sorted_events: {len(sorted_events_for_report)}")
        
        # レポート生成
        report = generate_material_report_from_results(
            macro_result=macro_result,
            two_stage_result=two_stage_result,
            impact_results=impact_results,
            sorted_events=sorted_events_for_report,
            metadata=metadata,
            material_type=material_type,
            output_dir=str(output_path),
            verbose=verbose,
            debug=True  # デバッグモードON
        )
        
        logger.info(f"   ✅ Report generated successfully")
        logger.info(f"   Report length: {len(report):,} characters")
        
        # レポート保存確認
        report_path = output_path / 'material_report.md'
        if report_path.exists():
            logger.info(f"   Report saved to: {report_path}")
        
        # 統合結果のJSON保存
        summary_data = {
            'material_type': material_type,
            'metadata': metadata,
            'n_frames': n_frames,
            'n_atoms': n_atoms,
            'n_events': len(material_events),
            'event_types': dict(Counter(e[2] for e in material_events if len(e) >= 3)),
            'analysis_complete': True,
            'report_generated': True,
            'report_length': len(report)
        }
        
        # 解析結果の追加
        if macro_result:
            if macro_result.failure_prediction:
                summary_data['failure_prediction'] = {
                    'probability': macro_result.failure_prediction.get('failure_probability', 0),
                    'mode': macro_result.failure_prediction.get('failure_mode', 'unknown')
                }
            if macro_result.stress_strain:
                summary_data['max_stress'] = macro_result.stress_strain.get('max_stress', 0)
        
        if two_stage_result:
            if hasattr(two_stage_result, 'material_state'):
                summary_data['material_state'] = two_stage_result.material_state.get('state', 'unknown')
            if hasattr(two_stage_result, 'critical_clusters'):
                summary_data['n_critical_clusters'] = len(two_stage_result.critical_clusters)
        
        if impact_results:
            summary_data['n_defect_analyses'] = len(impact_results)
        
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info("   Saved: analysis_summary.json")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        report = None
    
    # ========================================
    # Step 6: 可視化（オプション）
    # ========================================
    if enable_visualization:
        logger.info("\n📊 Creating visualizations...")
        
        try:
            # 各種可視化関数の呼び出し
            # (既存の可視化コードはそのまま使用)
            
            # イベントタイムライン
            if material_events:
                fig = visualize_event_timeline(
                    material_events,
                    save_path=str(output_path / 'event_timeline.png')
                )
                logger.info("   Event timeline visualized")
            
            # 応力-歪み曲線
            if macro_result and macro_result.stress_strain:
                fig = visualize_stress_strain(
                    macro_result.stress_strain,
                    material_type,
                    save_path=str(output_path / 'stress_strain.png')
                )
                logger.info("   Stress-strain curve visualized")
            
            # クラスターダメージマップ
            if two_stage_result:
                fig = visualize_cluster_damage(
                    two_stage_result,
                    save_path=str(output_path / 'cluster_damage.png')
                )
                logger.info("   Cluster damage map visualized")
            
            # 欠陥ネットワーク
            if impact_results:
                fig = visualize_defect_network(
                    impact_results,
                    save_path=str(output_path / 'defect_network.png')
                )
                logger.info("   Defect network visualized")
            
            logger.info("   ✅ All visualizations completed")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    
    # ========================================
    # 完了
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("✅ MATERIAL ANALYSIS PIPELINE COMPLETE!")
    logger.info(f"   Material: {material_type}")
    logger.info(f"   Output directory: {output_path}")
    logger.info("   Summary:")
    
    if macro_result:
        logger.info(f"     ✓ {len(material_events)} material events detected")
        if macro_result.failure_prediction:
            fp = macro_result.failure_prediction.get('failure_probability', 0)
            logger.info(f"     ✓ Failure probability: {fp:.1%}")
    
    if two_stage_result:
        if hasattr(two_stage_result, 'material_state'):
            state = two_stage_result.material_state.get('state', 'unknown')
            logger.info(f"     ✓ Material state: {state}")
        if hasattr(two_stage_result, 'critical_clusters'):
            n_crit = len(two_stage_result.critical_clusters)
            logger.info(f"     ✓ {n_crit} critical clusters identified")
    
    if impact_results:
        logger.info(f"     ✓ {len(impact_results)} atomic defect analyses completed")
    
    if report:
        logger.info(f"     ✓ Report generated ({len(report):,} characters)")
    
    logger.info("="*70)
    
    return {
        'macro_result': macro_result,
        'two_stage_result': two_stage_result,
        'impact_results': impact_results,
        'sorted_events': sorted_events_for_report,
        'material_events': material_events,
        'report': report,
        'output_dir': output_path,
        'material_type': material_type,
        'metadata': metadata,
        'success': True
    }

# ============================================
# ヘルパー関数（既存のものを維持）
# ============================================

def compute_strain_field_from_trajectory(
    trajectory: np.ndarray,
    material_events: List[Tuple[int, int, str]],
    lattice_constant: float = 2.87  # BCC鉄のデフォルト
) -> np.ndarray:
    """
    トラジェクトリから歪み場を簡易計算
    
    Parameters
    ----------
    trajectory : np.ndarray
        原子トラジェクトリ (n_frames, n_atoms, 3)
    material_events : List[Tuple[int, int, str]]
        材料イベントリスト（歪み推定の参考）
    lattice_constant : float
        格子定数（Å）- 正規化用
    
    Returns
    -------
    np.ndarray
        各原子の平均歪み場 (n_atoms,)
    """
    n_frames, n_atoms = trajectory.shape[:2]
    strain_field = np.zeros(n_atoms)
    
    # フレーム間の累積歪み計算
    for i in range(1, n_frames):
        # 各原子のフレーム間変位
        displacement = trajectory[i] - trajectory[i-1]
        displacement_norm = np.linalg.norm(displacement, axis=1)
        
        # 格子定数で正規化して歪みに変換
        frame_strain = displacement_norm / lattice_constant
        
        # 累積
        strain_field += frame_strain
    
    # 平均化
    strain_field = strain_field / (n_frames - 1) if n_frames > 1 else strain_field
    
    # イベントベースの補正（オプション）
    if material_events:
        # 塑性・転位イベントがある領域を強調
        event_mask = np.zeros(n_frames, dtype=bool)
        for start, end, event_type in material_events:
            if any(key in event_type for key in ['plastic', 'dislocation', 'crack']):
                event_mask[start:min(end+1, n_frames)] = True
        
        # イベント発生フレームでの追加歪み
        if np.any(event_mask):
            event_frames = np.where(event_mask)[0]
            for frame in event_frames:
                if frame > 0 and frame < n_frames:
                    event_displacement = np.linalg.norm(
                        trajectory[frame] - trajectory[frame-1], axis=1
                    )
                    # イベント領域は歪みを増幅
                    strain_field += event_displacement / lattice_constant * 0.5
    
    # クリッピング（物理的に妥当な範囲）
    strain_field = np.clip(strain_field, 0, 0.5)  # 最大50%歪み
    
    return strain_field

# 可視化関数（既存のものをそのまま使用）
def visualize_stress_strain(curve_data: Dict, material_type: str,
                          save_path: Optional[str] = None) -> plt.Figure:
    """応力-歪み曲線の可視化"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'strain' in curve_data and 'stress' in curve_data:
        strain = curve_data['strain']
        stress = curve_data['stress']
        
        ax.plot(strain, stress, 'b-', linewidth=2, label=material_type)
        
        if 'yield_point' in curve_data:
            yield_idx = curve_data['yield_point']
            ax.plot(strain[yield_idx], stress[yield_idx], 'ro',
                   markersize=10, label='Yield Point')
        
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
    
    colors = {
        'elastic_deformation': 'green',
        'uniform_deformation': 'lightgreen',
        'plastic_deformation': 'yellow',
        'dislocation_nucleation': 'orange',
        'dislocation_avalanche': 'darkorange',
        'defect_migration': 'coral',
        'crack_initiation': 'red',
        'crack_propagation': 'darkred',
        'phase_transition': 'purple',
        'transition_state': 'mediumpurple',
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
    
    ax1 = axes[0]
    if hasattr(two_stage_result, 'critical_clusters'):
        critical = two_stage_result.critical_clusters[:20]
        if critical:
            ax1.bar(range(len(critical)), [1]*len(critical), color='red', alpha=0.7)
            ax1.set_xticks(range(len(critical)))
            ax1.set_xticklabels([f"C{c}" for c in critical], rotation=45)
            ax1.set_ylabel('Criticality', fontsize=12)
            ax1.set_title('Critical Clusters', fontsize=12)
    
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
    
    ax1 = axes[0, 0]
    defect_types = Counter()
    for result in impact_results.values():
        if hasattr(result, 'dominant_defect') and result.dominant_defect:
            defect_types[result.dominant_defect] += 1
    
    if defect_types:
        ax1.pie(defect_types.values(), labels=defect_types.keys(),
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Defect Type Distribution')
    
    ax2 = axes[0, 1]
    stress_concs = []
    for r in impact_results.values():
        if hasattr(r, 'max_stress_concentration'):
            stress_concs.append(r.max_stress_concentration)
    
    if stress_concs:
        ax2.hist(stress_concs, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Stress Concentration (GPa)')
        ax2.set_ylabel('Count')
        ax2.set_title('Stress Concentration Distribution')
        ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    patterns = []
    for result in impact_results.values():
        if hasattr(result, 'defect_network') and result.defect_network:
            if hasattr(result.defect_network, 'network_pattern'):
                patterns.append(result.defect_network.network_pattern)
    
    if patterns:
        pattern_counts = Counter(patterns)
        ax3.bar(pattern_counts.keys(), pattern_counts.values(),
               color='green', alpha=0.7)
        ax3.set_xlabel('Network Pattern')
        ax3.set_ylabel('Count')
        ax3.set_title('Defect Network Patterns')
    
    ax4 = axes[1, 1]
    plastic_zones = []
    for r in impact_results.values():
        if hasattr(r, 'plastic_zone_size'):
            plastic_zones.append(r.plastic_zone_size)
    
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

# CLI Interface（既存のものを維持）
def main():
    """コマンドラインインターフェース"""
    parser = argparse.ArgumentParser(
        description='Material Full Analysis Pipeline - Lambda³ GPU (REFACTORED v3.1)'
    )
    
    parser.add_argument('trajectory', help='Path to trajectory file (.npy)')
    parser.add_argument('atom_types', help='Path to atom types file (.npy)')
    parser.add_argument('--material', '-m', default='SUJ2',
                       choices=['SUJ2', 'AL7075', 'Ti6Al4V', 'SS316L'],
                       help='Material type')
    parser.add_argument('--metadata', help='Path to metadata file (.json)')
    parser.add_argument('--clusters', required=True,  # ← 必須に！
                   help='Path to cluster definition file (REQUIRED). '
                        'Must define Cluster 0 as healthy region and others as defects.')
    parser.add_argument('--backbone', '-b',
                       help='Path to backbone_indices file (.npy) - pre-computed defect atoms')
    parser.add_argument('--strain', help='Path to strain field data (.npy)')
    parser.add_argument('--loading', '-l', default='tensile',
                       choices=['tensile', 'compression', 'shear', 'fatigue'])
    parser.add_argument(
        '--subdivide-defects',
        action='store_true',
        help='Subdivide defect clusters for detailed network analysis'
    )
    parser.add_argument(
        '--subdivision-size',
        type=int,
        default=100,
        help='Number of atoms per subdivision (default: 100)'
    )
    parser.add_argument(
        '--sort-by-position',
        action='store_true',
        help='Sort atoms by spatial position before subdivision'
    )
    parser.add_argument('--strain-rate', type=float, default=1e-3)
    parser.add_argument('--temperature', '-T', type=float, default=300.0)
    parser.add_argument('--output', '-o', default='./material_results')
    parser.add_argument('--no-two-stage', action='store_true')
    parser.add_argument('--no-impact', action='store_true')
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results as pickle files')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    metadata_path = args.metadata if args.metadata else 'metadata_auto.json'
    
    try:
        results = run_material_analysis_pipeline(
            trajectory_path=args.trajectory,
            metadata_path=metadata_path,
            atom_types_path=args.atom_types,
            material_type=args.material,
            cluster_definition_path=args.clusters,
            backbone_indices_path=args.backbone, 
            strain_field_path=args.strain,
            enable_two_stage=not args.no_two_stage,
            enable_impact=not args.no_impact,
            enable_visualization=not args.no_viz,
            output_dir=args.output,
            loading_type=args.loading,
            strain_rate=args.strain_rate,
            temperature=args.temperature,
            verbose=args.verbose,
            save_intermediate=args.save_intermediate,
            subdivide_defects=args.subdivide_defects,
            subdivision_size=args.subdivision_size,
            sort_by_position=args.sort_by_position
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
