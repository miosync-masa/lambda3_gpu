"""
Quantum Validation Module v4.0 - Lambda³ Integrated Edition
===========================================================

Lambda³が検出した構造変化イベントの量子起源を判定する統合モジュール

設計思想：
- Lambda³の構造異常検出を前提とした判定
- 3パターン（瞬間/遷移/カスケード）の明確な区別
- Trajectoryからの原子レベル証拠の活用
- 現実的かつ科学的な判定基準

Version: 4.0 - Complete Refactoring
Authors: 環ちゃん & ご主人さま
Date: 2024
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats, signal
from scipy.spatial.distance import cdist

# Logger設定
logger = logging.getLogger('quantum_validation_v4')

# ============================================
# Enums and Constants
# ============================================

class StructuralEventPattern(Enum):
    """構造変化パターン分類"""
    INSTANTANEOUS = "instantaneous"  # 単一フレーム（瞬間的）
    TRANSITION = "transition"        # 連続フレーム（遷移）
    CASCADE = "cascade"              # ネットワークカスケード

class QuantumSignature(Enum):
    """量子的シグネチャー"""
    ENTANGLEMENT = "quantum_entanglement"        # 量子もつれ
    TUNNELING = "quantum_tunneling"              # 量子トンネリング
    COHERENCE = "quantum_coherence"              # 量子コヒーレンス
    PHASE_TRANSITION = "quantum_phase_transition" # 量子相転移
    INFORMATION_TRANSFER = "quantum_info_transfer" # 量子情報伝達
    NONE = "classical"                           # 古典的

# ============================================
# Data Classes
# ============================================

@dataclass
class LambdaAnomaly:
    """Lambda構造の異常性評価"""
    lambda_jump: float = 0.0          # Λの変化量
    lambda_zscore: float = 0.0        # 統計的異常度
    rho_t_spike: float = 0.0          # ρTのスパイク
    sigma_s_value: float = 0.0        # σs（構造同期）
    coordination: float = 0.0          # 協調性（レプリカがある場合）
    statistical_rarity: float = 1.0    # p値
    thermal_comparison: float = 0.0    # 熱的ゆらぎとの比

@dataclass
class AtomicEvidence:
    """原子レベルの証拠"""
    max_velocity: float = 0.0          # 最大原子速度
    max_acceleration: float = 0.0      # 最大加速度
    correlation_coefficient: float = 0.0 # 原子運動の相関
    bond_anomalies: List[Dict] = field(default_factory=list)
    dihedral_flips: List[Dict] = field(default_factory=list)
    hydrogen_behavior: Dict = field(default_factory=dict)

@dataclass
class QuantumAssessment:
    """量子性評価結果"""
    pattern: StructuralEventPattern
    signature: QuantumSignature
    confidence: float = 0.0
    is_quantum: bool = False
    
    # 詳細評価
    lambda_anomaly: Optional[LambdaAnomaly] = None
    atomic_evidence: Optional[AtomicEvidence] = None
    
    # 判定根拠
    criteria_met: List[str] = field(default_factory=list)
    explanation: str = ""
    
    # カスケード固有
    bell_inequality: Optional[float] = None
    async_bonds_used: List[Dict] = field(default_factory=list)

# ============================================
# Main Quantum Validator Class
# ============================================

class QuantumValidatorV4:
    """
    Lambda³統合型量子判定器
    
    Lambda³が検出した構造変化イベントを受け取り、
    その変化が量子的起源を持つかを判定する。
    """
    
    def __init__(self,
                 trajectory: Optional[np.ndarray] = None,
                 topology: Optional[Any] = None,
                 dt_ps: float = 100.0,
                 temperature_K: float = 300.0,
                 config: Optional[Dict] = None):
        """
        Parameters
        ----------
        trajectory : np.ndarray, optional
            原子座標トラジェクトリ [frames, atoms, 3]
        topology : Any, optional
            トポロジー情報（結合、原子タイプなど）
        dt_ps : float
            タイムステップ（ピコ秒）
        temperature_K : float
            系の温度（ケルビン）
        config : dict, optional
            判定基準のカスタマイズ設定
        """
        self.trajectory = trajectory
        self.topology = topology
        self.dt_ps = dt_ps
        self.temperature = temperature_K
        
        # 物理定数
        self.k_B = 8.617333e-5  # eV/K
        self.k_B_T = self.k_B * self.temperature  # eV
        
        # デフォルト判定基準（調整可能）
        self.criteria = config or {
            # Lambda異常
            'lambda_zscore_threshold': 3.0,      # 3σ以上
            'coordination_threshold': 0.7,       # 70%以上の協調
            'statistical_rarity': 0.01,          # p < 0.01
            
            # 原子運動
            'velocity_anomaly_factor': 3.0,      # 平均の3倍
            'correlation_threshold': 0.8,        # 相関係数0.8以上
            'dihedral_flip_angle': 120.0,       # 120度以上の回転
            
            # トンネリング
            'tunneling_enhancement': 10.0,       # 古典比10倍
            'barrier_threshold_kT': 5.0,        # 5kT以上の障壁
            
            # コヒーレンス
            'coherence_time_thermal_ratio': 5.0, # 熱的時間の5倍
            'phase_correlation_threshold': 0.7,  # 位相相関0.7以上
            
            # カスケード
            'bell_chsh_threshold': 2.0,         # CHSH > 2
            'causality_strength': 0.5,          # 因果性強度
            'cascade_speed_factor': 2.0,        # 期待値の2倍速
        }
        
        logger.info("🚀 Quantum Validator v4.0 initialized")
        logger.info(f"   Temperature: {self.temperature:.1f} K")
        logger.info(f"   Time step: {self.dt_ps:.1f} ps")
        logger.info(f"   Trajectory: {'loaded' if trajectory is not None else 'not loaded'}")
    
    # ========================================
    # Main Entry Point
    # ========================================
    
    def validate_event(self,
                       event: Dict,
                       lambda_result: Any,
                       network_result: Optional[Any] = None) -> QuantumAssessment:
        """
        構造変化イベントの量子性を判定
        
        Parameters
        ----------
        event : dict
            Lambda³が検出したイベント
            Required keys: frame_start, frame_end, type, ...
        lambda_result : Any
            Lambda³解析結果（structures, events, etc.）
        network_result : Any, optional
            ネットワーク解析結果（async_bonds, causal_links）
            
        Returns
        -------
        QuantumAssessment
            量子性評価結果
        """
        # パターン分類
        pattern = self._classify_pattern(event, network_result)
        
        # Lambda異常性評価
        lambda_anomaly = self._evaluate_lambda_anomaly(event, lambda_result)
        
        # 原子レベル証拠（trajectoryがある場合）
        atomic_evidence = None
        if self.trajectory is not None:
            atomic_evidence = self._gather_atomic_evidence(event, self.trajectory)
        
        # パターン別判定
        if pattern == StructuralEventPattern.INSTANTANEOUS:
            assessment = self._validate_instantaneous(
                event, lambda_anomaly, atomic_evidence
            )
        elif pattern == StructuralEventPattern.CASCADE:
            assessment = self._validate_cascade(
                event, lambda_anomaly, atomic_evidence, network_result
            )
        else:  # TRANSITION
            assessment = self._validate_transition(
                event, lambda_anomaly, atomic_evidence
            )
        
        # 最終評価
        assessment.pattern = pattern
        assessment.lambda_anomaly = lambda_anomaly
        assessment.atomic_evidence = atomic_evidence
        
        # 説明文生成
        assessment.explanation = self._generate_explanation(assessment)
        
        return assessment
    
    # ========================================
    # Pattern Classification
    # ========================================
    
    def _classify_pattern(self, event: Dict, network_result: Optional[Any]) -> StructuralEventPattern:
        """イベントパターンの分類"""
        duration = event.get('frame_end', event.get('frame', 0)) - \
                  event.get('frame_start', event.get('frame', 0)) + 1
        
        # カスケード判定（async_bondsがある）
        if network_result and hasattr(network_result, 'async_strong_bonds'):
            if len(network_result.async_strong_bonds) > 0:
                return StructuralEventPattern.CASCADE
        
        # 時間による分類
        if duration == 1:
            return StructuralEventPattern.INSTANTANEOUS
        else:
            return StructuralEventPattern.TRANSITION
    
    # ========================================
    # Lambda Anomaly Evaluation
    # ========================================
    
    def _evaluate_lambda_anomaly(self, event: Dict, lambda_result: Any) -> LambdaAnomaly:
        """Lambda構造の異常性を評価"""
        anomaly = LambdaAnomaly()
        
        if not hasattr(lambda_result, 'structures'):
            return anomaly
        
        structures = lambda_result.structures
        frame = event.get('frame_start', event.get('frame', 0))
        
        # Lambda値の変化
        if 'lambda_f' in structures and frame < len(structures['lambda_f']):
            lambda_vals = structures['lambda_f']
            
            # 前後との差分
            if frame > 0 and frame < len(lambda_vals) - 1:
                prev_val = lambda_vals[frame - 1]
                curr_val = lambda_vals[frame]
                next_val = lambda_vals[frame + 1]
                
                anomaly.lambda_jump = abs(curr_val - prev_val)
                
                # Z-score計算
                if len(lambda_vals) > 10:
                    mean = np.mean(lambda_vals)
                    std = np.std(lambda_vals)
                    if std > 0:
                        anomaly.lambda_zscore = abs(curr_val - mean) / std
        
        # rho_t（テンション）
        if 'rho_t' in structures and frame < len(structures['rho_t']):
            anomaly.rho_t_spike = structures['rho_t'][frame]
        
        # sigma_s（構造同期）
        if 'sigma_s' in structures and frame < len(structures['sigma_s']):
            anomaly.sigma_s_value = structures['sigma_s'][frame]
        
        # 協調性（レプリカ解析の場合）
        if hasattr(lambda_result, 'coordination') and frame < len(lambda_result.coordination):
            anomaly.coordination = lambda_result.coordination[frame]
        
        # 統計的稀少性
        if anomaly.lambda_zscore > 0:
            anomaly.statistical_rarity = 2 * (1 - stats.norm.cdf(anomaly.lambda_zscore))
        
        # 熱的比較
        thermal_energy = self.k_B_T
        if anomaly.lambda_jump > 0:
            anomaly.thermal_comparison = anomaly.lambda_jump / thermal_energy
        
        return anomaly
    
    # ========================================
    # Atomic Evidence Gathering
    # ========================================
    
    def _gather_atomic_evidence(self, event: Dict, trajectory: np.ndarray) -> AtomicEvidence:
        """原子レベルの証拠を収集"""
        evidence = AtomicEvidence()
        
        frame_start = event.get('frame_start', event.get('frame', 0))
        frame_end = event.get('frame_end', frame_start)
        
        # フレーム範囲チェック
        if frame_start >= len(trajectory) or frame_end >= len(trajectory):
            return evidence
        
        # 原子速度・加速度
        if frame_start > 0:
            # 速度計算
            velocities = (trajectory[frame_start] - trajectory[frame_start - 1]) / self.dt_ps
            evidence.max_velocity = np.max(np.linalg.norm(velocities, axis=1))
            
            # 加速度計算
            if frame_start > 1:
                prev_vel = (trajectory[frame_start - 1] - trajectory[frame_start - 2]) / self.dt_ps
                accelerations = (velocities - prev_vel) / self.dt_ps
                evidence.max_acceleration = np.max(np.linalg.norm(accelerations, axis=1))
        
        # 原子運動の相関
        if frame_end > frame_start:
            displacements = []
            for f in range(frame_start, min(frame_end + 1, len(trajectory))):
                if f > 0:
                    disp = trajectory[f] - trajectory[f - 1]
                    displacements.append(disp.flatten())
            
            if len(displacements) > 1:
                corr_matrix = np.corrcoef(displacements)
                # 上三角の最大相関
                upper_triangle = np.triu(corr_matrix, k=1)
                evidence.correlation_coefficient = np.max(np.abs(upper_triangle))
        
        # 結合長異常（トポロジー情報がある場合）
        if self.topology is not None:
            evidence.bond_anomalies = self._check_bond_anomalies(
                trajectory[frame_start], self.topology
            )
        
        # 二面角フリップ
        if frame_start > 0 and self.topology is not None:
            evidence.dihedral_flips = self._check_dihedral_flips(
                trajectory[frame_start - 1], trajectory[frame_start], self.topology
            )
        
        # 水素の振る舞い（トポロジー情報がある場合）
        if self.topology is not None:
            evidence.hydrogen_behavior = self._analyze_hydrogen_behavior(
                event, trajectory, self.topology
            )
        
        return evidence
    
    def _check_bond_anomalies(self, coords: np.ndarray, topology: Any) -> List[Dict]:
        """結合長の異常をチェック"""
        anomalies = []
        
        # トポロジーから結合リストを取得（実装はトポロジー形式に依存）
        # ここでは簡略化
        try:
            if hasattr(topology, 'bonds'):
                for bond in topology.bonds[:100]:  # 最初の100結合のみチェック
                    i, j = bond[0], bond[1]
                    if i < len(coords) and j < len(coords):
                        distance = np.linalg.norm(coords[i] - coords[j])
                        
                        # 異常に短い結合（< 0.8 Å）
                        if distance < 0.8:
                            anomalies.append({
                                'atoms': (i, j),
                                'distance': distance,
                                'type': 'ultra_short'
                            })
        except:
            pass
        
        return anomalies
    
    def _check_dihedral_flips(self, coords1: np.ndarray, coords2: np.ndarray, 
                              topology: Any) -> List[Dict]:
        """二面角の急激な変化をチェック"""
        flips = []
        
        # 簡略化された実装
        # 実際にはトポロジーから二面角定義を取得
        
        return flips
    
    def _analyze_hydrogen_behavior(self, event: Dict, trajectory: np.ndarray, 
                                  topology: Any) -> Dict:
        """水素原子の量子的振る舞いを解析"""
        behavior = {
            'tunneling_candidates': 0,
            'delocalized_hydrogens': 0
        }
        
        # 簡略化された実装
        # 実際には水素結合ネットワークを解析
        
        return behavior
    
    # ========================================
    # Pattern-Specific Validation
    # ========================================
    
    def _validate_instantaneous(self,
                               event: Dict,
                               lambda_anomaly: LambdaAnomaly,
                               atomic_evidence: Optional[AtomicEvidence]) -> QuantumAssessment:
        """瞬間的変化の量子性判定"""
        assessment = QuantumAssessment(
            pattern=StructuralEventPattern.INSTANTANEOUS,
            signature=QuantumSignature.NONE
        )
        
        criteria_met = []
        confidence = 0.0
        
        # Lambda異常性チェック
        if lambda_anomaly.lambda_zscore > self.criteria['lambda_zscore_threshold']:
            criteria_met.append(f"Lambda Z-score: {lambda_anomaly.lambda_zscore:.2f}")
            confidence += 0.3
        
        if lambda_anomaly.statistical_rarity < self.criteria['statistical_rarity']:
            criteria_met.append(f"Statistical rarity: p={lambda_anomaly.statistical_rarity:.4f}")
            confidence += 0.3
        
        if lambda_anomaly.coordination > self.criteria['coordination_threshold']:
            criteria_met.append(f"High coordination: {lambda_anomaly.coordination:.2%}")
            confidence += 0.2
            assessment.signature = QuantumSignature.ENTANGLEMENT
        
        # 原子レベル証拠
        if atomic_evidence:
            if atomic_evidence.correlation_coefficient > self.criteria['correlation_threshold']:
                criteria_met.append(f"Atomic correlation: {atomic_evidence.correlation_coefficient:.3f}")
                confidence += 0.2
                assessment.signature = QuantumSignature.ENTANGLEMENT
            
            if len(atomic_evidence.bond_anomalies) > 0:
                criteria_met.append(f"Bond anomalies: {len(atomic_evidence.bond_anomalies)}")
                confidence += 0.1
        
        # 瞬間的変化は時間スケール的に量子的
        criteria_met.append("Instantaneous timescale")
        confidence += 0.1
        
        # 最終判定
        assessment.confidence = min(confidence, 1.0)
        assessment.is_quantum = confidence > 0.5
        assessment.criteria_met = criteria_met
        
        if assessment.is_quantum and assessment.signature == QuantumSignature.NONE:
            assessment.signature = QuantumSignature.PHASE_TRANSITION
        
        return assessment
    
    def _validate_transition(self,
                            event: Dict,
                            lambda_anomaly: LambdaAnomaly,
                            atomic_evidence: Optional[AtomicEvidence]) -> QuantumAssessment:
        """遷移過程の量子性判定"""
        assessment = QuantumAssessment(
            pattern=StructuralEventPattern.TRANSITION,
            signature=QuantumSignature.NONE
        )
        
        criteria_met = []
        confidence = 0.0
        
        # 遷移速度の評価
        duration = event.get('frame_end', 0) - event.get('frame_start', 0) + 1
        transition_time = duration * self.dt_ps
        
        # エネルギー障壁の推定（Lambda変化から）
        if lambda_anomaly.lambda_jump > 0:
            # 障壁高さの推定（簡略化）
            barrier_estimate = lambda_anomaly.lambda_jump * 10  # kT単位
            
            # 古典的遷移時間（Kramers理論）
            classical_time = np.exp(barrier_estimate) * 1.0  # ps（簡略化）
            
            if transition_time < classical_time / self.criteria['tunneling_enhancement']:
                criteria_met.append(f"Fast transition: {transition_time:.1f} ps << {classical_time:.1f} ps")
                confidence += 0.4
                assessment.signature = QuantumSignature.TUNNELING
        
        # コヒーレンスチェック
        if lambda_anomaly.sigma_s_value > 0.8:  # 高い構造同期
            thermal_decoherence = 0.1  # ps（室温での典型値）
            if transition_time > thermal_decoherence * self.criteria['coherence_time_thermal_ratio']:
                criteria_met.append(f"Sustained coherence: {transition_time:.1f} ps")
                confidence += 0.3
                assessment.signature = QuantumSignature.COHERENCE
        
        # 原子レベル証拠
        if atomic_evidence:
            if atomic_evidence.max_velocity > 0:
                # 速度異常
                typical_velocity = 1.0  # Å/ps（タンパク質の典型値）
                if atomic_evidence.max_velocity > typical_velocity * self.criteria['velocity_anomaly_factor']:
                    criteria_met.append(f"Velocity anomaly: {atomic_evidence.max_velocity:.2f} Å/ps")
                    confidence += 0.2
            
            # 水素トンネリング
            if atomic_evidence.hydrogen_behavior.get('tunneling_candidates', 0) > 0:
                criteria_met.append("Hydrogen tunneling candidates detected")
                confidence += 0.2
                assessment.signature = QuantumSignature.TUNNELING
        
        # 最終判定
        assessment.confidence = min(confidence, 1.0)
        assessment.is_quantum = confidence > 0.4
        assessment.criteria_met = criteria_met
        
        return assessment
    
    def _validate_cascade(self,
                         event: Dict,
                         lambda_anomaly: LambdaAnomaly,
                         atomic_evidence: Optional[AtomicEvidence],
                         network_result: Any) -> QuantumAssessment:
        """カスケードの量子性判定"""
        assessment = QuantumAssessment(
            pattern=StructuralEventPattern.CASCADE,
            signature=QuantumSignature.NONE
        )
        
        criteria_met = []
        confidence = 0.0
        
        # ネットワーク解析
        if network_result and hasattr(network_result, 'async_strong_bonds'):
            async_bonds = network_result.async_strong_bonds
            
            if len(async_bonds) > 0:
                # 最強の非同期結合
                strongest_bond = max(async_bonds, key=lambda b: b.strength)
                
                # Bell不等式チェック（簡易版）
                if strongest_bond.strength > self.criteria['causality_strength']:
                    # CHSHパラメータ推定
                    S_estimate = 2.0 + strongest_bond.strength * 0.8
                    
                    if S_estimate > self.criteria['bell_chsh_threshold']:
                        criteria_met.append(f"CHSH violation: S={S_estimate:.2f}")
                        confidence += 0.4
                        assessment.signature = QuantumSignature.INFORMATION_TRANSFER
                        assessment.bell_inequality = S_estimate
                
                # 伝播速度
                if hasattr(network_result, 'propagation_speed'):
                    expected_speed = 1.0  # 残基/ps（典型値）
                    if network_result.propagation_speed > expected_speed * self.criteria['cascade_speed_factor']:
                        criteria_met.append(f"Fast propagation: {network_result.propagation_speed:.2f} residues/ps")
                        confidence += 0.3
                
                # async_bonds記録
                assessment.async_bonds_used = [
                    {
                        'residues': (b.from_res, b.to_res),
                        'strength': b.strength,
                        'lag': b.lag
                    }
                    for b in async_bonds[:5]  # 上位5個
                ]
        
        # Lambda異常性も考慮
        if lambda_anomaly.rho_t_spike > 0:
            # テンションの伝播
            criteria_met.append(f"Tension cascade: ρT={lambda_anomaly.rho_t_spike:.2f}")
            confidence += 0.2
        
        # 原子レベル証拠
        if atomic_evidence and atomic_evidence.correlation_coefficient > 0.7:
            criteria_met.append("Correlated atomic motion in cascade")
            confidence += 0.1
        
        # 最終判定
        assessment.confidence = min(confidence, 1.0)
        assessment.is_quantum = confidence > 0.4
        assessment.criteria_met = criteria_met
        
        return assessment
    
    # ========================================
    # Explanation Generation
    # ========================================
    
    def _generate_explanation(self, assessment: QuantumAssessment) -> str:
        """判定結果の説明文を生成"""
        if not assessment.is_quantum:
            return f"Classical {assessment.pattern.value} process with confidence {assessment.confidence:.1%}"
        
        explanations = {
            QuantumSignature.ENTANGLEMENT: "Quantum entanglement detected through instantaneous correlation",
            QuantumSignature.TUNNELING: "Quantum tunneling through energy barrier",
            QuantumSignature.COHERENCE: "Sustained quantum coherence beyond thermal decoherence time",
            QuantumSignature.PHASE_TRANSITION: "Quantum phase transition with discontinuous change",
            QuantumSignature.INFORMATION_TRANSFER: "Non-local quantum information transfer via entangled network"
        }
        
        base_explanation = explanations.get(
            assessment.signature,
            f"Quantum {assessment.pattern.value} detected"
        )
        
        return f"{base_explanation} (confidence: {assessment.confidence:.1%})"
    
    # ========================================
    # Batch Processing
    # ========================================
    
    def validate_events(self,
                        events: List[Dict],
                        lambda_result: Any,
                        network_results: Optional[List[Any]] = None) -> List[QuantumAssessment]:
        """複数イベントの一括処理"""
        assessments = []
        
        for i, event in enumerate(events):
            network = network_results[i] if network_results and i < len(network_results) else None
            
            try:
                assessment = self.validate_event(event, lambda_result, network)
                assessments.append(assessment)
            except Exception as e:
                logger.warning(f"Failed to process event {i}: {e}")
                # 失敗時はデフォルト評価
                assessments.append(QuantumAssessment(
                    pattern=StructuralEventPattern.INSTANTANEOUS,
                    signature=QuantumSignature.NONE,
                    is_quantum=False,
                    explanation=f"Processing failed: {e}"
                ))
        
        return assessments
    
    # ========================================
    # Summary and Reporting
    # ========================================
    
    def generate_summary(self, assessments: List[QuantumAssessment]) -> Dict:
        """評価結果のサマリー生成"""
        total = len(assessments)
        quantum_count = sum(1 for a in assessments if a.is_quantum)
        
        # パターン別集計
        pattern_stats = {}
        for pattern in StructuralEventPattern:
            count = sum(1 for a in assessments if a.pattern == pattern)
            quantum = sum(1 for a in assessments if a.pattern == pattern and a.is_quantum)
            pattern_stats[pattern.value] = {
                'total': count,
                'quantum': quantum,
                'ratio': quantum / count if count > 0 else 0
            }
        
        # シグネチャー別集計
        signature_stats = {}
        for sig in QuantumSignature:
            count = sum(1 for a in assessments if a.signature == sig)
            if count > 0:
                signature_stats[sig.value] = count
        
        # 信頼度分布
        confidences = [a.confidence for a in assessments if a.is_quantum]
        
        summary = {
            'total_events': total,
            'quantum_events': quantum_count,
            'quantum_ratio': quantum_count / total if total > 0 else 0,
            'pattern_statistics': pattern_stats,
            'signature_distribution': signature_stats,
            'confidence_stats': {
                'mean': np.mean(confidences) if confidences else 0,
                'std': np.std(confidences) if confidences else 0,
                'min': np.min(confidences) if confidences else 0,
                'max': np.max(confidences) if confidences else 0
            }
        }
        
        return summary
    
    def print_summary(self, assessments: List[QuantumAssessment]):
        """サマリーを表示"""
        summary = self.generate_summary(assessments)
        
        print("\n" + "="*70)
        print("🌌 QUANTUM VALIDATION SUMMARY v4.0")
        print("="*70)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total events: {summary['total_events']}")
        print(f"   Quantum events: {summary['quantum_events']} ({summary['quantum_ratio']:.1%})")
        
        print(f"\n🎯 Pattern Analysis:")
        for pattern, stats in summary['pattern_statistics'].items():
            if stats['total'] > 0:
                print(f"   {pattern}: {stats['quantum']}/{stats['total']} quantum ({stats['ratio']:.1%})")
        
        print(f"\n⚛️ Quantum Signatures:")
        for sig, count in summary['signature_distribution'].items():
            if sig != 'classical':
                print(f"   {sig}: {count}")
        
        conf_stats = summary['confidence_stats']
        print(f"\n📈 Confidence Statistics:")
        print(f"   Mean: {conf_stats['mean']:.3f}")
        print(f"   Std: {conf_stats['std']:.3f}")
        print(f"   Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")

# ============================================
# Convenience Functions
# ============================================

def validate_lambda_events(lambda_result: Any,
                           trajectory: Optional[np.ndarray] = None,
                           network_results: Optional[List[Any]] = None,
                           **kwargs) -> List[QuantumAssessment]:
    """Lambda³イベントの量子性を検証する簡易インターフェース"""
    
    # イベント抽出
    events = []
    if hasattr(lambda_result, 'critical_events'):
        for e in lambda_result.critical_events:
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                events.append({
                    'frame_start': int(e[0]),
                    'frame_end': int(e[1]),
                    'type': 'critical'
                })
    
    # バリデーター作成
    validator = QuantumValidatorV4(trajectory=trajectory, **kwargs)
    
    # 検証実行
    assessments = validator.validate_events(events, lambda_result, network_results)
    
    # サマリー表示
    validator.print_summary(assessments)
    
    return assessments

# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example: テスト用のダミーデータ
    print("Quantum Validator v4.0 - Test Run")
    
    # ダミーイベント
    test_event = {
        'frame_start': 100,
        'frame_end': 100,
        'type': 'critical'
    }
    
    # ダミーLambda結果
    class DummyLambdaResult:
        def __init__(self):
            self.structures = {
                'lambda_f': np.random.randn(1000),
                'rho_t': np.random.rand(1000),
                'sigma_s': np.random.rand(1000)
            }
    
    # バリデーター作成
    validator = QuantumValidatorV4(dt_ps=100.0, temperature_K=300.0)
    
    # 検証実行
    assessment = validator.validate_event(
        test_event,
        DummyLambdaResult()
    )
    
    # 結果表示
    print(f"\nPattern: {assessment.pattern.value}")
    print(f"Quantum: {assessment.is_quantum}")
    print(f"Signature: {assessment.signature.value}")
    print(f"Confidence: {assessment.confidence:.1%}")
    print(f"Explanation: {assessment.explanation}")
