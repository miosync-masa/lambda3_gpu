"""
Lambda³ GPU処理結果の可視化モジュール
GPU解析結果を効率的に可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from ..analysis.md_lambda3_detector_gpu import MDLambda3Result
from ..analysis.two_stage_analyzer_gpu import TwoStageLambda3Result, ResidueLevelAnalysis
from ..types import ArrayType, NDArray


class Lambda3VisualizerGPU:
    """GPU解析結果の可視化クラス"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Parameters
        ----------
        style : str
            Matplotlibスタイル
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        
        self.color_palette = sns.color_palette("husl", 8)
        self.figure_cache = {}
    
    def visualize_results(self, 
                         result: MDLambda3Result,
                         save_path: Optional[str] = None,
                         show_extended: bool = True) -> plt.Figure:
        """
        Lambda³解析結果の包括的可視化
        
        Parameters
        ----------
        result : MDLambda3Result
            解析結果
        save_path : str, optional
            保存パス
        show_extended : bool
            拡張検出結果を表示するか
            
        Returns
        -------
        plt.Figure
            生成した図
        """
        # レイアウト設定
        if show_extended and 'extended_combined' in result.anomaly_scores:
            fig = plt.figure(figsize=(24, 20))
            gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)
        else:
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # 1. マルチスケール異常スコア
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_multiscale_anomalies(ax1, result)
        
        # 2. Lambda構造の進化
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_lambda_evolution(ax2, result)
        
        # 3. テンション場と同期
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_tension_sync(ax3, result)
        
        # 4. トポロジカルチャージ
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_topological_charge(ax4, result)
        
        # 5. 構造境界
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_boundaries(ax5, result)
        
        # 6. MD特徴
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_md_features(ax6, result)
        
        # 7. 位相空間（3D）
        ax7 = fig.add_subplot(gs[2, 2], projection='3d')
        self._plot_phase_space_3d(ax7, result)
        
        # 8. 検出パターン
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_detected_patterns(ax8, result)
        
        # 9. GPU性能統計
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_gpu_performance(ax9, result)
        
        # 10. サマリー統計
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_summary_stats(ax10, result)
        
        # 拡張検出結果（オプション）
        if show_extended and 'extended_combined' in result.anomaly_scores:
            ax11 = fig.add_subplot(gs[4, :])
            self._plot_extended_anomalies(ax11, result)
        
        # タイトル
        plt.suptitle(f'Lambda³ MD Analysis Results - {result.n_frames} frames (GPU)', 
                    fontsize=16, y=0.995)
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_multiscale_anomalies(self, ax: plt.Axes, result: MDLambda3Result):
        """マルチスケール異常スコアのプロット"""
        frames = np.arange(len(result.anomaly_scores['global']))
        
        # メインスコア
        if 'final_combined' in result.anomaly_scores:
            ax.plot(frames, result.anomaly_scores['final_combined'], 
                   'r-', label='Final Combined', alpha=0.8, linewidth=2)
        
        ax.plot(frames, result.anomaly_scores['global'], 
               'g-', label='Global', alpha=0.6)
        ax.plot(frames, result.anomaly_scores['local'], 
               'b-', label='Local', alpha=0.6)
        
        # 閾値ライン
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='2σ threshold')
        ax.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)
        
        # 境界位置をマーク
        if 'boundary_locations' in result.structural_boundaries:
            for loc in result.structural_boundaries['boundary_locations']:
                ax.axvline(x=loc, color='orange', alpha=0.3, linestyle=':')
        
        ax.set_title('Multi-scale Anomaly Scores', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Score (MAD-normalized)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_lambda_evolution(self, ax: plt.Axes, result: MDLambda3Result):
        """Lambda構造の進化"""
        # ΛF magnitude
        lf_mag = result.lambda_structures['lambda_F_mag']
        frames = np.arange(len(lf_mag))
        
        ax.semilogy(frames, lf_mag, 'b-', alpha=0.7, label='|ΛF|')
        
        # スムージング版も表示
        window = 100
        if len(lf_mag) > window:
            smoothed = np.convolve(lf_mag, np.ones(window)/window, mode='same')
            ax.semilogy(frames, smoothed, 'r-', linewidth=2, label='Smoothed')
        
        ax.set_title('Structural Flow |ΛF|', fontsize=12)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Magnitude (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_tension_sync(self, ax: plt.Axes, result: MDLambda3Result):
        """テンション場と同期率"""
        rho_t = result.lambda_structures['rho_T']
        sigma_s = result.lambda_structures['sigma_s']
        frames = np.arange(len(rho_t))
        
        # 2軸プロット
        ax2 = ax.twinx()
        
        # テンション場
        line1 = ax.plot(frames, rho_t, 'r-', alpha=0.7, label='ρT (Tension)')
        ax.set_ylabel('Tension Field ρT', color='r')
        ax.tick_params(axis='y', labelcolor='r')
        
        # 同期率
        line2 = ax2.plot(frames, sigma_s, 'b-', alpha=0.7, label='σs (Sync)')
        ax2.set_ylabel('Sync Rate σs', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim([0, 1])
        
        ax.set_title('Tension Field & Synchronization', fontsize=12)
        ax.set_xlabel('Frame')
        
        # 統合凡例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_topological_charge(self, ax: plt.Axes, result: MDLambda3Result):
        """トポロジカルチャージ"""
        q_cum = result.lambda_structures['Q_cumulative']
        frames = np.arange(len(q_cum))
        
        # 累積値
        ax.plot(frames, q_cum, 'g-', linewidth=2, label='Cumulative Q_Λ')
        
        # トレンドライン
        if len(q_cum) > 100:
            z = np.polyfit(frames, q_cum, 1)
            p = np.poly1d(z)
            ax.plot(frames, p(frames), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
        
        ax.set_title('Cumulative Topological Charge', fontsize=12)
        ax.set_xlabel('Frame')
        ax.set_ylabel('∫Q_Λ')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_boundaries(self, ax: plt.Axes, result: MDLambda3Result):
        """構造境界の可視化"""
        boundary_score = result.structural_boundaries['boundary_score']
        frames = np.arange(len(boundary_score))
        
        # スコア
        ax.plot(frames, boundary_score, 'k-', alpha=0.7)
        ax.fill_between(frames, 0, boundary_score, alpha=0.3, color='gray')
        
        # 検出された境界
        if 'boundary_locations' in result.structural_boundaries:
            for i, loc in enumerate(result.structural_boundaries['boundary_locations']):
                if i < 10:  # 最初の10個のみ
                    ax.axvline(x=loc, color='red', alpha=0.7, linewidth=2)
                    if 'boundary_strengths' in result.structural_boundaries:
                        strength = result.structural_boundaries['boundary_strengths'][i]
                        ax.text(loc, ax.get_ylim()[1]*0.9, f'{strength:.1f}', 
                               rotation=90, va='top', fontsize=8)
        
        ax.set_title('Structural Boundaries (ΔΛC)', fontsize=12)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Boundary Score')
        ax.grid(True, alpha=0.3)
    
    def _plot_md_features(self, ax: plt.Axes, result: MDLambda3Result):
        """MD特徴の可視化"""
        # RMSDとRgの2軸プロット
        if 'rmsd' in result.md_features and 'radius_of_gyration' in result.md_features:
            frames = np.arange(len(result.md_features['rmsd']))
            
            ax2 = ax.twinx()
            
            # RMSD
            line1 = ax.plot(frames, result.md_features['rmsd'], 
                           'b-', alpha=0.7, label='RMSD')
            ax.set_ylabel('RMSD (Å)', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            # Rg
            line2 = ax2.plot(frames, result.md_features['radius_of_gyration'], 
                            'r-', alpha=0.7, label='Rg')
            ax2.set_ylabel('Radius of Gyration (Å)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title('MD Features', fontsize=12)
            ax.set_xlabel('Frame')
            
            # 統合凡例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        else:
            ax.text(0.5, 0.5, 'MD Features\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_space_3d(self, ax: Axes3D, result: MDLambda3Result):
        """3D位相空間"""
        if 'lambda_F' in result.lambda_structures:
            lf = result.lambda_structures['lambda_F']
            
            # 3成分がある場合
            if lf.shape[1] >= 3:
                # カラーマップ用の時間
                colors = np.arange(len(lf))
                
                # 3Dプロット
                scatter = ax.scatter(lf[:, 0], lf[:, 1], lf[:, 2],
                                   c=colors, cmap='viridis', 
                                   alpha=0.6, s=1)
                
                # カラーバー
                cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
                cbar.set_label('Frame', fontsize=10)
                
                ax.set_title('ΛF Phase Space', fontsize=12)
                ax.set_xlabel('ΛF_x')
                ax.set_ylabel('ΛF_y')
                ax.set_zlabel('ΛF_z')
                
                # 視点設定
                ax.view_init(elev=20, azim=45)
            else:
                ax.text(0.5, 0.5, 0.5, '3D Data\nNot Available',
                       ha='center', va='center')
    
    def _plot_detected_patterns(self, ax: plt.Axes, result: MDLambda3Result):
        """検出されたパターン"""
        if result.detected_structures:
            patterns = sorted(result.detected_structures,
                            key=lambda x: x.get('strength', 0), reverse=True)[:8]
            
            names = [p['name'] for p in patterns]
            periods = [p.get('period', 0) for p in patterns]
            strengths = [p.get('strength', 0) for p in patterns]
            
            # 棒グラフ
            y_pos = np.arange(len(names))
            bars = ax.barh(y_pos, periods, alpha=0.7)
            
            # 強度で色分け
            norm = plt.Normalize(vmin=0, vmax=1)
            for bar, strength in zip(bars, strengths):
                bar.set_color(plt.cm.viridis(norm(strength)))
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names)
            ax.set_xlabel('Period (frames)')
            ax.set_title('Detected Patterns', fontsize=12)
            
            # カラーバー
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                               pad=0.1, shrink=0.8)
            cbar.set_label('Strength', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Patterns\nDetected',
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_gpu_performance(self, ax: plt.Axes, result: MDLambda3Result):
        """GPU性能統計"""
        if result.gpu_info:
            # パフォーマンスメトリクス
            metrics = {
                'Frames/sec': result.n_frames / result.computation_time,
                'Memory (GB)': result.gpu_info.get('memory_used', 0),
                'Atoms/frame': result.n_atoms,
                'Window size': result.window_steps
            }
            
            # テキスト表示
            text = "GPU Performance\n" + "-"*20 + "\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    text += f"{key}: {value:.2f}\n"
                else:
                    text += f"{key}: {value}\n"
            
            text += f"\nDevice: {result.gpu_info.get('device_name', 'Unknown')}"
            text += f"\nMode: {result.gpu_info.get('computation_mode', 'Unknown')}"
            
            ax.text(0.1, 0.9, text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.axis('off')
        ax.set_title('GPU Performance', fontsize=12)
    
    def _plot_summary_stats(self, ax: plt.Axes, result: MDLambda3Result):
        """サマリー統計"""
        # 統計情報収集
        stats = []
        
        # 境界統計
        if 'boundary_locations' in result.structural_boundaries:
            n_boundaries = len(result.structural_boundaries['boundary_locations'])
            stats.append(('Boundaries detected', n_boundaries))
        
        # 異常統計
        if 'final_combined' in result.anomaly_scores:
            scores = result.anomaly_scores['final_combined']
            stats.extend([
                ('Frames > 2σ', np.sum(scores > 2.0)),
                ('Max anomaly', f"{np.max(scores):.2f}"),
                ('Mean anomaly', f"{np.mean(scores):.2f}")
            ])
        
        # パターン統計
        stats.append(('Patterns found', len(result.detected_structures)))
        
        # 計算統計
        stats.extend([
            ('Total time (s)', f"{result.computation_time:.2f}"),
            ('Speed (fps)', f"{result.n_frames/result.computation_time:.1f}")
        ])
        
        # テーブル形式で表示
        table_data = [[k, str(v)] for k, v in stats]
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # スタイリング
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')
                cell.set_edgecolor('white')
        
        ax.axis('off')
        ax.set_title('Summary Statistics', fontsize=12)
    
    def _plot_extended_anomalies(self, ax: plt.Axes, result: MDLambda3Result):
        """拡張異常検出結果"""
        frames = np.arange(result.n_frames)
        
        # 各種拡張スコア
        extended_types = ['periodic', 'gradual', 'drift', 'rg_based', 'phase_space']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for score_type, color in zip(extended_types, colors):
            if score_type in result.anomaly_scores:
                scores = result.anomaly_scores[score_type]
                ax.plot(frames[:len(scores)], scores, 
                       color=color, alpha=0.6, label=score_type.capitalize())
        
        ax.set_title('Extended Anomaly Detection', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Score')
        ax.legend(loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)
    
    def create_animation(self,
                        result: MDLambda3Result,
                        feature: str = 'anomaly',
                        interval: int = 50,
                        save_path: Optional[str] = None) -> FuncAnimation:
        """
        時系列アニメーション作成
        
        Parameters
        ----------
        result : MDLambda3Result
            解析結果
        feature : str
            表示する特徴 ('anomaly', 'lambda_f', 'tension')
        interval : int
            フレーム間隔（ms）
        save_path : str, optional
            保存パス (.mp4 or .gif)
            
        Returns
        -------
        FuncAnimation
            アニメーションオブジェクト
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # データ選択
        if feature == 'anomaly':
            data = result.anomaly_scores.get('final_combined', 
                                           result.anomaly_scores['combined'])
            title = 'Anomaly Score Evolution'
        elif feature == 'lambda_f':
            data = result.lambda_structures['lambda_F_mag']
            title = 'Structural Flow Evolution'
        elif feature == 'tension':
            data = result.lambda_structures['rho_T']
            title = 'Tension Field Evolution'
        else:
            raise ValueError(f"Unknown feature: {feature}")
        
        frames = np.arange(len(data))
        
        # 初期化
        line1, = ax1.plot([], [], 'b-', alpha=0.7)
        point1, = ax1.plot([], [], 'ro', markersize=8)
        
        ax1.set_xlim(0, len(data))
        ax1.set_ylim(np.min(data)*1.1, np.max(data)*1.1)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Value')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        # ヒストグラム
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution')
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            return line1, point1
        
        def animate(frame):
            # 現在までのデータ
            current_frames = frames[:frame+1]
            current_data = data[:frame+1]
            
            # ライン更新
            line1.set_data(current_frames, current_data)
            point1.set_data([frame], [data[frame]])
            
            # ヒストグラム更新
            ax2.clear()
            ax2.hist(current_data, bins=30, alpha=0.7, color='blue')
            ax2.axvline(data[frame], color='red', linestyle='--', 
                       label=f'Current: {data[frame]:.2f}')
            ax2.legend()
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Count')
            
            return line1, point1
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(data), interval=interval, 
                           blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000/interval)
            else:
                anim.save(save_path, writer='ffmpeg', fps=1000/interval)
        
        return anim


def visualize_residue_analysis(result: TwoStageLambda3Result,
                             event_name: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    残基レベル解析結果の可視化
    
    Parameters
    ----------
    result : TwoStageLambda3Result
        2段階解析結果
    event_name : str, optional
        特定のイベント名（Noneの場合は全体サマリー）
    save_path : str, optional
        保存パス
        
    Returns
    -------
    plt.Figure
        生成した図
    """
    if event_name and event_name in result.residue_analyses:
        # 特定イベントの詳細
        analysis = result.residue_analyses[event_name]
        fig = _visualize_single_event(analysis, event_name)
    else:
        # 全体サマリー
        fig = _visualize_residue_summary(result)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _visualize_single_event(analysis: ResidueLevelAnalysis, 
                          event_name: str) -> plt.Figure:
    """単一イベントの詳細可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 実装は causality_viz_gpu.py に委譲
    # ここではプレースホルダー
    fig.suptitle(f"{event_name} - Residue Level Analysis", fontsize=16)
    
    return fig


def _visualize_residue_summary(result: TwoStageLambda3Result) -> plt.Figure:
    """残基解析の全体サマリー"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 重要残基ランキング
    ax1 = axes[0, 0]
    top_residues = sorted(result.global_residue_importance.items(), 
                         key=lambda x: x[1], reverse=True)[:20]
    
    residue_ids = [f"R{r[0]+1}" for r in top_residues]
    scores = [r[1] for r in top_residues]
    
    ax1.barh(residue_ids, scores, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Top 20 Important Residues', fontsize=12)
    ax1.invert_yaxis()
    
    # 2. ネットワーク統計
    ax2 = axes[0, 1]
    stats = result.global_network_stats
    
    labels = ['Causal', 'Sync', 'Async']
    values = [stats['total_causal_links'], 
             stats['total_sync_links'],
             stats['total_async_bonds']]
    
    ax2.pie(values, labels=labels, autopct='%1.1f%%', 
           startangle=90, colors=['red', 'green', 'blue'])
    ax2.set_title('Network Link Distribution', fontsize=12)
    
    # 3. イベント別統計
    ax3 = axes[1, 0]
    event_names = list(result.residue_analyses.keys())
    n_residues = [len(a.residue_events) for a in result.residue_analyses.values()]
    
    ax3.bar(event_names, n_residues, alpha=0.7, color='orange')
    ax3.set_xlabel('Event')
    ax3.set_ylabel('Number of Residues Involved')
    ax3.set_title('Residues per Event', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. GPU性能
    ax4 = axes[1, 1]
    gpu_times = [a.gpu_time for a in result.residue_analyses.values()]
    
    ax4.plot(event_names, gpu_times, 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('Event')
    ax4.set_ylabel('GPU Time (seconds)')
    ax4.set_title('GPU Processing Time per Event', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Residue-Level Analysis Summary', fontsize=16)
    plt.tight_layout()
    
    return fig
