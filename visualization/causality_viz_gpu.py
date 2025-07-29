"""
Lambda³ GPU版因果ネットワーク可視化モジュール
残基間の因果関係とネットワーク構造の高度な可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..analysis.two_stage_analyzer_gpu import ResidueLevelAnalysis, TwoStageLambda3Result
from ..types import ArrayType, NDArray


class CausalityVisualizerGPU:
    """因果ネットワークの可視化クラス"""
    
    def __init__(self):
        self.color_scheme = {
            'initiator': '#FF6B6B',
            'propagator': '#4ECDC4',
            'responder': '#45B7D1',
            'causal_link': '#2C3E50',
            'sync_link': '#27AE60',
            'async_link': '#E74C3C'
        }
    
    def visualize_residue_causality(self,
                                  analysis: ResidueLevelAnalysis,
                                  save_path: Optional[str] = None,
                                  interactive: bool = False) -> Any:
        """
        残基因果ネットワークの包括的可視化
        
        Parameters
        ----------
        analysis : ResidueLevelAnalysis
            残基レベル解析結果
        save_path : str, optional
            保存パス
        interactive : bool
            インタラクティブ版を作成するか
            
        Returns
        -------
        fig
            matplotlib.Figure または plotly.Figure
        """
        if interactive:
            return self._create_interactive_visualization(analysis, save_path)
        else:
            return self._create_static_visualization(analysis, save_path)
    
    def _create_static_visualization(self,
                                   analysis: ResidueLevelAnalysis,
                                   save_path: Optional[str]) -> plt.Figure:
        """静的な可視化（matplotlib）"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        # 1. タイムライン（適応ウィンドウ付き）
        ax1 = axes[0, 0]
        self._plot_residue_timeline(ax1, analysis)
        
        # 2. 因果ネットワーク
        ax2 = axes[0, 1]
        self._plot_causality_network(ax2, analysis)
        
        # 3. 非同期強結合
        ax3 = axes[0, 2]
        self._plot_async_bonds(ax3, analysis)
        
        # 4. ネットワーク統計
        ax4 = axes[1, 0]
        self._plot_network_stats(ax4, analysis)
        
        # 5. 信頼区間
        ax5 = axes[1, 1]
        self._plot_confidence_intervals(ax5, analysis)
        
        # 6. 伝播経路
        ax6 = axes[1, 2]
        self._plot_propagation_paths(ax6, analysis)
        
        plt.suptitle(f"{analysis.event_name} - Residue Causality Analysis", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_residue_timeline(self, ax: plt.Axes, analysis: ResidueLevelAnalysis):
        """残基イベントのタイムライン"""
        ax.set_title(f"{analysis.event_name} - Residue Event Timeline", fontsize=14)
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Residue ID")
        
        # イベントをロール別に分類
        initiators = []
        propagators = []
        responders = []
        
        for event in analysis.residue_events:
            if event.role == 'initiator':
                initiators.append(event)
            elif event.role == 'propagator':
                propagators.append(event)
            else:
                responders.append(event)
        
        # プロット
        for events, color, label in [
            (initiators, self.color_scheme['initiator'], 'Initiator'),
            (propagators, self.color_scheme['propagator'], 'Propagator'),
            (responders, self.color_scheme['responder'], 'Responder')
        ]:
            for event in events:
                width = event.end_frame - event.start_frame
                height = 0.4 + 0.4 * (100 / event.adaptive_window)
                
                rect = FancyBboxPatch(
                    (event.start_frame, event.residue_id - height/2),
                    width, height,
                    boxstyle="round,pad=0.1",
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7,
                    linewidth=1
                )
                ax.add_patch(rect)
                
                # ピーク強度をテキストで表示
                ax.text(event.start_frame + width/2, event.residue_id,
                       f"{event.peak_lambda_f:.1f}",
                       ha='center', va='center', fontsize=8)
        
        # 凡例用のダミーパッチ
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.color_scheme['initiator'], label='Initiator'),
            Patch(facecolor=self.color_scheme['propagator'], label='Propagator'),
            Patch(facecolor=self.color_scheme['responder'], label='Responder')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 軸設定
        all_residues = [e.residue_id for e in analysis.residue_events]
        if all_residues:
            ax.set_ylim(min(all_residues) - 1, max(all_residues) + 1)
            ax.set_xlim(analysis.macro_start, analysis.macro_end)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_causality_network(self, ax: plt.Axes, analysis: ResidueLevelAnalysis):
        """因果ネットワークグラフ"""
        ax.set_title("Causality Network", fontsize=14)
        
        # NetworkXグラフ構築
        G = nx.DiGraph()
        
        # ノード追加
        for event in analysis.residue_events:
            G.add_node(event.residue_id,
                      role=event.role,
                      peak=event.peak_lambda_f,
                      window=event.adaptive_window)
        
        # エッジ追加（因果リンク）
        for from_res, to_res, strength in analysis.causality_chain[:30]:  # 上位30
            G.add_edge(from_res, to_res, weight=strength)
        
        if len(G.nodes()) == 0:
            ax.text(0.5, 0.5, 'No causality network detected',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # レイアウト計算
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # ノードの描画
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            data = G.nodes[node]
            
            # 色をロールで決定
            if data.get('role') == 'initiator':
                node_colors.append(self.color_scheme['initiator'])
            elif data.get('role') == 'propagator':
                node_colors.append(self.color_scheme['propagator'])
            else:
                node_colors.append(self.color_scheme['responder'])
            
            # サイズをピーク強度で決定
            node_sizes.append(300 + data.get('peak', 1) * 100)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8, ax=ax)
        
        # エッジの描画（重みで太さを変える）
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        if weights:
            max_weight = max(weights)
            edge_widths = [w / max_weight * 3 for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths,
                                  alpha=0.6, edge_color='gray',
                                  arrows=True, arrowsize=15,
                                  connectionstyle="arc3,rad=0.1", ax=ax)
        
        # ラベル
        labels = {node: f"R{node+1}" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
        
        ax.axis('off')
    
    def _plot_async_bonds(self, ax: plt.Axes, analysis: ResidueLevelAnalysis):
        """非同期強結合の可視化"""
        ax.set_title("Async Strong Bonds (同期なき強い結びつき)", fontsize=14)
        
        if not analysis.async_strong_bonds:
            ax.text(0.5, 0.5, 'No async bonds detected',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # データ準備
        bond_data = []
        for bond in analysis.async_strong_bonds[:15]:  # 上位15
            res1, res2 = bond['residue_pair']
            bond_data.append({
                'pair': f"R{res1+1}-R{res2+1}",
                'causality': bond['causality'],
                'sync': abs(bond['sync_rate']),
                'lag': bond['optimal_lag']
            })
        
        # 散布図
        x = [b['sync'] for b in bond_data]
        y = [b['causality'] for b in bond_data]
        colors = [b['lag'] for b in bond_data]
        
        scatter = ax.scatter(x, y, c=colors, cmap='plasma',
                           s=200, alpha=0.7, edgecolors='black')
        
        # ラベル
        for i, b in enumerate(bond_data[:8]):  # 上位8個にラベル
            ax.annotate(b['pair'], (x[i], y[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        # 閾値ライン
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5,
                  label='Sync threshold')
        ax.axhline(y=0.15, color='blue', linestyle='--', alpha=0.5,
                  label='Causality threshold')
        
        # 非同期領域を強調
        ax.axvspan(0, 0.2, alpha=0.1, color='green', label='Async region')
        
        ax.set_xlabel("Synchronization Rate")
        ax.set_ylabel("Causality Strength")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Optimal Lag (frames)', fontsize=10)
    
    def _plot_network_stats(self, ax: plt.Axes, analysis: ResidueLevelAnalysis):
        """ネットワーク統計"""
        ax.set_title("Network Statistics", fontsize=14)
        ax.axis('off')
        
        # 統計テキスト
        stats_text = f"""
Network Type Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━
• Causal Links: {analysis.network_stats['n_causal']}
• Synchronous Links: {analysis.network_stats['n_sync']}
• Async Strong Bonds: {analysis.network_stats['n_async']}
• Async/Causal Ratio: {analysis.network_stats['n_async']/(analysis.network_stats['n_causal']+1e-10):.2%}

Adaptive Window Stats:
━━━━━━━━━━━━━━━━━━━━━━━━
• Mean Window: {analysis.network_stats['mean_adaptive_window']:.1f} frames
• GPU Processing: {analysis.gpu_time:.2f} seconds

Top Initiator Residues:
━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        for i, res_id in enumerate(analysis.initiator_residues[:5]):
            stats_text += f"\n  {i+1}. Residue {res_id+1}"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    
    def _plot_confidence_intervals(self, ax: plt.Axes, analysis: ResidueLevelAnalysis):
        """Bootstrap信頼区間"""
        ax.set_title("Bootstrap Confidence Intervals", fontsize=14)
        
        if not analysis.confidence_results:
            ax.text(0.5, 0.5, 'Confidence analysis not performed',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # 有意なペアのみ
        significant_results = [c for c in analysis.confidence_results if c.significant][:10]
        
        if not significant_results:
            ax.text(0.5, 0.5, 'No significant pairs found',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # プロット
        y_pos = np.arange(len(significant_results))
        
        for i, conf in enumerate(significant_results):
            res_i, res_j = conf.pair
            
            # エラーバー
            ax.plot([conf.ci_lower, conf.ci_upper], [i, i], 
                   'b-', linewidth=3, alpha=0.6)
            
            # 平均値
            ax.plot(conf.mean, i, 'ro', markersize=10)
            
            # ラベル
            ax.text(-0.15, i, f"R{res_i+1}-R{res_j+1}",
                   ha='right', va='center', fontsize=10)
            
            # 信頼スコア
            ax.text(conf.ci_upper + 0.05, i, f"{conf.confidence_score:.2f}",
                   ha='left', va='center', fontsize=9, color='green')
        
        # ゼロライン
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Correlation Coefficient")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_ylim(-0.5, len(significant_results)-0.5)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_title(f"95% Confidence Intervals (n={len(significant_results)} significant)")
    
    def _plot_propagation_paths(self, ax: plt.Axes, analysis: ResidueLevelAnalysis):
        """伝播経路の可視化"""
        ax.set_title("Key Propagation Paths", fontsize=14)
        
        if not analysis.key_propagation_paths:
            ax.text(0.5, 0.5, 'No propagation paths detected',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return
        
        # 上位5経路
        paths = analysis.key_propagation_paths[:5]
        
        # サンキーダイアグラムスタイル
        y_positions = np.linspace(0.8, 0.2, len(paths))
        
        for i, path in enumerate(paths):
            y = y_positions[i]
            
            # パスの長さに応じてx座標を配置
            x_positions = np.linspace(0.1, 0.9, len(path))
            
            # ノード描画
            for j, res_id in enumerate(path):
                # ノードの色（最初は赤、最後は青、中間は緑）
                if j == 0:
                    color = self.color_scheme['initiator']
                elif j == len(path) - 1:
                    color = self.color_scheme['responder']
                else:
                    color = self.color_scheme['propagator']
                
                circle = Circle((x_positions[j], y), 0.03,
                              facecolor=color, edgecolor='black',
                              transform=ax.transAxes)
                ax.add_patch(circle)
                
                # ラベル
                ax.text(x_positions[j], y, f"R{res_id+1}",
                       ha='center', va='center', fontsize=10,
                       transform=ax.transAxes)
                
                # 矢印
                if j < len(path) - 1:
                    ax.annotate('', xy=(x_positions[j+1]-0.03, y),
                               xytext=(x_positions[j]+0.03, y),
                               xycoords='axes fraction',
                               arrowprops=dict(arrowstyle='->',
                                             lw=2, alpha=0.6,
                                             color='gray'))
            
            # パスラベル
            ax.text(0.02, y, f"Path {i+1}:",
                   ha='right', va='center', fontsize=10,
                   transform=ax.transAxes, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _create_interactive_visualization(self,
                                        analysis: ResidueLevelAnalysis,
                                        save_path: Optional[str]) -> go.Figure:
        """インタラクティブな可視化（Plotly）"""
        # サブプロット作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residue Event Timeline', 'Causality Network 3D',
                          'Async Bonds Heatmap', 'Propagation Sankey'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}],
                  [{'type': 'heatmap'}, {'type': 'sankey'}]],
            row_heights=[0.5, 0.5]
        )
        
        # 1. タイムライン（インタラクティブ）
        self._add_interactive_timeline(fig, analysis, row=1, col=1)
        
        # 2. 3D因果ネットワーク
        self._add_3d_network(fig, analysis, row=1, col=2)
        
        # 3. 非同期結合ヒートマップ
        self._add_async_heatmap(fig, analysis, row=2, col=1)
        
        # 4. 伝播サンキーダイアグラム
        self._add_propagation_sankey(fig, analysis, row=2, col=2)
        
        # レイアウト更新
        fig.update_layout(
            title=f"{analysis.event_name} - Interactive Residue Analysis",
            height=1000,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def _add_interactive_timeline(self, fig: go.Figure, 
                                analysis: ResidueLevelAnalysis,
                                row: int, col: int):
        """インタラクティブタイムライン追加"""
        for event in analysis.residue_events:
            color = self.color_scheme[event.role]
            
            # Ganttチャートスタイル
            fig.add_trace(
                go.Scatter(
                    x=[event.start_frame, event.end_frame],
                    y=[event.residue_id, event.residue_id],
                    mode='lines+markers',
                    line=dict(color=color, width=10),
                    marker=dict(size=8),
                    name=event.role,
                    text=f"R{event.residue_id+1}: peak={event.peak_lambda_f:.2f}",
                    hoverinfo='text',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Frame", row=row, col=col)
        fig.update_yaxes(title_text="Residue ID", row=row, col=col)
    
    def _add_3d_network(self, fig: go.Figure,
                       analysis: ResidueLevelAnalysis,
                       row: int, col: int):
        """3D因果ネットワーク追加"""
        # NetworkXグラフ構築
        G = nx.DiGraph()
        
        for event in analysis.residue_events:
            G.add_node(event.residue_id, **vars(event))
        
        for from_res, to_res, strength in analysis.causality_chain[:50]:
            G.add_edge(from_res, to_res, weight=strength)
        
        if len(G.nodes()) == 0:
            return
        
        # 3Dレイアウト
        pos = nx.spring_layout(G, dim=3, k=3, iterations=50)
        
        # エッジトレース
        edge_trace = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            edge_trace.append(
                go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None],
                    mode='lines',
                    line=dict(width=G[edge[0]][edge[1]]['weight']*5,
                            color='gray'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # ノードトレース
        node_x = []
        node_y = []
        node_z = []
        node_color = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            # 色とサイズ
            data = G.nodes[node]
            if hasattr(data, 'role'):
                color = {'initiator': 0, 'propagator': 1, 'responder': 2}[data.role]
            else:
                color = 1
            node_color.append(color)
            
            size = 10 + getattr(data, 'peak_lambda_f', 1) * 5
            node_size.append(size)
            
            text = f"R{node+1}"
            if hasattr(data, 'peak_lambda_f'):
                text += f"<br>Peak: {data.peak_lambda_f:.2f}"
            node_text.append(text)
        
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Role",
                    ticktext=['Initiator', 'Propagator', 'Responder'],
                    tickvals=[0, 1, 2]
                )
            ),
            text=node_text,
            textposition='top center',
            hoverinfo='text',
            showlegend=False
        )
        
        # 追加
        for trace in edge_trace:
            fig.add_trace(trace, row=row, col=col)
        fig.add_trace(node_trace, row=row, col=col)
        
        fig.update_scenes(
            xaxis_title='', yaxis_title='', zaxis_title='',
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            zaxis_showticklabels=False
        )
    
    def _add_async_heatmap(self, fig: go.Figure,
                         analysis: ResidueLevelAnalysis,
                         row: int, col: int):
        """非同期結合ヒートマップ追加"""
        if not analysis.async_strong_bonds:
            return
        
        # 行列形式に変換
        residues = sorted(set(
            [b['residue_pair'][0] for b in analysis.async_strong_bonds] +
            [b['residue_pair'][1] for b in analysis.async_strong_bonds]
        ))
        
        n = len(residues)
        matrix = np.zeros((n, n))
        
        res_to_idx = {res: i for i, res in enumerate(residues)}
        
        for bond in analysis.async_strong_bonds:
            i = res_to_idx[bond['residue_pair'][0]]
            j = res_to_idx[bond['residue_pair'][1]]
            matrix[i, j] = bond['causality']
            matrix[j, i] = bond['causality']
        
        # ヒートマップ
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[f"R{r+1}" for r in residues],
                y=[f"R{r+1}" for r in residues],
                colorscale='Reds',
                showscale=True,
                hovertemplate='%{x} - %{y}<br>Causality: %{z:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Residue", row=row, col=col)
        fig.update_yaxes(title_text="Residue", row=row, col=col)
    
    def _add_propagation_sankey(self, fig: go.Figure,
                              analysis: ResidueLevelAnalysis,
                              row: int, col: int):
        """伝播サンキーダイアグラム追加"""
        if not analysis.key_propagation_paths:
            return
        
        # サンキー用データ準備
        source = []
        target = []
        value = []
        labels = []
        label_map = {}
        
        # 全ての残基をラベルに追加
        all_residues = set()
        for path in analysis.key_propagation_paths[:5]:
            all_residues.update(path)
        
        for i, res in enumerate(sorted(all_residues)):
            labels.append(f"R{res+1}")
            label_map[res] = i
        
        # パスをリンクに変換
        for path in analysis.key_propagation_paths[:5]:
            for i in range(len(path) - 1):
                source.append(label_map[path[i]])
                target.append(label_map[path[i+1]])
                value.append(1)
        
        # サンキーダイアグラム
        fig.add_trace(
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color="blue"
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(0,0,200,0.3)"
                )
            ),
            row=row, col=col
        )


def create_network_comparison(results: List[TwoStageLambda3Result],
                            event_names: List[str],
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    複数イベントのネットワーク比較
    
    Parameters
    ----------
    results : List[TwoStageLambda3Result]
        複数の解析結果
    event_names : List[str]
        比較するイベント名
    save_path : str, optional
        保存パス
        
    Returns
    -------
    plt.Figure
        比較図
    """
    n_events = len(event_names)
    fig, axes = plt.subplots(2, n_events, figsize=(6*n_events, 12))
    
    if n_events == 1:
        axes = axes.reshape(2, 1)
    
    visualizer = CausalityVisualizerGPU()
    
    for i, (result, event_name) in enumerate(zip(results, event_names)):
        if event_name not in result.residue_analyses:
            continue
        
        analysis = result.residue_analyses[event_name]
        
        # 上段：ネットワーク
        ax1 = axes[0, i]
        visualizer._plot_causality_network(ax1, analysis)
        ax1.set_title(f"{event_name}\nCausality Network")
        
        # 下段：統計比較
        ax2 = axes[1, i]
        stats = analysis.network_stats
        
        categories = ['Causal', 'Sync', 'Async']
        values = [stats['n_causal'], stats['n_sync'], stats['n_async']]
        
        bars = ax2.bar(categories, values, alpha=0.7,
                       color=['red', 'green', 'blue'])
        
        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom')
        
        ax2.set_ylabel('Number of Links')
        ax2.set_title(f"Network Statistics")
    
    plt.suptitle('Event Network Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
