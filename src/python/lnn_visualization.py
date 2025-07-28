"""
Advanced visualization and analysis tools for Liquid Neural Networks
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional
import jax.numpy as jnp
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LNNVisualizer:
    """Advanced visualization tools for LNN analysis"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_training_curves(self, loss_history: List[float], metrics: Dict = None):
        """Plot training loss and metrics curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curve
        axes[0].plot(loss_history, linewidth=2, color='blue')
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Metrics
        if metrics:
            for i, (name, values) in enumerate(metrics.items()):
                axes[1].plot(values, label=name, linewidth=2)
            axes[1].set_title('Training Metrics', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_phase_portraits(self, trajectories: List[Dict], title: str = "Phase Portraits"):
        """Plot phase portraits of network dynamics"""
        n_layers = len(trajectories)
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, traj in enumerate(trajectories):
            ax = axes[i]
            states = traj['states']
            
            if len(states) > 1:
                # 2D projection of state
                if states.ndim > 1:
                    x = states[:, 0] if states.shape[1] > 0 else states
                    y = states[:, 1] if states.shape[1] > 1 else np.zeros_like(x)
                else:
                    x = states
                    y = np.zeros_like(x)
                
                ax.plot(x, y, 'b-', alpha=0.7, linewidth=2)
                ax.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start')
                ax.scatter(x[-1], y[-1], c='red', s=100, marker='x', label='End')
                
                # Add arrows to show direction
                for j in range(0, len(x)-1, max(1, len(x)//10)):
                    ax.annotate('', xy=(x[j+1], y[j+1]), xytext=(x[j], y[j]),
                               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
            
            ax.set_title(f'Layer {i+1}', fontweight='bold')
            ax.set_xlabel('State Dimension 1')
            ax.set_ylabel('State Dimension 2')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase_portraits.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_stability_analysis(self, stability_metrics: List[Dict]):
        """Plot stability analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        stability_scores = [m['stability_score'] for m in stability_metrics]
        perturbation_effects = [m['mean_perturbation'] for m in stability_metrics]
        
        # Stability scores histogram
        axes[0, 0].hist(stability_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Stability Scores Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Stability Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Perturbation effects
        axes[0, 1].hist(perturbation_effects, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('Perturbation Effects Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Mean Perturbation Effect')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Stability vs Perturbation scatter
        axes[1, 0].scatter(perturbation_effects, stability_scores, alpha=0.6, s=50)
        axes[1, 0].set_title('Stability vs Perturbation', fontweight='bold')
        axes[1, 0].set_xlabel('Mean Perturbation Effect')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot of perturbation effects for each input
        if len(stability_metrics) > 0 and 'perturbation_effects' in stability_metrics[0]:
            all_effects = [m['perturbation_effects'] for m in stability_metrics]
            axes[1, 1].boxplot(all_effects, labels=[f'Input {i+1}' for i in range(len(all_effects))])
            axes[1, 1].set_title('Perturbation Effects by Input', fontweight='bold')
            axes[1, 1].set_xlabel('Input Number')
            axes[1, 1].set_ylabel('Perturbation Effect')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_expressivity_analysis(self, expressivity_metrics: List[Dict]):
        """Plot expressivity analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        layers = [m['layer'] for m in expressivity_metrics]
        mean_lengths = [m['mean_length'] for m in expressivity_metrics]
        std_lengths = [m['std_length'] for m in expressivity_metrics]
        max_lengths = [m['max_length'] for m in expressivity_metrics]
        
        # Mean trajectory lengths by layer
        axes[0, 0].bar(layers, mean_lengths, alpha=0.7, color='purple')
        axes[0, 0].set_title('Mean Trajectory Lengths by Layer', fontweight='bold')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Length')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation of trajectory lengths
        axes[0, 1].bar(layers, std_lengths, alpha=0.7, color='green')
        axes[0, 1].set_title('Trajectory Length Std Dev by Layer', fontweight='bold')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Std Dev')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max trajectory lengths
        axes[1, 0].bar(layers, max_lengths, alpha=0.7, color='red')
        axes[1, 0].set_title('Max Trajectory Lengths by Layer', fontweight='bold')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Max Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Expressivity scores
        expressivity_scores = [m['expressivity_score'] for m in expressivity_metrics]
        axes[1, 1].plot(layers, expressivity_scores, 'o-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Expressivity Scores by Layer', fontweight='bold')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Expressivity Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expressivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_dashboard(self, results: Dict):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Stability Scores', 
                          'Expressivity by Layer', 'Performance Metrics',
                          'Memory Usage', 'Trajectory Visualization'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        if 'training_loss' in results:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(results['training_loss']))),
                    y=results['training_loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Stability scores
        if 'stability' in results:
            stability_scores = [m['stability_score'] for m in results['stability']]
            fig.add_trace(
                go.Histogram(
                    x=stability_scores,
                    name='Stability Scores',
                    nbinsx=20,
                    marker_color='orange'
                ),
                row=1, col=2
            )
        
        # Expressivity by layer
        if 'expressivity' in results:
            layers = [m['layer'] for m in results['expressivity']]
            scores = [m['expressivity_score'] for m in results['expressivity']]
            fig.add_trace(
                go.Bar(
                    x=layers,
                    y=scores,
                    name='Expressivity',
                    marker_color='purple'
                ),
                row=2, col=1
            )
        
        # Performance metrics
        if 'performance' in results:
            perf = results['performance']
            metrics = ['mean_time', 'throughput']
            values = [perf.get(m, 0) for m in metrics]
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=values,
                    name='Performance',
                    marker_color='green'
                ),
                row=2, col=2
            )
        
        # Memory usage
        if 'memory' in results:
            memory = results['memory']
            if 'memory_usage_mb' in memory:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=memory['memory_usage_mb'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Memory Usage (MB)"},
                        gauge={'axis': {'range': [None, 1000]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 250], 'color': "lightgray"},
                                        {'range': [250, 500], 'color': "gray"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 500}}
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Liquid Neural Network Analysis Dashboard",
            showlegend=True,
            height=1200,
            width=1600
        )
        
        # Save dashboard
        fig.write_html(self.output_dir / 'interactive_dashboard.html')
        return fig
        
    def plot_comparison_charts(self, comparison_results: Dict):
        """Plot comparison charts between different models"""
        models = list(comparison_results.keys())
        
        # Extract metrics
        metrics = {}
        for model in models:
            if 'accuracy' in comparison_results[model]:
                acc = comparison_results[model]['accuracy']
                metrics[model] = {
                    'MSE': acc.get('mse', 0),
                    'MAE': acc.get('mae', 0),
                    'R2': acc.get('r2', 0)
                }
        
        if not metrics:
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # MSE comparison
        mse_values = [metrics[m]['MSE'] for m in models]
        axes[0, 0].bar(models, mse_values, alpha=0.7, color='red')
        axes[0, 0].set_title('Mean Squared Error Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE comparison
        mae_values = [metrics[m]['MAE'] for m in models]
        axes[0, 1].bar(models, mae_values, alpha=0.7, color='blue')
        axes[0, 1].set_title('Mean Absolute Error Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # R2 comparison
        r2_values = [metrics[m]['R2'] for m in models]
        axes[1, 0].bar(models, r2_values, alpha=0.7, color='green')
        axes[1, 0].set_title('R² Score Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance comparison
        perf_times = []
        for model in models:
            if 'performance' in comparison_results[model]:
                perf_times.append(comparison_results[model]['performance'].get('mean_time', 0))
            else:
                perf_times.append(0)
        
        axes[1, 1].bar(models, perf_times, alpha=0.7, color='purple')
        axes[1, 1].set_title('Inference Time Comparison', fontweight='bold')
        axes[1, 1].set_ylabel('Mean Time (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_network_architecture(self, model_config: Dict):
        """Visualize network architecture using NetworkX"""
        G = nx.DiGraph()
        
        # Add nodes
        layer_configs = model_config.get('layers', [])
        node_id = 0
        layer_nodes = {}
        
        for layer_idx, layer_config in enumerate(layer_configs):
            layer_nodes[layer_idx] = []
            for neuron_idx in range(layer_config.get('size', 1)):
                G.add_node(node_id, layer=layer_idx, neuron=neuron_idx)
                layer_nodes[layer_idx].append(node_id)
                node_id += 1
        
        # Add edges (simple feed-forward for now)
        for layer_idx in range(len(layer_configs) - 1):
            for from_node in layer_nodes[layer_idx]:
                for to_node in layer_nodes[layer_idx + 1]:
                    G.add_edge(from_node, to_node)
        
        # Create layout
        pos = {}
        for layer_idx, nodes in layer_nodes.items():
            for i, node in enumerate(nodes):
                pos[node] = (layer_idx, i - len(nodes)/2)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Draw nodes by layer
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for layer_idx, nodes in layer_nodes.items():
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                 node_color=colors[layer_idx % len(colors)],
                                 node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=20)
        
        # Add labels
        labels = {node: f'N{data["neuron"]}' for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title('Liquid Neural Network Architecture', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_architecture.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, results: Dict):
        """Generate a comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Liquid Neural Network Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 5px; }}
                .metric-value {{ font-weight: bold; font-size: 18px; color: #0066cc; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Liquid Neural Network Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <pre>{json.dumps(results.get('config', {}), indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
        """
        
        if 'accuracy' in results:
            acc = results['accuracy']
            html_content += f"""
                <div class="metric">
                    <div>MSE</div>
                    <div class="metric-value">{acc.get('mse', 0):.6f}</div>
                </div>
                <div class="metric">
                    <div>MAE</div>
                    <div class="metric-value">{acc.get('mae', 0):.6f}</div>
                </div>
                <div class="metric">
                    <div>R²</div>
                    <div class="metric-value">{acc.get('r2', 0):.6f}</div>
                </div>
            """
        
        if 'performance' in results:
            perf = results['performance']
            html_content += f"""
                <div class="metric">
                    <div>Mean Time</div>
                    <div class="metric-value">{perf.get('mean_time', 0):.6f}s</div>
                </div>
                <div class="metric">
                    <div>Throughput</div>
                    <div class="metric-value">{perf.get('throughput', 0):.2f} ops/s</div>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <h3>Training Curves</h3>
                <img src="training_curves.png" alt="Training Curves">
                
                <h3>Stability Analysis</h3>
                <img src="stability_analysis.png" alt="Stability Analysis">
                
                <h3>Expressivity Analysis</h3>
                <img src="expressivity_analysis.png" alt="Expressivity Analysis">
                
                <h3>Network Architecture</h3>
                <img src="network_architecture.png" alt="Network Architecture">
            </div>
            
            <div class="section">
                <h2>Interactive Dashboard</h2>
                <p><a href="interactive_dashboard.html">Open Interactive Dashboard</a></p>
            </div>
            
        </body>
        </html>
        """
        
        with open(self.output_dir / 'report.html', 'w') as f:
            f.write(html_content)
        
        print(f"Report generated: {self.output_dir / 'report.html'}")
