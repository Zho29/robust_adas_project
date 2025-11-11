"""
DSNet vs WUNet+YOLO Comparison Visualization
Uses matplotlib to create comprehensive visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Define colors
COLOR_DSNET = '#D55E00'  # Orange-red
COLOR_WUNET_YOLO = '#0072B2'  # Blue
COLOR_WUNET = '#56B4E9'  # Light blue
COLOR_YOLO = '#009E73'  # Green

def load_results(filename='dsnet_wunet_comparison_results.json'):
    """Load comparison results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_computational_comparison(results):
    """Figure 1: Computational Efficiency Comparison"""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('DSNet vs WUNet+YOLO: Computational Efficiency Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Extract data
    dsnet = results['dsnet']['computational']
    wunet_yolo = results['wunet_yolo']['computational']

    # Convert to appropriate units
    dsnet_params = dsnet['params'] / 1e6
    dsnet_flops = dsnet['flops'] / 1e9
    wunet_yolo_params = wunet_yolo['combined']['params'] / 1e6
    wunet_yolo_flops = wunet_yolo['combined']['flops'] / 1e9

    wunet_params = wunet_yolo['wunet']['params'] / 1e6
    wunet_flops = wunet_yolo['wunet']['flops'] / 1e9
    wunet_latency = wunet_yolo['wunet']['latency_ms']
    yolo_params = wunet_yolo['yolo']['params'] / 1e6
    yolo_flops = wunet_yolo['yolo']['flops'] / 1e9
    yolo_latency = wunet_yolo['yolo']['latency_ms']

    models = ['DSNet', 'WUNet+YOLO']

    # Subplot 1: Parameters
    ax1 = fig.add_subplot(gs[0, 0])
    params_data = [dsnet_params, wunet_yolo_params]
    bars = ax1.bar(models, params_data, color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Parameters (Million)', fontweight='bold')
    ax1.set_title('Model Parameters', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels
    for bar, val in zip(bars, params_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}M', ha='center', va='bottom', fontweight='bold')

    # Subplot 2: FLOPs
    ax2 = fig.add_subplot(gs[0, 1])
    flops_data = [dsnet_flops, wunet_yolo_flops]
    bars = ax2.bar(models, flops_data, color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('FLOPs (GFLOPs)', fontweight='bold')
    ax2.set_title('Computational Complexity', fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, flops_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}G', ha='center', va='bottom', fontweight='bold')

    # Subplot 3: Latency
    ax3 = fig.add_subplot(gs[0, 2])
    latency_data = [dsnet['latency_ms'], wunet_yolo['combined']['latency_ms']]
    bars = ax3.bar(models, latency_data, color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_ylabel('Latency (ms)', fontweight='bold')
    ax3.set_title('Inference Latency', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, latency_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold')

    # Subplot 4: FPS
    ax4 = fig.add_subplot(gs[1, 0])
    fps_data = [dsnet['fps'], wunet_yolo['combined']['fps']]
    bars = ax4.bar(models, fps_data, color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_ylabel('FPS', fontweight='bold')
    ax4.set_title('Frames Per Second', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    # Add reference lines
    ax4.axhline(y=30, color='green', linestyle='--', linewidth=2, alpha=0.5, label='30 FPS')
    ax4.axhline(y=60, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='60 FPS')
    ax4.legend(loc='upper left', fontsize=8)
    for bar, val in zip(bars, fps_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 5: Pipeline Breakdown (Parameters)
    ax5 = fig.add_subplot(gs[1, 1])
    pipeline_params = [wunet_params, yolo_params]
    bars = ax5.bar(['Pipeline'], pipeline_params, width=0.5,
                   color=[COLOR_WUNET, COLOR_YOLO], edgecolor='black',
                   linewidth=1.5, alpha=0.8, label=['WUNet', 'YOLO'])
    ax5.set_ylabel('Parameters (Million)', fontweight='bold')
    ax5.set_title('WUNet+YOLO Pipeline Breakdown', fontweight='bold', fontsize=12)
    ax5.legend(loc='upper right')
    ax5.grid(axis='y', alpha=0.3)
    # Add value labels
    y_offset = 0
    for i, val in enumerate(pipeline_params):
        ax5.text(0, y_offset + val/2, f'{val:.1f}M',
                ha='center', va='center', fontweight='bold', fontsize=9)
        y_offset += val

    # Subplot 6: Latency Breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    pipeline_latency = [wunet_latency, yolo_latency]
    bars = ax6.bar(['Pipeline'], pipeline_latency, width=0.5,
                   color=[COLOR_WUNET, COLOR_YOLO], edgecolor='black',
                   linewidth=1.5, alpha=0.8, label=['WUNet', 'YOLO'])
    ax6.set_ylabel('Latency (ms)', fontweight='bold')
    ax6.set_title('Latency Breakdown', fontweight='bold', fontsize=12)
    ax6.legend(loc='upper right')
    ax6.grid(axis='y', alpha=0.3)
    # Add value labels
    y_offset = 0
    for i, val in enumerate(pipeline_latency):
        ax6.text(0, y_offset + val/2, f'{val:.2f}ms',
                ha='center', va='center', fontweight='bold', fontsize=9)
        y_offset += val

    plt.tight_layout()
    plt.savefig('comparison_computational_efficiency.png', dpi=300, bbox_inches='tight')
    plt.savefig('comparison_computational_efficiency.pdf', bbox_inches='tight')
    print("✅ Saved: comparison_computational_efficiency.png")
    return fig

def create_map_performance(results):
    """Figure 2: mAP Performance Visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('WUNet+YOLO: Detection Accuracy Across Weather Conditions',
                 fontsize=16, fontweight='bold')

    # Extract mAP data
    map_data = results['wunet_yolo']['map']
    weather_conditions = ['Normal', 'Heavy Fog', 'Heavy Rain']
    map50_values = [
        map_data['normal']['mAP50'] * 100,
        map_data['fog_high']['mAP50'] * 100,
        map_data['rain_high']['mAP50'] * 100
    ]
    map50_95_values = [
        map_data['normal']['mAP50_95'] * 100,
        map_data['fog_high']['mAP50_95'] * 100,
        map_data['rain_high']['mAP50_95'] * 100
    ]

    # Subplot 1: mAP@0.5 across conditions
    ax1 = axes[0]
    bars = ax1.bar(weather_conditions, map50_values, color=COLOR_WUNET_YOLO,
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('mAP@0.5 (%)', fontweight='bold', fontsize=12)
    ax1.set_title('mAP@0.5 Performance', fontweight='bold', fontsize=13)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    # Add baseline reference line
    ax1.axhline(y=map50_values[0], color='red', linestyle='--',
                linewidth=2, alpha=0.6, label='Normal Baseline')
    ax1.legend(loc='lower right')
    # Add value labels
    for bar, val in zip(bars, map50_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

    # Subplot 2: mAP comparison
    ax2 = axes[1]
    x = np.arange(len(weather_conditions))
    width = 0.35
    bars1 = ax2.bar(x - width/2, map50_values, width, label='mAP@0.5',
                    color=COLOR_WUNET_YOLO, edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax2.bar(x + width/2, map50_95_values, width, label='mAP@0.5:0.95',
                    color=COLOR_WUNET, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('mAP (%)', fontweight='bold', fontsize=12)
    ax2.set_title('mAP Metrics Comparison', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(weather_conditions)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_map_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig('comparison_map_performance.pdf', bbox_inches='tight')
    print("✅ Saved: comparison_map_performance.png")
    return fig

def create_speedup_analysis(results):
    """Figure 3: Speedup and Efficiency Analysis"""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Performance Analysis: WUNet+YOLO vs DSNet',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extract data
    dsnet = results['dsnet']['computational']
    wunet_yolo = results['wunet_yolo']['computational']['combined']
    map_data = results['wunet_yolo']['map']

    # Calculate metrics
    speedup_latency = dsnet['latency_ms'] / wunet_yolo['latency_ms']
    speedup_fps = wunet_yolo['fps'] / dsnet['fps']
    reduction_flops = (dsnet['flops'] - wunet_yolo['flops']) / dsnet['flops'] * 100
    reduction_params = (dsnet['params'] - wunet_yolo['params']) / dsnet['params'] * 100

    avg_map = (map_data['normal']['mAP50'] + map_data['fog_high']['mAP50'] +
               map_data['rain_high']['mAP50']) / 3 * 100

    # Subplot 1: Speedup Metrics
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['Latency\nSpeedup', 'FPS\nSpeedup', 'FLOPs\nReduction (%)', 'Params\nReduction (%)']
    values = [speedup_latency, speedup_fps, reduction_flops, reduction_params]
    colors = [COLOR_WUNET_YOLO, COLOR_WUNET_YOLO, COLOR_WUNET, COLOR_WUNET]

    bars = ax1.barh(metrics, values, color=colors, edgecolor='black',
                    linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Value', fontweight='bold', fontsize=12)
    ax1.set_title('WUNet+YOLO Advantages over DSNet', fontweight='bold', fontsize=13)
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val, i in zip(bars, values, range(len(values))):
        width = bar.get_width()
        if i < 2:
            label = f'{val:.2f}x'
        else:
            label = f'{val:.1f}%'
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                label, ha='left', va='center', fontweight='bold', fontsize=11)

    # Subplot 2: Efficiency-Accuracy Trade-off
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(wunet_yolo['latency_ms'], avg_map, s=500, c=COLOR_WUNET_YOLO,
                edgecolors='black', linewidths=2, alpha=0.7, label='WUNet+YOLO')
    ax2.scatter(dsnet['latency_ms'], 0, s=500, marker='x', c=COLOR_DSNET,
                linewidths=3, label='DSNet (no mAP data)')
    ax2.set_xlabel('Latency (ms)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Average mAP@0.5 (%)', fontweight='bold', fontsize=11)
    ax2.set_title('Efficiency vs Accuracy', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, dsnet['latency_ms'] * 1.1])
    ax2.set_ylim([-10, 100])
    # Add annotations
    ax2.annotate(f'{wunet_yolo["latency_ms"]:.1f}ms\n{avg_map:.1f}%',
                xy=(wunet_yolo['latency_ms'], avg_map),
                xytext=(10, 10), textcoords='offset points',
                fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    ax2.annotate(f'{dsnet["latency_ms"]:.1f}ms\nN/A',
                xy=(dsnet['latency_ms'], 0),
                xytext=(10, -20), textcoords='offset points',
                fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.3))

    # Subplot 3: Real-time Performance
    ax3 = fig.add_subplot(gs[1, 1])
    models = ['DSNet', 'WUNet+YOLO']
    fps_values = [dsnet['fps'], wunet_yolo['fps']]
    bars = ax3.bar(models, fps_values, color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_ylabel('FPS', fontweight='bold', fontsize=11)
    ax3.set_title('Real-time Performance', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    # Add threshold lines
    ax3.axhline(y=30, color='green', linestyle='--', linewidth=2, alpha=0.5, label='30 FPS')
    ax3.axhline(y=60, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='60 FPS')
    ax3.axhline(y=120, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='120 FPS')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylim([0, max(fps_values) * 1.15])
    # Add value labels
    for bar, val in zip(bars, fps_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('comparison_speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('comparison_speedup_analysis.pdf', bbox_inches='tight')
    print("✅ Saved: comparison_speedup_analysis.png")
    return fig

def create_summary_dashboard(results):
    """Figure 4: Complete Summary Dashboard"""
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('DSNet vs WUNet+YOLO: Complete Performance Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

    # Extract all data
    dsnet = results['dsnet']['computational']
    wunet_yolo_comp = results['wunet_yolo']['computational']
    wunet_yolo = wunet_yolo_comp['combined']
    map_data = results['wunet_yolo']['map']

    dsnet_params = dsnet['params'] / 1e6
    dsnet_flops = dsnet['flops'] / 1e9
    wunet_yolo_params = wunet_yolo['params'] / 1e6
    wunet_yolo_flops = wunet_yolo['flops'] / 1e9

    # Summary statistics
    speedup_latency = dsnet['latency_ms'] / wunet_yolo['latency_ms']
    reduction_flops = (dsnet['flops'] - wunet_yolo['flops']) / dsnet['flops'] * 100

    # Subplot: Comparison Table
    ax2 = fig.add_subplot(gs[0, 0])
    ax2.axis('tight')
    ax2.axis('off')

    table_data = [
        ['Metric', 'DSNet', 'WUNet+YOLO', 'Winner'],
        ['Params (M)', f'{dsnet_params:.1f}', f'{wunet_yolo_params:.1f}', 'DSNet'],
        ['FLOPs (G)', f'{dsnet_flops:.1f}', f'{wunet_yolo_flops:.1f}', 'WUNet+YOLO'],
        ['Latency (ms)', f'{dsnet["latency_ms"]:.2f}', f'{wunet_yolo["latency_ms"]:.2f}', 'WUNet+YOLO'],
        ['FPS', f'{dsnet["fps"]:.1f}', f'{wunet_yolo["fps"]:.1f}', 'WUNet+YOLO'],
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white')

    # Style winner column
    for i in range(1, 5):
        cell = table[(i, 3)]
        if 'WUNet' in table_data[i][3]:
            cell.set_facecolor('#c8e6c9')
        else:
            cell.set_facecolor('#ffccbc')

    ax2.set_title('Performance Comparison Table', fontweight='bold', fontsize=11, pad=10)

    # Subplot: Pie chart for latency
    ax3 = fig.add_subplot(gs[0, 1])
    wunet_lat = wunet_yolo_comp['wunet']['latency_ms']
    yolo_lat = wunet_yolo_comp['yolo']['latency_ms']
    ax3.pie([wunet_lat, yolo_lat], labels=['WUNet', 'YOLO'],
            autopct='%1.1f%%', colors=[COLOR_WUNET, COLOR_YOLO],
            startangle=90, textprops={'fontweight': 'bold'})
    ax3.set_title('WUNet+YOLO Latency Distribution', fontweight='bold', fontsize=11)

    # Subplot: mAP performance
    ax4 = fig.add_subplot(gs[0, 2])
    weather = ['Normal', 'Fog', 'Rain']
    map_values = [
        map_data['normal']['mAP50'] * 100,
        map_data['fog_high']['mAP50'] * 100,
        map_data['rain_high']['mAP50'] * 100
    ]
    bars = ax4.bar(weather, map_values, color=COLOR_WUNET_YOLO,
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax4.set_ylabel('mAP@0.5 (%)', fontweight='bold', fontsize=10)
    ax4.set_title('mAP@0.5 Across Weather', fontweight='bold', fontsize=11)
    ax4.set_ylim([0, 100])
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, map_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot: Speedup metrics
    ax5 = fig.add_subplot(gs[1, 0])
    speedup_fps = wunet_yolo['fps'] / dsnet['fps']
    speedup_data = [speedup_latency, speedup_fps]
    bars = ax5.bar(['Latency', 'FPS'], speedup_data, color=COLOR_WUNET_YOLO,
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax5.set_ylabel('Speedup Factor', fontweight='bold', fontsize=10)
    ax5.set_title('WUNet+YOLO Speedup vs DSNet', fontweight='bold', fontsize=11)
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, speedup_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Subplot: FLOPs comparison
    ax6 = fig.add_subplot(gs[1, 1])
    flops_comp = [dsnet_flops, wunet_yolo_flops]
    bars = ax6.bar(['DSNet', 'WUNet+YOLO'], flops_comp,
                   color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax6.set_ylabel('FLOPs (G)', fontweight='bold', fontsize=10)
    ax6.set_title('FLOPs Comparison', fontweight='bold', fontsize=11)
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, flops_comp):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}G', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot: Latency comparison
    ax7 = fig.add_subplot(gs[1, 2])
    lat_comp = [dsnet['latency_ms'], wunet_yolo['latency_ms']]
    bars = ax7.bar(['DSNet', 'WUNet+YOLO'], lat_comp,
                   color=[COLOR_DSNET, COLOR_WUNET_YOLO],
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    ax7.set_ylabel('Latency (ms)', fontweight='bold', fontsize=10)
    ax7.set_title('Latency Comparison', fontweight='bold', fontsize=11)
    ax7.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, lat_comp):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('comparison_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig('comparison_summary_dashboard.pdf', bbox_inches='tight')
    print("✅ Saved: comparison_summary_dashboard.png")
    return fig

def main():
    """Main function to create all visualizations"""
    print("=" * 80)
    print("DSNet vs WUNet+YOLO: Creating Visualizations")
    print("=" * 80)
    print()

    # Load results
    try:
        results = load_results()
        print("✅ Loaded comparison results")
    except FileNotFoundError:
        print("❌ Error: dsnet_wunet_comparison_results.json not found!")
        print("   Please run compare_dsnet_wunet.py first.")
        return

    # Create visualizations
    print("\n Creating visualizations...")
    print("-" * 80)

    fig1 = create_computational_comparison(results)
    fig2 = create_map_performance(results)
    fig3 = create_speedup_analysis(results)
    fig4 = create_summary_dashboard(results)

    print("-" * 80)
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  • comparison_computational_efficiency.png")
    print("  • comparison_map_performance.png")
    print("  • comparison_speedup_analysis.png")
    print("  • comparison_summary_dashboard.png")
    print("  • (+ PDF versions of all figures)")
    print("\n" + "=" * 80)

    # Show plots
    plt.show()

if __name__ == '__main__':
    main()
