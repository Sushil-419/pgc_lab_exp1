import matplotlib.pyplot as plt
import os
import numpy as np

# Data
labels = ['Sequential', 'OpenMP', 'MPI', 'CUDA']
exec_times = [316.018358, 129.972436, 226.167575, 0.120403]
speedups = [1.00, 2.43, 1.40, 2624.21]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Set directory
out_dir = '../images'
if not os.path.exists(out_dir):
    out_dir = 'images'
os.makedirs(out_dir, exist_ok=True)

# 1. Execution Time Chart (Logarithmic Scale)
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, exec_times, color=colors, width=0.6)
plt.title('Execution Time Comparison (Log Scale)\n4000 × 4000 Matrix Multiplication', fontsize=14, pad=15)
plt.xlabel('Implementation', fontsize=12)
plt.ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{yval:.6f} s', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'execution_time_chart.png'), dpi=300)
plt.close()

# 2. Performance Comparison (Two panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Performance Comparison — Sequential vs OpenMP vs MPI vs CUDA', fontsize=16, y=1.05)

# Left Panel
bars1 = ax1.bar(labels, exec_times, color=colors, width=0.6)
ax1.set_title('Execution Time (Log Scale)', fontsize=14)
ax1.set_xlabel('Implementation', fontsize=12)
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.set_yscale('log')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{yval:.6f} s', ha='center', va='bottom', fontsize=11)

# Right Panel
bars2 = ax2.bar(labels, speedups, color=colors, width=0.6)
ax2.set_title('Speedup (Log Scale)', fontsize=14)
ax2.set_xlabel('Implementation', fontsize=12)
ax2.set_ylabel('Speedup (×)', fontsize=12)
ax2.set_yscale('log')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{yval:.2f}×', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Speedup Chart (Logarithmic Scale)
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, speedups, color=colors, width=0.6)
plt.title('Speedup Comparison (Log Scale)\nRelative to Sequential Baseline', fontsize=14, pad=15)
plt.xlabel('Implementation', fontsize=12)
plt.ylabel('Speedup (×) - Log Scale', fontsize=12)
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{yval:.2f}×', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'speedup_chart.png'), dpi=300)
plt.close()
