import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# --- NVIDIA A100 Hardware Specs ---
PEAK_COMPUTE = 19500  # GFLOPs/sec (FP32)
PEAK_BANDWIDTH = 1555 # GB/sec
RIDGE_POINT = PEAK_COMPUTE / PEAK_BANDWIDTH

# The 6 files generated for the A100 experiments
csv_files = [
    'resnet50_train_a100.csv', 'resnet50_eval_a100.csv',
    'vgg16_train_a100.csv', 'vgg16_eval_a100.csv',
    'vit_train_a100.csv', 'vit_eval_a100.csv'
]

# Map filenames to pretty plot labels for the legend
label_map = {
    'resnet50_train_a100.csv': 'ResNet-50 (Train)',
    'resnet50_eval_a100.csv': 'ResNet-50 (Eval)',
    'vgg16_train_a100.csv': 'VGG-16 (Train)',
    'vgg16_eval_a100.csv': 'VGG-16 (Eval)',
    'vit_train_a100.csv': 'ViT-B-16 (Train)',
    'vit_eval_a100.csv': 'ViT-B-16 (Eval)'
}

def parse_val(val_str, unit_str):
    val = float(val_str.replace(',', ''))
    unit = unit_str.strip()
    # Memory unit conversions to raw bytes
    if 'Kbyte' in unit or unit == 'KB': val *= 1024
    elif 'Mbyte' in unit or unit == 'MB': val *= (1024**2)
    elif 'Gbyte' in unit or unit == 'GB': val *= (1024**3)
    # Time unit conversions to seconds
    elif 'nsecond' in unit or unit == 'ns': val *= 1e-9
    elif 'usecond' in unit or unit in ('us', 'μs'): val *= 1e-6
    elif 'msecond' in unit or unit == 'ms': val *= 1e-3
    elif unit in ('second', 's'): val *= 1.0
    return val

print("=== ROOFLINE METRICS SUMMARY ===\n")

# Dictionaries to dynamically store the computed data for plotting
models_ai = {}
models_perf = {}

for filename in csv_files:
    if not os.path.exists(filename):
        print(f"Skipping {filename} - File not found.")
        continue

    fadd = fmul = ffma = hmma = dram_read = dram_write = total_time_sec = 0.0

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 15: continue
            
            metric_name = row[12]
            unit = row[13]
            try:
                val = parse_val(row[14], unit)
            except ValueError:
                continue
                
            if metric_name == 'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum': fadd += val
            elif metric_name == 'smsp__sass_thread_inst_executed_op_fmul_pred_on.sum': fmul += val
            elif metric_name == 'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum': ffma += val
            elif metric_name == 'smsp__sass_thread_inst_executed_op_hmma_pred_on.sum': hmma += val
            elif metric_name == 'dram__bytes_read.sum': dram_read += val
            elif metric_name == 'dram__bytes_write.sum': dram_write += val
            elif metric_name == 'gpu__time_duration.sum': total_time_sec += val

    # The Math: Standard Ops + Fused Ops + Tensor Core Ops (512 FLOPs per HMMA instruction)
    total_flops = fadd + fmul + (2 * ffma) + (512 * hmma)
    total_bytes = dram_read + dram_write
    
    ai = total_flops / total_bytes if total_bytes > 0 else 0
    actual_gflops_sec = (total_flops / 1e9) / total_time_sec if total_time_sec > 0 else 0

    # Store for plotting
    pretty_name = label_map.get(filename, filename)
    models_ai[pretty_name] = ai
    models_perf[pretty_name] = actual_gflops_sec

    print(f"[{pretty_name}]")
    print(f"  Total FLOPs:          {total_flops:,.0f} (HMMA: {hmma:,.0f})")
    print(f"  Total DRAM Bytes:     {total_bytes:,.0f}")
    if total_time_sec > 0:
        print(f"  Total GPU Time:       {total_time_sec:.4f} seconds")
    print(f"  Arithmetic Intensity: {ai:.4f} FLOPs/Byte")
    
    if actual_gflops_sec > 0:
        print(f"  Attained Perf:        {actual_gflops_sec:.2f} GFLOPs/sec\n")
    else:
        print(f"  Attained Perf:        MISSING (Needs gpu__time_duration.sum)\n")

# --- Plotting the Empirical Roofline Model ---
if models_ai and any(perf > 0 for perf in models_perf.values()):
    print("Generating Empirical Roofline Plot...")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the Theoretical Roofline
    ai_vals = np.linspace(0.1, 1000, 1000)
    perf_vals = np.minimum(PEAK_COMPUTE, ai_vals * PEAK_BANDWIDTH)

    ax.plot(ai_vals, perf_vals, color='black', linewidth=2, label='A100 Theoretical Roofline')

    # Plot the Ridge Point
    ax.plot(RIDGE_POINT, PEAK_COMPUTE, 'ro', markersize=8)
    ax.annotate(f'Ridge Point ({RIDGE_POINT:.2f})', 
                xy=(RIDGE_POINT, PEAK_COMPUTE), xytext=(RIDGE_POINT + 5, PEAK_COMPUTE - 500),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    # Plot the Empirical Models on the Roofline dynamically
    colors = ['blue', 'cyan', 'red', 'orange', 'purple', 'magenta']
    
    for (name, ai), color in zip(models_ai.items(), colors):
        perf = models_perf[name]
        if perf > 0:
            # Plot the actual empirical point
            ax.plot(ai, perf, marker='o', markersize=10, color=color, label=f'{name} (AI: {ai:.1f}, Perf: {perf:.1f} GFLOPs/s)')
            # Drop a vertical dashed line to the x-axis
            ax.vlines(x=ai, ymin=1, ymax=perf, color=color, linestyle='--', alpha=0.6)
            # Drop a horizontal dashed line to the y-axis
            ax.hlines(y=perf, xmin=0.1, xmax=ai, color=color, linestyle='--', alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=0.1, right=1000)
    ax.set_ylim(bottom=1, top=PEAK_COMPUTE * 1.5)
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Attained Performance (GFLOPs/sec)', fontsize=12)
    ax.set_title('Empirical Roofline Model: Architectures on NVIDIA A100', fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax.axvspan(0.1, RIDGE_POINT, color='red', alpha=0.05, label='Memory-Bound Region')
    ax.axvspan(RIDGE_POINT, 1000, color='green', alpha=0.05, label='Compute-Bound Region')

    # Ensure legend fits nicely
    ax.legend(loc='lower right', fontsize=10)

    # Save to a high-res image for the LaTeX report
    plt.savefig('a100_roofline_empirical.png', dpi=300, bbox_inches='tight')
    print("Roofline plot saved as 'a100_roofline_empirical.png'")
else:
    print("Cannot plot empirical Roofline without the gpu__time_duration.sum metric!")