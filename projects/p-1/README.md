# ImageNet Hardware Profiling Script

This repository contains a customized PyTorch ImageNet execution script designed strictly for hardware profiling and Roofline Model analysis. It isolates the computational workloads of neural networks (such as ResNet, VGG, and ViT) to measure single-precision FLOPs and DRAM traffic using NVIDIA Nsight Compute (`ncu`).

*Note: This code is a heavily modified version of the official PyTorch ImageNet example, which can be found here: [https://github.com/pytorch/examples/tree/main/imagenet](https://github.com/pytorch/examples/tree/main/imagenet).*

## Key Modifications for Profiling

To support specific assignment requirements (Complexity Estimation and isolated hardware measurements), the following features were added to the base PyTorch script:

* **Complexity Estimation (`torchsummary`)**: Dynamically extracts the required input shape for the specified model architecture and prints a parameter/memory footprint summary immediately after model initialization.
* **Targeted Profiling (`--profile-step`)**: Introduces a new command-line flag that specifies exactly which batch to profile. It allows the GPU to run a few "warm-up" batches to populate caches before `torch.cuda.profiler.start()` is triggered.
* **Isolated Compute Workload**: The `ncu` profiler hooks are strategically placed to wrap *only* the neural network's mathematical operations (forward pass, loss calculation, backward pass, and optimizer step) . Data loading and accuracy metric calculations are strictly excluded to prevent hardware metric distortion.
* **Early Termination**: Uses `sys.exit(0)` to instantly terminate the script the moment the target profiling step finishes, saving compute resources and preventing unnecessary validation loops.

## Prerequisites

Ensure you have the required packages installed in your environment before execution:
```bash
pip install torch torchvision torchsummary
```
*Note: NVIDIA Nsight Compute (`ncu`) must be installed and accessible in your system's PATH.*

## Usage Guide

This script is designed to be executed via the `ncu` CLI. We utilize the `--dummy` flag to pass synthetic data (bypassing disk I/O bottlenecks) and the `--profile-step 10` flag to measure the 11th batch after 10 warm-up iterations.

### 1. Profiling the Training Pipeline (Forward + Backward Pass)
To capture the full computational weight of the training step (including gradient computation and weight updates), run:

```bash
ncu --profile-from-start off \
--target-processes all \
--metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum \
python main.py --dummy -a resnet50 --epochs 1 -b 8 --profile-step 10
```

### 2. Profiling the Inference Pipeline (Forward Pass Only)
To measure strictly the inference workload (bypassing the training loop entirely), append the `-e` (or `--evaluate`) flag to the PyTorch execution command:

```bash
ncu --profile-from-start off \
--target-processes all \
--metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum \
python main.py -e --dummy -a resnet50 --epochs 1 -b 8 --profile-step 10
```

## Supported Architectures
You can swap `resnet50` in the commands above for any architecture natively supported by `torchvision.models` (e.g., `vit_b_16`, `vgg16`, `resnet18`).