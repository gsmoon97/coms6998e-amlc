# Project 1: Performance Modeling and Analysis

## Overview
Design and conduct an experiment to compare performance across different environments and neural network models. Accuracy is **not** the focus—completing full training is unnecessary.

***

## Experiment Design

### Environments
Compare **2–3 environments**, such as:
- Cloud vs. Bare Metal
- Different CPU flavors
- Different GPU flavors

### Neural Network Models
Use **2–3 neural network models** (e.g., ResNet, VGG, ViT).

### Execution
- Conduct **short, representative runs**
- Full training completion is **not required**

***

## Report Requirements

| Section | Weight | Description |
|---------|--------|-------------|
| **Experiment Design** | 10% | Clearly define the experiment objective and hypothesis being tested |
| **Complexity Estimation** | 20% | Estimate the complexity of the workload |
| **Measurement** | 15% | Collect data to evaluate performance using needed metrics |
| **Roofline Modeling** | 20% | Create roofline models using collected measurements (**mandatory**). Optionally use Nsight's built-in roofline tool for comparison |
| **Analysis** | 35% | -  In-depth analysis of performance metrics<br>-  How each environment influences performance<br>-  Influence of NN model choices on roofline model and performance outcomes |

**📅 Due Date:** March 27, 11:59 PM

***

## Useful Links

### Code & Data
- **PyTorch ImageNet examples:**  
  [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
- **ImageNet 1k dataset:**  
  [http://www.image-net.org/](http://www.image-net.org/)
  - You do **not** need the full dataset—extracting a subset is sufficient

### Tools
- **Nsight Roofline Guide:**  
  [https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline)

***

## Additional Discussions

### Q: Data Requirements
> **Question:** For Project 1, is running the dummy data provided in GitHub sufficient, or do we need to manually download a portion of ImageNet 1k? If the latter, how many training/validation images are sufficient?

> **Prof. Chung:** The data does not matter—garbage in, garbage out is fine. The key is the **application characteristics** and **system resource usage**. Just make sure the data size is reasonable so the key characteristics can be captured.

### Q: Roofline Model Plotting
> **Question:** Do we need to plot the Roofline model ourselves, or can we just use the one built into Nsight?

> **Nirav:** Either is fine. If Nsight is giving you a proper roofline model as discussed in class, then you don't have to plot it yourself.