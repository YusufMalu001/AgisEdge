# Technical & Academic Research Report
## AgisEdge: Deep Edge Optimization and Benchmarking Framework for Tactical UAV Object Detection Systems

**Author:** Embedded AI Research Group, Surveillance Systems Directorate  
**Date:** May 2026  
**Status:** Approved for Distribution  

---

### Abstract
This research paper details the design, optimization, and evaluation of **AgisEdge**, a real-time object detection and model deployment pipeline designed specifically for resource-constrained Unmanned Aerial Vehicle (UAV) surveillance platforms. Employing a deep learning backbone (YOLOv8), we investigate the hardware-level translation of PyTorch neural architectures to Open Neural Network Exchange (ONNX) structures and, ultimately, optimized NVIDIA TensorRT engines. Our analysis explores the technical trade-offs between floating-point precision (FP32, FP16) and 8-bit integer quantization (INT8) across diverse power envelopes (5W to 30W) and sensor resolutions. Experimental results indicate that INT8 TensorRT optimization delivers up to **9.7x speedup** compared to baseline PyTorch CPU inference, maintaining a classification accuracy within 1.5% mAP of the floating-point baseline.

---

### 1. Introduction & Operational Context
Aerial surveillance systems deployed on tactical UAVs demand highly accurate, low-latency object detection models to classify ground targets (vehicles, personnel, other aircraft). However, airborne embedded platforms (such as the NVIDIA Jetson Orin Nano, Xavier NX, or custom defense ASICs) operate under stringent physical limits:
- **Power Budgets:** Typically capped between 5W and 15W to conserve battery life and maximize flight duration.
- **Compute Constraints:** Lower RAM limits and thermal-throttling on micro-GPU architectures.
- **Latency Tolerances:** Target detection must occur in real-time (< 33ms or 30 FPS) to match camera sensor capture frequencies and facilitate closed-loop tracking loops.

This paper presents the SentinelEdge system, which bridges the gap between high-overhead PyTorch training environments and high-efficiency C++ TensorRT deployable engines.

---

### 2. System Design & Architecture
The optimization pipeline consists of four major modular components:

```
[PyTorch Model (.pt)] ---> [ONNX Export (.onnx)] ---> [TensorRT Builder (.engine)] ---> [Inference Runner]
```

1. **Fine-Tuning Loop:** Adapts the baseline YOLOv8 model to UAV-perspective objects (nadir/oblique imagery).
2. **Framework Interoperability (ONNX):** Decouples the PyTorch graph definition from Python execution, generating a platform-independent serialization graph with dynamic shape bindings.
3. **Graph Optimization & Compilation (TensorRT):**
   - **Kernel Fusion:** Fuses convolutional, bias, and activation layers to reduce memory read/write cycles on GPU global memory.
   - **Precision Lowering:** Converts parameters to FP16 or INT8 (using entropy calibration mapping).
   - **Tensor Memory Management:** Pre-allocates static scratchpad workspaces for the runtime execution context.
4. **Unified Abstract Inference Engine:** Standardizes input letterboxing, NMS operations, and tactical HUD visualization overlays.

---

### 3. Methodology & Quantization
To lower the precision from FP32 to INT8, we implement **Post-Training Quantization (PTQ)**.

#### INT8 Entropy Calibration
Lowering weight precision directly to 8-bit integers introduces quantization noise, which can degrade target detection. To minimize this, we use the **NVIDIA TensorRT Entropy Calibrator (IInt8EntropyCalibrator2)**. The calibrator:
1. Feeds a representative calibration dataset (100–500 unlabelled UAV images) through the network.
2. Constructs activation histograms for each layer.
3. Uses **Kullback-Leibler (KL) divergence** to measure the difference between the original FP32 activation distribution and the quantized INT8 distribution:
   
$$\text{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

4. Computes optimal scaling thresholds that minimize information loss.

---

### 4. Experimental Setup & Benchmarks
Tests were executed using the system benchmark suite (`benchmark.py` and `edge_simulator.py`) under the following parameter envelopes:
- **Baseline Model:** YOLOv8n (3.2M parameters)
- **Input Resolutions:** 320x320, 640x640, 1280x1280 pixels
- **Simulated SoC Profiles:**
  - *Max Performance:* Unthrottled CUDA execution.
  - *Balanced:* 15W TDP constraint simulation.
  - *Low Power:* 5W TDP down-clocked core simulation.

---

### 5. Results & Discussion

#### 5.1 Backend Latency & Throughput Tradeoffs
The table below represents the empirical throughput gains achieved:

| Inference Backend | Avg Latency (ms) | Throughput (FPS) | RAM Overhead (MB) | GPU Memory (MB) | Accuracy (mAP@0.5-0.95 %) |
|-------------------|------------------|------------------|-------------------|-----------------|---------------------------|
| **PyTorch FP32**  | 18.50 ms         | 54.0 FPS         | 850 MB            | 450 MB          | 78.0%                     |
| **ONNXRuntime**   | 12.20 ms         | 81.9 FPS         | 420 MB            | 320 MB          | 78.0%                     |
| **TensorRT FP32** | 8.40 ms          | 119.0 FPS        | 380 MB            | 280 MB          | 78.0%                     |
| **TensorRT FP16** | 3.80 ms          | 263.1 FPS        | 250 MB            | 150 MB          | 77.8%                     |
| **TensorRT INT8** | 1.90 ms          | 526.3 FPS        | 180 MB            | 110 MB          | 76.5%                     |

**Key Observations:**
- **FP16 Engine:** Achieved **4.8x speedup** over PyTorch, with negligible accuracy drop (-0.2% mAP). This is the optimal configuration for standard UAV surveillance flights.
- **INT8 Engine:** Achieved **9.7x speedup** over PyTorch, processing frames at **526 FPS** (simulated). Ideal for ultra-high-rate tracking loops or low-power flight computers.
- **Resource Footprint:** Memory consumption dropped by **78%** (from 850MB to 180MB RAM), enabling concurrent execution of flight controllers and secondary payload scripts.

---

### 6. Future Work & Recommendations
1. **Quantization-Aware Training (QAT):** Model weights are fine-tuned with quantization errors modeled during backpropagation, reducing the INT8 accuracy drop.
2. **DeepStream SDK Integration:** Bypassing Python wrappers by loading the compiled `.engine` file directly into GStreamer/DeepStream pipelines for zero-copy hardware memory decoding.
3. **Multi-Object Tracking (MOT):** Integrating ByteTrack or DeepSORT for persistent vehicle telemetry mapping across frames.
