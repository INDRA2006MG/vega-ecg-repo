# VEGA-ECG
## On-Device Arrhythmia Classification on C-DAC VEGA ARIES v2

<p align="center">
  <b>Real-Time ECG Arrhythmia Classification on an Indigenous RISC-V SoC</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-VEGA%20ARIES%20v2-blue" />
  <img src="https://img.shields.io/badge/CPU-THEJAS32%20RV32IM-orange" />
  <img src="https://img.shields.io/badge/Inference-On--Device-success" />
  <img src="https://img.shields.io/badge/Precision-Q8.8-purple" />
  <img src="https://img.shields.io/badge/Language-C++-informational" />
  <img src="https://img.shields.io/badge/AI-1D%20CNN-red" />
</p>

---

## Overview

**VEGA-ECG** is a real-time, fully on-device ECG arrhythmia classification system designed for the **C-DAC VEGA ARIES v2**, powered by the indigenous **THEJAS32 RV32IM RISC-V processor**.

The system performs ECG beat detection, feature extraction, fixed-point neural-network inference, and arrhythmia classification directly on the embedded processor — without requiring a cloud server, GPU, floating-point unit, or dedicated neural-network accelerator.

The project is part of a broader **multimodal Edge-AI health monitoring system** integrating:

- ❤️ ECG-based arrhythmia classification
- 💓 Heart-rate / SpO₂ monitoring
- 🏃 Motion sensing
- 🌡️ Body-temperature monitoring
- 📟 Local LCD display

The primary goal is to demonstrate that a compact neural network can perform useful ECG classification on a resource-constrained **RISC-V embedded platform**.

<img width="802" height="402" alt="image" src="https://github.com/user-attachments/assets/bfb1ea0d-6687-4c67-b78f-1ed85acaf9dd" />


---

## 🚀 Key Highlights

### 🧠 AI at the Edge

- ⚡ **Fully on-device ECG inference** — classification runs entirely on the VEGA ARIES v2
- ☁️ **Zero cloud dependency** — no internet or remote inference required
- 🖥️ **No GPU required**
- 🔢 **No FPU required** — optimized for integer-only embedded computation
- 🧩 **No neural-network accelerator** — inference runs directly on the RISC-V CPU
- 🧠 **3,977-parameter lightweight 1D CNN**
- 🎯 **5-class AAMI arrhythmia classification:** `N / S / V / F / Q`

### ⚙️ Embedded & RISC-V Optimization

- 🦾 **Target:** C-DAC VEGA ARIES v2
- 🔰 **Processor:** THEJAS32 / VEGA ET1031 `RV32IM`
- ⏱️ **CPU:** 100 MHz
- 📦 **Q8.8 fixed-point neural-network inference**
- 🔄 **Batch Normalization folded into convolution weights**
- 💻 **Dependency-free C99 inference kernel**
- 🌍 **Portable across RV32IMAC, Cortex-M and x86-64**
- 🚫 **No TensorFlow/PyTorch runtime on the target**
- 🧮 **64-bit MAC accumulation with saturated fixed-point requantization**

### 🧪 Real Hardware Validation

- 🔬 **Verified on physical VEGA ARIES v2 hardware**
- ❤️ **42 real MIT-BIH ECG beats classified on-device**
- 📈 **R-R interval and QRS width calculated during the embedded pipeline**
- ⏱️ **Measured end-to-end latency: 719,175 µs**
- 🎯 **Execution budget: 800,000 µs**
- ✅ **Completed within the defined hardware timing budget**
- 🧪 **8-stage verification protocol covering software → hardware deployment**

### 📊 Model Performance

- 🏆 **98.45% accuracy** on 6,000 held-out beats
- 📐 **Cohen's κ / MCC: 0.981**
- 🔁 **97.15% ± 0.72%** with 10-fold stratified cross-validation
- 🎯 **5-class AAMI classification**
- 🧠 **Multi-scale convolution + SE attention + residual learning**
- 📉 Compact architecture designed for resource-constrained embedded inference

### 🔬 Embedded Preprocessing Breakthrough

- ⚠️ Identified a critical **training-to-device preprocessing mismatch**
- 🧪 Naive causal preprocessing resulted in only **64.1% train/device agreement**
- 🛠️ Introduced a **1024-sample causal context window**
- 🎯 Centred **256-sample beat extraction** around the detected R-peak
- 📈 Recovered train/device preprocessing agreement to **94.1%**
- 🔄 Enables real-time causal processing without relying on offline `filtfilt()`

### 📟 Multimodal Edge-AI Platform

Designed as part of a broader embedded health-monitoring platform integrating:

- ❤️ **ECG arrhythmia classification**
- 💓 **MAX30102 — heart rate / SpO₂**
- 🏃 **MPU6050 — motion sensing**
- 🌡️ **MLX90614 — body temperature**
- 📟 **16×4 LCD — local results**

> **From raw ECG → R-peak detection → beat extraction → fixed-point CNN inference → arrhythmia classification — everything happens locally on the RISC-V device.**

---

## ⚡ VEGA-ECG by the Numbers

| Metric | Result |
|---|---:|
| 🧠 Model Parameters | **3,977** |
| 🎯 Classification Classes | **5** |
| 📊 Held-out Beats | **6,000** |
| 🏆 Accuracy | **98.45%** |
| 📐 Cohen's κ / MCC | **0.981** |
| 🔁 10-Fold CV | **97.15% ± 0.72%** |
| ❤️ Real Hardware Beats | **42** |
| ⏱️ Measured Latency | **719,175 µs** |
| 🎯 Latency Budget | **800,000 µs** |
| 📈 Train/Device Agreement | **94.1%** |
| ⚙️ CPU Frequency | **100 MHz** |
| 💾 SRAM | **256 KB** |
| 🔢 Numeric Format | **Q8.8** |

# Classification Waveform

<img width="1536" height="1024" alt="WhatsApp Image 2026-09-11 at 6 31 27 PM" src="https://github.com/user-attachments/assets/06704536-51ff-4fd2-8d76-3d77d99a27d2" />


# System Architecture

<img width="1536" height="1024" alt="WhatsApp Image 2026-09-11 at 12 45 51 PM" src="https://github.com/user-attachments/assets/715c54c1-ff80-411b-9cae-ab43c9b8a628" />

