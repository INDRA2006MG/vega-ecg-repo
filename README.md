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
  <img src="https://img.shields.io/badge/Language-C99-informational" />
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
- 📡 UART telemetry

The primary goal is to demonstrate that a compact neural network can perform useful ECG classification on a resource-constrained **RISC-V embedded platform**.

---

## Key Highlights

- **Fully on-device inference**
- **No cloud connectivity required**
- **No GPU**
- **No FPU**
- **No neural-network accelerator**
- **RV32IM RISC-V deployment**
- **100 MHz embedded processor**
- **3,977-parameter 1D CNN**
- **Q8.8 fixed-point inference**
- **Batch Normalization folded into convolution weights**
- **Dependency-free C99 inference kernel**
- **Portable across RV32IMAC, Cortex-M and x86-64**
- **Real MIT-BIH ECG beats tested on physical hardware**
- **42 real beats classified on-device**
- **End-to-end latency: 719,175 µs**
- **Latency budget: 800,000 µs**
- **98.45% held-out intra-patient classification accuracy**
- **97.15% ± 0.72% 10-fold stratified cross-validation**
- **94.1% train/device preprocessing agreement after causal-context correction**

---

# Classification Waveform

<img width="1536" height="1024" alt="WhatsApp Image 2026-09-11 at 6 31 27 PM" src="https://github.com/user-attachments/assets/06704536-51ff-4fd2-8d76-3d77d99a27d2" />


# System Architecture

<img width="1536" height="1024" alt="WhatsApp Image 2026-09-11 at 12 45 51 PM" src="https://github.com/user-attachments/assets/715c54c1-ff80-411b-9cae-ab43c9b8a628" />

