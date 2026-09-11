# VEGA-ECG: On-Device Arrhythmia Classification on the C-DAC VEGA ARIES v2 (THEJAS32 RISC-V)

Real-time ECG arrhythmia classification running entirely on-device on the
indigenous C-DAC **VEGA ARIES v2** board (**THEJAS32** SoC, VEGA ET1031
32-bit RV32IM RISC-V core, 100 MHz, no FPU, no NN accelerator) — part of a
multimodal Edge-AI health monitor (ECG + heart rate/SpO2 + motion + body
temperature), built for the **Embrix'26 Vegathon** (BIT Sathy + C-DAC).

No cloud, no GPU: a 3,977-parameter 1D CNN (`MultiScale-SE-CNN-Tiny` /
LME-SE-RCNN) is quantized to Q8.8 fixed-point with BatchNorm folded into the
convolution weights, and runs from a portable, dependency-free C inference
kernel — no HLS, no `ap_int`, just plain C99 that cross-compiles for
RV32IMAC, Cortex-M and x86-64 alike.

## Current status

Verified through a gated hardware protocol (Stage 01–08):

| Stage | What it checks | Status |
|---|---|---|
| 01 | Unit tests | ✅ Passed |
| 02 | Fixed-point vs. float quantization parity | ✅ Passed |
| 03 | On-target timing | ✅ Passed |
| 04 | End-to-end beat classification | ✅ Passed |
| 05 | Full end-to-end run on real hardware | ✅ Passed — 42 real MIT-BIH beats classified on-device, R-R interval/QRS width reported, 719,175 µs vs. an 800,000 µs budget |
| 06 | LCD text output | 🟡 In progress |
| 07 | On-device HR fusion (MAX30102) | 🟡 In progress |
| 08 | Unattended, timed end-to-end run | 🟡 In progress |

Model performance (intra-patient, 5-class AAMI: N/S/V/F/Q, 6,000 held-out
beats): **98.45% accuracy** (Cohen's κ / MCC = 0.981); 10-fold stratified
cross-validation: **97.15% ± 0.72%**. See `docs/` for the full technical
report and confusion matrix.

A hard preprocessing problem specific to embedded deployment was solved
along the way: training uses zero-phase (`filtfilt`) offline filtering,
but the device must extract beats causally, in real time. Naively doing so
dropped train/device agreement to 64.1%; a 1024-sample causal context
window with centred 256-sample extraction recovered **94.1%** agreement.
See `firmware/host/rpeak.c` and the technical report for the full analysis.

## Repository layout

```
firmware/
  host/            Desktop/host build of the inference kernel — used to
                    cross-verify the fixed-point C kernel against a NumPy
                    reference before anything touches hardware
                    (`make verify`, `make rpeak`, `make real`).
  arduino_vega/     The on-board sketch actually flashed to the VEGA ARIES v2
                    (Arduino-framework build: model_forward + rpeak + weights
                    + LCD driver, combined).
model/
  weights_keras.h   Q8.8 weights as exported from the trained Keras model.
  convert_weights.py  Transposes the Keras layout to the C layout used by
                    model_forward.c, with per-element assertions.
  test_vectors.h    Real, quantized MIT-BIH test beats + true labels, used
                    for on-target verification (Stage 02/05).
docs/
  ECG_on_VEGA_technical_report.pdf   Full write-up: architecture, metrics,
                    confusion matrix, hardware verification log, limitations.
  ECG_on_VEGA_paper.pdf              IEEE-style paper draft (VEGA-specific).
  ECG_on_VEGA_deck.pdf               Project slide deck.
images/
  fig1_pipeline.png   End-to-end pipeline: MIT-BIH → training → Q8.8
                    quantization → C kernel → VEGA ARIES v2 → UART output.
  fig2_beats.png      Real MIT-BIH beats (Normal vs. ventricular-ectopic)
                    as seen by the network, dequantized from Q8.8.
  fig_confmat.png     Confusion matrix, intra-patient 5-class AAMI split.
  fig_stages.png      Stage-gate hardware verification progress (01–08).
```

## Model architecture (as implemented)

```
input 256x1 (Q8.8, beat normalized to [-1,1], centred on R-peak)
├─ Conv1D(8, k=3)             ─┐
├─ Conv1D(8, k=5)              ├─ each: BN(folded) → ReLU → SE(8→4→8, hard_sigmoid)
└─ Conv1D(8, k=7, dilation=2) ─┘
        Add()   ← fused, not concatenated
ResBlock1: 8→16, k=3, 1x1 projection shortcut
MaxPool1D(2)                  256 → 128
ResBlock2: 16→16, k=3, identity shortcut
MaxPool1D(2)                  128 → 64
GlobalAveragePooling1D        → 16
Dense(16) + ReLU
Dense(5)                      → argmax (AAMI classes N/S/V/F/Q)
```

## Fixed-point notes

- Q8.8 throughout (`value = round(real * 256)`); BatchNorm already folded
  into the convolution weights at export time.
- All MACs accumulate in `int64_t`, requantized once with round-half-up
  `>> 8` and saturated to `int16_t`.
- `hard_sigmoid` is Keras's `clip(0.2x + 0.5, 0, 1)`, **not** a logistic
  sigmoid — every SE gate depends on getting this exact.
- Branch 3 is dilated (`k=7, dilation=2`, 13-tap span), not a plain k=7.

## Build & verify (host side)

```sh
cd firmware/host
make verify      # builds, runs the kernel, cross-checks against NumPy
make rpeak       # runs the R-peak detector against synthetic test vectors
make real        # validates against real, labelled MIT-BIH beats
```

## Hardware

- **Board:** C-DAC VEGA ARIES v2 (THEJAS32 SoC, VEGA ET1031 RV32IM RISC-V
  core, 100 MHz, 256 KB SRAM, 2 MB flash, no FPU, no accelerator)
- **Sensors:** MAX30102 (HR/SpO2), MPU6050 (motion), MLX90614 (body
  temperature), 16x2 character LCD, UART for telemetry

## Acknowledgements / references

MIT-BIH Arrhythmia Database (PhysioNet); AAMI EC57/EC38; de Chazal et al.
2004 (inter-patient heartbeat classification); Pan & Tompkins 1985 (QRS
detection); Hu, Shen & Sun 2018 (Squeeze-and-Excitation Networks); C-DAC
VEGA ARIES v2 / THEJAS32 SoC documentation. Full reference list in
`docs/ECG_on_VEGA_paper.pdf`.
