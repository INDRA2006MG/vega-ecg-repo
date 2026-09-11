#!/usr/bin/env python3
"""
Reference implementations of MultiScale_SE_CNN_Tiny, used to verify the C port.

Two references are built from the SAME Q8.8 weights that weights.c holds:

  forward_float(x)  -- dequantized weights (q/256), float32 activations.
                       This is the "intended math" of the architecture.
  forward_fixed(x)  -- integer simulator mirroring model_forward.c exactly
                       (int64 accumulators, round-half-up >> 8, int16 clamp).

If the C binary agrees with forward_fixed bit-for-bit, the C code faithfully
implements what we intended. Comparing forward_fixed against forward_float
then isolates how much the fixed-point activation path drifts, independent of
any C bug.

Note this does NOT prove agreement with the original float *Keras* model --
these weights are already Q8.8-rounded. That last comparison needs either
test_vectors.h or best_ecg_model_tiny.h5 from the training notebook.
"""

import re
import numpy as np
from pathlib import Path

SRC = Path(__file__).parent / "weights_keras.h"

DECL = re.compile(
    r"static\s+const\s+ap_int<16>\s+(\w+)\s*((?:\[\s*\d+\s*\])+)\s*=\s*\{(.*?)\}\s*;",
    re.S,
)


def load_weights():
    text = SRC.read_text()
    W = {}
    for m in DECL.finditer(text):
        name = m.group(1)
        shape = tuple(int(d) for d in re.findall(r"\d+", m.group(2)))
        body = re.sub(r"//[^\n]*", "", m.group(3))
        vals = np.array([int(v) for v in re.findall(r"-?\d+", body)], dtype=np.int64)
        assert vals.size == int(np.prod(shape)), f"{name}: {vals.size} vs {shape}"
        W[name] = vals.reshape(shape)
    return W


# ----------------------------------------------------------------- float ref

def _conv_same_float(x, w, b, dilation=1):
    """x: [T, Cin]; w (keras layout): [k, Cin, Cout]; b: [Cout] -> [T, Cout]"""
    T, Cin = x.shape
    k, Cin_w, Cout = w.shape
    assert Cin == Cin_w
    pad_left = (dilation * (k - 1)) // 2
    out = np.zeros((T, Cout), dtype=np.float64)
    out += b
    for kk in range(k):
        for t in range(T):
            it = t - pad_left + kk * dilation
            if 0 <= it < T:
                out[t] += x[it] @ w[kk]
    return out


def _hard_sigmoid_float(v):
    return np.clip(0.2 * v + 0.5, 0.0, 1.0)


def _se_float(x, w1, b1, w2, b2):
    s = x.mean(axis=0)                      # GlobalAveragePooling1D
    h = np.maximum(s @ w1 + b1, 0.0)        # Dense + relu
    e = _hard_sigmoid_float(h @ w2 + b2)    # Dense + hard_sigmoid
    return x * e


def forward_float(x_q88, W):
    """x_q88: int16 array [256]. Returns (class, logits_float)."""
    d = {k: v.astype(np.float64) / 256.0 for k, v in W.items()}
    x = (x_q88.astype(np.float64) / 256.0).reshape(-1, 1)

    b1 = np.maximum(_conv_same_float(x, d["w_b1"], d["b_b1"], 1), 0)
    b2 = np.maximum(_conv_same_float(x, d["w_b2"], d["b_b2"], 1), 0)
    b3 = np.maximum(_conv_same_float(x, d["w_b3"], d["b_b3"], 2), 0)

    b1 = _se_float(b1, d["w_se1_fc1"], d["b_se1_fc1"], d["w_se1_fc2"], d["b_se1_fc2"])
    b2 = _se_float(b2, d["w_se2_fc1"], d["b_se2_fc1"], d["w_se2_fc2"], d["b_se2_fc2"])
    b3 = _se_float(b3, d["w_se3_fc1"], d["b_se3_fc1"], d["w_se3_fc2"], d["b_se3_fc2"])

    fused = b1 + b2 + b3

    # ResBlock1: 8 -> 16, 1x1 shortcut
    a = np.maximum(_conv_same_float(fused, d["w_r1a"], d["b_r1a"]), 0)
    bb = _conv_same_float(a, d["w_r1b"], d["b_r1b"])
    sc = _conv_same_float(fused, d["w_r1s"], d["b_r1s"])
    x1 = np.maximum(bb + sc, 0)
    x1 = np.max(x1.reshape(-1, 2, 16), axis=1)     # MaxPool1D(2): 256 -> 128

    # ResBlock2: 16 -> 16, identity shortcut
    a = np.maximum(_conv_same_float(x1, d["w_r2a"], d["b_r2a"]), 0)
    bb = _conv_same_float(a, d["w_r2b"], d["b_r2b"])
    x2 = np.maximum(bb + x1, 0)
    x2 = np.max(x2.reshape(-1, 2, 16), axis=1)     # MaxPool1D(2): 128 -> 64

    g = x2.mean(axis=0)                            # GlobalAveragePooling1D
    f1 = np.maximum(g @ d["w_fc1"] + d["b_fc1"], 0)
    logits = f1 @ d["w_fc2"] + d["b_fc2"]
    return int(np.argmax(logits)), logits


# ----------------------------------------------------------------- fixed ref

def _requant(acc):
    """Mirror of requant() in model_forward.c: round-half-up, clamp int16."""
    r = (acc + 128) >> 8
    return np.clip(r, -32768, 32767).astype(np.int64)


def _conv_same_fixed(x, w, b, dilation=1):
    """x: int64 [T, Cin]; w keras layout [k, Cin, Cout]; b [Cout]"""
    T, Cin = x.shape
    k, _, Cout = w.shape
    pad_left = (dilation * (k - 1)) // 2
    acc = np.zeros((T, Cout), dtype=np.int64)
    acc += (b.astype(np.int64) << 8)
    for kk in range(k):
        for t in range(T):
            it = t - pad_left + kk * dilation
            if 0 <= it < T:
                acc[t] += x[it] @ w[kk].astype(np.int64)
    return _requant(acc)


def _gap_fixed(x):
    """Mirror global_avg_pool(): round-half-away-from-zero via trunc division."""
    n = x.shape[0]
    s = x.sum(axis=0)
    r = np.where(s >= 0, s + n // 2, s - n // 2)
    q = np.trunc(r / n).astype(np.int64)   # C integer division truncates toward 0
    return np.clip(q, -32768, 32767)


def _dense_fixed(x, w, b):
    acc = (b.astype(np.int64) << 8) + x @ w.astype(np.int64)
    return _requant(acc)


def _hard_sigmoid_fixed(v):
    lin = _requant(v * 51) + 128
    return np.clip(lin, 0, 256)


def _se_fixed(x, w1, b1, w2, b2):
    s = _gap_fixed(x)
    h = np.maximum(_dense_fixed(s, w1, b1), 0)
    e = _hard_sigmoid_fixed(_dense_fixed(h, w2, b2))
    return _requant(x * e)


def _clampadd(a, b):
    return np.clip(a + b, -32768, 32767)


def forward_fixed(x_q88, W):
    """x_q88: int16 [256]. Returns (class, logits_int)."""
    x = x_q88.astype(np.int64).reshape(-1, 1)

    b1 = np.maximum(_conv_same_fixed(x, W["w_b1"], W["b_b1"], 1), 0)
    b2 = np.maximum(_conv_same_fixed(x, W["w_b2"], W["b_b2"], 1), 0)
    b3 = np.maximum(_conv_same_fixed(x, W["w_b3"], W["b_b3"], 2), 0)

    b1 = _se_fixed(b1, W["w_se1_fc1"], W["b_se1_fc1"], W["w_se1_fc2"], W["b_se1_fc2"])
    b2 = _se_fixed(b2, W["w_se2_fc1"], W["b_se2_fc1"], W["w_se2_fc2"], W["b_se2_fc2"])
    b3 = _se_fixed(b3, W["w_se3_fc1"], W["b_se3_fc1"], W["w_se3_fc2"], W["b_se3_fc2"])

    fused = _clampadd(_clampadd(b1, b2), b3)

    a = np.maximum(_conv_same_fixed(fused, W["w_r1a"], W["b_r1a"]), 0)
    bb = _conv_same_fixed(a, W["w_r1b"], W["b_r1b"])
    sc = _conv_same_fixed(fused, W["w_r1s"], W["b_r1s"])
    x1 = np.maximum(_clampadd(bb, sc), 0)
    x1 = np.max(x1.reshape(-1, 2, 16), axis=1)

    a = np.maximum(_conv_same_fixed(x1, W["w_r2a"], W["b_r2a"]), 0)
    bb = _conv_same_fixed(a, W["w_r2b"], W["b_r2b"])
    x2 = np.maximum(_clampadd(bb, x1), 0)
    x2 = np.max(x2.reshape(-1, 2, 16), axis=1)

    g = _gap_fixed(x2)
    f1 = np.maximum(_dense_fixed(g, W["w_fc1"], W["b_fc1"]), 0)
    logits = _dense_fixed(f1, W["w_fc2"], W["b_fc2"])
    return int(np.argmax(logits)), logits
