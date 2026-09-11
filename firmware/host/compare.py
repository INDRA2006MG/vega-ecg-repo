#!/usr/bin/env python3
"""Compare the C kernel against the NumPy fixed-point and float references."""

import numpy as np
from ref_model import load_weights, forward_fixed, forward_float

W = load_weights()
X = np.load("test_inputs_q88.npy")

c_rows = [line.split() for line in open("c_out.txt") if line.strip()]
c_cls = np.array([int(r[1]) for r in c_rows])
c_logits = np.array([[int(v) for v in r[2:]] for r in c_rows])

fx_cls, fx_logits = [], []
fl_cls, fl_logits = [], []
for x in X:
    c, l = forward_fixed(x.astype(np.int16), W)
    fx_cls.append(c); fx_logits.append(l)
    c, l = forward_float(x.astype(np.int16), W)
    fl_cls.append(c); fl_logits.append(l)

fx_cls = np.array(fx_cls); fx_logits = np.array(fx_logits)
fl_cls = np.array(fl_cls); fl_logits = np.array(fl_logits)

print("=" * 68)
print("A. C kernel  vs  NumPy fixed-point simulator  (must be EXACT)")
print("=" * 68)
logit_match = np.array_equal(c_logits, fx_logits)
cls_match = np.array_equal(c_cls, fx_cls)
print(f"  class agreement : {(c_cls == fx_cls).sum()}/{len(c_cls)}")
print(f"  logits identical: {logit_match}")
if not logit_match:
    d = np.abs(c_logits - fx_logits)
    print(f"  max |diff| = {d.max()}   mismatching vectors = {(d.max(1) > 0).sum()}")
    for i in np.where(d.max(1) > 0)[0][:5]:
        print(f"    [{i}] C={c_logits[i]}  py={fx_logits[i]}")
print(f"  => {'PASS: C faithfully implements the intended fixed-point math' if (logit_match and cls_match) else 'FAIL: C and the simulator disagree'}")

print()
print("=" * 68)
print("B. Fixed-point  vs  float (same Q8.8 weights, float activations)")
print("   -- isolates drift from the fixed-point activation path")
print("=" * 68)
agree = (fx_cls == fl_cls).sum()
print(f"  class agreement : {agree}/{len(fx_cls)}  ({100*agree/len(fx_cls):.1f}%)")
fl_q = fl_logits * 256.0
err = np.abs(fx_logits - fl_q)
print(f"  logit abs error (Q8.8 LSBs): mean {err.mean():.1f}  max {err.max():.1f}")
print(f"  logit abs error (real units): mean {err.mean()/256:.4f}  max {err.max()/256:.4f}")

margin = []
for i in range(len(fl_logits)):
    s = np.sort(fl_logits[i])[::-1]
    margin.append(s[0] - s[1])
margin = np.array(margin)
print(f"  float top-1 margin (real units): mean {margin.mean():.3f}  min {margin.min():.3f}")

dis = np.where(fx_cls != fl_cls)[0]
if len(dis):
    print(f"  disagreeing vectors: {dis.tolist()}")
    for i in dis[:5]:
        print(f"    [{i}] fixed={fx_cls[i]} float={fl_cls[i]}  float margin={margin[i]:.4f}")
else:
    print("  no disagreements")

print()
print("=" * 68)
print("Class distribution over these synthetic inputs")
print("=" * 68)
names = ["Normal(N)", "SVEB(S)", "VEB(V)", "Fusion(F)", "Unknown(Q)"]
for c in range(5):
    print(f"  {c} {names[c]:12s} fixed={np.sum(fx_cls==c):3d}  float={np.sum(fl_cls==c):3d}")
