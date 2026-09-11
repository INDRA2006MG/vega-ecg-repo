#!/usr/bin/env python3
"""
Convert Keras-exported Q8.8 weights (weights_keras.h) into the C layout
used by our inference kernels.

Keras layouts                  Our layouts
-------------------------      -----------------------------
conv  w[k][in_ch][out_ch]  ->  w[out_ch][in_ch][k]
dense w[in_dim][out_dim]   ->  w[out_dim][in_dim]
bias  b[n]                 ->  b[n]   (unchanged)
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).parent / "weights_keras.h"
OUT_C = Path(__file__).parent / "weights.c"
OUT_H = Path(__file__).parent / "weights.h"

DECL = re.compile(
    r"static\s+const\s+ap_int<16>\s+(\w+)\s*((?:\[\s*\d+\s*\])+)\s*=\s*\{(.*?)\}\s*;",
    re.S,
)

# name -> kind: "conv", "dense", "bias"
KIND = {
    "w_b1": "conv", "w_b2": "conv", "w_b3": "conv",
    "w_r1a": "conv", "w_r1b": "conv", "w_r1s": "conv",
    "w_r2a": "conv", "w_r2b": "conv",
    "w_se1_fc1": "dense", "w_se1_fc2": "dense",
    "w_se2_fc1": "dense", "w_se2_fc2": "dense",
    "w_se3_fc1": "dense", "w_se3_fc2": "dense",
    "w_fc1": "dense", "w_fc2": "dense",
}

ORDER = [
    ("Branch 1 (kernel=3, 1 -> 8)",        ["w_b1", "b_b1"]),
    ("Branch 2 (kernel=5, 1 -> 8)",        ["w_b2", "b_b2"]),
    ("Branch 3 (kernel=7, 1 -> 8)",        ["w_b3", "b_b3"]),
    ("SE block 1 (8 -> 4 -> 8)",           ["w_se1_fc1", "b_se1_fc1", "w_se1_fc2", "b_se1_fc2"]),
    ("SE block 2 (8 -> 4 -> 8)",           ["w_se2_fc1", "b_se2_fc1", "w_se2_fc2", "b_se2_fc2"]),
    ("SE block 3 (8 -> 4 -> 8)",           ["w_se3_fc1", "b_se3_fc1", "w_se3_fc2", "b_se3_fc2"]),
    ("ResBlock 1 (8 -> 16, 1x1 shortcut)", ["w_r1a", "b_r1a", "w_r1b", "b_r1b", "w_r1s", "b_r1s"]),
    ("ResBlock 2 (16 -> 16, identity)",    ["w_r2a", "b_r2a", "w_r2b", "b_r2b"]),
    ("Classifier",                         ["w_fc1", "b_fc1", "w_fc2", "b_fc2"]),
]


def parse(text):
    arrays = {}
    for m in DECL.finditer(text):
        name = m.group(1)
        shape = [int(d) for d in re.findall(r"\d+", m.group(2))]
        body = re.sub(r"//[^\n]*", "", m.group(3))
        vals = [int(v) for v in re.findall(r"-?\d+", body)]
        n = 1
        for d in shape:
            n *= d
        if len(vals) != n:
            sys.exit(f"{name}: shape {shape} needs {n} values, found {len(vals)}")
        arrays[name] = (shape, vals)
    return arrays


def reshape(vals, shape):
    """Row-major nested list."""
    if len(shape) == 1:
        return list(vals)
    step = 1
    for d in shape[1:]:
        step *= d
    return [reshape(vals[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def flatten(x):
    if not isinstance(x, list):
        return [x]
    out = []
    for e in x:
        out.extend(flatten(e))
    return out


def convert(name, shape, vals):
    """Return (new_shape, nested_values)."""
    kind = KIND.get(name, "bias")
    a = reshape(vals, shape)
    if kind == "conv":
        k, ci, co = shape
        b = [[[a[kk][ii][oo] for kk in range(k)] for ii in range(ci)] for oo in range(co)]
        return [co, ci, k], b
    if kind == "dense":
        di, do = shape
        b = [[a[ii][oo] for ii in range(di)] for oo in range(do)]
        return [do, di], b
    return shape, a


def emit(name, shape, nested, per_line=16):
    """Fully-braced C initializer (no -Wmissing-braces noise)."""
    dims = "".join(f"[{d}]" for d in shape)
    lines = [f"const int16_t {name}{dims} = {{"]

    def row(vals, indent):
        pad = " " * indent
        if len(vals) <= per_line:
            return [pad + "{ " + " ".join(f"{v}," for v in vals) + " }"]
        out = [pad + "{"]
        for i in range(0, len(vals), per_line):
            out.append(pad + "    " + " ".join(f"{v}," for v in vals[i:i + per_line]))
        out.append(pad + "}")
        return out

    if len(shape) == 1:
        flat = flatten(nested)
        for i in range(0, len(flat), per_line):
            lines.append("    " + " ".join(f"{v}," for v in flat[i:i + per_line]))
    elif len(shape) == 2:
        for o, r in enumerate(nested):
            body = row(r, 4)
            body[-1] += ","
            lines.extend(body)
    else:  # [out_ch][in_ch][k]
        for o, oc in enumerate(nested):
            lines.append(f"    /* out {o} */")
            lines.append("    {")
            for ic in oc:
                body = row(ic, 8)
                body[-1] += ","
                lines.extend(body)
            lines.append("    },")
    lines.append("};")
    return "\n".join(lines)


def main():
    text = SRC.read_text()
    arrays = parse(text)

    missing = [n for _, g in ORDER for n in g if n not in arrays]
    if missing:
        sys.exit(f"missing arrays: {missing}")

    c_body = []
    h_body = []
    total = 0
    report = []

    for title, group in ORDER:
        c_body.append(f"\n/* ---- {title} ---- */")
        h_body.append(f"\n/* ---- {title} ---- */")
        for name in group:
            shape, vals = arrays[name]
            new_shape, nested = convert(name, shape, vals)

            # verification: element-wise transpose check
            a = reshape(vals, shape)
            kind = KIND.get(name, "bias")
            if kind == "conv":
                k, ci, co = shape
                for oo in range(co):
                    for ii in range(ci):
                        for kk in range(k):
                            assert nested[oo][ii][kk] == a[kk][ii][oo], name
            elif kind == "dense":
                di, do = shape
                for oo in range(do):
                    for ii in range(di):
                        assert nested[oo][ii] == a[ii][oo], name
            else:
                assert flatten(nested) == list(vals), name
            assert sorted(flatten(nested)) == sorted(vals), name

            n = len(vals)
            total += n
            report.append((name, shape, new_shape, n))

            dims = "".join(f"[{d}]" for d in new_shape)
            h_body.append(f"extern const int16_t {name}{dims};")
            c_body.append("")
            c_body.append(f"/* {name}: keras{shape} -> {new_shape} */")
            c_body.append(emit(name, new_shape, nested))

    header = (
        "/* Auto-generated by convert_weights.py — do not edit by hand.\n"
        " * Multi-Scale SE 1D-CNN (TINY), Q8.8 fixed point (scale = 256), BN folded.\n"
        " * Conv weights are [out_ch][in_ch][k]; dense weights are [out_dim][in_dim].\n"
        " */\n"
    )

    OUT_C.write_text(header + '#include "weights.h"\n' + "\n".join(c_body) + "\n")
    OUT_H.write_text(
        header
        + "#ifndef ECG_WEIGHTS_H\n#define ECG_WEIGHTS_H\n\n#include <stdint.h>\n\n"
        + "#define Q_FRAC_BITS 8\n#define Q_SCALE     256\n\n"
        + "\n".join(h_body)
        + "\n\n#endif /* ECG_WEIGHTS_H */\n"
    )

    w = max(len(r[0]) for r in report)
    print(f"{'array'.ljust(w)}  keras shape      ->  our shape        count")
    for name, s, ns, n in report:
        print(f"{name.ljust(w)}  {str(s).ljust(15)} ->  {str(ns).ljust(15)}  {n}")
    print(f"\ntotal parameters: {total}")
    print("all transpose assertions passed")


if __name__ == "__main__":
    main()
