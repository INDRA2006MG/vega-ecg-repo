/*
 * model_forward.c — Q8.8 fixed-point inference for MultiScale_SE_CNN_Tiny.
 *
 * Ported directly from the Keras definition (Ecg_2.ipynb, build_model_tiny /
 * se_block / residual_block):
 *
 *   b1 = ReLU(BN(Conv1D(8, k=3,            padding=same)(x)))
 *   b2 = ReLU(BN(Conv1D(8, k=5,            padding=same)(x)))
 *   b3 = ReLU(BN(Conv1D(8, k=7, dilation=2, padding=same)(x)))
 *   b1,b2,b3 = se_block(., ratio=2)   # squeeze 8->4, excite 4->8, hard_sigmoid
 *   fused = b1 + b2 + b3              # Add(), NOT concat
 *   x = residual_block(fused, filters=16, k=3)   # 8->16, 1x1 shortcut conv
 *   x = MaxPool1D(2)(x)                            # 256 -> 128
 *   x = residual_block(x, filters=16, k=3)         # 16->16, identity shortcut
 *   x = MaxPool1D(2)(x)                            # 128 -> 64
 *
 * Each MaxPool is fused into its residual block (residual_block_pooled) so the
 * full-length block output never needs a buffer -- see the RAM note there.
 *   x = GlobalAveragePooling1D()(x)                # 64x16 -> 16
 *   x = ReLU(Dense(16)(x))
 *   out = Dense(5)(x)                              # softmax at training time
 *                                                   # only; argmax is enough here
 *
 * BatchNorm is already folded into the branch/resblock conv weights (done at
 * export time — see weights.c). All activations are Q8.8 (value = round(real
 * * 256)), matching how weights.h / weights.c were quantized.
 *
 * Every multiply-accumulate uses an int64_t accumulator specifically to avoid
 * the inter-layer overflow bug that hit the earlier PYNQ/HLS build (that used
 * narrower accumulators and needed ad-hoc extra right-shifts as a workaround).
 * Requantization here is a single, uniform round-half-up ">> 8" after each
 * layer's accumulation — no per-layer shift tuning needed.
 *
 * PERF NOTE (found after ESP32 hardware testing showed ~99ms/beat inference,
 * vs the ~890us this file was written against on VEGA): every MAC below is
 * written as `acc += (int64_t)a * b`, which C promotes to a 64x64->64
 * multiply, not a 64x16->64 one — casting ONE operand to int64_t forces the
 * OTHER operand to be promoted to int64_t too before the multiply happens.
 * Xtensa LX6 (the ESP32's core) has no hardware 64-bit multiplier, so each
 * one of those lowers to a libgcc __multdi3 call emulating it out of several
 * 32-bit multiplies plus a real function-call - that's the ~100x slowdown,
 * not the model or the activation buffers.
 *
 * The actual operands are always int16_t (Q8.8), so their product always
 * fits safely in int32_t (max |32767*32767| ~= 1.07e9, well under INT32_MAX
 * ~= 2.15e9) - only SUMMING many such products can overflow int32_t, which
 * is what the int64_t accumulator is actually for. So every MAC below now
 * computes the product as a native int32_t*int32_t (a single-cycle hardware
 * MUL on Xtensa) and only widens to int64_t on the add into the accumulator
 * - same math, same overflow safety, no emulated 64-bit multiply anywhere.
 */

/* Arduino-ESP32's default build optimizes the whole sketch for size (-Os),
 * not speed, and the Tools menu doesn't expose a way to change that
 * project-wide. This is the one file where that actually costs real time -
 * ~500K MACs per beat (see the loop structure below) is exactly what -Os's
 * lighter inlining/unrolling hurts most. This pragma overrides the
 * optimization level for JUST this translation unit, independent of
 * whatever the rest of the sketch builds with - no IDE/platform.txt/
 * boards.txt changes needed. rpeak.c is untouched on purpose: it's already
 * fast enough (a few us/sample) that the size tradeoff isn't worth it there. */
#pragma GCC optimize("O2")

#include "model_forward.h"
#include "weights.h"

#include <string.h>
#include <limits.h>

#define FRAC_BITS 8
#define ONE_Q88   256      /* 1.0 in Q8.8 */

#define BRANCH_CH 8
#define SE_MID    4        /* BRANCH_CH / ratio(2) */
#define RES_CH    16
#define FC1_DIM   16

/* ---- small helpers ---- */

static int16_t clamp_i16(int64_t v) {
    if (v > INT16_MAX) return INT16_MAX;
    if (v < INT16_MIN) return INT16_MIN;
    return (int16_t)v;
}

/* Requantize a Q16.16 accumulator back to Q8.8.
 *
 * Round-half-up: add half an LSB, then arithmetic-shift (which floors).
 * Do NOT special-case negatives by subtracting half instead — flooring
 * already rounds them down, so subtracting first biases every negative
 * activation a full LSB low (acc = -256 would yield -2 instead of -1). */
static int16_t requant(int64_t acc) {
    return clamp_i16((acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS);
}

static void relu_inplace(int16_t *x, int n) {
    for (int i = 0; i < n; i++) if (x[i] < 0) x[i] = 0;
}

/* Keras hard_sigmoid: clip(0.2*x + 0.5, 0, 1) */
static int16_t hard_sigmoid_q88(int16_t x_q) {
    /* 0.2 in Q8.8 = 51 (round(0.2*256) = 51.2 -> 51) */
    int64_t acc = (int32_t)x_q * (int32_t)51;   /* native 32-bit multiply, see file header perf note */
    int32_t lin = (int32_t)requant(acc) + (ONE_Q88 / 2); /* + 0.5 */
    if (lin < 0) lin = 0;
    if (lin > ONE_Q88) lin = ONE_Q88;
    return (int16_t)lin;
}

/*
 * Generic 'same'-padding, stride-1 1D conv.
 * in:  [in_len][in_ch]   (time-major, channel-minor)
 * w:   [out_ch][in_ch][k]
 * b:   [out_ch]
 * out: [in_len][out_ch]  (length unchanged — 'same' padding)
 */
static void conv1d_same(const int16_t *in, int in_len, int in_ch,
                         int out_ch, int k, int dilation,
                         const int16_t *w, const int16_t *b,
                         int16_t *out) {
    int pad_left = (dilation * (k - 1)) / 2;

    for (int t = 0; t < in_len; t++) {
        for (int oc = 0; oc < out_ch; oc++) {
            int64_t acc = (int64_t)b[oc] << FRAC_BITS; /* promote Q8.8 bias to Q16.16 */
            for (int ic = 0; ic < in_ch; ic++) {
                const int16_t *wk = w + ((size_t)oc * in_ch + ic) * k;
                for (int kk = 0; kk < k; kk++) {
                    int it = t - pad_left + kk * dilation;
                    if (it < 0 || it >= in_len) continue;
                    int16_t xv = in[(size_t)it * in_ch + ic];
                    acc += (int32_t)xv * (int32_t)wk[kk];   /* native 32-bit multiply, see file header perf note */
                }
            }
            out[(size_t)t * out_ch + oc] = requant(acc);
        }
    }
}

/* Dense: in[in_dim], w[out_dim][in_dim], b[out_dim] -> out[out_dim] */
static void dense(const int16_t *in, int in_dim, int out_dim,
                   const int16_t *w, const int16_t *b, int16_t *out) {
    for (int od = 0; od < out_dim; od++) {
        int64_t acc = (int64_t)b[od] << FRAC_BITS;
        const int16_t *wr = w + (size_t)od * in_dim;
        for (int id = 0; id < in_dim; id++)
            acc += (int32_t)in[id] * (int32_t)wr[id];   /* native 32-bit multiply, see file header perf note */
        out[od] = requant(acc);
    }
}

/* GlobalAveragePooling1D: in[len][ch] -> out[ch] (plain average, no rescale) */
static void global_avg_pool(const int16_t *in, int len, int ch, int16_t *out) {
    for (int c = 0; c < ch; c++) {
        int64_t sum = 0;
        for (int t = 0; t < len; t++) sum += in[(size_t)t * ch + c];
        int64_t rounded = (sum >= 0) ? (sum + len / 2) : (sum - len / 2);
        out[c] = clamp_i16(rounded / len);
    }
}

static void add_inplace(int16_t *dst, const int16_t *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = clamp_i16((int64_t)dst[i] + src[i]);
}

/*
 * se_block: squeeze (GAP) -> fc1+ReLU -> fc2+hard_sigmoid -> channel-wise
 * scale of x in place. x: [len][ch], ch == BRANCH_CH here.
 */
static void se_block(int16_t *x, int len, int ch,
                      const int16_t *w_fc1, const int16_t *b_fc1,
                      const int16_t *w_fc2, const int16_t *b_fc2) {
    int16_t squeeze[BRANCH_CH];
    int16_t hidden[SE_MID];
    int16_t scale[BRANCH_CH];

    global_avg_pool(x, len, ch, squeeze);
    dense(squeeze, ch, SE_MID, w_fc1, b_fc1, hidden);
    relu_inplace(hidden, SE_MID);
    dense(hidden, SE_MID, ch, w_fc2, b_fc2, scale);
    for (int c = 0; c < ch; c++) scale[c] = hard_sigmoid_q88(scale[c]);

    for (int t = 0; t < len; t++)
        for (int c = 0; c < ch; c++) {
            int64_t prod = (int32_t)x[(size_t)t * ch + c] * (int32_t)scale[c];   /* native 32-bit multiply, see file header perf note */
            x[(size_t)t * ch + c] = requant(prod);
        }
}

/*
 * residual_block_pooled: the residual block with its trailing MaxPool1D(2)
 * fused in, so the full-length [len][16] block output is never materialized.
 *
 *   conv_a -> ReLU -> conv_b, plus shortcut (1x1 conv if the channel count
 *   changes, identity otherwise), then add -> ReLU -> maxpool(2).
 *
 * ReLU and max commute (both monotonic), so pooling before the ReLU gives
 * identical results and halves the work of the ReLU pass.
 *
 * Buffer discipline, which is what keeps RAM down:
 *   scratch_a  <- conv_a output, consumed by conv_b
 *   scratch_b  <- conv_b output
 *   scratch_a  <- REUSED for the projection shortcut (conv_a's output is
 *                 dead by then), so no third full-length buffer is needed
 *   out        <- [len/2][RES_CH], written last; may not alias `in`
 *
 * in:  [len][in_ch]
 * out: [len/2][RES_CH]
 */
static void residual_block_pooled(const int16_t *in, int len, int in_ch,
                                   const int16_t *w_a, const int16_t *b_a,
                                   const int16_t *w_b, const int16_t *b_b,
                                   const int16_t *w_s /* NULL if identity */,
                                   const int16_t *b_s,
                                   int16_t *out,
                                   int16_t *scratch_a, int16_t *scratch_b) {
    conv1d_same(in, len, in_ch, RES_CH, 3, 1, w_a, b_a, scratch_a);
    relu_inplace(scratch_a, len * RES_CH);
    conv1d_same(scratch_a, len, RES_CH, RES_CH, 3, 1, w_b, b_b, scratch_b);

    const int16_t *shortcut;
    if (w_s != NULL) {
        conv1d_same(in, len, in_ch, RES_CH, 1, 1, w_s, b_s, scratch_a);
        shortcut = scratch_a;
    } else {
        shortcut = in; /* identity: in_ch == RES_CH already */
    }

    for (int t = 0; t < len / 2; t++) {
        for (int c = 0; c < RES_CH; c++) {
            size_t i0 = (size_t)(2 * t) * RES_CH + c;
            size_t i1 = (size_t)(2 * t + 1) * RES_CH + c;
            int16_t v0 = clamp_i16((int64_t)scratch_b[i0] + shortcut[i0]);
            int16_t v1 = clamp_i16((int64_t)scratch_b[i1] + shortcut[i1]);
            int16_t m = (v0 > v1) ? v0 : v1;
            out[(size_t)t * RES_CH + c] = (m < 0) ? 0 : m;
        }
    }
}

/* ---- static activation buffers (see model_forward_ram_bytes()) ----
 *
 * `fused` doubles as the ResBlock1 input: the three branches are computed one
 * at a time into `branch` and accumulated into `fused`, so only one branch
 * buffer is live rather than three. `scratch_a`/`scratch_b` are shared by both
 * residual blocks. Fusing each MaxPool into its residual block removes the
 * two full-length [256][16] block-output buffers entirely.
 */
static int16_t branch[INPUT_LEN * BRANCH_CH];
static int16_t fused[INPUT_LEN * BRANCH_CH];
static int16_t scratch_a[INPUT_LEN * RES_CH];
static int16_t scratch_b[INPUT_LEN * RES_CH];
static int16_t pool1[(INPUT_LEN / 2) * RES_CH];
static int16_t pool2[(INPUT_LEN / 4) * RES_CH];

/* One branch: conv -> ReLU -> SE, accumulated into `fused`. */
static void branch_into_fused(const int16_t *in, int k, int dilation,
                               const int16_t *w_c, const int16_t *b_c,
                               const int16_t *w_f1, const int16_t *b_f1,
                               const int16_t *w_f2, const int16_t *b_f2,
                               int first) {
    conv1d_same(in, INPUT_LEN, 1, BRANCH_CH, k, dilation, w_c, b_c, branch);
    relu_inplace(branch, INPUT_LEN * BRANCH_CH);
    se_block(branch, INPUT_LEN, BRANCH_CH, w_f1, b_f1, w_f2, b_f2);
    if (first)
        memcpy(fused, branch, sizeof(branch));
    else
        add_inplace(fused, branch, INPUT_LEN * BRANCH_CH);
}

int model_forward(const int16_t in[INPUT_LEN], int16_t logits[NUM_CLASSES]) {
    /* ---- three parallel branches, fused by Add() ---- */
    branch_into_fused(in, 3, 1, (const int16_t *)w_b1, b_b1,
                      (const int16_t *)w_se1_fc1, b_se1_fc1,
                      (const int16_t *)w_se1_fc2, b_se1_fc2, 1);
    branch_into_fused(in, 5, 1, (const int16_t *)w_b2, b_b2,
                      (const int16_t *)w_se2_fc1, b_se2_fc1,
                      (const int16_t *)w_se2_fc2, b_se2_fc2, 0);
    branch_into_fused(in, 7, 2, (const int16_t *)w_b3, b_b3,
                      (const int16_t *)w_se3_fc1, b_se3_fc1,
                      (const int16_t *)w_se3_fc2, b_se3_fc2, 0);

    /* ---- ResBlock1 (8 -> 16, 1x1 projection shortcut) + MaxPool: 256 -> 128 ---- */
    residual_block_pooled(fused, INPUT_LEN, BRANCH_CH,
                          (const int16_t *)w_r1a, b_r1a,
                          (const int16_t *)w_r1b, b_r1b,
                          (const int16_t *)w_r1s, b_r1s,
                          pool1, scratch_a, scratch_b);

    /* ---- ResBlock2 (16 -> 16, identity shortcut) + MaxPool: 128 -> 64 ---- */
    residual_block_pooled(pool1, INPUT_LEN / 2, RES_CH,
                          (const int16_t *)w_r2a, b_r2a,
                          (const int16_t *)w_r2b, b_r2b,
                          NULL, NULL,
                          pool2, scratch_a, scratch_b);

    /* ---- classifier head ---- */
    int16_t gap[RES_CH];
    global_avg_pool(pool2, INPUT_LEN / 4, RES_CH, gap);

    int16_t fc1_out[FC1_DIM];
    dense(gap, RES_CH, FC1_DIM, (const int16_t *)w_fc1, b_fc1, fc1_out);
    relu_inplace(fc1_out, FC1_DIM);

    int16_t out_logits[NUM_CLASSES];
    dense(fc1_out, FC1_DIM, NUM_CLASSES, (const int16_t *)w_fc2, b_fc2, out_logits);

    if (logits) memcpy(logits, out_logits, sizeof(out_logits));

    int best = 0;
    for (int c = 1; c < NUM_CLASSES; c++)
        if (out_logits[c] > out_logits[best]) best = c;
    return best;
}

size_t model_forward_ram_bytes(void) {
    return sizeof(branch) + sizeof(fused)
         + sizeof(scratch_a) + sizeof(scratch_b)
         + sizeof(pool1) + sizeof(pool2);
}
