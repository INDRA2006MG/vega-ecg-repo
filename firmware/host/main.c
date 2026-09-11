/*
 * main.c — VEGA ARIES IoT firmware entry point.
 *
 * Built to be flashed in stages, matching the bring-up plan. Set STAGE below,
 * build, flash, check the gate, then move on. Do NOT jump straight to STAGE 5.
 *
 *   STAGE 2  replay the stored MIT-BIH vectors through the CNN
 *            GATE: logits bit-identical to the desktop `./run_real --dump`
 *   STAGE 3  stream a stored ECG segment through the R-peak detector
 *            GATE: peak indices identical to the desktop run
 *   STAGE 5  full path: detect -> cut -> normalize -> classify
 *            GATE: per-beat latency measured and well under 800 ms
 *
 * PORTING NOTE — the two SDK-specific things are isolated at the top:
 *   1. console output. The VEGA SDK normally retargets stdio to UART0, so
 *      printf() just works. If your build links without a syscall stub,
 *      replace dbg_puts()/dbg_dec() with the taurus UART calls; everything
 *      else in this file is plain C and does not change.
 *   2. the cycle counter, which uses the standard RISC-V `cycle` CSR rather
 *      than an SDK timer API, so it needs no header at all.
 */

#include <stdint.h>
#include <stdio.h>

#include "model_forward.h"
#include "rpeak.h"
#include "test_vectors.h"     /* 200 real beats + reference predictions */

#define STAGE 2

/* ---------- console ---------- */
static void dbg_puts(const char *s) { fputs(s, stdout); }
static void dbg_dec(long v)          { printf("%ld", v); }
static void dbg_nl(void)             { fputs("\n", stdout); fflush(stdout); }

/* ---------- cycle counter (standard RISC-V CSR, no SDK dependency) ---------- */
static inline uint32_t cycles(void) {
    uint32_t c;
    /* `.option arch, +zicsr` keeps this self-contained: newer GCC refuses
     * csrr unless the zicsr extension is enabled, and declaring it here
     * means the build needs no special -march flag. */
    __asm__ volatile (".option push\n"
                      ".option arch, +zicsr\n"
                      "csrr %0, cycle\n"
                      ".option pop"
                      : "=r"(c));
    return c;
}
#define CPU_HZ 100000000UL
static uint32_t cyc_to_us(uint32_t c) { return (uint32_t)(c / (CPU_HZ / 1000000UL)); }

static void banner(const char *stage) {
    dbg_nl();
    dbg_puts("=== ECG on VEGA — stage "); dbg_puts(stage); dbg_puts(" ===\n");
    dbg_puts("activation SRAM: "); dbg_dec((long)model_forward_ram_bytes());
    dbg_puts(" bytes\n");
    dbg_puts("rpeak state    : "); dbg_dec((long)sizeof(rpeak_state_t));
    dbg_puts(" bytes\n");
}

#if STAGE == 2
/* ---------------------------------------------------------------- stage 2 --
 * Replay the stored vectors. Integer math, so this is a yes/no test: the
 * logits must match the desktop build exactly. A mismatch means a real
 * portability bug (word size, shift behaviour, alignment), not rounding.
 * Capture the serial output to c_real.txt and run compare_real.py on it.
 */
int main(void) {
    banner("2 — CNN kernel");
    dbg_puts("idx cls l0 l1 l2 l3 l4\n");

    int agree = 0, correct = 0;
    uint32_t worst = 0, total = 0;

    for (int i = 0; i < NUM_TESTS; i++) {
        int16_t logits[NUM_CLASSES];
        uint32_t t0 = cycles();
        int cls = model_forward(test_inputs[i], logits);
        uint32_t dt = cycles() - t0;

        total += dt;
        if (dt > worst) worst = dt;
        if (cls == expected_class[i]) agree++;
        if (cls == true_label[i])     correct++;

        dbg_dec(i); dbg_puts(" "); dbg_dec(cls);
        for (int c = 0; c < NUM_CLASSES; c++) { dbg_puts(" "); dbg_dec(logits[c]); }
        dbg_nl();
    }

    dbg_puts("--- summary ---\n");
    dbg_puts("agreement with float model: "); dbg_dec(agree);
    dbg_puts("/"); dbg_dec(NUM_TESTS); dbg_nl();
    dbg_puts("accuracy vs annotations   : "); dbg_dec(correct);
    dbg_puts("/"); dbg_dec(NUM_TESTS); dbg_nl();
    dbg_puts("mean  us/beat: "); dbg_dec(cyc_to_us(total / NUM_TESTS)); dbg_nl();
    dbg_puts("worst us/beat: "); dbg_dec(cyc_to_us(worst)); dbg_nl();
    dbg_puts("EXPECT 198/200 and 198/200 — anything else is a port bug.\n");

    for (;;) { }
}

#elif STAGE == 3
/* ---------------------------------------------------------------- stage 3 --
 * Stream a stored ECG segment through the detector. Needs rpeak_vectors.h,
 * exported from the notebook by colab_export_rpeak_vectors.py.
 */
#include "rpeak_vectors.h"

int main(void) {
    banner("3 — R-peak detector");

    for (int r = 0; r < RPV_NUM_RECORDS; r++) {
        rpeak_state_t st;
        rpeak_init(&st);
        int n = 0;
        uint32_t t0 = cycles();
        for (int i = 0; i < RPV_LEN; i++) {
            uint32_t p;
            if (rpeak_push(&st, rpv_raw[r][i], &p)) {
                if (n < 8) { dbg_puts("  peak "); dbg_dec(p); dbg_nl(); }
                n++;
            }
        }
        uint32_t dt = cycles() - t0;
        dbg_puts("record "); dbg_puts(rpv_names[r]);
        dbg_puts(": detected "); dbg_dec(n);
        dbg_puts(" / annotated "); dbg_dec(rpv_ntrue[r]);
        dbg_puts("  ("); dbg_dec(cyc_to_us(dt) / RPV_LEN); dbg_puts(" us/sample)\n");
    }
    dbg_puts("Peak indices must match the desktop run exactly.\n");
    for (;;) { }
}

#elif STAGE == 5
/* ---------------------------------------------------------------- stage 5 --
 * Full path. Feeds stored samples in place of the ADC so the pipeline can be
 * proven before the analog front-end is trusted; swap feed_next_sample() for
 * the ADC read once this gate passes.
 */
#include "rpeak_vectors.h"

static const char *NAMES[NUM_CLASSES] = { "N", "S", "V", "F", "Q" };

int main(void) {
    banner("5 — full pipeline");

    rpeak_state_t st;
    rpeak_init(&st);

    const int16_t *sig = rpv_raw[0];
    int beats = 0;
    uint32_t worst = 0;

    for (int i = 0; i < RPV_LEN; i++) {
        uint32_t peak;
        if (!rpeak_push(&st, sig[i], &peak)) continue;

        int16_t beat[256];
        if (rpeak_extract_beat(sig, RPV_LEN, peak, beat) != 0) continue;

        uint32_t t0 = cycles();
        int cls = model_forward(beat, 0);
        uint32_t dt = cycles() - t0;
        if (dt > worst) worst = dt;

        beats++;
        dbg_puts("beat @"); dbg_dec(peak);
        dbg_puts("  -> "); dbg_puts(NAMES[cls]);
        dbg_puts("  ("); dbg_dec(cyc_to_us(dt)); dbg_puts(" us)\n");
    }

    dbg_puts("--- "); dbg_dec(beats); dbg_puts(" beats classified ---\n");
    dbg_puts("worst-case inference: "); dbg_dec(cyc_to_us(worst));
    dbg_puts(" us   (budget is 800000 us per beat)\n");
    for (;;) { }
}

#else
#error "Set STAGE to 2, 3 or 5"
#endif
