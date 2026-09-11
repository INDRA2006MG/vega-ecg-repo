#ifndef RPEAK_H
#define RPEAK_H

#include <stdint.h>
#include <stddef.h>

/*
 * Streaming fixed-point R-peak detector (Pan-Tompkins style) for the VEGA port.
 *
 * Ported from the validated Python reference (Untitled0_1.ipynb), which scored
 * 99.78% sensitivity / 100% PPV on MIT-BIH record 100. Two things had to change
 * for embedded use, and both matter:
 *
 *  1. The Python filters with `filtfilt` (zero-phase, needs the whole record).
 *     An MCU sees samples one at a time, so this uses a CAUSAL cascaded-biquad
 *     bandpass instead.
 *  2. The Python integrates with `np.convolve(..., mode='same')`, which centres
 *     the window and so adds no delay. A streaming moving-window integrator is
 *     inherently causal and lags by ~half its width.
 *
 * Together those introduce ~121 ms of lag, which is MORE than the +-100 ms
 * refinement window. So the refinement window is centred at
 * (detection_index - RPEAK_GROUP_DELAY), not at the detection index. Get this
 * wrong and every reported peak lands ~120 ms late, which then mis-centres every
 * 256-sample beat window fed to the CNN.
 */

#define RPEAK_FS            360     /* Hz - MIT-BIH sampling rate */
#define RPEAK_MWI_WIN       54      /* 150 ms moving-window integration */
#define RPEAK_REFRACTORY    72      /* 200 ms - no two peaks closer than this */
#define RPEAK_REFINE_WIN    36      /* +-100 ms search on the raw signal */
#define RPEAK_LEARN_SAMPLES (2 * RPEAK_FS)   /* 2 s threshold-learning phase */

/* Total lag of bandpass (~15) + derivative (2) + MWI ((54-1)/2 ~ 27) samples. */
#define RPEAK_GROUP_DELAY   44

/* Raw history must cover the delay plus the refinement window, with margin. */
#define RPEAK_RAW_HIST      256

/* Beat window layout used by BOTH rpeak_extract_beat() and
 * rpeak_extract_beat_ctx(): PRE samples before the R peak, POST after,
 * PRE+POST == 256 (the CNN's fixed input width). Exposed here (rather than
 * kept as a local const in each function, which is how it used to be) so
 * callers that need to know WHERE in the 256-sample window the R peak sits
 * - e.g. the QRS-width estimator below - have one source of truth instead
 * of a second copy of "90" that could silently drift out of sync. */
#define RPEAK_BEAT_PRE      90
#define RPEAK_BEAT_POST     166

typedef struct {
    /* cascaded biquad state (direct form II transposed), Q30 coefficients */
    int64_t s1[2], s2[2];

    int32_t deriv_hist[5];      /* last 5 bandpassed samples */
    int     deriv_n;

    int64_t mwi_buf[RPEAK_MWI_WIN];
    int     mwi_pos;
    int64_t mwi_sum;

    int16_t raw[RPEAK_RAW_HIST];    /* circular history of raw input */

    int64_t prev_int, prev_prev_int; /* for the 3-point local-max test */

    int64_t threshold;
    int64_t learn_max;
    int     learning;

    uint32_t n;                  /* samples consumed so far */
    uint32_t last_peak;          /* absolute index of the last accepted peak */
    int      have_peak;
} rpeak_state_t;

void rpeak_init(rpeak_state_t *st);

/*
 * Feed one raw sample. Returns 1 if a peak was confirmed, and writes its
 * ABSOLUTE sample index (into the original stream) to *peak_index.
 *
 * Note the peak is reported with roughly RPEAK_GROUP_DELAY + a few samples of
 * latency - it refers to a sample already in the past, which is exactly what
 * lets the 256-sample beat window be cut around it.
 */
int rpeak_push(rpeak_state_t *st, int16_t sample, uint32_t *peak_index);

/*
 * Batch helper: run the detector over a whole buffer. Returns the number of
 * peaks written to out_idx (capped at max_out).
 */
size_t rpeak_detect(const int16_t *sig, size_t n,
                    uint32_t *out_idx, size_t max_out);

/*
 * Cut the 256-sample beat window the CNN expects: 90 samples before the peak,
 * 166 after, then min-max normalize to [-1,1] in Q8.8 - matching the training
 * pipeline's normalize_beat(). Returns 0 on success, -1 if the window would run
 * off either end of the buffer.
 */
int rpeak_extract_beat(const int16_t *sig, size_t n, uint32_t peak,
                       int16_t out[256]);


/* ---- stage-04 fix: context-filtered beat extraction ------------------------
 *
 * rpeak_extract_beat() cuts from the RAW signal. Measured on 3,200 real beats,
 * that reproduces the training pipeline's verdict only 64% of the time. The
 * cause is baseline wander: normalize_beat() is min-max over the window, so a
 * drifting baseline rescales the whole beat even when the QRS shape is intact.
 *
 * Filtering only the 256-sample window does not fix it (86%): one cycle of
 * 0.5 Hz is 720 samples, so a 256-sample window cannot see the drift it is
 * meant to remove. Filtering a 1024-sample CONTEXT and cutting the middle 256
 * reaches 94% overall - and 100% SVEB, 99.5% VEB, 100% Fusion.
 *
 * Cost: a 1024-sample ring buffer instead of 512, plus one forward-backward
 * pass per beat. At 17 ms/beat against an 800 ms budget this is free.
 */
#define RPEAK_CTX_LEN     1024   /* ~2.8 s at 360 Hz */
#define RPEAK_FILT_SHIFT  12     /* fractional bits carried through the cascade;
                                  * without this the 0.993 poles of the 0.5 Hz
                                  * sections truncate the signal to nothing */

/*
 * Same contract as rpeak_extract_beat(), but the 256-sample window is cut from
 * a zero-phase-filtered 1024-sample context centred on the peak, matching how
 * the training data was prepared. Returns 0 on success, -1 if the context
 * would run off either end of the buffer.
 */
int rpeak_extract_beat_ctx(const int16_t *sig, size_t n, uint32_t peak,
                           int16_t out[256]);

/* ---------------------------------------------------------------------------
 * R-R interval and QRS width.
 *
 * Both are pure post-processing on data the pipeline already has by the time
 * a beat is classified - R-peak sample indices (rpeak_push() already reports
 * these) and the 256-sample beat window (rpeak_extract_beat*() already cuts
 * it). No new sensor, no new signal path. This is the direct, on-device
 * answer to "what parameter does your system use to detect an arrhythmia" -
 * the CNN's classification is a verdict, RR interval and QRS width are the
 * two numbers a clinician would actually look at to sanity-check it: RR
 * interval and its variability drive rate-based rhythm calls (bradycardia,
 * tachycardia, irregular rhythm), and QRS width flags conduction problems
 * (a wide QRS is the hallmark of a ventricular beat, e.g. AAMI class VEB).
 * ------------------------------------------------------------------------- */

/* R-R interval given two consecutive peaks' ABSOLUTE sample indices (as
 * reported by rpeak_push() / rpeak_detect()). Returns milliseconds, rounded
 * to the nearest ms. Caller is responsible for only calling this with two
 * peaks from the same continuous record (a gap or record boundary between
 * them would produce a meaningless interval). */
static inline int rpeak_rr_interval_ms(uint32_t peak, uint32_t prev_peak) {
    uint32_t d = peak - prev_peak;   /* peak > prev_peak is a caller invariant */
    return (int)((d * 1000u + RPEAK_FS / 2) / RPEAK_FS);
}

/* Instantaneous heart rate from an RR interval in ms. Returns 0 if the
 * interval is 0 (would otherwise divide by zero - can only happen from a
 * caller bug, e.g. passing the same peak twice). */
static inline int rpeak_bpm_from_rr_ms(int rr_ms) {
    return rr_ms > 0 ? (60000 + rr_ms / 2) / rr_ms : 0;
}

/*
 * QRS complex width, in samples, measured on an already-extracted 256-sample
 * beat window. r_index is where the R peak sits WITHIN that window - pass
 * RPEAK_BEAT_PRE for windows from rpeak_extract_beat() or
 * rpeak_extract_beat_ctx() (both cut RPEAK_BEAT_PRE samples before the peak).
 *
 * Returns -1 if no clear QRS boundary could be found (near-flat beat, or the
 * search window runs off either edge of the 256-sample buffer) rather than
 * fabricating a number. See rpeak.c for the algorithm.
 */
int rpeak_qrs_width_samples(const int16_t beat[256], int r_index);

/* Same as rpeak_qrs_width_samples(), converted to milliseconds (rounded to
 * the nearest ms). -1 propagates unchanged. */
int rpeak_qrs_width_ms(const int16_t beat[256], int r_index);

#endif /* RPEAK_H */
