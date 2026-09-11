/*
 * ecg_vega_aries_combined.ino   (REVISION 3)
 *
 * AD8232 ECG + CNN arrhythmia classifier, MAX30102 (HR + SpO2), MPU6050
 * (motion), MLX90614 (object + ambient temperature) and a 20x4 I2C LCD,
 * all on ONE VEGA ARIES board sharing one Serial port and one loop().
 *
 * rpeak.c/.h, model_forward.c/.h, weights.c/.h are UNCHANGED - the
 * classifier is independently validated at 99.00% on 200 real MIT-BIH
 * beats and nothing in this revision touches it.
 *
 * ======================================================================
 * REVISION 3 - FOUR REAL BUGS FOUND AND FIXED
 * ======================================================================
 * Revision 2 ran on hardware but exhibited: heart rate permanently 0.00
 * while SpO2 showed values, a temperature reading of 54.51C for BOTH
 * object and ambient, an LCD showing only solid blocks, and erratic
 * V/S/Q classification on a healthy user. Those were four separate
 * defects, not one. Each is fixed below and each fix is explained where
 * it lives. Summary:
 *
 * ----------------------------------------------------------------------
 * BUG 1 (why heart rate was ALWAYS 0.00 while SpO2 still showed numbers)
 * ----------------------------------------------------------------------
 * THE SENSOR'S OUTPUT RATE AND THIS CODE'S READ RATE WERE NEVER MATCHED.
 *
 * The MAX30102 writes samples into its own 32-deep hardware FIFO at a rate
 * set by SPO2_CONFIG and FIFO_CONFIG. Revision 2 configured 100Hz sampling
 * with 4x averaging - so the FIFO filled at 100/4 = 25 samples per second -
 * and then read exactly one sample from FIFO_DATA every 30ms, i.e. at
 * 33.3 reads per second, WITHOUT ever checking whether a new sample had
 * actually arrived.
 *
 * Reading faster than the sensor produces means roughly a quarter of all
 * reads happened with the FIFO empty (read pointer already caught up to
 * the write pointer). A MAX30102 read in that state returns the stale
 * previous sample again. So the PPG waveform this code reconstructed was
 * the real waveform with duplicated samples randomly spliced into it.
 *
 * That single fact explains the exact symptom pair, which is why it's
 * worth being precise about:
 *   - HEART RATE died completely. Beat detection works by finding the
 *     rise and fall of each pulse. A duplicated sample is a momentary
 *     flat spot; sprinkling flat spots through a waveform at random
 *     destroys the peak-to-trough excursions the detector measures. The
 *     detector therefore never saw a qualifying beat, and HR stayed 0.
 *   - SpO2 KEPT WORKING (or appeared to). calculateSpO2() computes an
 *     RMS ratio over a 100-sample buffer. RMS is a statistic over a set
 *     of values - it does not care what ORDER the samples are in or
 *     whether some are duplicates. So it kept producing a plausible-
 *     looking number from a waveform that was too corrupted to extract a
 *     heartbeat from. The SpO2 values in the logs (70%, 83%, 90% while HR
 *     read 0.00) were never trustworthy either; they just failed silently
 *     instead of failing visibly.
 *
 * FIX: read FIFO_WR_PTR and FIFO_RD_PTR first, compute how many samples
 * are genuinely waiting, and read exactly that many - never more. When the
 * FIFO is empty this now reads nothing at all instead of re-reading stale
 * data. Every sample handed to the beat detector is now a real, distinct
 * sample taken by the sensor's own crystal-accurate internal clock at
 * exactly PPG_SAMPLE_RATE_HZ, evenly spaced, regardless of how jittery
 * this sketch's polling happens to be. See max30102_service().
 *
 * ----------------------------------------------------------------------
 * BUG 2 (heart rate detection was brittle even with good samples)
 * ----------------------------------------------------------------------
 * MIN_PULSE_AMPLITUDE was a hardcoded 400 raw ADC counts. PPG pulse
 * amplitude varies by more than an order of magnitude with finger
 * pressure, skin, temperature, perfusion and LED current - a fixed
 * absolute threshold is guaranteed to be wrong for somebody. Worse, it was
 * applied AFTER exponential smoothing, which shrinks peak height further.
 *
 * FIX: replaced with an envelope-tracking threshold-crossing detector.
 * It continuously tracks the running minimum and maximum of the smoothed
 * IR signal and fires a beat when the signal crosses 65% of the way up
 * that envelope, re-arming below 35% (hysteresis, so one pulse can't
 * double-trigger on noise near the threshold). The threshold is therefore
 * always scaled to whatever signal this particular finger is actually
 * producing. Only one absolute floor remains (MIN_AC_AMPLITUDE), and it
 * exists solely to refuse to find "beats" in pure noise when no finger is
 * present. See processPPGSample().
 *
 * Beat timing also no longer uses millis(). Because samples now arrive at
 * a known fixed rate, the interval between two beats is counted in
 * SAMPLES and converted with the exact sample period. This is immune to
 * polling jitter, to the 144ms CNN inference stall, and to Serial/LCD
 * blocking - all of which corrupt a millis()-based interval.
 *
 * ----------------------------------------------------------------------
 * BUG 3 (why the ECG waveform was corrupted after every single beat)
 * ----------------------------------------------------------------------
 * This one is subtle and was silently damaging every beat's morphology.
 *
 * service_ecg() paced itself with `last_tick += SAMPLE_INTERVAL_US`, which
 * is the correct drift-free idiom - but ONLY while the loop keeps up. CNN
 * inference takes ~144ms (measured on this hardware). At 360Hz that is 52
 * sample periods. After inference returned, last_tick was 52 periods in
 * the past, so the next 52 calls ALL fired immediately, back to back,
 * taking 52 analogRead()s within about a millisecond - and then handed
 * them to the R-peak detector, whose filters assume every sample is
 * 2.78ms after the previous one.
 *
 * So after every heartbeat the detector received a burst of 52 samples
 * compressed ~50x in time but labelled as normal-rate. That injects a
 * violent artifact into the waveform immediately after each R-peak -
 * landing squarely on the ST segment and T-wave of the beat just
 * classified, and inside the context window used to classify the NEXT
 * beat. This is a very plausible contributor to the "healthy user reads as
 * V/S/Q" behaviour, entirely separate from electrode contact.
 *
 * FIX: if the clock is more than MAX_CATCHUP_SAMPLES periods behind, do
 * not try to catch up. Resynchronise last_tick to now, and count the lost
 * samples in ecg_dropped_samples (reported once a second). A real gap in
 * the signal is honest and the detector can cope with it; fabricated
 * samples with false timestamps are not and it cannot.
 *
 * ----------------------------------------------------------------------
 * BUG 4 (temperature reading 54.51C for object AND ambient simultaneously)
 * ----------------------------------------------------------------------
 * isPlausibleTemp() accepted anything from -60C to +300C. The MLX90614's
 * "ambient" reading is its own die temperature, which in a room is
 * 15-40C; 54.51C is not a real ambient reading, and object and ambient
 * landing on the identical value to 2 decimal places is the signature of a
 * failed bit-banged I2C read returning the same buffer twice, not of a
 * real measurement.
 *
 * FIX: separate, physically-motivated ranges for ambient (0-50C, it's a
 * die temperature in a room) and object (0-80C, skin plus headroom), a
 * single retry on failure, and the last good value is held rather than
 * being replaced by a bad one. Both values are now displayed, as
 * requested.
 *
 * ======================================================================
 * HOW THE THREE DIFFERENT SAMPLING RATES ARE KEPT CONSISTENT
 * ======================================================================
 * Three subsystems need three different, unrelated rates on one core with
 * no RTOS. Each is paced by its own clock check, and NONE uses delay():
 *
 *   ECG        360 Hz  (RPEAK_FS - fixed by the CNN's training data)
 *              paced by micros(), drift-free accumulate, with the
 *              resync-on-long-stall rule from Bug 3 above.
 *
 *   PPG        25 Hz   (the MAX30102's own FIFO write rate)
 *              NOT paced by this sketch at all - the sensor's internal
 *              clock is the timebase. This code just polls often enough
 *              (every 20ms, i.e. 50Hz, comfortably faster than 25Hz) to
 *              drain whatever has arrived. Polling faster than the data
 *              rate is now harmless because the pointer check means an
 *              early poll simply finds nothing and returns.
 *
 *   Temp/LCD   1 Hz display refresh, 3 s temperature refresh
 *              paced by millis(). The MLX90614's bit-banged read costs
 *              ~127ms, so it deliberately runs on only every third
 *              display refresh; the display reprints the held value in
 *              between.
 *
 * The honest tradeoff, unchanged from revision 2 and inherent to one core:
 * ECG sampling still pauses during CNN inference (~144ms, once per beat).
 * Revision 3 no longer papers over that pause with fabricated samples - it
 * reports it. If [ECG] dropped counts are high, the fix is to speed up
 * inference (confirm the build actually uses -O2; 561,680 MACs taking
 * 144ms works out to ~4 MMAC/s, which is slow enough to suspect the
 * optimiser is off) rather than to change this pacing logic.
 *
 * ======================================================================
 * PINS - VERIFY AGAINST YOUR BOARD'S SILKSCREEN BEFORE WIRING
 * ======================================================================
 *   ECG_ANALOG_PIN     A0   (AD8232 output)          - PLACEHOLDER, verify
 *   LEAD_OFF_PLUS_PIN  2    (AD8232 LO+)              - PLACEHOLDER, verify
 *   LEAD_OFF_MINUS_PIN 3    (AD8232 LO-)              - PLACEHOLDER, verify
 *   AD8232_SDN_PIN     4    (AD8232 SDN)              - PLACEHOLDER, verify
 *   MLX_SDA_PIN        5    (MLX90614 software SDA)
 *   MLX_SCL_PIN        6    (MLX90614 software SCL)
 *   Wire1 hardware I2C0     (MAX30102 0x57, MPU6050 0x68, LCD 0x27/0x3F)
 *
 * LCD POWER NOTE: the 20x4 module and its PCF8574 backpack are 5V parts.
 * Power the module's VCC from 5V. If it is on 3.3V the backpack will still
 * ACK on I2C (so the sketch will report "backpack detected") while the
 * panel itself never reaches a usable operating point - which looks
 * exactly like a software failure but is not one.
 */

#include <Arduino.h>
#include <stdio.h>
#include <math.h>
extern "C" {
#include "rpeak.h"
#include "model_forward.h"
}

#include <Wire.h>
#include <VEGA_MLX90614.h>
#include "lcd_i2c_pcf8574.h"


/* =====================================================================
 * SECTION 1 - ECG (AD8232 + R-peak detector + CNN classifier)
 * ===================================================================== */

#define STREAM_MODE 0
#define PLOTTER_DECIMATION 3

/* =====================================================================
 * REVISION 5 - LIVE WAVEFORM STREAMING ALONGSIDE EVERYTHING ELSE
 * =====================================================================
 * Previously the sketch could stream the raw waveform (STREAM_MODE 1/2,
 * bare numbers for the Arduino Serial Plotter) OR print classification and
 * vitals text, but never both - the Plotter needs a line to contain
 * nothing but a number, so any text line breaks it. That made "show the
 * live ECG trace AND the arrhythmia class AND HR/SpO2/temperature on one
 * dashboard" impossible over a single serial link.
 *
 * Revision 5 solves that with a tagged line protocol. Every line starts
 * with a one-letter tag, so a parser can route it and a human can still
 * read it:
 *
 *   W:<s1>,<s2>,...       raw ECG samples, WAVE_BATCH per line
 *   E:<cls>,<rr>,<bpm>,<qrs>,<rel>   one line per classified beat
 *   V:<hr>,<spo2>,<obj>,<amb>,<motion>,<ir>   one line per second
 *
 * All three interleave freely on the same port. The Arduino Serial Plotter
 * can no longer be used (it never could alongside text) - the dashboard
 * bridge is the intended consumer, and it plots far better anyway.
 *
 * BANDWIDTH AND THE COST TO ECG SAMPLING - the reason for decimation:
 * every byte written to Serial blocks sampling while it transmits. At
 * 115200 baud that is ~11.5 bytes per millisecond. Streaming all 360
 * samples/sec at ~5 bytes each would be ~1800 B/s = ~156ms of every second
 * spent transmitting, i.e. 15% of all ECG samples lost to printing the
 * waveform. Decimating by 3 gives a 120Hz trace - visually indistinguishable
 * for a monitoring display, since the QRS complex is only ~10Hz of real
 * content - at ~600 B/s, about 5%. Batching WAVE_BATCH samples per line
 * amortises the "W:" prefix and newline across the batch.
 *
 * If your board's UART tolerates it, raising SERIAL_BAUD to 230400 halves
 * this cost again and is the single cheapest improvement available here.
 * It is left at 115200 by default because an unsupported baud rate fails
 * as silent garbage, which is a bad thing to discover during a demo. */
#define SERIAL_BAUD        115200
/* REVISION 7: ALWAYS ON. No toggle, no switching between "readable on the
 * Serial Monitor" and "feeds the dashboard" - one firmware does both at
 * once. The human-readable [ECG]/[VITALS] lines and the machine-readable
 * W: waveform lines simply interleave; the Serial Monitor shows everything
 * and stream_bridge.py picks out what it needs.
 *
 * The budget works out: verbose beat lines ~360 B/s, the once-a-second
 * vitals block ~150 B/s, and the decimated waveform ~620 B/s. Around
 * 1.1 kB/s against 11.5 kB/s available at 115200 baud - roughly 10%. */
#define STREAM_WAVEFORM    1
#define WAVE_DECIMATE      3     /* send every Nth sample: 360/3 = 120 Hz */
#define WAVE_BATCH         12    /* samples per W: line (~10 lines/sec) */

#define SAMPLING_RATE_HZ   RPEAK_FS
#define SAMPLE_INTERVAL_US (1000000UL / SAMPLING_RATE_HZ)

/* Bug 3 fix: how many sample periods we're willing to "catch up" by
 * firing back-to-back before we give up and resynchronise instead. 2
 * absorbs ordinary loop jitter (a fast I2C read, a Serial write) while
 * still refusing to fabricate the ~52-sample burst that a CNN inference
 * stall would otherwise produce. */
#define MAX_CATCHUP_SAMPLES 2

#define ECG_ANALOG_PIN       A0
#define LEAD_OFF_PLUS_PIN    2
#define LEAD_OFF_MINUS_PIN   3
#define AD8232_SDN_PIN       4

#define SIGNAL_QUALITY_WINDOW_SAMPLES  360
#define SIGNAL_QUALITY_MIN_PP          300

#define BUF_LEN       4096
#define BUF_MASK      (BUF_LEN - 1)
static int16_t hist[BUF_LEN];
static uint32_t total_samples = 0;

/* Diagnostic only (see the long note in service_ecg): how many times the
 * sample clock fell more than MAX_CATCHUP_SAMPLES periods behind since the
 * last report - i.e. how often a long stall (CNN inference, I2C, Serial)
 * occurred. Counting only; it does NOT change pacing behaviour. */
static uint32_t ecg_behind_events = 0;

#define PENDING_QUEUE_LEN 8
static uint32_t pending_peaks[PENDING_QUEUE_LEN];
static uint8_t  pending_head = 0, pending_tail = 0, pending_count = 0;

static inline bool pending_push(uint32_t v) {
  if (pending_count >= PENDING_QUEUE_LEN) return false;
  pending_peaks[pending_tail] = v;
  pending_tail = (uint8_t)((pending_tail + 1) % PENDING_QUEUE_LEN);
  pending_count++;
  return true;
}
static inline bool pending_peek(uint32_t *v) {
  if (pending_count == 0) return false;
  *v = pending_peaks[pending_head];
  return true;
}
static inline void pending_pop(void) {
  pending_head = (uint8_t)((pending_head + 1) % PENDING_QUEUE_LEN);
  pending_count--;
}

static rpeak_state_t rp_state;

static const char *CLASS_SHORT[NUM_CLASSES] = { "N", "S", "V", "F", "Q" };

#define CLASS_BIAS_N_Q88 500

static int latest_hr_bpm = 0;     /* ECG-derived HR (from RR interval) */
static int latest_class_id = 0;   /* class index * 100 (kept for STREAM_MODE 2 compat) */

static bool was_leads_off = false;
static uint16_t sq_min = 65535, sq_max = 0;
static int sq_count = 0;
static bool was_weak_signal = false;
static bool last_beat_reliable = false;   /* drives the '*' unreliable marker on the LCD */

static void classify_pending_beat_if_ready(void) {
  static int16_t ctx_scratch[RPEAK_CTX_LEN];
  static int16_t beat[256];
  static int16_t logits[NUM_CLASSES];
  static uint32_t last_reported_peak = 0;
  static bool have_last_peak = false;

  uint32_t peak_abs;
  if (!pending_peek(&peak_abs)) return;
  if (total_samples < peak_abs + (RPEAK_CTX_LEN / 2)) return;
  pending_pop();

  uint32_t start_abs = peak_abs - (RPEAK_CTX_LEN / 2);
  for (int i = 0; i < RPEAK_CTX_LEN; i++) {
    ctx_scratch[i] = hist[(start_abs + i) & BUF_MASK];
  }

  unsigned long t0 = micros();
  int ok = rpeak_extract_beat_ctx(ctx_scratch, RPEAK_CTX_LEN,
                                   RPEAK_CTX_LEN / 2, beat);
  unsigned long t1 = micros();
  if (ok) {
#if STREAM_MODE == 0
    Serial.println("# extract_beat_ctx: not enough context (shouldn't happen - check BUF_LEN/timing)");
#endif
    return;
  }

  unsigned long t2 = micros();
  int cls = model_forward(beat, logits);
  unsigned long t3 = micros();

  int32_t biased_n_score = (int32_t)logits[0] + CLASS_BIAS_N_Q88;
  if (biased_n_score > logits[cls]) cls = 0;

  int rr_ms = -1, bpm = -1;
  if (have_last_peak) {
    rr_ms = rpeak_rr_interval_ms(peak_abs, last_reported_peak);
    bpm = rpeak_bpm_from_rr_ms(rr_ms);
  }
  int qrs_ms = rpeak_qrs_width_ms(beat, RPEAK_BEAT_PRE);
  int p_ms = rpeak_p_wave_width_ms(beat, RPEAK_BEAT_PRE);
  int p_mag = rpeak_p_wave_peak_mag(beat, RPEAK_BEAT_PRE);

  /* Lead-polarity diagnostic: beat[] is centred on the R-peak and Q8.8
   * scaled. A correctly-wired lead reads strongly POSITIVE here on a
   * normal beat. Consistently negative means the trace is inverted
   * (usually RA/LA swapped), which makes every normal beat look like a
   * PVC to the CNN regardless of the true rhythm. */
  int16_t r_peak_amplitude = beat[RPEAK_BEAT_PRE];

  /* Plausibility gate: the CNN has no "this is noise" class, so a noise
   * spike gets confidently classified as something (usually V, since a
   * sharp isolated deflection is what random noise most resembles). Two
   * independent signals that a detection isn't a real heartbeat: the
   * amplitude-window weak-signal flag, and an implausible instantaneous
   * rate. When either holds the beat is still logged but does not update
   * what the display trusts. */
  const int PLAUSIBLE_MIN_BPM = 30;
  const int PLAUSIBLE_MAX_BPM = 220;
  bool rate_plausible = (bpm < 0) || (bpm >= PLAUSIBLE_MIN_BPM && bpm <= PLAUSIBLE_MAX_BPM);
  bool reliable = !was_weak_signal && !was_leads_off && rate_plausible;

  if (reliable) {
    latest_class_id = cls * 100;
    if (bpm >= 0) latest_hr_bpm = bpm;
  }
  last_beat_reliable = reliable;

#if STREAM_MODE == 0
  /* REVISION 7: the original readable format, restored. Compressing this
   * was optimising the wrong thing - this is the line you actually read
   * while working on the board, and it carries what matters when something
   * looks off: the RR interval, the QRS width, and the raw logits that
   * show how close the decision was. stream_bridge.py parses this same
   * line, so there is no second machine-readable format to keep in sync. */
  Serial.print(reliable ? "[ECG] beat @" : "[ECG] (unreliable, ignored) beat @");
  Serial.print(peak_abs);
  Serial.print("  -> ");
  Serial.print(CLASS_SHORT[cls]);
  Serial.print("   RR ");
  if (rr_ms >= 0) { Serial.print(rr_ms); Serial.print("ms/"); Serial.print(bpm); Serial.print("bpm"); }
  else Serial.print("--");
  Serial.print("   QRS ");
  if (qrs_ms >= 0) { Serial.print(qrs_ms); Serial.print("ms"); }
  else Serial.print("--");
  Serial.print("   P ");
  if (p_ms >= 0) { Serial.print(p_ms); Serial.print("ms"); }
  else Serial.print("--");
  Serial.print("(mag ");
  Serial.print(p_mag);
  Serial.print(")");
  if (r_peak_amplitude < 0) Serial.print("   Rpeak NEGATIVE-check lead polarity!");
  Serial.print("   prep ");
  Serial.print(t1 - t0);
  Serial.print(" us   infer ");
  Serial.print(t3 - t2);
  Serial.print(" us   logits N/S/V/F/Q ");
  for (int c = 0; c < NUM_CLASSES; c++) {
    Serial.print(logits[c]);
    if (c < NUM_CLASSES - 1) Serial.print("/");
  }
  Serial.println();
#endif

  /* last_reported_peak advances even on an unreliable beat: RR math needs
   * the true last DETECTED peak position, or the next beat's interval is
   * computed against a stale timestamp and comes out wrong too. */
  last_reported_peak = peak_abs;
  have_last_peak = true;
}

static void service_ecg(void) {
  static unsigned long last_tick = 0;
  unsigned long now = micros();

  /* Signed difference so this stays correct across micros() rollover. */
  long behind_us = (long)(now - last_tick);
  if (behind_us < (long)SAMPLE_INTERVAL_US) return;

  /* ---- Bug 3 fix (see file header) ----
   * If we're only slightly behind, accumulate normally - that's the
   * drift-free idiom and it's what we want for ordinary jitter.
   * If we're MANY periods behind, real time was lost to something long
   * (CNN inference at ~144ms = 52 periods, an I2C stall, a Serial flush).
   * ==================================================================
   * REVISION 4: THE REVISION 3 "FIX" HERE WAS WRONG AND IS REVERTED
   * ==================================================================
   * Revision 3 resynchronised the clock on a long stall instead of
   * catching up, reasoning that a genuine gap is more honest than
   * fabricated samples. On hardware that made classification MUCH worse -
   * every beat came out at 200-300bpm and got flagged unreliable. Here is
   * why, and it is a reasoning error worth stating plainly rather than
   * quietly reverting:
   *
   * RR intervals are computed from the DIFFERENCE IN total_samples between
   * two R-peaks, multiplied by the nominal sample period. That arithmetic
   * is only valid if total_samples advances once per real 2.78ms of
   * elapsed time. Resyncing without advancing total_samples breaks exactly
   * that invariant: 144ms of real time passes while the counter does not
   * move, so every subsequent interval is measured short. With ~50% of
   * samples being dropped, a real 1000ms RR interval was measured as
   * ~470ms and reported as ~128bpm. That is precisely the failure seen in
   * the field logs, and it then tripped the plausibility gate, so nothing
   * reached the display at all.
   *
   * The revision 2 behaviour restored below keeps total_samples in step
   * with real time, which is what every downstream timing calculation
   * depends on. The compressed-burst artifact it produces is real, but it
   * is a morphology distortion the detector tolerates - whereas a broken
   * time base corrupts RR, bpm, the plausibility gate, and beat-window
   * alignment all at once. Correct timing first, clean morphology second.
   *
   * ecg_behind_events still counts how often a long stall happened, purely
   * as a diagnostic - it no longer changes behaviour. */
  if (behind_us > (long)(SAMPLE_INTERVAL_US * MAX_CATCHUP_SAMPLES)) {
    ecg_behind_events++;
  }
  last_tick += SAMPLE_INTERVAL_US;

  static float dc_baseline = -1.0f;

  bool leads_off = digitalRead(LEAD_OFF_PLUS_PIN) || digitalRead(LEAD_OFF_MINUS_PIN);
  if (leads_off) {
#if STREAM_MODE == 0
    if (!was_leads_off) Serial.println("# [ECG] leads off - pausing sampling");
#endif
    was_leads_off = true;
    return;
  }
  if (was_leads_off) {
    rpeak_init(&rp_state);
    dc_baseline = -1.0f;
    was_leads_off = false;
    sq_min = 65535; sq_max = 0; sq_count = 0;
#if STREAM_MODE == 0
    Serial.println("# [ECG] leads on - resuming (detector state reset)");
#endif
  }

  uint16_t raw = analogRead(ECG_ANALOG_PIN);
  if (dc_baseline < 0.0f) dc_baseline = (float)raw;

  if (raw < sq_min) sq_min = raw;
  if (raw > sq_max) sq_max = raw;
  sq_count++;
  if (sq_count >= SIGNAL_QUALITY_WINDOW_SAMPLES) {
    uint16_t pp = sq_max - sq_min;
    bool weak = pp < SIGNAL_QUALITY_MIN_PP;
#if STREAM_MODE == 0
    if (weak && !was_weak_signal) {
      Serial.print("# [ECG] weak signal (");
      Serial.print(pp);
      Serial.println("pp) - check electrode contact/gel");
    } else if (!weak && was_weak_signal) {
      Serial.println("# [ECG] signal amplitude back to normal");
    }
#endif
    was_weak_signal = weak;
    sq_min = 65535; sq_max = 0; sq_count = 0;
  }

  dc_baseline += 0.0005f * ((float)raw - dc_baseline);
  int16_t sample = (int16_t)((float)raw - dc_baseline);

  hist[total_samples & BUF_MASK] = sample;

#if STREAM_WAVEFORM
  /* Live waveform out. Collected into a batch and emitted as one line so
   * the per-line overhead ("W:" + newline) is paid once per WAVE_BATCH
   * samples instead of once each. The emit blocks for roughly 5ms at
   * 115200 baud, about ten times a second - which the sample clock's
   * catch-up logic absorbs the same way it absorbs the CNN inference. */
  {
    static int16_t wave_buf[WAVE_BATCH];
    static uint8_t wave_n = 0;
    if ((total_samples % WAVE_DECIMATE) == 0) {
      wave_buf[wave_n++] = sample;
      if (wave_n >= WAVE_BATCH) {
        Serial.print("W:");
        for (uint8_t i = 0; i < WAVE_BATCH; i++) {
          Serial.print(wave_buf[i]);
          if (i < WAVE_BATCH - 1) Serial.print(",");
        }
        Serial.println();
        wave_n = 0;
      }
    }
  }
#endif

#if STREAM_MODE == 1 || STREAM_MODE == 2
  if ((total_samples % PLOTTER_DECIMATION) == 0) {
#endif
#if STREAM_MODE == 1
  Serial.println(sample);
#elif STREAM_MODE == 2
  Serial.print("ECG:");
  Serial.print(sample);
  Serial.print(",HR:");
  Serial.print(latest_hr_bpm);
  Serial.print(",Class_ID:");
  Serial.println(latest_class_id);
#endif
#if STREAM_MODE == 1 || STREAM_MODE == 2
  }
#endif

  uint32_t peak_abs;
  int got = rpeak_push(&rp_state, sample, &peak_abs);
  total_samples++;

#if STREAM_MODE != 1
  if (got) {
    pending_push(peak_abs);
  }
  classify_pending_beat_if_ready();
#endif
}


/* =====================================================================
 * SECTION 2 - VITALS (MAX30102 HR+SpO2, MPU6050 motion, MLX90614 temp)
 * ===================================================================== */

TwoWire Wire1(0);   /* hardware I2C0 - MAX30102 + MPU6050 + LCD backpack */

#define MLX_SDA_PIN 5
#define MLX_SCL_PIN 6
VEGA_MLX90614 mlx(MLX_SDA_PIN, MLX_SCL_PIN);

#define MAX30102_ADDR 0x57
#define MPU6050_ADDR  0x68

#define MAX_REG_INTR_STATUS_1   0x00
#define MAX_REG_INTR_STATUS_2   0x01
#define MAX_REG_FIFO_WR_PTR     0x04
#define MAX_REG_OVF_COUNTER     0x05
#define MAX_REG_FIFO_RD_PTR     0x06
#define MAX_REG_FIFO_DATA       0x07
#define MAX_REG_FIFO_CONFIG     0x08
#define MAX_REG_MODE_CONFIG     0x09
#define MAX_REG_SPO2_CONFIG     0x0A
#define MAX_REG_LED1_PA         0x0C
#define MAX_REG_LED2_PA         0x0D

/* ---------------------------------------------------------------------
 * MAX30102 SAMPLE RATE - the single most important constant in this file
 * ---------------------------------------------------------------------
 * These three values MUST agree with each other, and PPG_SAMPLE_RATE_HZ
 * must equal what the register settings actually produce, because the
 * beat detector converts sample counts into milliseconds using it. Get
 * this wrong and heart rate is silently scaled by the error ratio.
 *
 *   SPO2_CONFIG = 0x27 -> bits[6:5]=01  ADC range 4096nA
 *                         bits[4:2]=001 sample rate 100 Hz
 *                         bits[1:0]=11  pulse width 411us (18-bit)
 *   FIFO_CONFIG = 0x5F -> bits[7:5]=010 average 4 samples
 *                         bit[4]  =1    FIFO rollover ENABLED
 *                         bits[3:0]=1111 almost-full at 15
 *
 * Effective FIFO write rate = 100 Hz / 4 (averaging) = 25 Hz.
 *
 * Note rollover is now ENABLED (it was disabled in revision 2, where
 * FIFO_CONFIG was 0x4F). With rollover disabled, a FIFO that fills up
 * stops accepting new samples entirely until it is read - so any hiccup
 * in polling could wedge the sensor into permanently returning stale
 * data. Enabled, the oldest sample is discarded instead and the stream
 * stays live. */
#define PPG_SAMPLE_RATE_HZ    25
#define PPG_SAMPLE_PERIOD_MS  (1000.0f / (float)PPG_SAMPLE_RATE_HZ)   /* 40.0 ms */

/* ---------------------------------------------------------------------
 * REVISION 4: RAISED 20 -> 40 ms. This was the other half of the
 * revision 3 regression, and it repeated a mistake revision 2 had ALREADY
 * learned on this exact hardware.
 *
 * Revision 2's own comment recorded that dropping this interval to ~10ms
 * produced "mostly Q/V/S classifications on a healthy user", and that
 * raising it to 30ms fixed that - because every poll is a blocking I2C
 * transaction interleaved into the 360Hz ECG stream. Revision 3 lowered it
 * to 20ms AND made each poll more expensive (a FIFO-pointer read on top of
 * the data read), measured on hardware at up to 8455us per poll versus
 * 1784us before. At 20ms that is roughly 40% of all CPU time spent inside
 * the vitals poll, and the ECG classification degraded exactly as revision
 * 2 had already documented it would.
 *
 * 80ms is chosen from the measured numbers rather than guessed. The field
 * logs give two data points: a poll that found NO new sample cost 1501us,
 * and a poll that found some cost 8455us. So roughly 1.5ms is fixed
 * per-transaction overhead (the FIFO pointer read) and ~7ms is the data
 * read - and that ~7ms is transaction setup cost, not per-byte cost, since
 * six bytes at any sane I2C clock is well under 1ms. Two consequences:
 *   - polling MORE often does not get data sooner, it just pays the 1.5ms
 *     pointer read more times for nothing;
 *   - draining SEVERAL samples in one burst transaction is nearly free
 *     compared to fetching them one per poll.
 * So: poll at 80ms and take both samples that accumulated, instead of
 * polling at 20ms and mostly finding nothing. Same 25Hz of PPG data, same
 * even spacing (the sensor's clock still defines it), roughly half the
 * CPU cost of the 40ms version and a quarter of revision 3's 20ms.
 *
 * Motion detection now updates at 12.5Hz rather than 50Hz, which is still
 * far faster than a fall or a hand movement needs. */
#define SENSOR_POLL_INTERVAL_MS 80

/* Hard cap on samples drained per poll. At 80ms against a 40ms production
 * rate we expect exactly 2; 4 gives headroom for clock drift or a late
 * poll without unbounded blocking, and 4 samples = 24 bytes still fits
 * comfortably inside the standard 32-byte Wire buffer in one transaction. */
#define MAX_FIFO_DRAIN_PER_POLL 4

/* Finger presence: raw IR DC level below this means nothing is on the
 * sensor.
 *
 * REVISION 5: lowered 50000 -> 15000. The field logs showed "Place
 * finger..." almost continuously even while a finger was on the sensor,
 * with only occasional moments where detection latched - which means the
 * IR level was hovering right around the old threshold rather than
 * comfortably above it. The absolute IR reading depends heavily on LED
 * current, how hard the finger presses, skin, ambient light leaking into
 * the package, and whether the sensor sits flush - so a threshold picked
 * for one setup is routinely wrong for another. 15000 is still far above
 * the no-finger baseline (typically a few thousand) while admitting a
 * light touch.
 *
 * The V: line now reports the raw IR value every second, so this can be
 * set from measurement instead of guesswork: look at <ir> with no finger,
 * then with a finger, and put the threshold between the two. */
const uint32_t FINGER_THRESHOLD = 15000;
const int FINGER_DEBOUNCE_COUNT = 5;

/* Physiological bounds for accepting a PPG-derived beat interval. */
const float MIN_HR = 40.0f;
const float MAX_HR = 180.0f;

const int SPO2_WINDOW = 100;      /* 100 samples @25Hz = 4.0 s of data */
const float IR_SMOOTHING_ALPHA = 0.35f;

/* Bug 2 fix: the ONLY absolute amplitude constant left. This is not a
 * beat threshold - the beat threshold is computed from the signal
 * envelope. This is purely a floor below which we declare "there is no
 * pulsatile signal here at all" and refuse to report beats, so that
 * sensor noise with a finger half-on can't be mistaken for a heartbeat.
 * At 8x smaller than revision 2's hardcoded 400, this admits weak-
 * perfusion fingers that used to be rejected outright. */
const float MIN_AC_AMPLITUDE = 50.0f;

/* Envelope tracker decay per sample. At 25Hz, 0.01 gives roughly a 4
 * second time constant - fast enough to follow a finger being repositioned,
 * slow enough not to chase individual pulses. */
const float ENV_DECAY = 0.01f;

/* Threshold-crossing levels as a fraction of the tracked envelope.
 * Crossing UP through HI fires a beat; the detector re-arms only after
 * falling back below LO. The gap between them is the hysteresis that
 * stops one pulse double-triggering. */
const float BEAT_THRESH_HI = 0.65f;
const float BEAT_THRESH_LO = 0.35f;

/* Beat timeout in samples rather than milliseconds, for the same reason
 * beat intervals are counted in samples - immunity to loop stalls.
 * 4 seconds at 25Hz. */
const uint32_t HR_TIMEOUT_SAMPLES = 4 * PPG_SAMPLE_RATE_HZ;

static unsigned long lastSensorPollMs = 0;

/* Stall instrumentation - pure measurement, no behavioural effect. */
static unsigned long max_vitals_stall_us = 0;

/* MAX30102 diagnostic counters, reported once a second on the V: line.
 * These exist because "IR reads 0" told us the sensor was silent but not
 * WHY, and there are three distinct reasons it can go silent:
 *   max_ptr_fail  - the FIFO pointer register read itself failed (I2C
 *                   problem, wrong address, sensor not responding)
 *   max_empty     - pointers read fine and genuinely reported no new
 *                   sample (sensor configured wrong, or in shutdown, or
 *                   not actually sampling)
 *   max_read_fail - pointers said data was waiting but the FIFO data read
 *                   failed (usually a Wire buffer or bus problem)
 * Whichever of these is non-zero identifies the fault immediately instead
 * of requiring another guess-and-reflash cycle. */
static uint16_t max_ptr_fail = 0;
static uint16_t max_empty = 0;
static uint16_t max_read_fail = 0;

/* MAX30102 LED drive current, ~0.2mA per count. Starts at the value that
 * was measured working on this hardware; max30102_autorange() only moves
 * it if the reading is provably unusable (clipped at the ADC ceiling, or
 * down at the no-finger floor), so in normal operation it never fires. */
static uint8_t led_pa = 0x32;      /* ~10mA - the known-good value */
#define LED_PA_MIN  0x08           /* ~1.6mA  */
#define LED_PA_MAX  0x7F           /* ~25mA   */
#define IR_SATURATED    250000UL   /* 18-bit full scale is 262143 */
#define IR_TOO_WEAK      25000UL   /* plenty above the ~6600 no-finger floor */

uint32_t redValue = 0;
uint32_t irValue = 0;

/* PPG beat detector state (Bug 2 fix) */
static float irSmoothed = 0.0f;
static float irMin = 0.0f, irMax = 0.0f;
static float irDC = 0.0f;          /* slowly-tracked PPG baseline */
static bool  dcInitialised = false;
static float lastAC = 0.0f;        /* most recent pulse envelope, for display */
/* ~2 second time constant at 25Hz (corner ~0.08Hz) - far below the 0.67Hz
 * of a 40bpm pulse, so it removes baseline wander without attenuating the
 * heartbeat itself. */
#define IR_DC_ALPHA 0.02f
static bool  envelopeInitialised = false;
static bool  aboveThreshold = false;
static uint32_t ppgSampleCount = 0;      /* monotonic count of genuinely-new FIFO samples */
static uint32_t lastBeatSample = 0;
static bool  haveLastBeatSample = false;
static bool  pulseAmplitudeOK = false;

bool fingerPresent = false;
int fingerAboveCount = 0;
int fingerBelowCount = 0;
float currentHR = 0.0f;      /* PPG-derived instantaneous HR */
float filteredHR = 0.0f;     /* smoothed PPG HR - this is what gets displayed */

uint32_t redBuffer[SPO2_WINDOW];
uint32_t irBuffer[SPO2_WINDOW];
int sampleIndex = 0;
float spo2 = 0.0f;
bool spo2Valid = false;

int16_t ax = 0, ay = 0, az = 0;
const long MOTION_THRESHOLD = 3000;
int16_t prevAx = 0, prevAy = 0, prevAz = 0;
bool motionInitialized = false;
bool motionDetected = false;

/* MPU6050 at default +-2g full scale: 16384 LSB per g. */
#define ACCEL_LSB_PER_G 16384.0f

double objectTempC = 0.0;
double ambientTempC = 0.0;
bool mlxObjectOK = false;
bool mlxAmbientOK = false;

unsigned long lastPrintTime = 0;
unsigned long lastTempReadTime = 0;
#define TEMP_READ_INTERVAL_MS 3000

/* ---- 20x4 I2C LCD ---- */
#define LCD_COLS 20
#define LCD_ROWS 4
static uint8_t lcdAddr = 0x00;
static LCD_I2C_PCF8574 *lcd = nullptr;
static bool lcdOK = false;

bool writeMAXRegister(uint8_t reg, uint8_t value)
{
  Wire1.beginTransmission(MAX30102_ADDR);
  Wire1.write(reg);
  Wire1.write(value);
  return (Wire1.endTransmission() == 0);
}

bool readMAXRegister(uint8_t reg, uint8_t &value)
{
  Wire1.beginTransmission(MAX30102_ADDR);
  Wire1.write(reg);
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(MAX30102_ADDR, (uint8_t)1) != 1) return false;
  value = Wire1.read();
  return true;
}

/* Reads FIFO_WR_PTR, OVF_COUNTER and FIFO_RD_PTR in ONE burst - they are
 * contiguous at 0x04/0x05/0x06, so this costs a single I2C transaction
 * instead of three. This runs on every poll, so keeping it to one
 * transaction directly reduces how long ECG sampling is blocked. */
static bool readMAXFifoPointers(uint8_t &wr, uint8_t &ovf, uint8_t &rd)
{
  Wire1.beginTransmission(MAX30102_ADDR);
  Wire1.write(MAX_REG_FIFO_WR_PTR);
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(MAX30102_ADDR, (uint8_t)3) != 3) {
    while (Wire1.available()) Wire1.read();
    return false;
  }
  wr  = Wire1.read();
  ovf = Wire1.read();
  rd  = Wire1.read();
  return true;
}

/* Burst-reads `count` complete samples (6 bytes each: 3 red, 3 IR) in a
 * single I2C transaction. The MAX30102 auto-increments within FIFO_DATA,
 * so one transaction can carry several samples - much cheaper than one
 * transaction each, which matters because ECG sampling is stalled for the
 * duration. */
/* Reads ONE sample (6 bytes: 3 red, 3 IR) in a single transaction.
 *
 * REVISION 6: this replaces a multi-sample burst read. The burst asked for
 * up to 24 bytes in one requestFrom() and bailed out entirely if it got
 * back fewer than it asked for. Arduino cores commonly ship a 32-byte Wire
 * buffer, but not all of them do, and this board's core is not one I can
 * check from here - if its buffer is smaller than the request, every burst
 * fails, the function returns 0, and the IR value stays stuck at its
 * initial 0. Which is exactly the symptom in the field log.
 *
 * A single 6-byte read is what revision 2 used, and revision 2 demonstrably
 * got real data out of this exact sensor on this exact board. Reading one
 * sample per transaction and looping is marginally more expensive than one
 * burst, and it removes a whole class of silent failure. Proven beats
 * clever here. */
static bool readMAX30102One(uint32_t &red, uint32_t &ir)
{
  Wire1.beginTransmission(MAX30102_ADDR);
  Wire1.write(MAX_REG_FIFO_DATA);
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(MAX30102_ADDR, (uint8_t)6) != 6) {
    while (Wire1.available()) Wire1.read();
    return false;
  }
  uint32_t r = 0, i = 0;
  r  = ((uint32_t)Wire1.read() << 16);
  r |= ((uint32_t)Wire1.read() << 8);
  r |=  (uint32_t)Wire1.read();
  i  = ((uint32_t)Wire1.read() << 16);
  i |= ((uint32_t)Wire1.read() << 8);
  i |=  (uint32_t)Wire1.read();
  red = r & 0x03FFFF;    /* 18-bit ADC - mask off unused high bits */
  ir  = i & 0x03FFFF;
  return true;
}

void resetMAXMeasurement()
{
  currentHR = 0.0f;
  filteredHR = 0.0f;
  irSmoothed = 0.0f;
  irMin = irMax = 0.0f;
  irDC = 0.0f;
  dcInitialised = false;
  lastAC = 0.0f;
  envelopeInitialised = false;
  aboveThreshold = false;
  haveLastBeatSample = false;
  pulseAmplitudeOK = false;
  sampleIndex = 0;
  spo2 = 0.0f;
  spo2Valid = false;
  for (int i = 0; i < SPO2_WINDOW; i++)
  {
    redBuffer[i] = 0;
    irBuffer[i] = 0;
  }
}

bool initMAX30102()
{
  uint8_t dummy;
  if (!readMAXRegister(0xFF, dummy)) return false;   /* part ID - presence check */

  if (!writeMAXRegister(MAX_REG_MODE_CONFIG, 0x40)) return false;  /* soft reset */
  delay(100);

  writeMAXRegister(MAX_REG_FIFO_WR_PTR, 0x00);
  writeMAXRegister(MAX_REG_OVF_COUNTER, 0x00);
  writeMAXRegister(MAX_REG_FIFO_RD_PTR, 0x00);

  /* See the PPG_SAMPLE_RATE_HZ comment block above - these two registers
   * are what make the effective rate 25Hz, and PPG_SAMPLE_RATE_HZ must
   * match them. */
  writeMAXRegister(MAX_REG_FIFO_CONFIG, 0x5F);   /* avg 4, rollover ON, a_full 15 */
  writeMAXRegister(MAX_REG_MODE_CONFIG, 0x03);   /* SpO2 mode: red + IR */

  /* RESTORED to the value that was working. 0x27 = 4096nA range, 100Hz,
   * 411us/18-bit. Unchanged from revision 2. */
  writeMAXRegister(MAX_REG_SPO2_CONFIG, 0x27);

  /* RESTORED to 0x32 (~10mA), the value in use when heart rate was
   * reading 91-94 bpm correctly on this hardware.
   *
   * Revision 5 raised this to 0x60 (~19mA) on the reasoning that more
   * drive current would give the beat detector more headroom. Nothing
   * measured called for it, and on real hardware it drove the IR channel
   * to 262143 - which is 0x3FFFF, the exact full-scale value of the
   * 18-bit ADC. A clipped channel is flat at the top, so the pulsatile
   * component the detector measures is gone entirely and HR reads 0
   * regardless of how good the detector is. SpO2 went on producing a
   * frozen 89.64% because its ratio math divides two clipped values
   * without complaint.
   *
   * The lesson being applied here: a constant that is producing correct
   * readings on real hardware does not get changed for a theoretical
   * improvement. */
  writeMAXRegister(MAX_REG_LED1_PA, led_pa);     /* red */
  writeMAXRegister(MAX_REG_LED2_PA, led_pa);     /* IR  */

  writeMAXRegister(MAX_REG_FIFO_WR_PTR, 0x00);
  writeMAXRegister(MAX_REG_OVF_COUNTER, 0x00);
  writeMAXRegister(MAX_REG_FIFO_RD_PTR, 0x00);
  delay(100);
  return true;
}

bool readMPU6050(int16_t &x, int16_t &y, int16_t &z)
{
  Wire1.beginTransmission(MPU6050_ADDR);
  Wire1.write(0x3B);
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(MPU6050_ADDR, (uint8_t)6) != 6)
  {
    while (Wire1.available()) Wire1.read();
    return false;
  }
  /* ------------------------------------------------------------------
   * REVISION 4 BUG FIX - UNSEQUENCED READS
   * ------------------------------------------------------------------
   * This was previously written as:
   *     x = ((int16_t)Wire1.read() << 8) | Wire1.read();
   * which is UNDEFINED BEHAVIOUR in C++. The two Wire1.read() calls in one
   * expression are unsequenced, so the compiler is free to evaluate the
   * right-hand one first - producing a byte-swapped result. Each read() has
   * the side effect of consuming a byte from the I2C buffer, so the order
   * genuinely matters and is genuinely unspecified.
   *
   * This is a plausible explanation for the field logs showing Accel X
   * swinging between roughly -30700 and +28700 on consecutive reads while
   * Y and Z sat still: a byte-swapped low-magnitude value lands at a wild
   * magnitude, and a swapped sign bit flips it. Those readings then tripped
   * MOTION_THRESHOLD constantly, and motionDetected gates SpO2
   * accumulation - so this bug could suppress SpO2 as well.
   *
   * Fixed by reading each byte into its own variable first, which sequences
   * the calls explicitly. */
  uint8_t xh = (uint8_t)Wire1.read();
  uint8_t xl = (uint8_t)Wire1.read();
  uint8_t yh = (uint8_t)Wire1.read();
  uint8_t yl = (uint8_t)Wire1.read();
  uint8_t zh = (uint8_t)Wire1.read();
  uint8_t zl = (uint8_t)Wire1.read();
  x = (int16_t)(((uint16_t)xh << 8) | xl);
  y = (int16_t)(((uint16_t)yh << 8) | yl);
  z = (int16_t)(((uint16_t)zh << 8) | zl);
  return true;
}

bool initMPU6050()
{
  Wire1.beginTransmission(MPU6050_ADDR);
  Wire1.write(0x6B);
  Wire1.write(0x00);     /* clear sleep bit */
  return (Wire1.endTransmission() == 0);
}

/* ---- Bug 4 fix: physically-motivated temperature validation ----
 * The MLX90614's "ambient" channel is the sensor die's own temperature.
 * In any room this sits between roughly 0 and 50C - a reading of 54.51C
 * (as seen in the field logs, identically for object AND ambient) is not
 * a measurement, it is a failed bit-banged I2C read returning whatever
 * was in the buffer. The old -60..300C window let that straight through.
 * Object gets a wider window because it legitimately reads skin, forehead
 * or a warm surface, but still nowhere near 300C. */
#define TEMP_AMBIENT_MIN_C   0.0
#define TEMP_AMBIENT_MAX_C  50.0
#define TEMP_OBJECT_MIN_C    0.0
#define TEMP_OBJECT_MAX_C   80.0

static bool isPlausibleAmbient(double t) {
  return !isnan(t) && t >= TEMP_AMBIENT_MIN_C && t <= TEMP_AMBIENT_MAX_C;
}
static bool isPlausibleObject(double t) {
  return !isnan(t) && t >= TEMP_OBJECT_MIN_C && t <= TEMP_OBJECT_MAX_C;
}

/* Reads both MLX90614 channels, retrying once on an implausible result.
 * On failure the previous good value is KEPT rather than overwritten -
 * a held-but-stale reading is more useful on the display than a garbage
 * one, and the OK flags tell the display which it is looking at. */
static void readTemperatures(void)
{
  for (int attempt = 0; attempt < 2; attempt++) {
    double rawAmbient = mlx.mlx90614ReadAmbientTempC();
    double rawObject  = mlx.mlx90614ReadTargetTempC();

    bool aOK = isPlausibleAmbient(rawAmbient);
    bool oOK = isPlausibleObject(rawObject);

    /* Both channels returning bit-identical values is the signature of a
     * duplicated/failed read rather than a real measurement - a real IR
     * sensor and its own die essentially never agree to this precision. */
    if (aOK && oOK && fabs(rawObject - rawAmbient) < 0.005) {
      aOK = oOK = false;
    }

    if (aOK) { ambientTempC = rawAmbient; mlxAmbientOK = true; }
    if (oOK) { objectTempC  = rawObject;  mlxObjectOK  = true; }

    if (aOK && oOK) return;      /* good read, done */

    if (attempt == 0) {
      delay(5);                  /* brief settle, then one retry */
      continue;
    }

    /* Second attempt also failed - mark whichever channel is bad as stale
     * so the display shows N/A rather than a number we don't believe. */
    if (!aOK) mlxAmbientOK = false;
    if (!oOK) mlxObjectOK = false;
  }
}

/* ---------------------------------------------------------------------
 * PPG beat detection - Bug 2 fix
 * ---------------------------------------------------------------------
 * Called exactly once per genuinely-new FIFO sample, so "one call" always
 * means "one PPG_SAMPLE_PERIOD_MS of elapsed time" - which is what makes
 * counting beat intervals in samples valid.
 *
 * Method: track a slow envelope (running min/max) of the smoothed IR
 * signal, then fire a beat when the signal crosses upward through 65% of
 * that envelope, re-arming only once it drops back under 35%. Because
 * both levels are derived from the envelope, the effective threshold
 * scales itself to whatever amplitude this finger is producing - which is
 * exactly what the old fixed 400-count threshold could not do. */
static void processPPGSample(uint32_t rawIr)
{
  ppgSampleCount++;

  if (rawIr < FINGER_THRESHOLD) return;

  if (!envelopeInitialised) {
    irSmoothed = (float)rawIr;
    irMin = irMax = irSmoothed;
    envelopeInitialised = true;
    return;
  }

  irSmoothed += IR_SMOOTHING_ALPHA * ((float)rawIr - irSmoothed);

  /* ------------------------------------------------------------------
   * DC REMOVAL - the fix for "IR looks healthy but HR stays 0"
   * ------------------------------------------------------------------
   * A PPG reading is a large, slowly-wandering DC level with a small
   * pulsatile component riding on top. The DC part moves as the finger
   * settles, warms, and presses harder - the field log showed it climbing
   * from 128349 to 145452 in about nineteen seconds, roughly 900 counts
   * per second, while a real pulse is only a few hundred counts.
   *
   * Running the envelope tracker directly on that signal fails in a
   * specific and total way. On a rising baseline every new sample is a
   * new maximum, so irMax follows it exactly, while irMin only creeps
   * upward at ENV_DECAY per sample and stays seconds behind. The envelope
   * therefore measures the DRIFT, not the pulse. The upper threshold sits
   * at irMin + 0.65*(irSmoothed - irMin), which the signal is always
   * above, so aboveThreshold latches true on the first sample; and the
   * re-arm level sits far below where a rising signal ever returns, so it
   * never clears. One beat, then silence - exactly what the board did.
   *
   * Subtracting a slowly-tracked baseline leaves the pulse centred on
   * zero, which is what the envelope and threshold logic below was
   * written for. The time constant is ~2 seconds (0.02 per sample at
   * 25Hz, a corner around 0.08Hz), comfortably below the 0.67Hz of even a
   * 40bpm pulse, so it removes the wander without eating the signal. This
   * is the same technique the ECG path has always used on its own
   * baseline; the PPG path simply never had it. */
  if (!dcInitialised) { irDC = irSmoothed; dcInitialised = true; }
  irDC += IR_DC_ALPHA * (irSmoothed - irDC);
  const float pulse = irSmoothed - irDC;      /* zero-centred pulse */

  /* Envelope: jump instantly to a new extreme, decay slowly back toward
   * the signal otherwise. Now tracking the pulse, not the baseline. */
  if (pulse > irMax) irMax = pulse;
  else               irMax -= (irMax - pulse) * ENV_DECAY;

  if (pulse < irMin) irMin = pulse;
  else               irMin += (pulse - irMin) * ENV_DECAY;

  float ac = irMax - irMin;
  lastAC = ac;
  if (ac < MIN_AC_AMPLITUDE) {
    /* No real pulsatile content - refuse to manufacture beats from noise. */
    pulseAmplitudeOK = false;
    return;
  }
  pulseAmplitudeOK = true;

  const float hi = irMin + ac * BEAT_THRESH_HI;
  const float lo = irMin + ac * BEAT_THRESH_LO;

  if (!aboveThreshold && irSmoothed > hi) {
    aboveThreshold = true;    /* rising edge through the upper level = one beat */

    if (haveLastBeatSample) {
      uint32_t deltaSamples = ppgSampleCount - lastBeatSample;
      if (deltaSamples > 0) {
        float interval_ms = (float)deltaSamples * PPG_SAMPLE_PERIOD_MS;
        float bpm = 60000.0f / interval_ms;
        if (bpm >= MIN_HR && bpm <= MAX_HR) {
          currentHR = bpm;
          if (filteredHR == 0.0f) filteredHR = bpm;
          else filteredHR = (0.75f * filteredHR) + (0.25f * bpm);
        }
      }
    }
    lastBeatSample = ppgSampleCount;
    haveLastBeatSample = true;

  } else if (aboveThreshold && irSmoothed < lo) {
    aboveThreshold = false;   /* re-arm (hysteresis) */
  }

  /* Timed out with no qualifying beat - clear the displayed HR rather
   * than leaving a stale number on screen. Counted in samples, so a loop
   * stall can't trip it prematurely. */
  if (haveLastBeatSample && (ppgSampleCount - lastBeatSample) > HR_TIMEOUT_SAMPLES) {
    filteredHR = 0.0f;
    currentHR = 0.0f;
    haveLastBeatSample = false;
    aboveThreshold = false;
  }
}

void calculateSpO2()
{
  if (sampleIndex < SPO2_WINDOW) { spo2Valid = false; return; }

  double redMean = 0.0, irMean = 0.0;
  for (int i = 0; i < SPO2_WINDOW; i++) { redMean += redBuffer[i]; irMean += irBuffer[i]; }
  redMean /= SPO2_WINDOW;
  irMean /= SPO2_WINDOW;
  if (redMean <= 0 || irMean <= 0) { spo2Valid = false; return; }

  double redSquareSum = 0.0, irSquareSum = 0.0;
  for (int i = 0; i < SPO2_WINDOW; i++)
  {
    double redACSample = (double)redBuffer[i] - redMean;
    double irACSample = (double)irBuffer[i] - irMean;
    redSquareSum += redACSample * redACSample;
    irSquareSum += irACSample * irACSample;
  }
  double redAC = sqrt(redSquareSum / SPO2_WINDOW);
  double irAC = sqrt(irSquareSum / SPO2_WINDOW);
  if (redAC <= 0 || irAC <= 0) { spo2Valid = false; return; }

  double redRatio = redAC / redMean;
  double irRatio = irAC / irMean;
  if (irRatio <= 0) { spo2Valid = false; return; }

  double R = redRatio / irRatio;
  if (R < 0.3 || R > 2.0) { spo2Valid = false; return; }

  double calculatedSpO2 = 110.0 - (25.0 * R);
  if (calculatedSpO2 > 100.0) calculatedSpO2 = 100.0;
  if (calculatedSpO2 < 70.0) calculatedSpO2 = 70.0;

  if (!spo2Valid) spo2 = calculatedSpO2;
  else spo2 = (0.8 * spo2) + (0.2 * calculatedSpO2);
  spo2Valid = true;

  /* Slide the window by half so SpO2 refreshes every ~2s instead of
   * every ~4s, while still averaging over a full 4s of data. */
  for (int i = 0; i < SPO2_WINDOW / 2; i++)
  {
    redBuffer[i] = redBuffer[i + SPO2_WINDOW / 2];
    irBuffer[i] = irBuffer[i + SPO2_WINDOW / 2];
  }
  sampleIndex = SPO2_WINDOW / 2;
}

/* ---------------------------------------------------------------------
 * MAX30102 service - Bug 1 fix (the FIFO pointer check)
 * ---------------------------------------------------------------------
 * This is where "match the sampling frequencies" actually happens. We do
 * not assume anything about how much time has passed since the last poll,
 * and we do not read a fixed number of samples. We ask the sensor how
 * many new samples it has, and take exactly that many. */
static int max30102_service(void)
{
  uint8_t wr = 0, ovf = 0, rd = 0;
  bool ptr_ok = readMAXFifoPointers(wr, ovf, rd);

  int available;
  if (!ptr_ok) {
    /* REVISION 6 FALLBACK. If the pointer read fails we can no longer tell
     * whether a new sample exists - but going permanently silent is the
     * worse outcome, and that is what revision 5 did. Degrade to revision
     * 2's behaviour instead: read one sample anyway. The beat detector may
     * see an occasional duplicate, which costs some timing accuracy; a
     * sensor that reports nothing at all costs everything. */
    max_ptr_fail++;
    available = 1;
  } else {
    /* FIFO pointers are 5-bit and wrap at 32. */
    available = (int)wr - (int)rd;
    if (available < 0) available += 32;
    if (available == 0) { max_empty++; return 0; }
  }

  if (available > MAX_FIFO_DRAIN_PER_POLL) available = MAX_FIFO_DRAIN_PER_POLL;

  int got = 0;
  uint32_t reds[MAX_FIFO_DRAIN_PER_POLL];
  uint32_t irs[MAX_FIFO_DRAIN_PER_POLL];
  for (int s = 0; s < available; s++) {
    if (!readMAX30102One(reds[s], irs[s])) { max_read_fail++; break; }
    got++;
  }
  if (got <= 0) return 0;

  for (int s = 0; s < got; s++) {
    redValue = reds[s];
    irValue  = irs[s];

    /* Finger presence with debounce, evaluated per real sample. */
    if (irValue >= FINGER_THRESHOLD) {
      fingerAboveCount++;
      fingerBelowCount = 0;
      if (!fingerPresent && fingerAboveCount >= FINGER_DEBOUNCE_COUNT) {
        resetMAXMeasurement();
        fingerPresent = true;
        fingerAboveCount = 0;
      }
    } else {
      fingerBelowCount++;
      fingerAboveCount = 0;
      if (fingerPresent && fingerBelowCount >= FINGER_DEBOUNCE_COUNT) {
        resetMAXMeasurement();
        fingerPresent = false;
        fingerBelowCount = 0;
      }
    }

    if (!fingerPresent) continue;

    /* Motion gates SpO2 accumulation (a 4-second RMS window is genuinely
     * ruined by movement) but NOT beat detection. Revision 2 gated both,
     * which meant ordinary hand tremor on a breadboard-mounted sensor
     * could suppress heart rate entirely. The beat detector has its own
     * hysteresis and physiological rate limits to reject motion spikes. */
    processPPGSample(irValue);

    if (!motionDetected) {
      if (sampleIndex < SPO2_WINDOW) {
        redBuffer[sampleIndex] = redValue;
        irBuffer[sampleIndex] = irValue;
        sampleIndex++;
      }
      if (sampleIndex >= SPO2_WINDOW) calculateSpO2();
    }
  }

  return got;
}

/* Keeps the IR channel inside the ADC's usable range.
 *
 * Called once a second, and only with a finger present - with no finger
 * the reading is just the sensor's own dark/ambient floor and says nothing
 * about what drive current a real measurement will need.
 *
 * Steps rather than jumps, because each change invalidates the beat
 * detector's envelope and the SpO2 window (both are built from absolute
 * signal levels), so resetMAXMeasurement() has to run and the next couple
 * of seconds of data are discarded. Small steps converge in a few seconds
 * and stay converged; large ones oscillate. */
static void max30102_autorange(void) {
  if (!fingerPresent) return;

  uint8_t before = led_pa;

  if (irValue >= IR_SATURATED && led_pa > LED_PA_MIN) {
    led_pa = (led_pa >= LED_PA_MIN + 0x08) ? (uint8_t)(led_pa - 0x08) : LED_PA_MIN;
  } else if (irValue < IR_TOO_WEAK && led_pa < LED_PA_MAX) {
    led_pa = (led_pa <= LED_PA_MAX - 0x08) ? (uint8_t)(led_pa + 0x08) : LED_PA_MAX;
  }

  if (led_pa != before) {
    writeMAXRegister(MAX_REG_LED1_PA, led_pa);
    writeMAXRegister(MAX_REG_LED2_PA, led_pa);
    resetMAXMeasurement();
  }
}

static void service_vitals_and_motion(void) {
  if (millis() - lastSensorPollMs < SENSOR_POLL_INTERVAL_MS) return;
  lastSensorPollMs = millis();
  unsigned long stall_t0 = micros();

  readMPU6050(ax, ay, az);

  if (motionInitialized) {
    long motionDelta = abs((long)ax - prevAx) + abs((long)ay - prevAy) + abs((long)az - prevAz);
    motionDetected = motionDelta > MOTION_THRESHOLD;
  }
  prevAx = ax; prevAy = ay; prevAz = az;
  motionInitialized = true;

  int n = max30102_service();
  if (n < 0) {
    /* I2C failure talking to the MAX30102 - treat as finger removed so
     * stale HR/SpO2 don't linger on the display. */
    fingerPresent = false;
    fingerAboveCount = 0;
    fingerBelowCount = 0;
    resetMAXMeasurement();
  }

  unsigned long stall_us = micros() - stall_t0;
  if (stall_us > max_vitals_stall_us) max_vitals_stall_us = stall_us;
}


/* =====================================================================
 * SECTION 3 - DISPLAY (20x4 LCD + Serial)
 * ===================================================================== */

/* Formats `val` to one decimal place without snprintf's "%f", which isn't
 * guaranteed across every Arduino-core libc (would print "?" on some). */
static void formatFloat1(float val, char *buf, size_t bufSize) {
  bool neg = val < 0;
  if (neg) val = -val;
  long tenths = (long)(val * 10.0f + 0.5f);
  long whole = tenths / 10;
  long frac = tenths % 10;
  snprintf(buf, bufSize, "%s%ld.%ld", neg ? "-" : "", whole, frac);
}

/* Total acceleration magnitude in g - a single number that proves the IMU
 * is live and reacts visibly when the board is moved or tilted. ~1.00g
 * at rest in any orientation. */
static float accelMagnitudeG(void) {
  float fx = (float)ax / ACCEL_LSB_PER_G;
  float fy = (float)ay / ACCEL_LSB_PER_G;
  float fz = (float)az / ACCEL_LSB_PER_G;
  return sqrtf(fx * fx + fy * fy + fz * fz);
}

/* 20x4 layout - every subsystem visible at once:
 *
 *   Row 0:  ECG:N* HR:072bpm      classifier + ECG-derived rate
 *   Row 1:  PPG:071 SpO2: 98%     MAX30102 - both values, as requested
 *   Row 2:  Obj:36.5C Amb:29.4C   MLX90614 - both channels, as requested
 *   Row 3:  Motion:STILL 1.01g    MPU6050
 *
 * The '*' after the ECG class means the last beat failed the plausibility
 * gate (weak signal / leads off / impossible rate), so the class shown is
 * the last one that passed rather than the newest one. */
static void lcdUpdate(void) {
  if (!lcd) return;
  char line[LCD_COLS + 1];
  char a[12], b[12];

  /* Row 0 - ECG classification and ECG-derived heart rate */
  if (was_leads_off) {
    snprintf(line, sizeof(line), "ECG: LEADS OFF");
  } else if (latest_hr_bpm > 0) {
    snprintf(line, sizeof(line), "ECG:%s%s HR:%3dbpm",
             CLASS_SHORT[latest_class_id / 100],
             last_beat_reliable ? " " : "*",
             latest_hr_bpm);
  } else {
    snprintf(line, sizeof(line), "ECG:%s%s HR:--",
             CLASS_SHORT[latest_class_id / 100],
             last_beat_reliable ? " " : "*");
  }
  lcd->setCursor(0, 0);
  lcd->printPadded(line, LCD_COLS);

  /* Row 1 - MAX30102 heart rate AND SpO2 */
  if (!fingerPresent) {
    snprintf(line, sizeof(line), "PPG: place finger");
  } else {
    int hr = (int)(filteredHR + 0.5f);
    if (hr > 0 && spo2Valid) {
      snprintf(line, sizeof(line), "PPG:%3d SpO2:%3d%%", hr, (int)(spo2 + 0.5f));
    } else if (hr > 0) {
      snprintf(line, sizeof(line), "PPG:%3d SpO2: --", hr);
    } else if (spo2Valid) {
      snprintf(line, sizeof(line), "PPG:--- SpO2:%3d%%", (int)(spo2 + 0.5f));
    } else {
      snprintf(line, sizeof(line), "PPG: measuring...");
    }
  }
  lcd->setCursor(0, 1);
  lcd->printPadded(line, LCD_COLS);

  /* Row 2 - MLX90614 object AND ambient temperature */
  if (mlxObjectOK) formatFloat1((float)objectTempC, a, sizeof(a));
  else             snprintf(a, sizeof(a), "--");
  if (mlxAmbientOK) formatFloat1((float)ambientTempC, b, sizeof(b));
  else              snprintf(b, sizeof(b), "--");
  snprintf(line, sizeof(line), "Obj:%sC Amb:%sC", a, b);
  lcd->setCursor(0, 2);
  lcd->printPadded(line, LCD_COLS);

  /* Row 3 - MPU6050 motion state and magnitude */
  formatFloat1(accelMagnitudeG(), a, sizeof(a));
  snprintf(line, sizeof(line), "Motion:%s %sg",
           motionDetected ? "MOVE " : "STILL", a);
  lcd->setCursor(0, 3);
  lcd->printPadded(line, LCD_COLS);
}

static void service_print_and_temp(void) {
  if (millis() - lastPrintTime < 1000) return;
  lastPrintTime = millis();
  unsigned long pt_stall_t0 = micros();

  /* MLX90614's bit-banged read costs ~127ms, so it is deliberately
   * decoupled from the 1Hz display refresh and runs every 3s. The display
   * reprints the held value in between. */
  if (millis() - lastTempReadTime >= TEMP_READ_INTERVAL_MS) {
    lastTempReadTime = millis();
    readTemperatures();
  }

  /* ------------------------------------------------------------------
   * REVISION 4: SERIAL OUTPUT CUT DOWN HARD
   * ------------------------------------------------------------------
   * Revision 3 printed three long lines once a second. Measured on
   * hardware, this function cost ~82ms per call even on passes that
   * skipped the MLX90614 read - that is pure Serial transmit time (at
   * 115200 baud, ~11.5 bytes per ms, so ~940 bytes of text), and every
   * one of those milliseconds is ECG sampling time lost. It also
   * overflowed the host serial tool, which is what produced the constant
   * "data too long to fit in transmit buffer for read" spam that made the
   * log almost unreadable.
   *
   * Now: ONE compact line per second, roughly 80 characters (~7ms). The
   * per-beat [ECG] lines still carry the full diagnostic detail, which is
   * where it actually belongs. */
  /* V:<hr>,<spo2>,<objC*10>,<ambC*10>,<motion>,<ir>,<ecgbpm>,<class>,<i2c_us>
   * -1 = not available. Temperatures are sent as tenths of a degree so the
   * whole line stays integer-only and trivial to parse.
   *
   * <ir> is the raw MAX30102 IR reading, added in revision 5 specifically
   * to diagnose "I put my finger on and HR stayed blank". Finger presence
   * is decided by comparing exactly this number against FINGER_THRESHOLD,
   * so seeing it live tells you immediately whether the sensor is picking
   * up your finger at all or whether the threshold is simply set wrong for
   * your mounting - which is otherwise pure guesswork. */
  /* REVISION 7: back to the readable two-line block. */
  Serial.print("[VITALS] ");
  if (!fingerPresent) {
    Serial.print("Place finger...");
  } else {
    Serial.print("HR: ");
    Serial.print(filteredHR, 2);
    Serial.print(" bpm");
    Serial.print("  SpO2: ");
    Serial.print(spo2, 2);
    Serial.print("%");
  }

  if (motionDetected) Serial.print("  [MOTION]");

  Serial.print("  Accel X: "); Serial.print(ax);
  Serial.print("  Accel Y: "); Serial.print(ay);
  Serial.print("  Accel Z: "); Serial.print(az);

  Serial.print("  Temp: ");
  if (mlxObjectOK) { Serial.print(objectTempC, 2); Serial.print("C"); }
  else Serial.print("N/A");

  Serial.print(" (Ambient: ");
  if (mlxAmbientOK) { Serial.print(ambientTempC, 2); Serial.print("C)"); }
  else Serial.print("N/A)");

  /* Raw IR and the MAX30102 fault counters ride on the end of this line.
   * IR is what finger detection actually compares against, so seeing it
   * live is the difference between tuning FINGER_THRESHOLD from
   * measurement and guessing at it. p/e/r are pointer-read failures,
   * genuinely-empty polls, and data-read failures since the last report -
   * all three zero alongside a non-zero IR means the sensor is healthy. */
  max30102_autorange();

  Serial.print("  IR: ");
  Serial.print(irValue);
  if (irValue >= IR_SATURATED) Serial.print(" SAT!");
  Serial.print(" LED:0x");
  Serial.print(led_pa, HEX);
  /* Pulse envelope after DC removal. This is the number the beat detector
   * actually thresholds against, so if HR is blank this says immediately
   * whether there is a pulse to find (hundreds of counts) or not
   * (single digits). */
  Serial.print(" AC:");
  Serial.print((int)(lastAC + 0.5f));
  Serial.print(" [p");
  Serial.print(max_ptr_fail);
  Serial.print(" e");
  Serial.print(max_empty);
  Serial.print(" r");
  Serial.print(max_read_fail);
  Serial.println("]");
  max_ptr_fail = max_empty = max_read_fail = 0;

  Serial.print("[ECG-HR] last RR-derived HR: ");
  if (latest_hr_bpm > 0) { Serial.print(latest_hr_bpm); Serial.println(" bpm"); }
  else Serial.println("--");

  lcdUpdate();

  (void)pt_stall_t0;
  max_vitals_stall_us = 0;
  ecg_behind_events = 0;
}


/* =====================================================================
 * SETUP
 * ===================================================================== */
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  /* ---- ECG (AD8232) ---- */
  pinMode(LEAD_OFF_PLUS_PIN, INPUT);
  pinMode(LEAD_OFF_MINUS_PIN, INPUT);
  pinMode(AD8232_SDN_PIN, OUTPUT);
  digitalWrite(AD8232_SDN_PIN, HIGH);
  rpeak_init(&rp_state);

  /* ---- Hardware I2C0 ---- */
  Wire1.begin();
  delay(100);

  Serial.println();
  Serial.println("Initializing MAX30102...");
  if (initMAX30102()) {
    Serial.print("MAX30102 SUCCESS - FIFO rate ");
    Serial.print(PPG_SAMPLE_RATE_HZ);
    Serial.println(" Hz (100Hz sampling, 4x averaging), rollover enabled");
  } else {
    Serial.println("MAX30102 FAILED");
  }

  Serial.println("Initializing MPU6050...");
  if (initMPU6050()) Serial.println("MPU6050 SUCCESS");
  else Serial.println("MPU6050 FAILED");

  /* ---- MLX90614 (software I2C on its own pins) ---- */
  Serial.println("Initializing MLX90614 (VEGA software I2C, pins 5/6)...");
  readTemperatures();
  if (mlxAmbientOK) {
    Serial.print("MLX90614 SUCCESS - ambient ");
    Serial.print(ambientTempC, 2);
    Serial.println("C");
  } else {
    Serial.println("MLX90614 FAILED (check wiring on pins 5/6, separate from I2C0)");
  }

  resetMAXMeasurement();

  /* ---- 20x4 I2C LCD ---- */
  Serial.println("Detecting LCD I2C backpack on Wire1 (I2C0)...");
  lcdAddr = lcd_i2c_autodetect(Wire1);
  if (lcdAddr != 0x00) {
    lcd = new LCD_I2C_PCF8574(Wire1, lcdAddr, LCD_COLS, LCD_ROWS);
    lcdOK = lcd->begin();
    if (lcdOK) {
      /* NOTE: this only means the PCF8574 expander ACKed every write. The
       * expander ACKs unconditionally and there is no return path from the
       * HD44780 behind it, so this is NOT proof the panel initialised.
       * Look at the screen. */
      Serial.println("LCD backpack responding (ACK != panel initialised)");

      /* ================================================================
       * LCD CONTRAST SELF-TEST - revision 5
       * ================================================================
       * Reported symptom: backlight on, nothing else, ever. That single
       * observation cannot distinguish between two very different faults,
       * so this test forces them apart before the demo starts.
       *
       * For 4 seconds every cell on the screen is filled with the solid
       * block glyph (0xFF), which is the highest-contrast thing an HD44780
       * can display - it lights every pixel. Then the screen clears and
       * normal text appears. What you see decides the diagnosis:
       *
       *   BLOCKS APPEAR, then text does NOT
       *       -> The panel and wiring are fine and initialisation worked.
       *          Contrast is set so that thin text strokes are invisible
       *          while fully-lit cells still show. Turn the blue trimmer
       *          pot on the backpack during the block phase until the
       *          blocks are crisp black, then keep turning slightly until
       *          text is readable. This is a five-second fix, not a code
       *          problem.
       *
       *   NOTHING APPEARS AT ALL, not even blocks
       *       -> Either contrast is at a hard extreme (turn the pot through
       *          its FULL range during the block phase - it is often 10+
       *          turns end to end on these small pots), or the module is
       *          not getting 5V. The PCF8574 backpack happily ACKs on I2C
       *          at 3.3V, which is why the sketch reports it as present,
       *          while the LCD's own drive circuit never reaches a usable
       *          operating point. Measure VCC at the module.
       *
       *   BLOCKS APPEAR AND SO DOES TEXT
       *       -> Working. Nothing to do.
       *
       * Revision 2 had a one-row version of this and I removed it in
       * revision 3, reasoning that a row of blocks on a panel whose init
       * had failed was indistinguishable from the failure itself. That was
       * right about one row and wrong about the idea: a FULL screen of
       * blocks, immediately followed by a clear, is unambiguous - an
       * uninitialised panel shows blocks on alternating rows permanently
       * and never clears. */
      Serial.println("[LCD] self-test: filling screen with blocks for 4s.");
      Serial.println("[LCD]   Blocks but no text afterwards -> turn the contrast pot.");
      Serial.println("[LCD]   Nothing at all -> full pot sweep, then check module VCC is 5V.");
      for (uint8_t row = 0; row < LCD_ROWS; row++) {
        lcd->setCursor(0, row);
        for (uint8_t col = 0; col < LCD_COLS; col++) lcd->write(0xFF);
      }
      delay(4000);
      lcd->clear();

      lcd->setCursor(0, 0);
      lcd->print("ECG + Vitals");
      lcd->setCursor(0, 1);
      lcd->print("Starting up...");
      lcd->setCursor(0, 2);
      lcd->print("Place finger on");
      lcd->setCursor(0, 3);
      lcd->print("MAX30102 sensor");
      /* Revision 2 painted a full row of 0xFF block glyphs here as a
       * "contrast test". That is removed: on a panel whose init had
       * failed it was indistinguishable from the failure symptom itself,
       * and it actively contributed to the all-blocks appearance being
       * reported from the field. If you need a contrast reference, the
       * four text rows above serve the same purpose without ambiguity. */
    } else {
      Serial.println("LCD FAILED (address found but a write wasn't ACKed - check wiring)");
    }
  } else {
    Serial.println("LCD SKIPPED (no backpack detected - Serial output still works)");
  }

  Serial.println();
  Serial.println("==================================================");
  Serial.println(" AD8232 ECG (CNN) + MAX30102 + MPU6050 + MLX90614");
  Serial.println(" VEGA ARIES - COMBINED, revision 9");
  Serial.println("==================================================");
  Serial.print("# ECG sampling ");
  Serial.print(SAMPLING_RATE_HZ);
  Serial.print(" Hz   PPG ");
  Serial.print(PPG_SAMPLE_RATE_HZ);
  Serial.print(" Hz (sensor-clocked)   vitals poll ");
  Serial.print(SENSOR_POLL_INTERVAL_MS);
  Serial.println(" ms   display 1 Hz   temp 3 s");
  Serial.println("# REMINDER: verify LEAD_OFF_PLUS/MINUS and SDN pins against this board's pinout");
  Serial.println("# LCD must be powered from 5V - a 3.3V module ACKs on I2C but never displays");
  Serial.println();
}


/* =====================================================================
 * LOOP - three independently self-paced services, none of which can
 * block or skip the others. No delay() anywhere in this sketch.
 * ===================================================================== */
void loop() {
  service_ecg();
  service_vitals_and_motion();
  service_print_and_temp();
}
