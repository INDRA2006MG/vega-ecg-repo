#ifndef SENSOR_THRESHOLDS_H
#define SENSOR_THRESHOLDS_H

/*
 * sensor_thresholds.h - decision thresholds for the sensors stage 06/07
 * still need to add.
 *
 * The ECG path already has its own decision layer: the CNN's argmax
 * (model_forward.h) plus RR interval / QRS width (rpeak.h). This file is
 * the same kind of thing for the two sensors left in scope - MAX30102
 * (heart rate + SpO2, I2C) and an external accelerometer for motion/fall.
 *
 * BOARD NOTE: roadmap.html assumed ARIES IoT v2.0 with an onboard BMI088,
 * which would need no wiring or driver at all. The board actually in use is
 * plain ARIES (no onboard IMU), so the fall-detection sensor is an external
 * MPU6050 over I2C, same as originally planned before that roadmap update -
 * wiring and driver both still need doing (stage 06).
 *
 * These are DEMO thresholds, not medical-device thresholds: picked to be
 * robust in front of a live audience (movement, imperfect finger contact,
 * one fixed unpersonalized cutoff for everyone) rather than diagnostic-
 * grade. Say so if asked, same as the QRS-width caveat in
 * faculty_qa_briefing.md - a real number from a real sensor, not a
 * clinically validated one.
 *
 * WHERE THESE NUMBERS COME FROM
 *   Heart rate / SpO2 ranges - standard adult resting vital-sign reference
 *   ranges (resting HR 60-100 bpm, SpO2 >=95% normal, <90% hypoxemia) used
 *   throughout consumer and clinical pulse oximetry; not specific to this
 *   project.
 *   Fall detection - the free-fall / impact / post-impact-stillness shape
 *   is the standard threshold-based wearable fall detector (see e.g.
 *   Bourke et al. 2007, "Evaluation of a threshold-based tri-axial
 *   accelerometer fall detection algorithm"); the g-values below are a
 *   commonly cited starting point from that literature, NOT measured on
 *   this project's own hardware - there is no way to validate a fall
 *   threshold without a real MPU6050 on the actual mounting point, which is
 *   stage-06 board work this session cannot do.
 *
 * All values that get compared against live sensor readings are plain
 * integers (bpm, percent, milli-g) so the fusion layer stays fixed-point,
 * matching the rest of this codebase - no floating point needed anywhere
 * in this file.
 */

/* ---------------------------------------------------------------------------
 * MAX30102 - heart rate (bpm), from the PPG signal.
 *
 * This is a SECOND, independent heart-rate measurement - rpeak.h already
 * derives one from the ECG's R-R interval. The two should agree; a fusion
 * layer that checks PPG-HR against ECG-HR gets a free plausibility check
 * ("is this device even on the person correctly") that neither sensor
 * alone can give itself. See classify_health_state() below for where that
 * could plug in later - not implemented, since neither driver exists yet.
 * ------------------------------------------------------------------------- */
#define HR_BRADYCARDIA_BPM      50   /* below this: clinically bradycardic.
                                       * (60 is the textbook cutoff; 50 avoids
                                       * flagging a fit, resting teammate as
                                       * "abnormal" mid-demo - tighten to 60
                                       * for a stricter, less forgiving demo) */
#define HR_TACHYCARDIA_BPM     120   /* above this: clinically tachycardic.
                                       * (100 is the textbook cutoff; 120
                                       * gives headroom for "just walked over
                                       * to the sensor" without a false alarm) */
#define HR_CRITICAL_LOW_BPM     35   /* below this: treat as an emergency (or
                                       * a sensor fault), not routine
                                       * bradycardia - escalate straight to
                                       * CRITICAL rather than WARNING */
#define HR_CRITICAL_HIGH_BPM   160   /* above this: same, for tachycardia */

/* MAX30102 also reports raw IR amplitude, which doubles as a "finger
 * present" signal - a reading below this is "no contact", not "flatlining".
 * PLACEHOLDER: the right cutoff depends on the sensor's ambient light and
 * exact mounting, so this needs confirming against the real hardware once
 * it's wired (stage 06) - do not trust this number for a go/no-go decision
 * until then. */
#define HR_MIN_IR_FOR_VALID   50000  /* TODO: confirm on real hardware */

/* ---------------------------------------------------------------------------
 * MAX30102 - SpO2 (%)
 * ------------------------------------------------------------------------- */
#define SPO2_NORMAL_MIN_PCT     95   /* >=95%: normal */
#define SPO2_MILD_HYPOXEMIA_PCT 90   /* 90-94%: mild - worth a soft alert */
#define SPO2_CRITICAL_PCT       85   /* <85%: urgent - buzzer + OLED, not
                                       * just logged to the dashboard */

/* ---------------------------------------------------------------------------
 * External accelerometer (MPU6050) - motion / fall.
 *
 * Accelerometer only: a gyro-fused (Kalman/complementary-filter) detector
 * is more accurate but is not a hackathon-timeline task: threshold-based on
 * the acceleration vector magnitude is what stage 06/07 should build.
 *
 * All values are in units of g (1g = 9.80665 m/s^2 = the magnitude
 * sqrt(ax^2+ay^2+az^2) reads when the sensor is sitting still). Convert
 * raw MPU6050 counts to milli-g on read (counts * 1000 / sensitivity for
 * the configured range) and compare against the _MILLI constants below -
 * keeps the whole path integer, no floating point on-device.
 * ------------------------------------------------------------------------- */
#define FALL_FREEFALL_G_MILLI      400  /* magnitude below this = free-fall
                                          * (weightlessness during the drop,
                                          * before impact) */
#define FALL_IMPACT_G_MILLI       2500  /* magnitude above this, shortly
                                          * after a free-fall window = impact */
#define FALL_FREEFALL_MIN_MS        100 /* free-fall must last at least this
                                          * long - filters out a quick bump or
                                          * jolt that never actually goes
                                          * weightless */
#define FALL_IMPACT_WINDOW_MS        500 /* impact must follow the free-fall
                                          * window within this long, or it is
                                          * treated as unrelated */
#define FALL_STILLNESS_WINDOW_MS   2000  /* after impact, magnitude must stay
                                          * near 1g for this long to CONFIRM a
                                          * fall - a dropped phone/device
                                          * bounces and keeps moving; a person
                                          * who has fallen mostly doesn't */
#define FALL_STILLNESS_BAND_MILLI   300  /* "near 1g" = within +-this many
                                          * milli-g of 1000 during the
                                          * stillness check */

/* ---------------------------------------------------------------------------
 * Temperature - non-contact IR sensor (e.g. MLX90614), replacing the GPS
 * slot after GPS was confirmed unusable indoors at the venue.
 *
 * NOTE ON WHERE THIS IS ACTUALLY USED: the temperature sensor is being read
 * by a companion Arduino board (MAX30102/MPU6050/temp all moved off VEGA
 * onto a separate microcontroller, to sidestep the still-unconfirmed
 * taurus/i2c.h transaction sequence under deadline pressure - see
 * arduino/ecg_companion_sensors/). classify_health_state() below is NOT
 * being extended to take a temp argument, on purpose: it already has 13
 * passing precedence tests in test_fusion.c and changing its signature this
 * close to the deadline risks breaking something for a sensor whose
 * fusion logic now lives off-VEGA anyway. The Python bridge script that
 * merges VEGA's ECG stream with the Arduino's vitals/motion/temp stream is
 * the actual fusion point now - it re-implements this same precedence
 * shape with temperature added, referencing these same numbers, rather
 * than this C function. If sensor fusion ever moves back onto VEGA itself,
 * these constants are ready and classify_health_state() is the place to
 * wire them in.
 *
 * A non-contact IR thermometer reads skin/forehead temperature, which runs
 * a bit below oral/core body temperature (commonly ~0.3-0.5 C lower) -
 * these cutoffs are set for a non-contact skin reading, not core temp; say
 * so if asked, same demo-not-clinical caveat as HR/SpO2 above.
 * ------------------------------------------------------------------------- */
#define TEMP_HYPOTHERMIA_C        35   /* below this (skin, non-contact): worth
                                         * flagging - cold room/AC vent airflow
                                         * on the sensor is a likely false
                                         * trigger during a demo, so this is
                                         * WARNING not CRITICAL by itself */
#define TEMP_FEVER_C               38   /* above this (skin, non-contact):
                                         * elevated - textbook oral fever is
                                         * 38.0C but skin reads a bit low, so
                                         * this already has some margin built
                                         * in rather than needing a separate
                                         * offset */
#define TEMP_CRITICAL_LOW_C        34   /* below this: escalate regardless of
                                         * other signals (or the sensor isn't
                                         * making good contact/reading air) */
#define TEMP_CRITICAL_HIGH_C       39   /* above this: escalate regardless -
                                         * high fever territory */

/* ---------------------------------------------------------------------------
 * Fusion: one combined severity from ECG rhythm + HR + SpO2 + fall.
 *
 * Mirrors model_forward.h's CLASS_NAMES ordering - named here for
 * readability at call sites rather than redefined there, so NUM_CLASSES /
 * the ordering itself still has exactly one source of truth.
 * ------------------------------------------------------------------------- */
#define ECG_CLASS_NORMAL   0
#define ECG_CLASS_SVEB     1
#define ECG_CLASS_VEB      2
#define ECG_CLASS_FUSION   3
#define ECG_CLASS_UNKNOWN  4

typedef enum {
    HEALTH_NORMAL = 0,
    HEALTH_WARNING,     /* one signal is outside its normal range - worth a
                          * dashboard note, not yet an alarm */
    HEALTH_CRITICAL      /* buzzer + OLED alert territory */
} health_severity_t;

/*
 * Combines the ECG classifier's verdict with MAX30102 and fall-detector
 * readings into one severity. Precedence, highest first:
 *
 *   1. A fall is a safety event independent of everything else - it wins
 *      outright. Someone who falls with an otherwise-normal rhythm is still
 *      CRITICAL.
 *   2. SpO2 or HR past its own CRITICAL cutoff - escalate regardless of
 *      what the ECG says (a single sensor pegged at a dangerous value does
 *      not need a second signal to confirm it).
 *   3. A dangerous-shaped beat (VEB/Fusion) COMBINED with an abnormal HR or
 *      SpO2 - two independent signals agreeing is a materially stronger
 *      case than either alone, so this escalates from WARNING to CRITICAL
 *      even though neither one alone crossed its CRITICAL cutoff.
 *   4. Any single abnormal signal alone (VEB/Fusion/SVEB rhythm, or HR/SpO2
 *      outside normal but not critical) - WARNING.
 *   5. Otherwise - NORMAL.
 *
 * hr_bpm and spo2_pct take -1 for "not yet a valid reading" (e.g. no finger
 * contact, or the sensor hasn't produced its first value yet) and are
 * skipped rather than compared. ecg_class takes -1 for "no beat classified
 * yet". fall_detected is 1 only on the beat/tick where the free-fall ->
 * impact -> stillness sequence just completed, not held high afterward -
 * the caller (stage 07, once written) is responsible for latching a
 * CRITICAL fall state for display even after this function stops being
 * called with fall_detected=1.
 */
static inline health_severity_t classify_health_state(
    int ecg_class, int hr_bpm, int spo2_pct, int fall_detected)
{
    if (fall_detected) return HEALTH_CRITICAL;

    if (spo2_pct >= 0 && spo2_pct < SPO2_CRITICAL_PCT)
        return HEALTH_CRITICAL;
    if (hr_bpm >= 0 && (hr_bpm <= HR_CRITICAL_LOW_BPM || hr_bpm >= HR_CRITICAL_HIGH_BPM))
        return HEALTH_CRITICAL;

    int rhythm_dangerous = (ecg_class == ECG_CLASS_VEB || ecg_class == ECG_CLASS_FUSION);
    int hr_abnormal   = (hr_bpm   >= 0 && (hr_bpm <= HR_BRADYCARDIA_BPM || hr_bpm >= HR_TACHYCARDIA_BPM));
    int spo2_abnormal = (spo2_pct >= 0 && spo2_pct < SPO2_NORMAL_MIN_PCT);

    if (rhythm_dangerous && (hr_abnormal || spo2_abnormal))
        return HEALTH_CRITICAL;

    if (rhythm_dangerous)              return HEALTH_WARNING;
    if (ecg_class == ECG_CLASS_SVEB)   return HEALTH_WARNING;
    if (hr_abnormal || spo2_abnormal)  return HEALTH_WARNING;

    return HEALTH_NORMAL;
}

#endif /* SENSOR_THRESHOLDS_H */
