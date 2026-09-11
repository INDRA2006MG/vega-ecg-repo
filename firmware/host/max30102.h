#ifndef MAX30102_H
#define MAX30102_H

#include <stdint.h>
#include "sensor_thresholds.h"   /* HR_MIN_IR_FOR_VALID */

/*
 * max30102.h - driver for the MAX30102 heart-rate/SpO2 sensor, stage 06's
 * first new sensor (I2C, address 0x57).
 *
 * SCOPE OF THIS FIRST STEP: get real IR/Red samples flowing off the sensor
 * and printed over serial - NOT a finished heart-rate/SpO2 algorithm yet.
 * That is deliberate and matches how every other stage in this project got
 * built: stage 03 first proved raw peak indices matched the desktop before
 * anything was built on top of the R-peak detector; stage 04 measured real
 * data before picking a preprocessing fix. A red/IR-ratio SpO2 calibration
 * and a PPG peak detector are both easier to get right once we've actually
 * looked at what this sensor's waveform looks like on real hardware, rather
 * than guessing blind. sensor_thresholds.h's HR_BRADYCARDIA_BPM /
 * SPO2_CRITICAL_PCT etc. are already decided and waiting for this driver to
 * start feeding them real numbers.
 *
 * WHAT'S SOLID vs WHAT'S A PLACEHOLDER
 * The register map, init sequence and FIFO packing below are straight from
 * the MAX30102 datasheet - not board-specific, should be right as written.
 * The two functions that actually talk to the I2C bus (max30102_i2c_write,
 * max30102_i2c_read) are the SDK-specific part, exactly like main.c's
 * console/cycle-counter isolation - their declarations are here, but the
 * body needs writing against <taurus/i2c.h>'s real API once we have it
 * (see max30102.c for the placeholder and what to check for).
 */

/* ---- I2C address (7-bit, not shifted) ---- */
#define MAX30102_I2C_ADDR   0x57

/* ---- registers used here (see the MAX30102 datasheet for the full map) ---- */
#define MAX30102_REG_INTR_STATUS_1  0x00
#define MAX30102_REG_INTR_ENABLE_1  0x02
#define MAX30102_REG_INTR_ENABLE_2  0x03
#define MAX30102_REG_FIFO_WR_PTR    0x04
#define MAX30102_REG_OVF_COUNTER    0x05
#define MAX30102_REG_FIFO_RD_PTR    0x06
#define MAX30102_REG_FIFO_DATA      0x07
#define MAX30102_REG_FIFO_CONFIG    0x08
#define MAX30102_REG_MODE_CONFIG    0x09
#define MAX30102_REG_SPO2_CONFIG    0x0A
#define MAX30102_REG_LED1_PA        0x0C   /* Red LED current */
#define MAX30102_REG_LED2_PA        0x0D   /* IR LED current */
#define MAX30102_REG_PART_ID        0xFF   /* must read 0x15 - "is this chip even here" check */

/* One FIFO sample: 18-bit Red + 18-bit IR, right-justified after unpacking. */
typedef struct {
    uint32_t red;
    uint32_t ir;
} max30102_sample_t;

/*
 * Call once at firmware startup, BEFORE max30102_init() - brings up the I2C
 * bus itself (clock/divider config), separate from bringing up the sensor
 * chip that sits on that bus. See max30102.c for the bus number/speed used
 * and why those specific numbers are still guesses pending i2c.c.
 */
void max30102_i2c_bus_init(void);

/*
 * Bring the sensor up: soft reset, confirm PART_ID, configure the FIFO
 * (4-sample averaging + rollover so a slow reader can't desync from the
 * write pointer), SpO2 mode (both LEDs on, 18-bit samples, 100 Hz), and a
 * moderate starting LED current.
 *
 * Returns 0 on success, -1 if PART_ID didn't read back 0x15 (wrong wiring,
 * wrong address, or the chip isn't powered - check this before suspecting
 * anything more complicated).
 */
int max30102_init(void);

/*
 * How many unread samples are sitting in the FIFO right now (computed from
 * the write/read pointer registers, so it costs two register reads, not a
 * FIFO drain). Call this before max30102_read_fifo() so you know how many
 * samples to ask for.
 */
int max30102_samples_available(void);

/*
 * Read up to max_out samples out of the FIFO (oldest first) into out[].
 * Returns how many it actually read - may be less than max_out if the FIFO
 * had fewer samples available, and may be less than
 * max30102_samples_available() returned a moment ago if an interrupt/other
 * bus traffic drained some in between; that's normal, not an error.
 */
int max30102_read_fifo(max30102_sample_t *out, int max_out);

/*
 * "Is a finger actually on the sensor" check - see HR_MIN_IR_FOR_VALID in
 * sensor_thresholds.h. Kept here rather than duplicated at every call site.
 */
static inline int max30102_finger_present(uint32_t ir_sample) {
    return ir_sample >= (uint32_t)HR_MIN_IR_FOR_VALID;
}

#endif /* MAX30102_H */
