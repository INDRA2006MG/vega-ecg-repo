/*
 * max30102.c - see max30102.h for scope. Register map and init sequence are
 * from the MAX30102 datasheet - solid. The I2C primitives below are now
 * anchored on the REAL <taurus/i2c.h> function names (confirmed from the
 * board's own header, 29 Aug), but the exact byte SEQUENCE inside them is
 * still an INFERENCE, not confirmed - see the big comment below before
 * flashing this. Compare against ~/ecg_vega_kernel/taurus/drivers/i2c.c
 * (the actual implementation, not just the header - `cat` it and send it
 * over) before trusting this on real hardware.
 */

#include "max30102.h"
#include "i2c.h"      /* taurus's I2C header - the real one, not a guess */
#include <stddef.h>

/* ============================================================================
 * SDK-SPECIFIC: I2C primitives.
 *
 * taurus/i2c.h gives i2c_start(i2c_number, read_length, Read),
 * i2c_data_write(i2c_number, buf, len), i2c_data_read(i2c_number, buf, len),
 * i2c_stop(i2c_number) - but no slave-address parameter anywhere on those
 * four, and no generic "read/write register N of device at address A"
 * helper the way EEPROM and SSD1306 each got their own
 * i2c_WriteByte_EEPROM()/i2c_WriteByte_SSD1306() wrapper. That means
 * whatever sequence turns "slave address 0x57, register REG, byte VAL" into
 * start/write/stop calls for THOSE two devices already exists inside
 * i2c.c, worked out by whoever wrote it - it just isn't exposed as a
 * reusable function for a new device like the MAX30102.
 *
 * BEST-EFFORT INFERENCE below, reasoning from how i2c_WriteByte_EEPROM's
 * signature (i2c_number, data, word_address, slave_address) implies the
 * pieces fit together: i2c_start() opens the bus and (based on the
 * function needing to know read_length up front, typical for a FIFO/ISR-
 * driven controller that needs to know how many bytes to expect) primes it
 * for a transfer of known length; the slave address + R/W bit is then sent
 * as the first byte via i2c_data_write(), same as it would be bit-banged by
 * hand; i2c_stop() closes it. A register READ needs a repeated start: write
 * the slave address (write bit) + register number, i2c_start() again with
 * Read=1, write the slave address (read bit), i2c_data_read() the payload.
 *
 * THIS IS A GUESS. Confirm it against i2c.c - specifically how
 * i2c_WriteByte_EEPROM() and i2c_ReadByte_EEPROM() are implemented, since
 * those are a WORKING reference for the exact same kind of "address, then
 * register, then data" transaction this driver needs. If their sequence
 * differs from what's below, copy theirs, don't trust this comment over
 * the working code. sensor_thresholds.h's HR_MIN_IR_FOR_VALID has the same
 * "confirm on real hardware" flag - this is the same category of gap.
 * ========================================================================= */

#define MAX30102_I2C_BUS  I2C_0   /* first I2C bus - confirm this is where
                                    * SDA/SCL were actually wired; I2C_1 is
                                    * the other option taurus/i2c.h defines */

static int max30102_i2c_write(uint8_t reg, uint8_t val) {
    uint8_t addr_w = (uint8_t)((MAX30102_I2C_ADDR << 1) | I2C_WRITE);
    uint8_t payload[2] = { reg, val };

    if (i2c_start(MAX30102_I2C_BUS, 0, I2C_WRITE) != 0) return -1;
    if (i2c_data_write(MAX30102_I2C_BUS, &addr_w, 1) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }
    if (i2c_data_write(MAX30102_I2C_BUS, payload, 2) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }
    i2c_stop(MAX30102_I2C_BUS);
    return 0;
}

static int max30102_i2c_read(uint8_t reg, uint8_t *buf, int len) {
    uint8_t addr_w = (uint8_t)((MAX30102_I2C_ADDR << 1) | I2C_WRITE);
    uint8_t addr_r = (uint8_t)((MAX30102_I2C_ADDR << 1) | I2C_READ);

    /* phase 1: write the register pointer, no stop (repeated start) */
    if (i2c_start(MAX30102_I2C_BUS, 0, I2C_WRITE) != 0) return -1;
    if (i2c_data_write(MAX30102_I2C_BUS, &addr_w, 1) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }
    if (i2c_data_write(MAX30102_I2C_BUS, &reg, 1) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }

    /* phase 2: repeated start, then read `len` bytes */
    if (i2c_start(MAX30102_I2C_BUS, (UC)len, I2C_READ) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }
    if (i2c_data_write(MAX30102_I2C_BUS, &addr_r, 1) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }
    if (i2c_data_read(MAX30102_I2C_BUS, buf, (UC)len) != 0) { i2c_stop(MAX30102_I2C_BUS); return -1; }

    i2c_stop(MAX30102_I2C_BUS);
    return 0;
}

/* Call once at firmware startup, before max30102_init() - sets up the bus
 * clock divider. System_Clock/I2C_Clock units aren't given in the header;
 * 100000000 (100 MHz, matches CPU_HZ in main.c) and 100000 (standard-mode
 * 100 kHz I2C) are reasonable starting guesses, not confirmed values. */
void max30102_i2c_bus_init(void) {
    i2c_initialize(MAX30102_I2C_BUS);
    i2c_configure(MAX30102_I2C_BUS, 100000000UL, 100000UL);
}

/* ============================================================================
 * Everything below this line is from the MAX30102 datasheet, not the SDK -
 * should not need changing once the two functions above are real.
 * ========================================================================= */

int max30102_init(void) {
    /* soft reset: MODE_CONFIG bit 6 */
    if (max30102_i2c_write(MAX30102_REG_MODE_CONFIG, 0x40) != 0) return -1;
    /* datasheet: reset bit clears itself when done, typically <1 ms - a
     * real port needs a short delay here (the SDK's own delay/sleep call,
     * not shown - another thing to confirm once taurus's API is known) */

    uint8_t part_id = 0;
    if (max30102_i2c_read(MAX30102_REG_PART_ID, &part_id, 1) != 0) return -1;
    if (part_id != 0x15) return -1;   /* wrong chip, wrong address, or unpowered */

    /* FIFO: average 4 samples per reported sample (SMP_AVE=100), enable
     * rollover so a slow reader loses old data instead of the FIFO jamming */
    if (max30102_i2c_write(MAX30102_REG_FIFO_CONFIG, 0x50) != 0) return -1;

    /* reset FIFO pointers so the first read starts clean */
    if (max30102_i2c_write(MAX30102_REG_FIFO_WR_PTR, 0x00) != 0) return -1;
    if (max30102_i2c_write(MAX30102_REG_OVF_COUNTER, 0x00) != 0) return -1;
    if (max30102_i2c_write(MAX30102_REG_FIFO_RD_PTR, 0x00) != 0) return -1;

    /* SpO2 mode config: ADC range 4096 nA, 100 Hz sample rate, 411 us pulse
     * width -> 18-bit resolution (SPO2_ADC_RGE=01, SPO2_SR=001, LED_PW=11) */
    if (max30102_i2c_write(MAX30102_REG_SPO2_CONFIG, 0x27) != 0) return -1;

    /* moderate LED current to start (~7 mA each, register value 0x24) -
     * this is the first thing to retune once real waveforms are visible:
     * too low and the signal is buried in noise, too high and it saturates */
    if (max30102_i2c_write(MAX30102_REG_LED1_PA, 0x24) != 0) return -1;
    if (max30102_i2c_write(MAX30102_REG_LED2_PA, 0x24) != 0) return -1;

    /* mode: SpO2 (Red + IR both active) */
    if (max30102_i2c_write(MAX30102_REG_MODE_CONFIG, 0x03) != 0) return -1;

    return 0;
}

int max30102_samples_available(void) {
    uint8_t wr = 0, rd = 0;
    if (max30102_i2c_read(MAX30102_REG_FIFO_WR_PTR, &wr, 1) != 0) return -1;
    if (max30102_i2c_read(MAX30102_REG_FIFO_RD_PTR, &rd, 1) != 0) return -1;
    /* both pointers wrap at 32 (5-bit registers) */
    return (int)((wr - rd) & 0x1F);
}

int max30102_read_fifo(max30102_sample_t *out, int max_out) {
    int avail = max30102_samples_available();
    if (avail < 0) return -1;
    if (avail > max_out) avail = max_out;

    for (int i = 0; i < avail; i++) {
        /* each sample is 6 bytes: 3 (Red) + 3 (IR), each 18 bits left-packed
         * into 3 bytes (top 6 bits of the first byte are always 0) */
        uint8_t buf[6];
        if (max30102_i2c_read(MAX30102_REG_FIFO_DATA, buf, 6) != 0) return i;

        out[i].red = (((uint32_t)buf[0] << 16) | ((uint32_t)buf[1] << 8) | buf[2]) & 0x3FFFF;
        out[i].ir  = (((uint32_t)buf[3] << 16) | ((uint32_t)buf[4] << 8) | buf[5]) & 0x3FFFF;
    }
    return avail;
}
