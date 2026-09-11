/*
 * lcd_i2c_pcf8574.h  (REVISION 2 - hardened init)
 *
 * Minimal HD44780 character-LCD driver over a PCF8574 I2C backpack ("MH"
 * branded and most clones), written directly against a caller-supplied
 * TwoWire instance (this project passes Wire1, the board's hardware I2C0
 * object that MAX30102/MPU6050 already share) instead of the global
 * `Wire`.
 *
 * ======================================================================
 * WHAT CHANGED IN REVISION 2, AND WHY (the "screen shows solid blocks"
 * bug)
 * ======================================================================
 * Symptom on real hardware: backlight on, two rows of solid blocks, never
 * any text - even though Serial printed "LCD SUCCESS".
 *
 * "LCD SUCCESS" was misleading and is now documented as such. begin()
 * reports whether every I2C write was ACKed - but the thing ACKing is the
 * PCF8574, a dumb 8-bit I/O expander. It ACKs whatever you send it
 * regardless of whether the HD44780 behind it understood anything. So an
 * ACK proves the backpack is alive; it proves NOTHING about the LCD
 * controller's state. Solid blocks on alternating rows is the textbook
 * signature of an HD44780 that never received a valid Function Set - it's
 * sitting in its power-on default (8-bit, 1-line) with undefined DDRAM,
 * and every cell renders as the all-pixels-lit glyph.
 *
 * Three fixes, all aimed at "the controller missed the wake-up sequence":
 *
 * 1. THE WAKE-UP SEQUENCE NOW RUNS TWICE. This is the single highest-value
 *    change. The HD44780 wake-up is timing-sensitive and unforgiving: if
 *    the controller's internal power-on reset hadn't finished when the
 *    first 0x03 nibble arrived, that nibble is swallowed and the whole
 *    sequence desynchronises - every subsequent nibble is then interpreted
 *    as the wrong half of a byte, so Function Set never lands and the
 *    display stays in its default 1-line mode showing blocks. Running the
 *    full sequence a second time re-synchronises it, because the sequence
 *    is specifically designed to be self-synchronising from ANY starting
 *    state (that's why it sends 0x03 three times). Costs ~20ms once at
 *    boot; removes an entire class of "sometimes it works" failures.
 *
 * 2. LONGER POWER-ON DELAY. Was delay(50) before the first nibble; now
 *    delay(100) after construction plus delay(50) after the first
 *    expander write. The datasheet minimum is 40ms after Vcc reaches
 *    4.5V - but on this board the LCD's 5V rail is shared with three other
 *    I2C sensors that are being initialised at the same time, so the rail
 *    settles later than a bare LCD's would. The old 50ms had no margin.
 *
 * 3. EXPLICIT POST-COMMAND DELAYS. Most HD44780 commands need 37us to
 *    execute; clear/home need 1.52ms. The previous version leaned on the
 *    50us tail inside pulseEnable() to cover this. That happens to be
 *    enough for the 37us commands, but it left clear/home under-delayed on
 *    a fast bus. Now stated explicitly per command instead of relied upon
 *    as a side effect.
 *
 * Also added: forceReinit(), so the sketch can re-run initialisation at
 * runtime if the display is ever seen to be wedged, without a power cycle.
 *
 * ======================================================================
 * WHY NOT LiquidCrystal_I2C
 * ======================================================================
 * The common Arduino Library Manager versions of that library hardcode the
 * global `Wire` object inside their .cpp - there is no constructor/begin()
 * overload to hand them a different TwoWire instance. This board's
 * hardware I2C0 lives on a custom object (`TwoWire Wire1(0)`), not the
 * global `Wire`, so that library would either fail to compile (no global
 * `Wire` symbol) or silently drive the wrong bus if one happens to exist.
 * This driver sidesteps that by being bus-agnostic: pass whichever TwoWire
 * instance is correct at construction.
 *
 * PCF8574 pin mapping (standard for these backpacks):
 *   P0 = RS      P1 = RW (tied low/unused - always write)
 *   P2 = E       P3 = Backlight
 *   P4..P7 = D4..D7 (upper nibble of the byte sent over I2C)
 *
 * Only 4-bit mode is implemented (all these backpacks are wired for it -
 * there's no way to reach the LCD's D0-D3 pins through this chip anyway).
 *
 * ======================================================================
 * IF IT STILL SHOWS BLOCKS AFTER THIS REVISION
 * ======================================================================
 * Then it is not a software problem, and there are exactly two things
 * left, in order of likelihood:
 *   1. CONTRAST. Turn the blue trimmer pot on the backpack while powered.
 *      At the wrong setting, correctly-rendered text is invisible and the
 *      panel reads as blank or as faint blocks. This costs ten seconds to
 *      rule out and should be done first.
 *   2. LOGIC VOLTAGE. These backpacks and the HD44780 behind them are 5V
 *      parts. If the LCD module is powered from 3.3V it will ACK on I2C
 *      (the PCF8574 is happy) while the LCD's own contrast/drive circuit
 *      never reaches a usable operating point. Power the module's VCC from
 *      5V. The I2C data lines themselves are fine either way here, because
 *      the PCF8574's outputs are open-drain-ish quasi-bidirectional.
 */

#ifndef LCD_I2C_PCF8574_H
#define LCD_I2C_PCF8574_H

#include <Arduino.h>
#include <Wire.h>

class LCD_I2C_PCF8574 {
public:
  LCD_I2C_PCF8574(TwoWire &wire, uint8_t addr, uint8_t cols, uint8_t rows)
    : _wire(wire), _addr(addr), _cols(cols), _rows(rows), _backlight(0x08) {}

  /* Runs the HD44780 initialisation.
   *
   * RETURN VALUE CAVEAT - read this before trusting it: true means every
   * I2C write was ACKed by the PCF8574 expander. It does NOT mean the LCD
   * controller behind the expander initialised correctly, because the
   * expander ACKs unconditionally and there is no return path from the
   * HD44780 through this backpack to check (RW is tied low). Treat true as
   * "the backpack is wired and alive", not as "the display works". The
   * only real verification is looking at the screen. */
  bool begin() {
    _ok = true;

    /* HD44780 needs >=40ms after its supply rises before it will accept
     * anything. Generous here because this board brings up MAX30102,
     * MPU6050 and MLX90614 on the same rails at the same time. */
    delay(100);
    expanderWrite(0x00);       /* all control/data lines low, backlight per _backlight */
    delay(50);

    /* Run the wake-up sequence TWICE - see fix #1 in the file header.
     * The sequence is self-synchronising from any controller state, so a
     * second pass recovers a controller that swallowed the first pass. */
    for (uint8_t attempt = 0; attempt < 2; attempt++) {
      /* Datasheet fig. 24 "Initializing by Instruction", 4-bit interface.
       * Each of these pulses only D4-D7 (the wired nibble) with 0b0011. */
      writeNibble(0x03, 0);
      delayMicroseconds(4500);   /* >4.1ms */
      writeNibble(0x03, 0);
      delayMicroseconds(4500);   /* >4.1ms */
      writeNibble(0x03, 0);
      delayMicroseconds(200);    /* >100us */
      writeNibble(0x02, 0);      /* NOW switch the interface to 4-bit */
      delayMicroseconds(200);

      /* From here on, full bytes are sent as two nibbles. */
      command(0x28); delayMicroseconds(100);  /* function set: 4-bit, 2-line, 5x8 font */
      command(0x08); delayMicroseconds(100);  /* display off */
      command(0x01); delay(3);                /* clear (needs 1.52ms) */
      command(0x06); delayMicroseconds(100);  /* entry mode: increment, no shift */
      command(0x0C); delayMicroseconds(100);  /* display on, cursor off, blink off */
    }

    clear();
    home();
    return _ok;
  }

  /* Re-runs begin() from scratch. Exposed so the sketch can recover a
   * display that got wedged mid-run (e.g. a corrupted I2C transaction
   * while sharing the bus with the MAX30102/MPU6050) without needing a
   * power cycle. Cheap enough to call occasionally; ~20ms. */
  bool forceReinit() { return begin(); }

  void clear() { command(0x01); delay(3); }
  void home()  { command(0x02); delay(3); }

  void setCursor(uint8_t col, uint8_t row) {
    /* DDRAM base address per visible row on 16x2/20x4 panels. Note rows
     * 2/3 continue rows 0/1 in DDRAM - that layout is exactly why a
     * controller stuck in 1-line mode renders as blocks on alternating
     * physical rows. */
    static const uint8_t rowOffsets[4] = { 0x00, 0x40, 0x14, 0x54 };
    if (row >= _rows) row = _rows - 1;
    command(0x80 | (col + rowOffsets[row]));
    delayMicroseconds(100);
  }

  void print(const char *s) { while (*s) writeChar(*s++); }

  /* Writes one raw character code (e.g. 0xFF, the solid-block glyph on
   * virtually every HD44780 character ROM) - print() takes a
   * null-terminated string, which can't carry an arbitrary single byte
   * as conveniently as this. */
  void write(uint8_t ch) { writeChar(ch); }

  /* Pads/truncates to `width` so leftover characters from a shorter
   * previous string on the same row don't linger on screen. */
  void printPadded(const char *s, uint8_t width) {
    uint8_t n = 0;
    while (s[n] && n < width) { writeChar(s[n]); n++; }
    for (; n < width; n++) writeChar(' ');
  }

  void backlight(bool on) { _backlight = on ? 0x08 : 0x00; expanderWrite(0x00); }

  bool ok() const { return _ok; }

private:
  TwoWire &_wire;
  uint8_t _addr, _cols, _rows, _backlight;
  bool _ok = false;

  bool expanderWrite(uint8_t data) {
    _wire.beginTransmission(_addr);
    _wire.write((uint8_t)(data | _backlight));
    return (_wire.endTransmission() == 0);
  }

  void pulseEnable(uint8_t data) {
    expanderWrite(data | 0x04);   /* E high */
    delayMicroseconds(2);         /* HD44780 needs E high >=450ns; the I2C
                                   * transaction itself already far exceeds
                                   * that, this is belt-and-braces */
    expanderWrite(data & ~0x04);  /* E low - this edge latches the nibble */
    delayMicroseconds(50);        /* >=37us command execution time */
  }

  void writeNibble(uint8_t nibble4, uint8_t rs) {
    uint8_t data = (uint8_t)((nibble4 << 4) | (rs ? 0x01 : 0x00));
    if (!expanderWrite(data)) _ok = false;
    pulseEnable(data);
  }

  void sendByte(uint8_t val, uint8_t rs) {
    writeNibble((val >> 4) & 0x0F, rs);
    writeNibble(val & 0x0F, rs);
  }

  void command(uint8_t cmd) { sendByte(cmd, 0); }
  void writeChar(uint8_t ch) { sendByte(ch, 1); }
};

/* Probes a single I2C address on `wire` for an ACK (no data read/written
 * beyond the address byte). */
static inline bool i2c_probe_addr(TwoWire &wire, uint8_t addr) {
  wire.beginTransmission(addr);
  return (wire.endTransmission() == 0);
}

/* Returns the I2C address to use, or 0x00 if nothing was found at all.
 * Tries the two common PCF8574 defaults first (0x27 for PCF8574, 0x3F for
 * PCF8574A), then falls back to scanning the whole 7-bit range and picking
 * the first responder that isn't MAX30102 (0x57) or MPU6050 (0x68). */
static inline uint8_t lcd_i2c_autodetect(TwoWire &wire) {
  if (i2c_probe_addr(wire, 0x27)) { Serial.println("[LCD] found backpack at 0x27"); return 0x27; }
  if (i2c_probe_addr(wire, 0x3F)) { Serial.println("[LCD] found backpack at 0x3F"); return 0x3F; }

  Serial.println("[LCD] 0x27/0x3F both silent - scanning full I2C0 bus...");
  for (uint8_t addr = 0x08; addr <= 0x77; addr++) {
    if (addr == 0x57 || addr == 0x68) continue;   /* MAX30102 / MPU6050 */
    if (i2c_probe_addr(wire, addr)) {
      Serial.print("[LCD] scan found device at 0x");
      Serial.print(addr, HEX);
      Serial.println(" - assuming this is the LCD backpack (verify if display stays blank)");
      return addr;
    }
  }
  Serial.println("[LCD] no I2C device found at all besides MAX30102/MPU6050 - check wiring/power");
  return 0x00;
}

#endif
