/**
 * ============================================================================
 * Project: ESP32 Wearable Parkinson's Tremor Controller & Zero-Crossing Damper
 * Target Hardware: ESP32 DevKitC (ESP32-WROOM-32)
 * Sensor: MPU6050 6-DOF IMU (I2C: GPIO 21 SDA, GPIO 22 SCL)
 * Motor Drivers:
 *   - DRV8833 Dual H-Bridge:
 *       * Channel 1: IN1=GPIO 25, IN2=GPIO 26 -> OUT1 & OUT2 (Motors 1 & 2)
 *       * Channel 2: IN3=GPIO 27, IN4=GPIO 14 -> OUT3 & OUT4 (Motors 3 & 4)
 *       * Driver Sleep / Enable: EEP=GPIO 32 (driven HIGH to keep driver awake)
 *   - ULN2003 Auxiliary Darlington Driver:
 *       * Auxiliary Channel: IN1=GPIO 4 -> OUT1 (Motor 5)
 * Actuators: 4 to 5 ERM Coin Vibration Motors (3V nominal)
 *
 * BUGFIXES & CRITICAL RESOLUTIONS v3.6:
 *   1. FIXED CHANNEL 2 NOT WORKING (LEDC Hardware Clock Divider Overflow):
 *      - Root Cause: 175 Hz with 8-bit resolution requires a clock divider of 457,142,
 *        which EXCEEDS the ESP32 hardware 18-bit maximum (262,143).
 *        Minimum frequency supported at 8-bit on ESP32 is 305.18 Hz!
 *        As a result, ledcSetup for Timer 1 (Channel 2: GPIO 27 & 14) failed silently!
 *      - Fix: Set PWM_FREQ = 1000 Hz (1 kHz standard for DRV8833 and ERM motors).
 *        All channels now initialize cleanly and output full power!
 *   2. FIXED MACHINE STOPS READING INPUT (Zero Time-Debt / Non-blocking loop):
 *      - Root Cause A: Repeated start vs STOP on MPU6050. MPU6050 requires repeated
 *        start (Wire.endTransmission(false)) to latch register 0x3B. Sending STOP
 *        reset the internal pointer to 0x00 on clone chips, causing continuous 0-reads
 *        and triggering infinite 25ms recovery delays.
 *      - Root Cause B: Slip debt accumulation (lastSampleTimeUs += SAMPLE_INTERVAL_US)
 *        flooded the I2C bus and starved serial input.
 *      - Root Cause C: Removed blocking delay(10) calls inside setMotorOutputs().
 *      - Fix: Moved handleSerialCommands() to TOP of loop(), reset lastSampleTimeUs = nowUs,
 *        and used non-blocking 15ms Wire timeout.
 *   3. CHANNEL 2 HARDWARE VERIFICATION & DIAGNOSTIC ECHO:
 *      - Confirms hardware timer allocation at startup in Serial Monitor.
 *      - 'i' key flips Channel 2 rotation direction in software.
 *      - 'q' key toggles between Dual H-Bridge and Quad Single-Ended modes.
 * ============================================================================
 */

#include <Wire.h>
#include <esp_arduino_version.h>
#include <math.h>

// ============================================================================
// HARDWARE PIN DEFINITIONS
// ============================================================================
#define PIN_SDA         21
#define PIN_SCL         22

// DRV8833 Dual H-Bridge Channel 1 (OUT1 & OUT2)
#define MOTOR1_IN1      25  // OUT1 PWM (+)
#define MOTOR1_IN2      26  // OUT2 GND (-)

// DRV8833 Dual H-Bridge Channel 2 (OUT3 & OUT4)
#define MOTOR2_IN3      27  // OUT3 PWM (+)
#define MOTOR2_IN4      14  // OUT4 GND (-)

// DRV8833 Sleep / Enable Pin
#define DRV_SLEEP_PIN   32  // Must be HIGH to wake DRV8833

// ULN2003 Auxiliary Driver (Motor 5)
#define MOTOR_AUX       4   // ULN2003 IN1 -> OUT1

// LEDC PWM Carrier Setup
// CRITICAL FIX: 1000 Hz (1 kHz) is within ESP32 8-bit LEDC hardware limits (min 305.18 Hz).
// 175 Hz caused divider overflow and prevented Channel 2 from initializing!
const int PWM_FREQ        = 1000; // 1 kHz carrier frequency for smooth ERM DC vibration
const int PWM_RESOLUTION  = 8;    // 8-bit resolution (0 - 255)

#define CH_M1_IN1   0
#define CH_M1_IN2   1
#define CH_M2_IN3   2
#define CH_M2_IN4   3
#define CH_M_AUX    4

void writeHBridgePWM(uint8_t pin, uint8_t channel, uint8_t duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, duty);
#else
  ledcWrite(channel, duty);
#endif
}

// ============================================================================
// MOTOR WIRING & DRIVE MODES
// ============================================================================
enum MotorWiringMode {
  WIRING_DUAL_HBRIDGE = 0,    // OUT1/OUT2 differential + OUT3/OUT4 differential (DEFAULT)
  WIRING_QUAD_ALL_OUTS = 1    // OUT1, OUT2, OUT3, OUT4 all driven HIGH PWM simultaneously to GND
};

volatile MotorWiringMode motorWiringMode = WIRING_DUAL_HBRIDGE;
volatile bool ch2Inverted = false; // Toggle with 'i' to flip Ch2 rotation direction

// ============================================================================
// TELEMETRY & DISPLAY MODES
// ============================================================================
enum TelemetryFormat {
  TELEMETRY_MONITOR_DASHBOARD = 0,  // Clean human-readable report in Serial Monitor (DEFAULT)
  TELEMETRY_RAW_CSV,                // Raw CSV (ax,ay,az,gx,gy,gz) for Python predict_serial.py
  TELEMETRY_SERIAL_PLOTTER          // Waveforms for Arduino IDE Serial Plotter
};

volatile TelemetryFormat outputMode = TELEMETRY_MONITOR_DASHBOARD;

// ============================================================================
// CONTROL MODES
// ============================================================================
enum OperatingMode {
  MODE_MONITOR_ONLY = 0,         // Motors OFF
  MODE_ADAPTIVE_FEEDBACK,        // Autonomous severity-dependent intensity (DEFAULT)
  MODE_ANTIPHASE_INTERFERENCE,   // Destructive interference antiphase bursts at pi rad
  MODE_HOST_ML_CONTROL,          // Legacy steady ON/OFF from Python ('1'/'0')
  MODE_ML_PHASE_FEEDBACK         // ML gate + local phase-timed motor pulses (G/D)
};

volatile OperatingMode currentMode = MODE_MONITOR_ONLY; // Safe startup; host must explicitly enable feedback

// ============================================================================
// SIGNAL PROCESSING & PARAMETERS
// ============================================================================
const uint32_t SAMPLE_INTERVAL_US = 10000; // 100 Hz = 10,000 microseconds
const float    FS                  = 100.0f;
const int      RMS_WINDOW_SIZE     = 100;

// Accelerometer Sensitivity (+/- 8g scale = 4096 LSB/g)
// Earth gravity 1g = 4096 counts (prevents motor vibration ADC saturation!)
const float GRAVITY_NOMINAL_LSB = 4096.0f;

// Tremor Severity Thresholds (Scaled for +/- 8g)
float TREMOR_DETECT_THRESHOLD = 45.0f;   // Responsive detection
float TREMOR_LOW_THRESHOLD    = 75.0f;
float TREMOR_HIGH_THRESHOLD   = 180.0f;
float HYSTERESIS_BAND         = 10.0f;

// Actuator PWM Duty Cycles (0 - 255)
const uint8_t PWM_DUTY_LOW     = 160;  // Gentle tactile sensation
const uint8_t PWM_DUTY_MEDIUM  = 205;  // Standard therapeutic vibration
const uint8_t PWM_DUTY_HIGH    = 235;  // High vibration
const uint8_t PWM_DUTY_MAX_CAP = 240;  // NOT a guaranteed voltage limit: check motor rail and rating
// EXPERIMENTAL ML-GATED, PHASE-TIMED PULSES (not mechanical tremor cancellation).
// 1 kHz PWM carrier controls approximate motor drive; it is NOT the tremor frequency.
// Start conservatively with the motors off the body and check supply voltage/current.
const uint8_t ML_PHASE_PWM_DEFAULT = 110; // Default intensity (0..255)
volatile uint8_t mlPhasePwmDuty = 110;    // LIVE intensity set by Python 'V' command (80..240)
const uint32_t ML_PULSE_DURATION_MS = 30;
const uint32_t HOST_TIMEOUT_MS = 1200;
const uint32_t SENSOR_TIMEOUT_MS = 250;
const uint32_t ML_MAX_LATE_US = 15000;
const uint32_t ACTUATOR_LEAD_US = 0;    // No assumed latency compensation; calibrate experimentally
const uint8_t PHASE_GYRO_AXIS = 0;      // 0=x, 1=y, 2=z; choose axis aligned with measured tremor

// Serial command parse state for multi-byte commands (e.g., 'V' + intensity byte)
volatile bool awaitingIntensityByte = false;

const uint32_t LEVEL_DWELL_TIME_MS = 300;

// Voluntary Motion Rejection
const float VOLUNTARY_GYRO_LIMIT   = 7500.0f;
const int   TREMOR_STREAK_REQUIRED = 30; // 30 samples = 0.3s sustained rhythmic oscillation
int         tremorStreakCount      = 0;

// Frequency Limits (3.5 - 7.5 Hz)
const float    TREMOR_FREQ_MIN_HZ  = 3.5f;
const float    TREMOR_FREQ_MAX_HZ  = 7.5f;
const uint32_t MIN_CYCLE_PERIOD_US = (uint32_t)(1000000.0f / TREMOR_FREQ_MAX_HZ); // 133,333 us
const uint32_t MAX_CYCLE_PERIOD_US = (uint32_t)(1000000.0f / TREMOR_FREQ_MIN_HZ); // 285,714 us

// ============================================================================
// DSP DATA STRUCTURES: BUTTERWORTH FILTER & ZERO-CROSSING
// ============================================================================
// 1. Dynamic Gravity Tracking (Highpass DC removal)
float dcAx = 0.0f, dcAy = 0.0f, dcAz = 4096.0f;
const float DC_ALPHA = 0.0314f;

// 2. 2nd-Order Butterworth Bandpass Filter (3.5 - 7.5 Hz @ 100 Hz fs)
// Direct Form II Transposed Biquad coefficients
const float bpf_b0 = 0.109930f, bpf_b1 = 0.0f, bpf_b2 = -0.109930f;
const float bpf_a1 = -1.688680f, bpf_a2 = 0.780130f;
float bpf_d1 = 0.0f, bpf_d2 = 0.0f;
// Independent signed gyro filter for phase; DO NOT take abs/magnitude before zero crossings.
float phaseBpfD1 = 0.0f, phaseBpfD2 = 0.0f;

// 3. Moving-Window RMS Intensity
float rmsBuffer[RMS_WINDOW_SIZE];
int   rmsBufIdx = 0;
float runningSumSq = 0.0f;
float currentTremorRMS = 0.0f;

// 4. Schmitt-Trigger Zero-Crossing & Phase Tracker
float    prevFilteredSignal  = 0.0f;
uint32_t lastZeroCrossingUs  = 0;
uint32_t estimatedPeriodUs   = 200000;
float    estimatedFrequency  = 5.0f;
float    currentPhaseRad     = 0.0f;
bool     phaseValid          = false;
bool     negativeArmed       = false;
float    crossingPulseMarker = 0.0f;
float    signalEnvelope      = 40.0f;

// State Flags
uint8_t  activeMotorPWM = 0;
bool     tremorDetected = false;
bool     isVoluntaryMotion = false;
bool     mlGateEnabled = false;
uint32_t lastHostMlMsgMs = 0;
uint32_t lastGoodSensorMs = 0;
uint32_t lastMlPulsedCrossingUs = 0;
uint32_t lastLevelChangeTime = 0;

// Antiphase Stimulation Control (pi rad = 180 deg phase offset)
float    targetStimPhaseOffsetRad = 3.14159265f;
uint32_t stimBurstDurationMs      = 45;
uint32_t stimBurstStartTimeMs     = 0;
bool     stimBurstActive          = false;

// Timing
uint32_t lastSampleTimeUs = 0;
uint32_t lastLogTimeMs    = 0;
uint32_t lastWatchdogTimeMs = 0;
const uint32_t MONITOR_LOG_INTERVAL_MS = 200; // 5 Hz refresh for readable Serial Monitor

// ============================================================================
// SELF-HEALING MPU6050 DRIVER (ANTI-FREEZE & REPEATED-START LATCHING)
// ============================================================================
uint8_t mpuAddress = 0x68;
uint32_t consecutiveI2cErrors = 0;
uint32_t consecutiveZeroReads = 0;
uint32_t totalSleepRecoveries = 0;

bool writeMPUReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission(true) == 0); // Writing registers terminates with STOP
}

bool readMPUReg(uint8_t reg, uint8_t *val) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(reg);
  // CRITICAL: Repeated start (false) is mandatory for MPU6050 to latch register address!
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)mpuAddress, 1, (int)true) != 1) return false;
  *val = Wire.read();
  return true;
}

/**
 * Re-awakens MPU6050 and configures +/- 8g range.
 * Uses Internal 8MHz RC oscillator for instant zero-delay wakeup without blocking.
 */
void reawakenMPU() {
  totalSleepRecoveries++;

  // Step 1: Write 0x00 to PWR_MGMT_1 to wake up and select internal 8MHz RC oscillator
  // (RC oscillator starts immediately with zero PLL lock wait time!)
  writeMPUReg(0x6B, 0x00);
  delayMicroseconds(500);

  // Step 2: DLPF Mode 3 (42 Hz cutoff) to filter out high-frequency motor vibration
  writeMPUReg(0x1A, 0x03);

  // Step 3: 100 Hz output rate divider
  writeMPUReg(0x19, 0x09);

  // Step 4: ACCEL_CONFIG: 0x10 = +/- 8g scale (4096 LSB/g)
  // Essential: prevents multiple ERM motors from clipping the accelerometer ADC!
  writeMPUReg(0x1C, 0x10);

  // Step 5: GYRO_CONFIG: +/- 250 dps (131 LSB/dps)
  writeMPUReg(0x1B, 0x00);

  // Step 6: User Control reset
  writeMPUReg(0x6A, 0x00);
}

void recoverI2CBus() {
  Wire.end();

  pinMode(PIN_SDA, INPUT_PULLUP);
  pinMode(PIN_SCL, OUTPUT);

  // Clock SCL up to 32 times to force hung slave to release SDA
  for (int i = 0; i < 32; i++) {
    digitalWrite(PIN_SCL, LOW);
    delayMicroseconds(5);
    digitalWrite(PIN_SCL, HIGH);
    delayMicroseconds(5);
    if (digitalRead(PIN_SDA) == HIGH) {
      break;
    }
  }

  // Generate STOP condition
  pinMode(PIN_SDA, OUTPUT);
  digitalWrite(PIN_SDA, LOW);
  delayMicroseconds(5);
  digitalWrite(PIN_SCL, HIGH);
  delayMicroseconds(5);
  digitalWrite(PIN_SDA, HIGH);
  delayMicroseconds(5);

  pinMode(PIN_SDA, INPUT_PULLUP);
  pinMode(PIN_SCL, INPUT_PULLUP);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000); // 100 kHz standard stable speed
  Wire.setTimeOut(15);   // 15ms timeout prevents any permanent I2C hang

  reawakenMPU();
}

bool initMPU6050Hardware() {
  pinMode(PIN_SDA, INPUT_PULLUP);
  pinMode(PIN_SCL, INPUT_PULLUP);

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000);
  Wire.setTimeOut(15);

  mpuAddress = 0x68;
  Wire.beginTransmission(mpuAddress);
  if (Wire.endTransmission(true) != 0) {
    mpuAddress = 0x69;
    Wire.beginTransmission(mpuAddress);
    if (Wire.endTransmission(true) != 0) {
      return false;
    }
  }

  reawakenMPU();
  return true;
}

/**
 * Direct non-blocking 14-byte IMU read with Earth-gravity vector validation
 */
bool readMotionData(int16_t out[6]) {
  Wire.beginTransmission(mpuAddress);
  Wire.write(0x3B); // ACCEL_XOUT_H
  // Repeated start (false) is required for MPU6050 to retain the 0x3B pointer!
  if (Wire.endTransmission(false) != 0) {
    consecutiveI2cErrors++;
    if (consecutiveI2cErrors >= 3) {
      recoverI2CBus();
      consecutiveI2cErrors = 0;
    }
    return false;
  }

  if (Wire.requestFrom((int)mpuAddress, 14, (int)true) != 14) {
    consecutiveI2cErrors++;
    if (consecutiveI2cErrors >= 3) {
      recoverI2CBus();
      consecutiveI2cErrors = 0;
    }
    return false;
  }

  int16_t raw[7];
  for (int i = 0; i < 7; ++i) {
    uint8_t high = Wire.read();
    uint8_t low  = Wire.read();
    raw[i] = (int16_t)(((uint16_t)high << 8) | low);
  }

  // =========================================================================
  // ZERO-READ SLEEP DETECTION:
  // On Earth (+/- 8g scale), gravity magnitude is ~4096 counts.
  // If |accel| < 600 LSB (less than 0.15g), MPU has reset into SLEEP MODE (0x40).
  // =========================================================================
  long accMagSq = (long)raw[0] * raw[0] + (long)raw[1] * raw[1] + (long)raw[2] * raw[2];
  if (accMagSq < 360000L) { // Magnitude < 600 counts
    consecutiveZeroReads++;
    if (consecutiveZeroReads >= 2) {
      reawakenMPU();
      consecutiveZeroReads = 0;
    }
    return false; // Reject bogus zero sample to protect the filter and RMS window!
  }

  consecutiveZeroReads = 0;
  consecutiveI2cErrors = 0;

  out[0] = raw[0]; out[1] = raw[1]; out[2] = raw[2]; // ax, ay, az
  out[3] = raw[4]; out[4] = raw[5]; out[5] = raw[6]; // gx, gy, gz
  return true;
}

// ============================================================================
// DSP ALGORITHMS: 2ND-ORDER BUTTERWORTH BPF & INTENSITY
// ============================================================================

/**
 * 2nd-Order Butterworth Bandpass Filter (3.5 - 7.5 Hz) Direct Form II Transposed
 */
inline float processBiquadBPF(float in) {
  float out = bpf_b0 * in + bpf_d1;
  bpf_d1 = bpf_b1 * in - bpf_a1 * out + bpf_d2;
  bpf_d2 = bpf_b2 * in - bpf_a2 * out;
  return out;
}

inline float processSignedGyroBPF(float in) {
  float out = bpf_b0 * in + phaseBpfD1;
  phaseBpfD1 = bpf_b1 * in - bpf_a1 * out + phaseBpfD2;
  phaseBpfD2 = bpf_b2 * in - bpf_a2 * out;
  return out;
}

void updateTremorIntensity(float filteredSample) {
  float oldSample = rmsBuffer[rmsBufIdx];
  float newSq = filteredSample * filteredSample;
  float oldSq = oldSample * oldSample;

  rmsBuffer[rmsBufIdx] = filteredSample;
  rmsBufIdx = (rmsBufIdx + 1) % RMS_WINDOW_SIZE;

  runningSumSq += (newSq - oldSq);
  if (runningSumSq < 0.0f) runningSumSq = 0.0f;
  currentTremorRMS = sqrtf(runningSumSq / (float)RMS_WINDOW_SIZE);
}

void updateZeroCrossingAndFrequency(float filteredSample, uint32_t nowUs) {
  signalEnvelope = 0.98f * signalEnvelope + 0.02f * fabsf(filteredSample);
  float deadband = 0.25f * signalEnvelope;
  if (deadband < 8.0f) deadband = 8.0f;

  crossingPulseMarker = 0.0f;

  if (filteredSample < -deadband) {
    negativeArmed = true;
  }

  if (negativeArmed && (prevFilteredSignal <= 0.0f) && (filteredSample > 0.0f)) {
    uint32_t cycleDeltaUs = nowUs - lastZeroCrossingUs;

    if (cycleDeltaUs >= MIN_CYCLE_PERIOD_US && cycleDeltaUs <= MAX_CYCLE_PERIOD_US) {
      estimatedPeriodUs = (uint32_t)(0.70f * (float)estimatedPeriodUs + 0.30f * (float)cycleDeltaUs);
      estimatedFrequency = 1000000.0f / (float)estimatedPeriodUs;
      lastZeroCrossingUs = nowUs;
      phaseValid = true;
      negativeArmed = false;
      crossingPulseMarker = 100.0f;
    } else if (cycleDeltaUs > MAX_CYCLE_PERIOD_US) {
      lastZeroCrossingUs = nowUs;
      negativeArmed = false;
    }
  }

  prevFilteredSignal = filteredSample;

  if (nowUs - lastZeroCrossingUs > (uint32_t)(1.40f * (float)estimatedPeriodUs)) {
    phaseValid = false;
  }
}

void updatePhaseEstimator(uint32_t nowUs) {
  if (!phaseValid || estimatedPeriodUs == 0) {
    currentPhaseRad = 0.0f;
    return;
  }
  uint32_t elapsedUs = nowUs - lastZeroCrossingUs;
  float fraction = (float)elapsedUs / (float)estimatedPeriodUs;
  if (fraction > 1.0f) fraction = fmodf(fraction, 1.0f);
  currentPhaseRad = fraction * 2.0f * 3.14159265f;
}

// ============================================================================
// DUAL-CHANNEL & QUAD-OUTPUT MOTOR CONTROL (4-5 MOTORS SIMULTANEOUS)
// ============================================================================

/**
 * Controls DRV8833 Channel 1 (OUT1/OUT2), Channel 2 (OUT3/OUT4), and ULN2003 (AUX)
 * Non-blocking operation: Drives both channels synchronously in the EXACT same direction!
 */
void setMotorOutputs(uint8_t duty, bool forward = true) {
  if (duty > PWM_DUTY_MAX_CAP) duty = PWM_DUTY_MAX_CAP;

  if (duty == 0) {
    writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 0);
    writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
    writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 0);
    writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 0);
    writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  0);
    activeMotorPWM = 0;
    return;
  }

  // Micro-stagger (500us non-blocking) between channel energization to protect battery rail
  if (activeMotorPWM == 0) {
    if (motorWiringMode == WIRING_QUAD_ALL_OUTS) {
      writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, duty);
      writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, duty);
      delayMicroseconds(500);
      writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, duty);
      writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, duty);
      delayMicroseconds(500);
      writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  duty);
    } else {
      // Channel 1
      if (forward) {
        writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
        writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, duty);
      } else {
        writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 0);
        writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, duty);
      }
      delayMicroseconds(500);

      // Channel 2
      bool effCh2Forward = forward ^ ch2Inverted;
      if (effCh2Forward) {
        writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 0);
        writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, duty);
      } else {
        writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 0);
        writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, duty);
      }
      delayMicroseconds(500);

      writeHBridgePWM(MOTOR_AUX, CH_M_AUX, duty);
    }
  } else {
    // Steady-state continuous update
    if (motorWiringMode == WIRING_QUAD_ALL_OUTS) {
      writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, duty);
      writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, duty);
      writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, duty);
      writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, duty);
      writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  duty);
    } else {
      if (forward) {
        writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
        writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, duty);
      } else {
        writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 0);
        writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, duty);
      }

      bool effCh2Forward = forward ^ ch2Inverted;
      if (effCh2Forward) {
        writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 0);
        writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, duty);
      } else {
        writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 0);
        writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, duty);
      }

      writeHBridgePWM(MOTOR_AUX, CH_M_AUX, duty);
    }
  }

  activeMotorPWM = duty;
}

/**
 * Comprehensive 7-Stage Diagnostic Test ('T' key)
 * Verifies every OUT pin and channel combination so you can observe each motor vibrating!
 */
void runMotorDiagnosticTest() {
  Serial.println(F("\n========================================================"));
  Serial.println(F(">>> EXECUTING COMPREHENSIVE MULTI-MOTOR TEST (OUT1..OUT4 + AUX) <<<"));
  Serial.println(F("========================================================"));

  // Test 1: Channel 1 Only (OUT1 & OUT2)
  Serial.println(F(">> [TEST 1/7] DRV8833 Channel 1 ONLY (OUT1 & OUT2 / Motors 1 & 2)..."));
  writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
  writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 220);
  writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 0);
  writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 0);
  writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  0);
  delay(1200);
  setMotorOutputs(0);
  delay(200);

  // Test 2: Channel 2 Only - Forward Polarity (OUT3 & OUT4)
  Serial.println(F(">> [TEST 2/7] DRV8833 Channel 2 ONLY (OUT3 (+) & OUT4 (-))..."));
  writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 0);
  writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
  writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 0);
  writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 220);
  writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  0);
  delay(1200);
  setMotorOutputs(0);
  delay(200);

  // Test 3: Channel 2 Only - Reverse Polarity (OUT4 (+) & OUT3 (-))
  Serial.println(F(">> [TEST 3/7] DRV8833 Channel 2 REVERSE POLARITY (OUT4 (+) & OUT3 (-))..."));
  Serial.println(F("   (Notice which polarity rotates your Channel 2 motor stronger!)"));
  writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 0);
  writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
  writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 0);
  writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 220);
  writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  0);
  delay(1200);
  setMotorOutputs(0);
  delay(200);

  // Test 4: BOTH Channels Together (Dual H-Bridge Same Direction)
  Serial.print(F(">> [TEST 4/7] BOTH CHANNELS TOGETHER (OUT1/2 + OUT3/4) [Ch2 Invert="));
  Serial.print(ch2Inverted ? F("ON") : F("OFF"));
  Serial.println(F("]..."));
  setMotorOutputs(220, true);
  delay(1500);
  setMotorOutputs(0);
  delay(200);

  // Test 5: QUAD Mode (All 4 OUTs Active High PWM simultaneously)
  Serial.println(F(">> [TEST 5/7] QUAD MODE: All 4 OUTs (OUT1, OUT2, OUT3, OUT4) Active High..."));
  writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 220);
  writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 220);
  writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 220);
  writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 220);
  writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  0);
  delay(1500);
  setMotorOutputs(0);
  delay(200);

  // Test 6: Auxiliary ULN2003 Only (Motor 5 on GPIO 4)
  Serial.println(F(">> [TEST 6/7] Auxiliary ULN2003 ONLY (GPIO 4 / Motor 5)..."));
  writeHBridgePWM(MOTOR1_IN1, CH_M1_IN1, 0);
  writeHBridgePWM(MOTOR1_IN2, CH_M1_IN2, 0);
  writeHBridgePWM(MOTOR2_IN3, CH_M2_IN3, 0);
  writeHBridgePWM(MOTOR2_IN4, CH_M2_IN4, 0);
  writeHBridgePWM(MOTOR_AUX,  CH_M_AUX,  220);
  delay(1200);
  setMotorOutputs(0);
  delay(200);

  // Test 7: ALL 5 MOTORS SIMULTANEOUSLY!
  Serial.println(F(">> [TEST 7/7] ALL 5 MOTORS SIMULTANEOUSLY (Full System Vibration)..."));
  setMotorOutputs(220, true);
  delay(2000);
  setMotorOutputs(0);

  Serial.println(F(">> Comprehensive Motor Diagnostic Test COMPLETE. All Motors Halted.\n"));
}

void enterSafeState() {
  setMotorOutputs(0);
  activeMotorPWM = 0;
  stimBurstActive = false;
}

// ============================================================================
// ADAPTIVE & ANTIPHASE MECHANISMS
// ============================================================================
void updateAdaptiveFeedback(uint32_t nowMs) {
  if (tremorDetected) {
    uint8_t targetPWM = PWM_DUTY_MEDIUM;
    if (currentTremorRMS > TREMOR_HIGH_THRESHOLD) targetPWM = PWM_DUTY_HIGH;
    else if (currentTremorRMS < TREMOR_LOW_THRESHOLD) targetPWM = PWM_DUTY_LOW;

    if (activeMotorPWM != targetPWM && (nowMs - lastLevelChangeTime >= LEVEL_DWELL_TIME_MS)) {
      setMotorOutputs(targetPWM);
      lastLevelChangeTime = nowMs;
    }
  } else {
    if (activeMotorPWM != 0 && (nowMs - lastLevelChangeTime >= LEVEL_DWELL_TIME_MS)) {
      setMotorOutputs(0);
      lastLevelChangeTime = nowMs;
    }
  }
}

void updateAntiphaseInterference(uint32_t nowMs) {
  if (!tremorDetected || !phaseValid) {
    if (stimBurstActive && (nowMs - stimBurstStartTimeMs >= stimBurstDurationMs)) {
      setMotorOutputs(0);
      stimBurstActive = false;
    }
    return;
  }

  // Antiphase target: pi radians (180 deg)
  float phaseDiff = fabsf(currentPhaseRad - targetStimPhaseOffsetRad);
  if (phaseDiff > 3.14159265f) phaseDiff = 2.0f * 3.14159265f - phaseDiff;

  if (!stimBurstActive && (phaseDiff < 0.28f)) {
    uint8_t dynamicDuty = (currentTremorRMS > TREMOR_HIGH_THRESHOLD) ? PWM_DUTY_HIGH : PWM_DUTY_MEDIUM;
    setMotorOutputs(dynamicDuty);
    stimBurstActive = true;
    stimBurstStartTimeMs = nowMs;
  }

  if (stimBurstActive && (nowMs - stimBurstStartTimeMs >= stimBurstDurationMs)) {
    setMotorOutputs(0);
    stimBurstActive = false;
  }
}

/**
 * ML-gated phase-timed stimulation DEMONSTRATION.
 * A Python ML positive decision only enables this controller; each pulse is
 * scheduled relative to a fresh zero crossing of a SIGNED gyroscope axis.
 * 180 degrees here is a temporal phase target, not verified opposing force.
 */
void updateMlPhaseFeedback(uint32_t nowUs, uint32_t nowMs) {
  if (currentMode != MODE_ML_PHASE_FEEDBACK) return;

  // DRV8833 outputs must always be used as differential H-bridge pairs.
  motorWiringMode = WIRING_DUAL_HBRIDGE;
  bool hostFresh = (nowMs - lastHostMlMsgMs <= HOST_TIMEOUT_MS);
  bool sensorFresh = (nowMs - lastGoodSensorMs <= SENSOR_TIMEOUT_MS);
  bool signalValid = phaseValid && estimatedPeriodUs > 0 &&
      estimatedFrequency >= TREMOR_FREQ_MIN_HZ && estimatedFrequency <= TREMOR_FREQ_MAX_HZ;

  if (!mlGateEnabled || !hostFresh || !sensorFresh || !signalValid) {
    enterSafeState();
    if (!hostFresh || !sensorFresh) {
      mlGateEnabled = false;
      currentMode = MODE_MONITOR_ONLY;
    }
    return;
  }

  // Remove drive on time even when no fresh IMU sample arrives.
  if (stimBurstActive && nowMs - stimBurstStartTimeMs >= ML_PULSE_DURATION_MS) {
    setMotorOutputs(0);
    stimBurstActive = false;
  }

  // Exactly one pulse per validated sensor zero-crossing cycle, nominal pi delay.
  if (!stimBurstActive && lastZeroCrossingUs != lastMlPulsedCrossingUs) {
    uint32_t phaseDelayUs = estimatedPeriodUs / 2;
    uint32_t leadUs = (ACTUATOR_LEAD_US < phaseDelayUs) ? ACTUATOR_LEAD_US : 0;
    uint32_t targetUs = lastZeroCrossingUs + phaseDelayUs - leadUs;
    int32_t latenessUs = (int32_t)(nowUs - targetUs);
    if (latenessUs >= 0 && latenessUs <= (int32_t)ML_MAX_LATE_US) {
      setMotorOutputs(mlPhasePwmDuty);
      stimBurstActive = true;
      stimBurstStartTimeMs = nowMs;
      lastMlPulsedCrossingUs = lastZeroCrossingUs;
    }
  }
}

// ============================================================================
// SERIAL CLI & TELEMETRY
// ============================================================================
void handleSerialCommands() {
  while (Serial.available() > 0) {
    char cmd = Serial.read();

    // Multi-byte command: if awaiting an intensity byte after 'V', consume it now
    if (awaitingIntensityByte) {
      uint8_t rawByte = (uint8_t)cmd;
      // Clamp to safe PWM range: 0 means OFF, otherwise 60..240
      if (rawByte == 0) {
        mlPhasePwmDuty = 0;
      } else if (rawByte < 60) {
        mlPhasePwmDuty = 60;   // Minimum duty to overcome motor static friction
      } else if (rawByte > PWM_DUTY_MAX_CAP) {
        mlPhasePwmDuty = PWM_DUTY_MAX_CAP;
      } else {
        mlPhasePwmDuty = rawByte;
      }
      awaitingIntensityByte = false;
      Serial.print(F("[CMD] Motor intensity set to PWM="));
      Serial.println(mlPhasePwmDuty);
      continue;
    }

    if (cmd == '\r' || cmd == '\n' || cmd == ' ') continue;

    switch (cmd) {
      case 'M':
      case 'm':
        outputMode = TELEMETRY_MONITOR_DASHBOARD;
        Serial.println(F("\n[CMD] Switched to: SERIAL MONITOR DIAGNOSTIC DASHBOARD"));
        break;
      case 'R':
      case 'r':
        outputMode = TELEMETRY_RAW_CSV;
        Serial.println(F("\n[CMD] Switched to: RAW CSV MODE (for Python predict_serial.py)"));
        break;
      case 'P':
      case 'p':
        outputMode = TELEMETRY_SERIAL_PLOTTER;
        Serial.println(F("\n[CMD] Switched to: ARDUINO SERIAL PLOTTER MODE"));
        break;
      case 'T':
      case 't':
        Serial.println(F("[SAFETY] Legacy multi-motor self-test disabled in ML-phase build"));
        break;
      case 'I':
      case 'i':
        ch2Inverted = !ch2Inverted;
        Serial.print(F("\n[CMD] Channel 2 Polarity Invert: "));
        Serial.println(ch2Inverted ? F("ENABLED (Direction Flipped in Software)") : F("DISABLED (Standard Direction)"));
        if (activeMotorPWM > 0) setMotorOutputs(activeMotorPWM);
        break;
      case 'Q':
      case 'q':
        Serial.println(F("[SAFETY] Unsafe quad-output H-bridge mode disabled"));
        break;
      case 'V':  // Variable intensity command: next byte is the PWM duty value
      case 'v':
        awaitingIntensityByte = true;
        break;
      case 'G':  // Repeated positive ML gate packets serve as the host heartbeat
        if (currentMode != MODE_ML_PHASE_FEEDBACK) {
          enterSafeState();
          lastMlPulsedCrossingUs = lastZeroCrossingUs; // Wait for the next fresh cycle
          currentMode = MODE_ML_PHASE_FEEDBACK;
          Serial.print(F("[ML] Phase-pulse feedback enabled; PWM="));
          Serial.println(mlPhasePwmDuty);
        }
        mlGateEnabled = true;
        lastHostMlMsgMs = millis();
        break;
      case 'D':  // ML negative / explicit disable
        mlGateEnabled = false;
        currentMode = MODE_MONITOR_ONLY;
        enterSafeState();
        break;
      case '1':
        currentMode = MODE_HOST_ML_CONTROL;
        setMotorOutputs(PWM_DUTY_MEDIUM);
        Serial.println(F("\n[CMD] ALL MOTORS ACTIVE (Manual '1')"));
        break;
      case '0':
        currentMode = MODE_HOST_ML_CONTROL;
        setMotorOutputs(0);
        Serial.println(F("\n[CMD] ALL MOTORS OFF (Manual '0')"));
        break;
      case '2':
        currentMode = MODE_ADAPTIVE_FEEDBACK;
        Serial.println(F("\n[CMD] Switched to: MODE_ADAPTIVE_FEEDBACK (Autonomous On-Chip)"));
        break;
      case '4':
        currentMode = MODE_ANTIPHASE_INTERFERENCE;
        Serial.println(F("\n[CMD] Switched to: MODE_ANTIPHASE_INTERFERENCE (Destructive Interference)"));
        break;
      case 'S':
      case 's':
        mlGateEnabled = false;
        enterSafeState();
        currentMode = MODE_MONITOR_ONLY;
        Serial.println(F("\n[CMD] EMERGENCY STOP: Motors Halted."));
        break;
      default:
        Serial.print(F("\n[CMD] Unknown key: "));
        Serial.println(cmd);
        break;
    }
  }
}

void printSerialMonitorReport(int16_t ax, int16_t ay, int16_t az, int16_t gx, int16_t gy, int16_t gz) {
  if (tremorDetected) {
    Serial.print(F(">>> [STATUS: PARKINSONIAN TREMOR] <<< | Freq: "));
    Serial.print(estimatedFrequency, 1);
    Serial.print(F(" Hz | RMS: "));
    Serial.print(currentTremorRMS, 0);
    Serial.print(F(" LSB | ZC-Phase: "));
    Serial.print(currentPhaseRad * 180.0f / 3.14159f, 0);
    Serial.print(F(" deg | ALL MOTORS: ACTIVE (PWM "));
    Serial.print(activeMotorPWM);
    Serial.println(F(")"));
  } else if (isVoluntaryMotion) {
    Serial.print(F("    [STATUS: VOLUNTARY GESTURE]    | Freq: "));
    Serial.print(estimatedFrequency, 1);
    Serial.print(F(" Hz | RMS: "));
    Serial.print(currentTremorRMS, 0);
    Serial.print(F(" LSB | (Waving/Reaching Rejected) | Motors: OFF"));
    Serial.println();
  } else {
    Serial.print(F("    [STATUS: NORMAL / RESTING]     | Freq: --.- Hz | RMS: "));
    Serial.print(currentTremorRMS, 0);
    Serial.print(F(" LSB | Zero-Crossing: IDLE        | Motors: OFF"));
    Serial.println();
  }
}

void logTelemetry(int16_t ax, int16_t ay, int16_t az, int16_t gx, int16_t gy, int16_t gz, float tremorSignal) {
  if (outputMode == TELEMETRY_MONITOR_DASHBOARD) {
    printSerialMonitorReport(ax, ay, az, gx, gy, gz);
  } else if (outputMode == TELEMETRY_RAW_CSV) {
    // Upscale by 4 to match original 16384 counts/g expected by Dataset.csv and Python ML models
    Serial.print((int32_t)ax * 4); Serial.print(',');
    Serial.print((int32_t)ay * 4); Serial.print(',');
    Serial.print((int32_t)az * 4); Serial.print(',');
    Serial.print(gx); Serial.print(',');
    Serial.print(gy); Serial.print(',');
    Serial.println(gz);
  } else {
    // Arduino Serial Plotter
    float deadband = 0.25f * signalEnvelope;
    Serial.print(F("Tremor_Signal:"));     Serial.print(tremorSignal, 1);           Serial.print(F(" "));
    Serial.print(F("Zero_Crossing:"));     Serial.print(crossingPulseMarker, 1);    Serial.print(F(" "));
    Serial.print(F("Deadband_Pos:"));      Serial.print(deadband, 1);               Serial.print(F(" "));
    Serial.print(F("Deadband_Neg:"));      Serial.print(-deadband, 1);              Serial.print(F(" "));
    Serial.print(F("Motor_PWM:"));         Serial.print(activeMotorPWM);            Serial.print(F(" "));
    Serial.print(F("Freq_Hz:"));           Serial.println(estimatedFrequency * 20.0f, 1);
  }
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n========================================================"));
  Serial.println(F("  WEARABLE PARKINSON'S TREMOR CONTROLLER v3.6"));
  Serial.println(F("  Dual-Channel DRV8833 (OUT1..OUT4) + ULN2003 (4-5 Motors)"));
  Serial.println(F("  With Butterworth BPF, Zero-Crossing, and Anti-Freeze"));
  Serial.println(F("========================================================"));

  // Configure DRV8833 sleep pin and drive HIGH to wake driver
  pinMode(DRV_SLEEP_PIN, OUTPUT);
  digitalWrite(DRV_SLEEP_PIN, HIGH);

  // Configure motor output GPIOs
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);
  pinMode(MOTOR2_IN3, OUTPUT);
  pinMode(MOTOR2_IN4, OUTPUT);
  pinMode(MOTOR_AUX,  OUTPUT);

  digitalWrite(MOTOR1_IN1, LOW);
  digitalWrite(MOTOR1_IN2, LOW);
  digitalWrite(MOTOR2_IN3, LOW);
  digitalWrite(MOTOR2_IN4, LOW);
  digitalWrite(MOTOR_AUX,  LOW);

  // Configure LEDC PWM Timers (1 kHz carrier)
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  bool ok1 = ledcAttach(MOTOR1_IN1, PWM_FREQ, PWM_RESOLUTION);
  bool ok2 = ledcAttach(MOTOR1_IN2, PWM_FREQ, PWM_RESOLUTION);
  bool ok3 = ledcAttach(MOTOR2_IN3, PWM_FREQ, PWM_RESOLUTION);
  bool ok4 = ledcAttach(MOTOR2_IN4, PWM_FREQ, PWM_RESOLUTION);
  bool ok5 = ledcAttach(MOTOR_AUX,  PWM_FREQ, PWM_RESOLUTION);
  Serial.print(F(">> [PWM INIT] LEDC v3.x Attach Status: "));
  Serial.println((ok1 && ok2 && ok3 && ok4 && ok5) ? F("ALL CHANNELS SUCCESS") : F("WARNING ON SOME PINS"));
#else
  double f1 = ledcSetup(CH_M1_IN1, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR1_IN1, CH_M1_IN1);

  double f2 = ledcSetup(CH_M1_IN2, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR1_IN2, CH_M1_IN2);

  double f3 = ledcSetup(CH_M2_IN3, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR2_IN3, CH_M2_IN3);

  double f4 = ledcSetup(CH_M2_IN4, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR2_IN4, CH_M2_IN4);

  double f5 = ledcSetup(CH_M_AUX, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_AUX, CH_M_AUX);

  Serial.print(F(">> [PWM INIT] Channel 1 Freq: ")); Serial.print(f1);
  Serial.print(F(" Hz | Channel 2 Freq: ")); Serial.print(f3);
  Serial.print(F(" Hz | AUX Freq: ")); Serial.print(f5); Serial.println(F(" Hz"));
  if (f3 > 0.0) {
    Serial.println(F(">> [SUCCESS] Channel 2 Hardware Timer Initialized Successfully!"));
  } else {
    Serial.println(F("## [ERROR] Channel 2 Timer failed! Check pin assignments."));
  }
#endif

  // Motors OFF at boot. Hardware test must never run automatically on a wearable.
  setMotorOutputs(0);

  // Initialize MPU6050 with +/- 8g range
  Serial.println(F(">> Initializing MPU6050 (+/- 8g range to prevent motor saturation)..."));
  if (!initMPU6050Hardware()) {
    Serial.println(F("## WARNING: MPU6050 not acknowledging. Attempting bus recovery..."));
    recoverI2CBus();
  }
  Serial.println(F(">> MPU6050 Online. Butterworth BPF Active."));

  // Pre-seed DC gravity baseline
  for (int i = 0; i < RMS_WINDOW_SIZE; i++) rmsBuffer[i] = 0.0f;
  long initAx = 0, initAy = 0, initAz = 0;
  int validInitCount = 0;
  for (int i = 0; i < 40; i++) {
    int16_t m[6];
    if (readMotionData(m)) {
      initAx += m[0]; initAy += m[1]; initAz += m[2];
      validInitCount++;
    }
    delay(5);
  }
  if (validInitCount > 0) {
    dcAx = (float)initAx / validInitCount;
    dcAy = (float)initAy / validInitCount;
    dcAz = (float)initAz / validInitCount;
  }

  Serial.println(F("========================================================"));
  Serial.println(F("  COMMAND KEYS IN SERIAL MONITOR:"));
  Serial.println(F("  'm' -> Human-readable Diagnostic Predictions (Default)"));
  Serial.println(F("  'r' -> Raw CSV stream for Python predict_serial.py"));
  Serial.println(F("  'p' -> Arduino Serial Plotter waveforms"));
  Serial.println(F("  't' -> Disabled for safety in ML-phase build"));
  Serial.println(F("  'i' -> Toggle Channel 2 Polarity Invert (Flip Direction)"));
  Serial.println(F("  'q' -> Disabled for safety (dual H-bridge only)"));
  Serial.println(F("  'G' -> ML phase-timed pulse gate ON + host heartbeat"));
  Serial.println(F("  'D' -> ML phase-timed pulse gate OFF"));
  Serial.println(F("  'V' -> Set motor intensity (next byte = PWM 0-240)"));
  Serial.println(F("  '1'/'0' legacy constant-drive testing (NOT ML phase feedback)"));
  Serial.println(F("========================================================\n"));

  lastSampleTimeUs = micros();
  lastGoodSensorMs = millis();
}

// ============================================================================
// MAIN DETERMINISTIC 100 HZ LOOP
// ============================================================================
void loop() {
  // 1. Process Serial Commands FIRST on every loop pass (Guarantees zero input lag!)
  handleSerialCommands();

  uint32_t nowUs = micros();
  uint32_t nowMs = millis();

  // 2. Background MPU Sleep Watchdog (Every 1000 ms)
  if (nowMs - lastWatchdogTimeMs >= 1000) {
    lastWatchdogTimeMs = nowMs;
    uint8_t pwr = 0;
    if (readMPUReg(0x6B, &pwr)) {
      if (pwr & 0x40) { // If SLEEP bit is set
        reawakenMPU();
        if (outputMode == TELEMETRY_MONITOR_DASHBOARD) {
          Serial.println(F("[MPU WATCHDOG] Cleared sleep bit (0x6B) -> Sensor re-awakened."));
        }
      }
    }
  }

  // 3. Precise 100 Hz Kinematic Acquisition & DSP Loop
  if (nowUs - lastSampleTimeUs >= SAMPLE_INTERVAL_US) {
    lastSampleTimeUs = nowUs; // CRITICAL: Reset directly to nowUs to prevent slip accumulation!

    int16_t imu[6];
    if (readMotionData(imu)) {
      lastGoodSensorMs = millis();
      int16_t ax = imu[0], ay = imu[1], az = imu[2];
      int16_t gx = imu[3], gy = imu[4], gz = imu[5];

      // Gravity baseline tracking
      dcAx = (1.0f - DC_ALPHA) * dcAx + DC_ALPHA * (float)ax;
      dcAy = (1.0f - DC_ALPHA) * dcAy + DC_ALPHA * (float)ay;
      dcAz = (1.0f - DC_ALPHA) * dcAz + DC_ALPHA * (float)az;

      float dynAx = (float)ax - dcAx;
      float dynAy = (float)ay - dcAy;
      float dynAz = (float)az - dcAz;

      float dynamicAcc = sqrtf(dynAx * dynAx + dynAy * dynAy + dynAz * dynAz);
      float gyroMag = sqrtf((float)gx * (float)gx + (float)gy * (float)gy + (float)gz * (float)gz);

      // 3.5 - 7.5 Hz Butterworth Bandpass Filter execution
      float tremorSignal = processBiquadBPF(dynamicAcc);

      // Intensity & Zero-Crossing tracking
      updateTremorIntensity(tremorSignal);
      float phaseAxis = (PHASE_GYRO_AXIS == 1) ? (float)gy :
                        (PHASE_GYRO_AXIS == 2) ? (float)gz : (float)gx;
      float signedPhaseSignal = processSignedGyroBPF(phaseAxis);
      updateZeroCrossingAndFrequency(signedPhaseSignal, nowUs);
      updatePhaseEstimator(nowUs);

      // Biomechanical Voluntary Motion Gating
      bool isMotionActive = (currentTremorRMS >= TREMOR_DETECT_THRESHOLD);
      bool isParkinsonianFreq = phaseValid && (estimatedFrequency >= TREMOR_FREQ_MIN_HZ && estimatedFrequency <= TREMOR_FREQ_MAX_HZ);
      isVoluntaryMotion = (gyroMag > VOLUNTARY_GYRO_LIMIT) || (gyroMag > 2200.0f && !isParkinsonianFreq);

      if (isMotionActive && isParkinsonianFreq && !isVoluntaryMotion) {
        if (tremorStreakCount < TREMOR_STREAK_REQUIRED) tremorStreakCount++;
      } else {
        if (tremorStreakCount > 0) tremorStreakCount--;
      }

      tremorDetected = (tremorStreakCount >= TREMOR_STREAK_REQUIRED);

      // Execute Control Action
      switch (currentMode) {
        case MODE_MONITOR_ONLY:
          enterSafeState();
          break;
        case MODE_ADAPTIVE_FEEDBACK:
          updateAdaptiveFeedback(nowMs);
          break;
        case MODE_ANTIPHASE_INTERFERENCE:
          updateAntiphaseInterference(nowMs);
          break;
        case MODE_HOST_ML_CONTROL:
          // Legacy steady ON/OFF (not used by ML phase controller)
          break;
        case MODE_ML_PHASE_FEEDBACK:
          // ML gate determines eligibility; local scheduler below times pulses.
          break;
      }

      // Telemetry Output Cadence
      if (outputMode == TELEMETRY_RAW_CSV) {
        // Every successful 100 Hz sample: required by the trained 100 Hz ML pipeline.
        logTelemetry(ax, ay, az, gx, gy, gz, tremorSignal);
      } else {
        uint32_t logRate = (outputMode == TELEMETRY_MONITOR_DASHBOARD) ? MONITOR_LOG_INTERVAL_MS : 20;
        if (nowMs - lastLogTimeMs >= logRate) {
          lastLogTimeMs = nowMs;
          logTelemetry(ax, ay, az, gx, gy, gz, tremorSignal);
        }
      }
    }
  }
  // Pulse timing and safety run on every loop iteration, including failed IMU reads.
  updateMlPhaseFeedback(micros(), millis());
}
