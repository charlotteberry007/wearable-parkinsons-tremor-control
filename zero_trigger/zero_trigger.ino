/**
 * ============================================================================
 * Standalone Zero-Crossing & Frequency Estimator for Tremor Monitoring
 * Target: ESP32 with MPU6050 (SDA: GPIO 21, SCL: GPIO 22)
 * ============================================================================
 */

#include <Wire.h>
#include <MPU6050.h>
#include <math.h>

MPU6050 mpu;

// Sampling Timing: 100 Hz = 10,000 microseconds
const uint32_t SAMPLE_INTERVAL_US = 10000;
uint32_t lastSampleUs = 0;

// Physiological Parkinsonian Frequency Limits (3.5 Hz - 7.5 Hz)
const float    FREQ_MIN_HZ         = 3.5f;
const float    FREQ_MAX_HZ         = 7.5f;
const uint32_t MIN_CYCLE_PERIOD_US = (uint32_t)(1000000.0f / FREQ_MAX_HZ); // 133,333 us (7.5 Hz)
const uint32_t MAX_CYCLE_PERIOD_US = (uint32_t)(1000000.0f / FREQ_MIN_HZ); // 285,714 us (3.5 Hz)

// Baseline (Gravity) Tracker
float dcBaseline = 16384.0f;
const float DC_ALPHA = 0.0314f; // ~0.5 Hz highpass cutoff at 100 Hz

// 2nd-Order Butterworth Bandpass Filter (3.5 - 7.5 Hz @ 100 Hz fs)
// Direct Form II Transposed coefficients
const float b0 = 0.109930f, b1 = 0.0f, b2 = -0.109930f;
const float a1 = -1.688680f, a2 = 0.780130f;
float d1 = 0.0f, d2 = 0.0f;

// Zero-Crossing Detector State Variables
float    prevSignal          = 0.0f;
bool     negativeArmed       = false;      // Schmitt trigger arming state
uint32_t lastCrossingUs      = 0;
uint32_t estimatedPeriodUs   = 200000;     // Default 200 ms (5.0 Hz)
float    estimatedFrequency  = 0.0f;
bool     frequencyValid      = false;
float    crossingPulseMarker = 0.0f;       // Serial plotter visual pulse

// Exponential moving average for noise deadband estimation
float signalEnvelope = 100.0f;

/**
 * 2nd-order Direct Form II Transposed Biquad Filter
 */
float applyBandpassFilter(float in) {
  float out = b0 * in + d1;
  d1 = b1 * in - a1 * out + d2;
  d2 = b2 * in - a2 * out;
  return out;
}

/**
 * Robust Zero-Crossing Detector with Schmitt Trigger and Period Gating
 * 
 * @param signal Current filtered, zero-mean kinematic sample
 * @param nowUs  Current microsecond timestamp
 */
void processZeroCrossing(float signal, uint32_t nowUs) {
  // 1. Dynamic Deadband: 25% of recent envelope to avoid triggering on sensor noise
  signalEnvelope = 0.98f * signalEnvelope + 0.02f * fabsf(signal);
  float deadband = 0.25f * signalEnvelope;
  if (deadband < 25.0f) deadband = 25.0f; // Minimum noise floor clamp

  crossingPulseMarker = 0.0f; // Clear marker for this sample

  // 2. Arm Schmitt Trigger: Only arm if signal dives below the negative deadband
  if (signal < -deadband) {
    negativeArmed = true;
  }

  // 3. Detect Positive-Going Zero-Crossing
  if (negativeArmed && (prevSignal <= 0.0f) && (signal > 0.0f)) {
    uint32_t cycleDeltaUs = nowUs - lastCrossingUs;

    // 4. Validate plausible physiological period (133 ms - 286 ms)
    if (cycleDeltaUs >= MIN_CYCLE_PERIOD_US && cycleDeltaUs <= MAX_CYCLE_PERIOD_US) {
      // 5. Smooth period using a 1st-order IIR filter (prevents single-cycle jumps)
      estimatedPeriodUs = (uint32_t)(0.70f * (float)estimatedPeriodUs + 0.30f * (float)cycleDeltaUs);
      estimatedFrequency = 1000000.0f / (float)estimatedPeriodUs;
      
      frequencyValid = true;
      lastCrossingUs = nowUs;
      negativeArmed = false;
      crossingPulseMarker = 300.0f; // Trigger visual spike for Serial Plotter
    } 
    else if (cycleDeltaUs > MAX_CYCLE_PERIOD_US) {
      // Cycle took too long (> 286 ms); reset timing baseline
      lastCrossingUs = nowUs;
      negativeArmed = false;
      frequencyValid = false;
    }
  }

  prevSignal = signal;

  // 6. Timeout: If no valid crossing occurs within 1.35x expected period, invalidate
  if (nowUs - lastCrossingUs > (uint32_t)(1.35f * (float)estimatedPeriodUs)) {
    frequencyValid = false;
    estimatedFrequency = 0.0f;
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);

  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println(F("MPU6050 Connection Failed! Check SDA(21)/SCL(22)."));
    while (1);
  }

  // Internal digital low-pass filter (42 Hz) to reject high-frequency mechanical noise
  mpu.setDLPFMode(3);

  Serial.println(F("Filtered_Signal,Deadband_High,Deadband_Low,Zero_Crossing_Trigger,Estimated_Freq_Hz"));
  lastSampleUs = micros();
}

void loop() {
  uint32_t nowUs = micros();

  // 100 Hz Fixed-Rate Execution Loop
  if (nowUs - lastSampleUs >= SAMPLE_INTERVAL_US) {
    lastSampleUs += SAMPLE_INTERVAL_US;

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Compute 3D Acceleration Magnitude
    float accMag = sqrtf((float)ax * ax + (float)ay * ay + (float)az * az);

    // Subtract gravity DC offset
    dcBaseline = (1.0f - DC_ALPHA) * dcBaseline + DC_ALPHA * accMag;
    float dynamicAcc = accMag - dcBaseline;

    // Apply 3.5 - 7.5 Hz Butterworth Bandpass Filter
    float filteredTremor = applyBandpassFilter(dynamicAcc);

    // Execute Zero-Crossing Detection & Frequency Estimation
    processZeroCrossing(filteredTremor, nowUs);

    // Format for Arduino Serial Plotter
    float deadband = 0.25f * signalEnvelope;
    Serial.print(filteredTremor);          Serial.print(F(" "));
    Serial.print(deadband);                Serial.print(F(" "));
    Serial.print(-deadband);               Serial.print(F(" "));
    Serial.print(crossingPulseMarker);     Serial.print(F(" "));
    Serial.println(estimatedFrequency * 50.0f); // Scaled for display alongside waveform
  }
}
