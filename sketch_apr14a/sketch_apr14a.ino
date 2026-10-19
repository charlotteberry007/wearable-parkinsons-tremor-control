#include <Wire.h>
#include <MPU6050.h>
#include <LiquidCrystal_I2C.h>
#include <esp_arduino_version.h>

MPU6050 mpu;
LiquidCrystal_I2C lcd(0x27, 16, 2);
bool lcdPresent = false; // Auto-detected at startup

// Pin Definitions
#define MOTOR_PIN 4  // ULN2003 IN1 connected to GPIO 4

// Sampling Rate Setup (100 Hz = 10ms period)
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 10;

// Motor PWM Parameters (150-200 Hz sensory vibration)
const int PWM_FREQ = 175;        // 175 Hz sensory stimulation frequency
const int PWM_CHANNEL = 0;       // Used for ESP32 Core v2.x
const int PWM_RESOLUTION = 8;    // 0-255 duty cycle
const int MOTOR_INTENSITY = 200; // ~78% duty cycle for ERM motor

bool tremorActive = false;

// Universal PWM Write Helper (compatible with ESP32 Core v2.x and v3.x+)
void setMotorPWM(int duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(MOTOR_PIN, duty);
#else
  ledcWrite(PWM_CHANNEL, duty);
#endif
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA = GPIO 21, SCL = GPIO 22

  // Auto-detect if LCD is physically connected at I2C address 0x27
  Wire.beginTransmission(0x27);
  if (Wire.endTransmission() == 0) {
    lcdPresent = true;
    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("System Ready");
    lcd.setCursor(0, 1);
    lcd.print("Monitoring...");
  } else {
    Serial.println("No I2C LCD detected. Running in Headless / Motor mode.");
  }

  mpu.initialize();
  if (!mpu.testConnection()) {
    if (lcdPresent) {
      lcd.setCursor(0, 0);
      lcd.print("MPU6050 Error!");
    }
    Serial.println("MPU6050 connection failed! Check wiring on GPIO 21 (SDA) and GPIO 22 (SCL).");
    while (1);
  }

  // Setup ESP32 PWM Peripheral for Motor (Version-agnostic)
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(MOTOR_PIN, PWM_FREQ, PWM_RESOLUTION);
#else
  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(MOTOR_PIN, PWM_CHANNEL);
#endif

  setMotorPWM(0); // Ensure motor is off initially
}

void loop() {
  unsigned long currentTime = millis();

  // 1. Precise 100 Hz Kinematic Sampling Loop
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Stream raw values over Serial: ax,ay,az,gx,gy,gz
    Serial.print(ax); Serial.print(",");
    Serial.print(ay); Serial.print(",");
    Serial.print(az); Serial.print(",");
    Serial.print(gx); Serial.print(",");
    Serial.print(gy); Serial.print(",");
    Serial.println(gz);
  }

  // 2. Non-Blocking Feedback Command Processing
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    if (cmd == '1' && !tremorActive) {
      tremorActive = true;
      setMotorPWM(MOTOR_INTENSITY); // Turn ON Vibrotactile Motor
      
      if (lcdPresent) {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("STATUS: TREMOR");
        lcd.setCursor(0, 1);
        lcd.print("Stimulation ON ");
      }
    } 
    else if (cmd == '0' && tremorActive) {
      tremorActive = false;
      setMotorPWM(0); // Turn OFF Motor
      
      if (lcdPresent) {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("STATUS: NORMAL");
        lcd.setCursor(0, 1);
        lcd.print("Monitoring...   ");
      }
    }
  }
}