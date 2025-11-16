#include <WiFi.h>
#include <PubSubClient.h>
#include <TinyGPS++.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>

#ifndef MQTT_MAX_PACKET_SIZE
#define MQTT_MAX_PACKET_SIZE 512
#endif

// -----------------------------
// User configuration
// -----------------------------
const char *WIFI_SSID = "Andrikos";          // Replace with ground-station SSID
const char *WIFI_PASSWORD = "";              // Replace with ground-station password
const char *MQTT_SERVER = "5.196.78.28";     // Broker that archives uploaded logs
const uint16_t MQTT_PORT = 1883;
const char *MQTT_TOPIC = "flight/location_data";  // Topic consumed by ground station

// GPIO pin that can optionally be tied to a landing switch. Leave at -1 to disable.
const int LANDING_SWITCH_PIN = -1;
const bool LANDING_SWITCH_ACTIVE_LOW = true;

// Logging tunables
static constexpr const char *LOG_FILE = "/flight_log.jsonl";
static constexpr uint32_t GPS_LOG_INTERVAL_MS = 1000;    // Log every second
static constexpr uint32_t WIFI_RETRY_INTERVAL_MS = 5000;  // Retry Wi-Fi after 5 s
static constexpr uint8_t MAX_WIFI_ATTEMPTS = 3;

// -----------------------------
// Globals
// -----------------------------
TinyGPSPlus gps;
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
unsigned long lastGpsLog = 0;
unsigned long lastWifiAttempt = 0;
bool uploadPending = false;
bool uploadRunning = false;

// Forward declarations
void logSample();
void flushLogsToBroker();
bool safePublish(const String &payload);
String buildIsoTimestamp();
bool landingConfirmed();

void setup() {
  Serial.begin(38400);
  if (!SPIFFS.begin(true)) {
    Serial.println("[boot] SPIFFS init failed");
  }

  if (LANDING_SWITCH_PIN >= 0) {
    pinMode(LANDING_SWITCH_PIN, LANDING_SWITCH_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  }

  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);

  Serial.println("[boot] Offline logger ready");
}

void loop() {
  while (Serial.available() > 0) {
    gps.encode(Serial.read());
  }

  if (gps.location.isUpdated() && millis() - lastGpsLog > GPS_LOG_INTERVAL_MS) {
    logSample();
    lastGpsLog = millis();
    uploadPending = true;
  }

  if (WiFi.status() == WL_CONNECTED) {
    mqttClient.loop();
    if (uploadPending && !uploadRunning && landingConfirmed()) {
      uploadRunning = true;
      flushLogsToBroker();
      uploadRunning = false;
      uploadPending = SPIFFS.exists(LOG_FILE);  // Retry if anything left
      if (!uploadPending) {
        Serial.println("[upload] All samples uploaded, disconnecting Wi-Fi");
        WiFi.disconnect(true);
      }
    }
    return;
  }

  if (landingConfirmed() && millis() - lastWifiAttempt > WIFI_RETRY_INTERVAL_MS) {
    Serial.println("[wifi] Attempting to connect...");
    lastWifiAttempt = millis();
    uint8_t attempts = 0;
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED && attempts < MAX_WIFI_ATTEMPTS) {
      delay(500);
      attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.print("[wifi] Connected, IP: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("[wifi] Unable to connect, will retry");
      WiFi.disconnect(true);
    }
  }
}

void logSample() {
  if (!gps.location.isValid()) {
    return;
  }
  File file = SPIFFS.open(LOG_FILE, FILE_APPEND);
  if (!file) {
    Serial.println("[log] Failed to open log file");
    return;
  }
  StaticJsonDocument<256> doc;
  doc["utc_time"] = buildIsoTimestamp();
  doc["latitude"] = gps.location.lat();
  doc["longitude"] = gps.location.lng();
  if (gps.altitude.isValid()) {
    doc["altitude"] = gps.altitude.meters();
  }
  if (gps.speed.isValid()) {
    doc["speed_mps"] = gps.speed.mps();
  }
  serializeJson(doc, file);
  file.println();
  file.close();
  Serial.print("[log] Sample stored -> ");
  serializeJson(doc, Serial);
  Serial.println();
}

void flushLogsToBroker() {
  if (!SPIFFS.exists(LOG_FILE)) {
    Serial.println("[upload] No samples to send");
    return;
  }

  File input = SPIFFS.open(LOG_FILE, FILE_READ);
  File retry = SPIFFS.open("/retry.jsonl", FILE_WRITE);
  if (!input) {
    Serial.println("[upload] Failed to open log file");
    return;
  }
  if (!retry) {
    Serial.println("[upload] Failed to open retry file");
    input.close();
    return;
  }

  while (input.available()) {
    String line = input.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) {
      continue;
    }
    if (!safePublish(line)) {
      retry.println(line);
    }
  }
  input.close();
  retry.close();

  SPIFFS.remove(LOG_FILE);
  if (SPIFFS.exists("/retry.jsonl")) {
    File retryFile = SPIFFS.open("/retry.jsonl", FILE_READ);
    size_t remaining = retryFile.size();
    retryFile.close();
    if (remaining == 0) {
      SPIFFS.remove("/retry.jsonl");
    } else {
      SPIFFS.rename("/retry.jsonl", LOG_FILE);
    }
  }
}

bool safePublish(const String &payload) {
  if (!mqttClient.connected()) {
    while (!mqttClient.connected()) {
      String clientId = String("esp32-flight-") + String((uint32_t)ESP.getEfuseMac(), HEX);
      if (mqttClient.connect(clientId.c_str())) {
        break;
      }
      delay(250);
    }
  }
  bool ok = mqttClient.publish(MQTT_TOPIC, payload.c_str(), true);
  Serial.printf("[upload] %s -> %s\n", ok ? "sent" : "failed", payload.c_str());
  return ok;
}

String buildIsoTimestamp() {
  if (gps.time.isValid() && gps.date.isValid()) {
    char buffer[25];
    snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02dZ",
             gps.date.year(), gps.date.month(), gps.date.day(),
             gps.time.hour(), gps.time.minute(), gps.time.second());
    return String(buffer);
  }
  return String(millis());
}

bool landingConfirmed() {
  if (LANDING_SWITCH_PIN < 0) {
    return true;
  }
  int level = digitalRead(LANDING_SWITCH_PIN);
  return LANDING_SWITCH_ACTIVE_LOW ? (level == LOW) : (level == HIGH);
}
