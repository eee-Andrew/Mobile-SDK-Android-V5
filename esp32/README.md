# ESP32 Offline Flight Logger

This sketch configures an ESP32 as a black-box recorder for GPS samples while the
aircraft is away from Wi-Fi coverage. Samples are buffered in SPIFFS and replayed
via MQTT when the drone lands within range of the ground station.

## Features

- Logs timestamp, latitude, longitude, altitude (when available) and airspeed to
  `/flight_log.jsonl` once per second.
- Keeps Wi-Fi disabled while airborne. After landing, it repeatedly attempts to
  join the configured SSID and uploads samples to `flight/location_data`.
- Retries failed MQTT publishes without losing samples by moving them into a
  retry file.
- Optional landing switch input lets you delay the upload attempts until the
  drone is safely on the ground.

## Deployment

1. Open `esp32/offline_logger/offline_logger.ino` in the Arduino IDE or
   PlatformIO.
2. Update `WIFI_SSID`, `WIFI_PASSWORD`, `MQTT_SERVER`, and `MQTT_TOPIC` to match
   your ground station.
3. (Optional) Set `LANDING_SWITCH_PIN` to the GPIO tied to your landing gear or
   arm switch.
4. Flash the sketch to the ESP32. Connect the GPS module UART to the hardware
   serial port used in the sketch (default: `Serial` at 38 400 baud).
5. Verify that `/flight_log.jsonl` grows with JSON entries while airborne.
6. After landing, confirm that the ESP32 connects to Wi-Fi and publishes the
   buffered entries.

## Testing

The MQTT replay logic was designed to run inside PlatformIO's `unity` test
framework. Use `pio test -e esp32` with an attached board or the Arduino IDE's
Serial Monitor to confirm:

- GPS samples continue logging when Wi-Fi is down.
- `/retry.jsonl` only exists when there are publish failures.
- All data in `/flight_log.jsonl` is removed after a successful upload.
