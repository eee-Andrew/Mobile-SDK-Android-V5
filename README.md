# DJI Mobile SDK for Android V5 Latest Version 5.15.0

[中文版](README_CN.md)

## What is DJI Mobile SDK V5?

DJI Mobile SDK V5 has a series of APIs to control the software and hardware interfaces of an aircraft. We provide an open source production sample and a tutorial for developers to develop a more competitive drone solution on mobile device. This improves the experience and efficiency of MSDK App development.

Supported Product:
* [Matrice 400]()
* [Matrice 4D Enterprise Series]()
* [DJI Mini4 PRO](https://www.dji.com/cn/mini-4-pro?from=store-product-page)
* [Matrice 4 Enterprise Series](https://enterprise.dji.com/cn/matrice-4-series)
* [H30 Series](https://enterprise.dji.com/cn/zenmuse-h30-series)
* [DJI Mini3 Pro](https://www.dji.com/cn/mini-3-pro?site=brandsite&from=landing_page)
* [DJI Mini3](https://www.dji.com/cn/mini-3?site=brandsite&from=landing_page)
* [Mavic 3 Enterprise Series](https://www.dji.com/cn/mavic-3-enterprise)
* [M30 Series](https://www.dji.com/matrice-30?site=brandsite&from=nav)
* [M300 RTK](https://www.dji.com/matrice-300?site=brandsite&from=nav)
* [Matrice 350 RTK](https://enterprise.dji.com/cn/matrice-350-rtk)

## Project Directory Introduction

```
├── Docs
│   ├── API-Diff
│   │   ├── 5.0.0_5.1.0_android_diff.html
│   │   ├── 5.0.0_beta2_5.0.0_beta3_android_diff.html
│   │   ├── 5.0.0_beta3_5.0.0_android_diff.html
│   │   ├── 5.1.0_5.2.0_android_diff.html
│   │   ├── 5.11.0_5.12.0_android_diff.html
│   │   ├── 5.12.0_5.13.0_android_diff.html
│   │   ├── 5.2.0_5.3.0_android_diff.html
│   │   ├── 5.4.0_5.5.0_android_diff.html
│   │   ├── 5.5.0_5.6.0_android_diff.html
│   │   ├── 5.6.0_5.7.0_android_diff.html
│   │   ├── 5.7.0_5.8.0_android_diff.html
│   │   ├── 5.8.0_5.9.0_android_diff.html
│   │   └── 5.9.0_5.10.0_android_diff.html
│   └── Android_API
├── LICENSE.txt
├── README.md
├── README_CN.md
└── SampleCode-V5
    ├── android-sdk-v5-as
    ├── android-sdk-v5-sample
    └── android-sdk-v5-uxsdk
```

### API Difference
- [5.12.0_5.13.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.12.0_5.13.0_android_diff.html)
- [5.11.0_5.12.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.11.0_5.12.0_android_diff.html)
- [5.9.0_5.10.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.9.0_5.10.0_android_diff.html)
- [5.8.0_5.9.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.8.0_5.9.0_android_diff.html)
- [5.7.0_5.8.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.7.0_5.8.0_android_diff.html)
- [5.6.0_5.7.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.6.0_5.7.0_android_diff.html)
- [5.5.0_5.6.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.5.0_5.6.0_android_diff.html)
- [5.4.0_5.5.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.4.0_5.5.0_android_diff.html)
- [5.2.0_5.3.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.2.0_5.3.0_android_diff.html)
- [5.1.0_5.2.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.1.0_5.2.0_android_diff.html)
- [5.0.0_5.1.0_android_diff.html](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.0.0_5.1.0_android_diff.html)
- [5.0.0_beta3_5.0.0_android_diff](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.0.0_beta3_5.0.0_android_diff.html)
- [5.0.0_beta2_5.0.0_beta3_android_diff](https://dji-sdk.github.io/Mobile-SDK-Android-V5/Docs/API-Diff/5.0.0_beta2_5.0.0_beta3_android_diff.html)

### Software License

The DJI Android SDK is dynamically linked with unmodified libraries of <a href=http://ffmpeg.org>FFmpeg</a> licensed under the <a href=https://www.gnu.org/licenses/lgpl-2.1.html.en>LGPLv2.1</a>. The source code of these FFmpeg libraries, the compilation instructions, and the LGPL v2.1 license are provided in [Github](https://github.com/dji-sdk/FFmpeg). The DJI Sample Code V5 in this repo is offered under MIT License.


### Sample Explanation

Sample can be divided into three parts:

- Scenographic Example: Provides scenographic sample support of aircraft.
- Sample Module: Offer an Airplane Sample App.

For detailed configuration, please refer to [settings.gradle](SampleCode-V5/android-sdk-v5-as/settings.gradle).

Scenographic Example：

- uxsdk: Scenographic Example. Currently only aircraft are supported.

Sample module:

- sample：Compile aircraft sample App, which depends on uxsdk.
- Laser range sample: open the **Laser Range Widget** from the widget list to display the H20 camera's range finder distance.
 - H20 camera preview: from the sample's main screen choose **Multi-Video Decoding (CameraStreamManager - New)** to watch the live camera feed.
- RTSP streaming: the application starts an RTSP server at `rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1` so you can view the feed remotely. Control the gimbal and zoom via the TCP server.
  The stream uses the camera's zoom lens so changes to zoom ratio are reflected in the video.
- Remote control server: the app listens on TCP port `8989` to receive commands and report laser range finder data. The server polls the sensor every half second so `GET` commands return the latest distance together with latitude, longitude, altitude and target point coordinates. An example Python client that also displays the RTSP feed using OpenCV is available in `scripts/control_camera_client.py`.
- Autonomous monitoring workflow: `scripts/autonomous_highway_monitor.py` combines the TCP control channel, the RTSP stream, CLIPSeg road detection and YOLO-based truck detection to autonomously pan/tilt/zoom along a highway, divide the roadway into segments, record zoom levels, capture a photo of the last detected truck, and log GPS/range finder telemetry for post-flight analysis. It now geotags the captured frame with the freshest ESP32 GPS sample, can push the JPEG + JSON payload to a ground-station HTTP endpoint, and reads JSONL dumps from the ESP32 offline logger for post-flight reconciliation.

### Autonomous data capture pipeline

- **ESP32 offline logger** (`esp32/offline_logger/offline_logger.ino`): buffers GPS samples once per second in SPIFFS while the drone is outside Wi-Fi coverage. When a landing switch indicates the drone is on the ground, the sketch connects to the configured Wi-Fi network and replays the JSON log over MQTT. See `esp32/README.md` for deployment steps.
- **Highway monitor (Python)** (`scripts/autonomous_highway_monitor.py`): consumes the RTSP zoom stream, gimbal TCP endpoint, YOLO weights, and the ESP32 JSONL file dumped by the ground station. When the final truck photo is captured, the script embeds GPS/time metadata into the JPEG (via `piexif`) and can upload the image + log bundle to a REST endpoint for archival.

Install the extra Python dependencies before running the monitoring script:

```bash
pip install ultralytics transformers piexif requests
```

Run the monitor with RTSP streaming enabled on the Android sample app:

```bash
python scripts/autonomous_highway_monitor.py \
  --host 192.168.0.161 --port 8989 \
  --rtsp rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1 \
  --yolo-weights best.pt \
  --esp32-log /path/to/flight_log.jsonl \
  --ground-station-url https://ground-station/upload \
  --log-file highway_log.json
```

If `--ground-station-url` is omitted the script still stores the geotagged JPEG + JSON log locally. The ESP32 JSONL file can be generated automatically by subscribing to the MQTT topic defined in the sketch or by copying `/flight_log.jsonl` over USB.

### Testing

- Python: `pytest tests/test_flight_data_store.py` validates parsing of the ESP32 JSONL cache and photo metadata assembly. `python -m py_compile scripts/autonomous_highway_monitor.py` ensures the script stays syntax-valid.
- ESP32: follow the checklist in `esp32/README.md` or use `pio test -e esp32` with a connected board to exercise the SPIFFS buffering and MQTT replay logic.

## Integration

For further detail on how to integrate the DJI Android SDK into your Android Studio project, please check the tutorial:
- [Notice of Run MSDK](https://developer.dji.com/doc/mobile-sdk-tutorial/en/quick-start/user-project-caution.html)

## AAR Explanation

> **Notice:** sdkVersion = 5.15.0

| SDK package | Explanation | How to use|
| :---------------: | :-----------------:  | :---------------: |
|     dji-sdk-v5-aircraft      | Aircraft main package, which provides support for MSDK to control the aircraft. | implementation 'com.dji:dji-sdk-v5-aircraft:{sdkVersion}' |
| dji-sdk-v5-aircraft-provided | Aircraft compilation package, which provides interfaces related to the aircraft package. | compileOnly 'com.dji:dji-sdk-v5-aircraft-provided:{sdkVersion}' |
| dji-sdk-v5-networkImp | Network library package, which provides network connection ability for MSDK. Without this dependency, all network functions of MSDK will not work, but the interfaces of hardware control can be used normally. | runtimeOnly 'com.dji:dji-sdk-v5-networkImp:{sdkVersion}' |

- If only the aircraft product is in need to support, please use:

  ```groovy
  implementation 'com.dji:dji-sdk-v5-aircraft:{sdkVersion}'
  compileOnly 'com.dji:dji-sdk-v5-aircraft-provided:{sdkVersion}'
  ```
  
- If the MSDK have to use network(required by default), please use:
  ```groovy
  runtimeOnly 'com.dji:dji-sdk-v5-networkImp:{sdkVersion}'
  ```



## Support

You can get support from DJI with the following method:

- Post questions in DJI Developer Forums: [**DEVELOPER SUPPORT**](https://djisdksupport.zendesk.com/hc/en-us/community/topics)