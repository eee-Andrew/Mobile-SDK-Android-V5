package dji.v5.ux.sample.util

import android.os.Handler
import android.os.Looper
import android.util.Log
import dji.v5.manager.KeyManager
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.CameraKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.value.camera.CameraVideoStreamSourceType
import dji.sdk.keyvalue.value.camera.LaserMeasureInformation
import dji.sdk.keyvalue.value.camera.ZoomRatiosRange
import dji.sdk.keyvalue.value.common.CameraLensType
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.Attitude
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotation
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotationMode
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread
import kotlin.math.abs

/**
 * Simple TCP server to receive camera control commands and send range finder data.
 * Commands:
 *   SET <yaw> <pitch> <zoom>  - set gimbal orientation and zoom ratio
 *   GET                      - reply with RANGE <distance> LAT <lat> LON <lon>
 *                               ALT <alt> TX <x> TY <y> YAW <yaw> HEADING <heading>
 */
object RangeControlServer {
    private const val TAG = "RangeControlServer"
    private const val ZOOM_LENS_RETRY_DELAY_MS = 500L
    private const val ZOOM_LENS_MAX_ATTEMPTS = 10
    private var server: ServerSocket? = null
    private var laserInfo: LaserMeasureInformation? = null
    private val laserInfoKey = KeyTools.createCameraKey(
        CameraKey.KeyLaserMeasureInformation,
        ComponentIndexType.LEFT_OR_MAIN,
        CameraLensType.CAMERA_LENS_ZOOM
    )
    private val zoomKey = KeyTools.createCameraKey(
        CameraKey.KeyCameraZoomRatios,
        ComponentIndexType.LEFT_OR_MAIN,
        CameraLensType.CAMERA_LENS_ZOOM
    )
    private val streamSourceKey = KeyTools.createCameraKey(
        CameraKey.KeyCameraVideoStreamSource,
        ComponentIndexType.LEFT_OR_MAIN,
        CameraLensType.CAMERA_LENS_ZOOM
    )
    private val zoomRangeKey = KeyTools.createCameraKey(
        CameraKey.KeyCameraZoomRatiosRange,
        ComponentIndexType.LEFT_OR_MAIN,
        CameraLensType.CAMERA_LENS_ZOOM
    )
    private val attitudeKey = KeyTools.createKey(FlightControllerKey.KeyAircraftAttitude)
    private val zoomHandler = Handler(Looper.getMainLooper())
    @Volatile
    private var zoomRange: ZoomRatiosRange? = null
    private var pollingThread: Thread? = null

    @JvmStatic
    @JvmOverloads
    fun start(port: Int = 8989) {
        if (server != null) return
        server = ServerSocket(port)
        // switch the live stream to the zoom lens so zoom ratios are visible
        ensureZoomLensSelected()
        // enable the laser range finder so distance values can be returned
        enableLaserModule()
        // fetch zoom range in the background so we can clamp incoming values
        thread(start = true) {
            while (!Thread.currentThread().isInterrupted && zoomRange == null) {
                val range = KeyManager.getInstance().getValue<ZoomRatiosRange>(zoomRangeKey)
                if (range != null) {
                    zoomRange = range
                    break
                }
                Thread.sleep(500)
            }
        }
        // poll the laser measurement so latest values are available
        pollingThread = thread(start = true) {
            while (!server!!.isClosed) {
                val info = KeyManager.getInstance().getValue<LaserMeasureInformation>(laserInfoKey)
                if (info != null) {
                    laserInfo = info
                }
                Thread.sleep(500)
            }
        }
        thread {
            while (!server!!.isClosed) {
                try {
                    val socket = server!!.accept()
                    handleClient(socket)
                } catch (_: Exception) {
                }
            }
        }
    }

    private fun handleClient(socket: Socket) {
        thread {
            socket.use { sock ->
                val reader = BufferedReader(InputStreamReader(sock.getInputStream()))
                val writer = BufferedWriter(OutputStreamWriter(sock.getOutputStream()))
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    val parts = line!!.trim().split(" ")
                    when (parts[0].uppercase()) {
                        "SET" -> if (parts.size >= 4) {
                            val yaw = parts[1].toDoubleOrNull() ?: 0.0
                            val pitch = parts[2].toDoubleOrNull() ?: 0.0
                            val zoom = parts[3].toDoubleOrNull() ?: 1.0
                            setOrientationAndZoom(yaw, pitch, zoom)
                        }
                        "ZOOM" -> if (parts.size >= 2) {
                            val zoom = parts[1].toDoubleOrNull()
                            if (zoom != null) {
                                applyZoom(zoom)
                            }
                        }
                        "GET" -> {
                            val info = getLaserInfo()
                            val distance = info?.distance ?: -1.0
                            val loc = info?.location3D
                            val lat = loc?.latitude ?: 0.0
                            val lon = loc?.longitude ?: 0.0
                            val alt = loc?.altitude ?: 0.0
                            val point = info?.targetPoint
                            val tx = point?.x ?: 0.0
                            val ty = point?.y ?: 0.0
                            val attitude = getAircraftAttitude()
                            val yaw = attitude?.yaw ?: Double.NaN
                            val heading = if (yaw.isNaN()) Double.NaN else normalizeHeading(yaw)
                            writer.write("RANGE $distance LAT $lat LON $lon ALT $alt TX $tx TY $ty YAW $yaw HEADING $heading\n")
                            writer.flush()
                        }
                    }
                }
            }
        }
    }

    private fun ensureZoomLensSelected(attempt: Int = 0) {
        val keyManager = KeyManager.getInstance()
        val current = keyManager.getValue<CameraVideoStreamSourceType>(streamSourceKey)
        if (current == CameraVideoStreamSourceType.ZOOM_CAMERA) {
            if (attempt > 0) {
                Log.i(TAG, "Zoom lens selected after ${attempt + 1} attempts")
            }
            return
        }
        keyManager.setValue(streamSourceKey, CameraVideoStreamSourceType.ZOOM_CAMERA, null)
        if (attempt >= ZOOM_LENS_MAX_ATTEMPTS - 1) {
            Log.w(TAG, "Zoom lens still unavailable after $ZOOM_LENS_MAX_ATTEMPTS attempts (last=$current)")
            return
        }
        zoomHandler.postDelayed({ ensureZoomLensSelected(attempt + 1) }, ZOOM_LENS_RETRY_DELAY_MS)
    }

    private fun enableLaserModule() {
        val key = KeyTools.createCameraKey(
            CameraKey.KeyLaserMeasureEnabled,
            ComponentIndexType.LEFT_OR_MAIN,
            CameraLensType.CAMERA_LENS_ZOOM
        )
        KeyManager.getInstance().setValue(key, true, null)
    }

    private fun setOrientationAndZoom(yaw: Double, pitch: Double, zoom: Double) {
        val rotateKey = KeyTools.createKey(GimbalKey.KeyRotateByAngle, ComponentIndexType.LEFT_OR_MAIN)
        val rotation = GimbalAngleRotation().apply {
            mode = GimbalAngleRotationMode.ABSOLUTE_ANGLE
            this.yaw = yaw
            this.pitch = pitch
            duration = 2.0
        }
        KeyManager.getInstance().performAction(rotateKey, rotation, null)
        scheduleZoomUpdate(zoom)
    }

    private fun scheduleZoomUpdate(requestedZoom: Double) {
        val clamped = clampZoom(requestedZoom)
        zoomHandler.postDelayed({ setZoomInternal(clamped) }, 200)
    }

    private fun applyZoom(requestedZoom: Double) {
        val clamped = clampZoom(requestedZoom)
        zoomHandler.post { setZoomInternal(clamped) }
    }

    private fun setZoomInternal(value: Double) {
        ensureZoomLensSelected()
        KeyManager.getInstance().setValue(zoomKey, value, null)
    }

    private fun clampZoom(value: Double): Double {
        val range = zoomRange
        if (range != null) {
            val gears = range.gears
            if (gears != null && gears.isNotEmpty()) {
                val minGear = gears.minOrNull()
                val maxGear = gears.maxOrNull()
                if (minGear != null && maxGear != null && minGear <= maxGear) {
                    val bounded = value.coerceIn(minGear.toDouble(), maxGear.toDouble())
                    if (range.isContinuous) {
                        return bounded
                    }
                    var closest = gears[0]
                    var smallestDelta = abs(gears[0].toDouble() - bounded)
                    for (gear in gears) {
                        val delta = abs(gear.toDouble() - bounded)
                        if (delta < smallestDelta) {
                            smallestDelta = delta
                            closest = gear
                        }
                    }
                    return closest.toDouble()
                }
            }
        }
        return value.coerceIn(1.0, 200.0)
    }

    private fun getLaserInfo(): LaserMeasureInformation? {
        return laserInfo
    }

    private fun getAircraftAttitude(): Attitude? {
        return KeyManager.getInstance().getValue(attitudeKey)
    }

    private fun normalizeHeading(yaw: Double): Double {
        var heading = yaw % 360.0
        if (heading < 0) {
            heading += 360.0
        }
        return heading
    }

    @JvmStatic
    fun stop() {
        pollingThread?.interrupt()
        pollingThread = null
        try {
            server?.close()
        } catch (_: Exception) {
        } finally {
            server = null
        }
    }
}

