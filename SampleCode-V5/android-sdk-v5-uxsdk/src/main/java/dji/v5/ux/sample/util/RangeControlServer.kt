package dji.v5.ux.sample.util

import dji.v5.manager.KeyManager
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.CameraKey
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.CameraLensType
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotation
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotationMode
import dji.sdk.keyvalue.value.camera.LaserMeasureInformation
import dji.sdk.keyvalue.value.camera.CameraVideoStreamSourceType
import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.ServerSocket
import java.net.Socket
import kotlin.concurrent.thread

/**
 * Simple TCP server to receive camera control commands and send range finder data.
 * Commands:
 *   SET <yaw> <pitch> <zoom>  - set gimbal orientation and zoom ratio
 *   GET                      - reply with RANGE <distance> LAT <lat> LON <lon>
 */
object RangeControlServer {
    private var server: ServerSocket? = null
    private var laserInfo: LaserMeasureInformation? = null
    private val laserInfoKey = KeyTools.createCameraKey(
        CameraKey.KeyLaserMeasureInformation,
        ComponentIndexType.LEFT_OR_MAIN,
        CameraLensType.CAMERA_LENS_ZOOM
    )
    private var pollingThread: Thread? = null

    @JvmStatic
    @JvmOverloads
    fun start(port: Int = 8989) {
        if (server != null) return
        server = ServerSocket(port)
        // switch the live stream to the zoom lens so zoom ratios are visible
        setZoomLens()
        // enable the laser range finder so distance values can be returned
        enableLaserModule()
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
                            writer.write("RANGE $distance LAT $lat LON $lon ALT $alt TX $tx TY $ty\n")
                            writer.flush()
                        }
                    }
                }
            }
        }
    }

    private fun setZoomLens() {
        val key = KeyTools.createCameraKey(
            CameraKey.KeyCameraVideoStreamSource,
            ComponentIndexType.LEFT_OR_MAIN,
            CameraLensType.CAMERA_LENS_ZOOM
        )
        KeyManager.getInstance().setValue(key, CameraVideoStreamSourceType.ZOOM_CAMERA, null)
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

        val zoomKey = KeyTools.createCameraKey(
            CameraKey.KeyCameraZoomRatios,
            ComponentIndexType.LEFT_OR_MAIN,
            CameraLensType.CAMERA_LENS_ZOOM
        )
        KeyManager.getInstance().setValue(zoomKey, zoom, null)
    }

    private fun getLaserInfo(): LaserMeasureInformation? {
        return laserInfo
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

