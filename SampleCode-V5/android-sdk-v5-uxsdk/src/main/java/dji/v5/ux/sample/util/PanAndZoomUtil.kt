package dji.v5.ux.sample.util

import dji.sdk.keyvalue.key.CameraKey
import dji.sdk.keyvalue.key.GimbalKey
import dji.v5.manager.KeyManager
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.CameraLensType
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotation
import dji.sdk.keyvalue.value.gimbal.GimbalAngleRotationMode
import io.reactivex.rxjava3.core.Observable
import io.reactivex.rxjava3.disposables.CompositeDisposable
import io.reactivex.rxjava3.schedulers.Schedulers
import java.util.concurrent.TimeUnit

/**
 * Utility to slowly pan the gimbal and zoom the camera.
 */
object PanAndZoomUtil {
    private val disposables = CompositeDisposable()

    /** Start rotating the gimbal from -50° to +50° in 5° steps every 3 seconds.
     *  Each step increases the zoom ratio by 2. */
    @JvmStatic
    fun start() {
        val rotateKey = KeyTools.createKey(GimbalKey.KeyRotateByAngle, ComponentIndexType.LEFT_OR_MAIN)
        val zoomKey = KeyTools.createCameraKey(
            CameraKey.KeyCameraZoomRatios,
            ComponentIndexType.LEFT_OR_MAIN,
            CameraLensType.CAMERA_LENS_ZOOM
        )

        disposables.clear()
        var yaw = -50
        var zoom = 1.0

        disposables.add(
            Observable.interval(0, 3, TimeUnit.SECONDS)
                .takeWhile { yaw <= 50 }
                .observeOn(Schedulers.io())
                .subscribe {
                    val rotation = GimbalAngleRotation().apply {
                        mode = GimbalAngleRotationMode.ABSOLUTE_ANGLE
                        this.yaw = yaw.toDouble()
                        duration = 2.0
                    }
                    KeyManager.getInstance().performAction(rotateKey, rotation, null)
                    KeyManager.getInstance().setValue(zoomKey, zoom, null)
                    yaw += 5
                    zoom += 2.0
                }
        )
    }

    /** Stop the pan and zoom routine. */
    @JvmStatic
    fun stop() {
        disposables.clear()
    }
}
