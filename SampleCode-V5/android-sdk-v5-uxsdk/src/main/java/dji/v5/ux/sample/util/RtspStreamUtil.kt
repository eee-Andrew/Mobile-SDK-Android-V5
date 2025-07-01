package dji.v5.ux.sample.util

import dji.v5.manager.datacenter.MediaDataCenter
import dji.v5.manager.datacenter.livestream.LiveStreamSettings
import dji.v5.manager.datacenter.livestream.LiveStreamType
import dji.v5.manager.datacenter.livestream.settings.RtspSettings

/**
 * Utility to start and stop the built-in RTSP server so the camera stream can
 * be viewed remotely.
 */
object RtspStreamUtil {
    private val streamManager = MediaDataCenter.getInstance().liveStreamManager
    private var isStreaming = false

    /**
     * Start the RTSP server using the credentials and port parsed from [url].
     */
    @JvmStatic
    fun start(url: String) {
        if (isStreaming) return
        val regex = Regex("rtsp://([^:]+):([^@]+)@[^:]+:(\\d+)/.*")
        val match = regex.find(url)
        val username = match?.groupValues?.get(1) ?: "user"
        val password = match?.groupValues?.get(2) ?: "password"
        val port = match?.groupValues?.get(3)?.toIntOrNull() ?: 8554

        val settings = RtspSettings.Builder()
            .setUserName(username)
            .setPassWord(password)
            .setPort(port)
            .build()

        val config = LiveStreamSettings.Builder()
            .setLiveStreamType(LiveStreamType.RTSP)
            .setRtspSettings(settings)
            .build()

        streamManager.liveStreamSettings = config
        streamManager.startStream(null)
        isStreaming = true
    }

    /** Stop the RTSP server if it is running. */
    @JvmStatic
    fun stop() {
        if (!isStreaming) return
        streamManager.stopStream(null)
        isStreaming = false
    }
}
