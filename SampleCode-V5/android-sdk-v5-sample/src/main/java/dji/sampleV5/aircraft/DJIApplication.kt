package dji.sampleV5.aircraft

import android.app.Application
import dji.sampleV5.aircraft.models.MSDKManagerVM
import dji.sampleV5.aircraft.models.globalViewModels
import dji.v5.ux.sample.util.PanAndZoomUtil
import dji.v5.ux.sample.util.RtspStreamUtil
import dji.v5.ux.sample.util.RangeControlServer

/**
 * Class Description
 *
 * @author Hoker
 * @date 2022/3/1
 *
 * Copyright (c) 2022, DJI All Rights Reserved.
 */
open class DJIApplication : Application() {

    private val msdkManagerVM: MSDKManagerVM by globalViewModels()

    override fun onCreate() {
        super.onCreate()

        // Ensure initialization is called first
        msdkManagerVM.initMobileSDK(this)

        // Start demo camera controls and streaming so they persist across activities
        PanAndZoomUtil.start()
        RtspStreamUtil.start("rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1")
        RangeControlServer.start()
    }

    override fun onTerminate() {
        RtspStreamUtil.stop()
        PanAndZoomUtil.stop()
        RangeControlServer.stop()
        super.onTerminate()
    }

}
