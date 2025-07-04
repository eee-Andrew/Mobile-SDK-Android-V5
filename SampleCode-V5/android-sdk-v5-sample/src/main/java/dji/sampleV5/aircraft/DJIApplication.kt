package dji.sampleV5.aircraft

import android.app.Application
import dji.sampleV5.aircraft.models.MSDKManagerVM
import dji.sampleV5.aircraft.models.globalViewModels
import dji.v5.ux.sample.util.RtspStreamUtil
import dji.v5.ux.sample.util.RangeControlServer
import androidx.lifecycle.Observer

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
    private val connectionObserver = Observer<Pair<Boolean, Int>> { resultPair ->
        if (resultPair.first) {
            RtspStreamUtil.start("rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1")
        } else {
            RtspStreamUtil.stop()
        }
    }

    override fun onCreate() {
        super.onCreate()

        // Ensure initialization is called first
        msdkManagerVM.initMobileSDK(this)

        // Start servers for remote control and video streaming
        msdkManagerVM.lvProductConnectionState.observeForever(connectionObserver)
        RangeControlServer.start()
    }

    override fun onTerminate() {
        msdkManagerVM.lvProductConnectionState.removeObserver(connectionObserver)
        RtspStreamUtil.stop()
        RangeControlServer.stop()
        super.onTerminate()
    }

}
