package dji.v5.ux.core.widget.laserrange

import dji.sdk.keyvalue.key.CameraKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.camera.LaserMeasureInformation
import dji.v5.ux.core.base.DJISDKModel
import dji.v5.ux.core.base.WidgetModel
import dji.v5.ux.core.communication.ObservableInMemoryKeyedStore
import dji.v5.ux.core.util.DataProcessor
import io.reactivex.rxjava3.core.Flowable

/**
 * Widget Model for the [LaserRangeWidget] used to access the camera laser range finder data.
 */
class LaserRangeWidgetModel(
    djiSdkModel: DJISDKModel,
    keyedStore: ObservableInMemoryKeyedStore
) : WidgetModel(djiSdkModel, keyedStore) {

    private val laserInfoProcessor = DataProcessor.create<LaserMeasureInformation?>(null)
    private val rangeStateProcessor = DataProcessor.create<RangeState>(RangeState.ProductDisconnected)

    /**
     * Range finder state containing the measured distance if available.
     */
    val rangeState: Flowable<RangeState>
        get() = rangeStateProcessor.toFlowable()

    override fun inSetup() {
        bindDataProcessor(
            KeyTools.createCameraKey(CameraKey.KeyLaserMeasureInformation),
            laserInfoProcessor
        )
    }

    override fun updateStates() {
        if (productConnectionProcessor.value) {
            val info = laserInfoProcessor.value
            if (info != null) {
                rangeStateProcessor.onNext(RangeState.CurrentRange(info.distance))
            } else {
                rangeStateProcessor.onNext(RangeState.RangeUnavailable)
            }
        } else {
            rangeStateProcessor.onNext(RangeState.ProductDisconnected)
        }
    }

    override fun inCleanup() {
        // Nothing to clean
    }

    /**
     * Class representing range finder state.
     */
    sealed class RangeState {
        /** Product is disconnected */
        object ProductDisconnected : RangeState()

        /** Range finder data unavailable */
        object RangeUnavailable : RangeState()

        /** Current range value from the laser sensor */
        data class CurrentRange(val distance: Double) : RangeState()
    }
}
