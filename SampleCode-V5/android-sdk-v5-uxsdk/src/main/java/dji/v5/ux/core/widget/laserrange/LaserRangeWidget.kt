package dji.v5.ux.core.widget.laserrange

import android.content.Context
import android.util.AttributeSet
import io.reactivex.rxjava3.core.Flowable
import dji.v5.ux.R
import dji.v5.ux.core.base.DJISDKModel
import dji.v5.ux.core.base.SchedulerProvider
import dji.v5.ux.core.base.WidgetSizeDescription
import dji.v5.ux.core.base.widget.BaseTelemetryWidget
import dji.v5.ux.core.communication.ObservableInMemoryKeyedStore
import dji.v5.ux.core.extension.getDistanceString
import dji.v5.ux.core.extension.getString
import dji.v5.ux.core.util.UnitConversionUtil
import dji.v5.ux.core.widget.laserrange.LaserRangeWidget.ModelState
import dji.v5.ux.core.widget.laserrange.LaserRangeWidgetModel.RangeState
import java.text.DecimalFormat

/**
 * Widget that displays the distance measured by the camera laser range finder.
 */
open class LaserRangeWidget @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
    widgetTheme: Int = 0
) : BaseTelemetryWidget<ModelState>(
    context,
    attrs,
    defStyleAttr,
    WidgetType.TEXT,
    widgetTheme,
    R.style.UXSDKLaserRangeWidget
) {

    override val metricDecimalFormat: DecimalFormat = DecimalFormat("###0.0")
    override val imperialDecimalFormat: DecimalFormat = DecimalFormat("###0")

    private val widgetModel: LaserRangeWidgetModel by lazy {
        LaserRangeWidgetModel(
            DJISDKModel.getInstance(),
            ObservableInMemoryKeyedStore.getInstance()
        )
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        if (!isInEditMode) {
            widgetModel.setup()
        }
    }

    override fun onDetachedFromWindow() {
        if (!isInEditMode) {
            widgetModel.cleanup()
        }
        super.onDetachedFromWindow()
    }

    override fun reactToModelChanges() {
        addReaction(widgetModel.productConnection
            .observeOn(SchedulerProvider.ui())
            .subscribe { widgetStateDataProcessor.onNext(ModelState.ProductConnected(it)) })
        addReaction(widgetModel.rangeState
            .observeOn(SchedulerProvider.ui())
            .subscribe { updateUI(it) })
    }

    private fun updateUI(state: RangeState) {
        widgetStateDataProcessor.onNext(ModelState.RangeStateUpdated(state))
        when (state) {
            RangeState.ProductDisconnected, RangeState.RangeUnavailable -> {
                valueString = getString(R.string.uxsdk_string_default_value)
                unitString = null
            }
            is RangeState.CurrentRange -> {
                valueString = getDecimalFormat(UnitConversionUtil.UnitType.METRIC).format(state.distance)
                unitString = getDistanceString(UnitConversionUtil.UnitType.METRIC)
            }
        }
    }

    override fun getIdealDimensionRatioString(): String? = null

    override val widgetSizeDescription: WidgetSizeDescription =
        WidgetSizeDescription(WidgetSizeDescription.SizeType.OTHER,
            widthDimension = WidgetSizeDescription.Dimension.EXPAND,
            heightDimension = WidgetSizeDescription.Dimension.WRAP)

    @SuppressWarnings
    override fun getWidgetStateUpdate(): Flowable<ModelState> {
        return super.getWidgetStateUpdate()
    }

    /** Widget state updates for [LaserRangeWidget] */
    sealed class ModelState {
        data class ProductConnected(val boolean: Boolean) : ModelState()
        data class RangeStateUpdated(val rangeState: RangeState) : ModelState()
    }
}
