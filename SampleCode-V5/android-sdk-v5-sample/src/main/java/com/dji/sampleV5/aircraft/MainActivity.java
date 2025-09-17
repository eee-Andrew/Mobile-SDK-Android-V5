package com.dji.sampleV5.aircraft;

import android.os.Bundle;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import dji.sdk.keyvalue.key.FlightControllerKey;
import dji.sdk.keyvalue.key.KeyTools;
import dji.sdk.keyvalue.value.common.EmptyMsg;
import dji.sdk.keyvalue.value.common.LocationCoordinate3D;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.manager.KeyManager;
import dji.v5.manager.intelligent.IntelligentFlightManager;
import dji.v5.manager.intelligent.flyto.FlyToTarget;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d("MainActivity", "App started");
        // Fly to the requested location when the app launches
        startFlyToMission(41.1380417, 24.9127228, 10.0);
    }

    private void startFlyToMission(double latitude, double longitude, double altitude) {
        FlyToTarget target = new FlyToTarget();
        // Allow manual override with the sticks if something goes wrong.
        // Older SDKs may not implement setExitOnRCInput so call it via
        // reflection and ignore failures.
        try {
            java.lang.reflect.Method m =
                target.getClass().getMethod("setExitOnRCInput", boolean.class);
            m.invoke(target, true);
        } catch (Exception e) {
            Log.w("MainActivity", "setExitOnRCInput not supported: " + e.getMessage());
        }
        LocationCoordinate3D location = new LocationCoordinate3D(latitude, longitude, altitude);
        target.setTargetLocation(location);
        target.setMaxSpeed(10);
        target.setSecurityTakeoffHeight(20);

        IntelligentFlightManager.getInstance().getFlyToMissionManager().startMission(target, null,
            new CommonCallbacks.CompletionCallback() {
                @Override
                public void onSuccess() {
                    Log.d("MainActivity", "Fly-To started");
                }

                @Override
                public void onFailure(IDJIError error) {
                    Log.e("MainActivity", "Fly-To failed: " + error.description());
                }
            });
    }

    /** Call this after completing gimbal operations to return to home */
    private void returnHome() {
        KeyManager.getInstance().performAction(
            KeyTools.createKey(FlightControllerKey.KeyStartGoHome),
            new CommonCallbacks.CompletionCallbackWithParam<EmptyMsg>() {
                @Override
                public void onSuccess(EmptyMsg emptyMsg) {
                    Log.d("MainActivity", "RTH started");
                }

                @Override
                public void onFailure(@NonNull IDJIError error) {
                    Log.e("MainActivity", "RTH failed: " + error.description());
                }
            });
    }
}
