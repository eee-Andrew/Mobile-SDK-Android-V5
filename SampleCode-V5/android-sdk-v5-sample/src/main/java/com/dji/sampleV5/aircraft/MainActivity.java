package com.dji.sampleV5.aircraft;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.dji.wpmzsdk.common.data.Template;

import dji.sampleV5.aircraft.models.MissionGlobalModel;
import dji.sampleV5.aircraft.utils.KMZTestUtil;
import dji.sampleV5.aircraft.utils.wpml.WaypointInfoModel;
import dji.sdk.wpmz.value.mission.WaylineLocationCoordinate2D;
import dji.sdk.wpmz.value.mission.WaylineMission;
import dji.sdk.wpmz.value.mission.WaylineMissionConfig;
import dji.sdk.wpmz.value.mission.WaylineWaypoint;
import dji.v5.common.callback.CommonCallbacks;
import dji.v5.common.error.IDJIError;
import dji.v5.manager.aircraft.waypoint3.WPMZManager;
import dji.v5.manager.aircraft.waypoint3.WaypointMissionManager;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.d("MainActivity", "App started");
        startWaypointMission(37.4219999, -122.0840575, 30.0);
    }

    private void startWaypointMission(double latitude, double longitude, double altitude) {
        WPMZManager.getInstance().init(getApplicationContext());

        WaylineWaypoint waypoint = new WaylineWaypoint();
        waypoint.setWaypointIndex(0);
        waypoint.setLocation(new WaylineLocationCoordinate2D(latitude, longitude));
        waypoint.setHeight(altitude);
        waypoint.setEllipsoidHeight(altitude);

        WaypointInfoModel infoModel = new WaypointInfoModel();
        infoModel.setWaylineWaypoint(waypoint);
        infoModel.setActionInfos(new ArrayList<>());

        List<WaypointInfoModel> waypointList = new ArrayList<>();
        waypointList.add(infoModel);

        WaylineMission mission = KMZTestUtil.createWaylineMission();
        MissionGlobalModel globalModel = new MissionGlobalModel();
        WaylineMissionConfig config = KMZTestUtil.createMissionConfig(globalModel);
        Template template = KMZTestUtil.createTemplate(waypointList);

        String kmzPath = new File(getExternalFilesDir(null), "simple_waypoint.kmz").getAbsolutePath();
        WPMZManager.getInstance().generateKMZFile(kmzPath, mission, config, template);

        WaypointMissionManager.getInstance().pushKMZFileToAircraft(kmzPath,
            new CommonCallbacks.CompletionCallbackWithProgress<Double>() {
                @Override
                public void onProgressUpdate(Double progress) {
                    Log.d("MainActivity", "Upload: " + progress);
                }

                @Override
                public void onSuccess() {
                    Log.d("MainActivity", "Mission uploaded");
                    startMission(kmzPath);
                }

                @Override
                public void onFailure(IDJIError error) {
                    Log.e("MainActivity", "Upload failed: " + error.description());
                }
            });
    }

    private void startMission(String kmzPath) {
        String missionName = new File(kmzPath).getName();
        List<Integer> waylines = WaypointMissionManager.getInstance().getAvailableWaylineIDs(kmzPath);
        WaypointMissionManager.getInstance().startMission(missionName, waylines,
            new CommonCallbacks.CompletionCallback() {
                @Override
                public void onSuccess() {
                    Log.d("MainActivity", "Mission started");
                }

                @Override
                public void onFailure(IDJIError error) {
                    Log.e("MainActivity", "Start failed: " + error.description());
                }
            });
    }
}
