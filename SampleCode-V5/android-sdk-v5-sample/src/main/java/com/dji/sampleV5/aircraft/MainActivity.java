package com.dji.sampleV5.aircraft;

import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        Python py = Python.getInstance();
        py.getModule("drone_control").callAttr("go_to_waypoint",
                22.5362, 113.9454, 20.0);
    }
}
