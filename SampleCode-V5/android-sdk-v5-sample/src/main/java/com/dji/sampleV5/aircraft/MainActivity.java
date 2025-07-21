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
        // For initial testing, simply fly to a set altitude using Python
        py.getModule("drone_control").callAttr("fly_to_altitude", 20.0);
    }
}
