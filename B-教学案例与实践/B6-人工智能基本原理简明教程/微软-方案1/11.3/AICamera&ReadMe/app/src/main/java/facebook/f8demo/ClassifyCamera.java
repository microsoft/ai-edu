package facebook.f8demo;

import android.Manifest;
import android.app.ActionBar;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.text.method.ScrollingMovementMethod;
import android.util.Size;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;

public class ClassifyCamera extends AppCompatActivity {
    private static final String TAG = "F8DEMO";
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private static final int INPUT_SIZE = 28;

    private String predictedClass = "none";
    private AssetManager mgr;


    private Button btnDetectObject, btnToggleCamera;
    private ImageView image;
    private Bitmap mBitmap;
    private Canvas canvas;
    private Paint paint;
    private TextView result;
    private boolean run_HWC = false;



    static {
        System.loadLibrary("native-lib");
    }

    public native String classificationFromCaffe2(int h, int w, int[] D, boolean r_hwc);
    public native void initCaffe2(AssetManager mgr);
    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                initCaffe2(mgr);
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_classify_camera);
        mgr = getResources().getAssets();
        new SetUpNeuralNetwork().execute();
        image = (ImageView) findViewById(R.id.image);
        result = (TextView) findViewById(R.id.result);
        result.setMovementMethod(new ScrollingMovementMethod());

        btnToggleCamera = (Button) findViewById(R.id.clear);
        btnDetectObject = (Button) findViewById(R.id.detect);

        mBitmap = Bitmap.createBitmap(280, 280, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(mBitmap);
        canvas.drawColor(Color.BLACK);
        paint = new Paint();
        paint.setColor(Color.WHITE);
        paint.setStrokeWidth(35);
        canvas.drawBitmap(mBitmap, new Matrix(), paint);
        image.setImageBitmap(mBitmap);
        image.setOnTouchListener(new View.OnTouchListener() {
            int startX;
            int startY;
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        // 获取手按下时的坐标
                        startX = (int) motionEvent.getX();
                        startY = (int) motionEvent.getY();
                        break;
                    case MotionEvent.ACTION_MOVE:
                        // 获取手移动后的坐标
                        int endX = (int) motionEvent.getX();
                        int endY = (int) motionEvent.getY();
                        // 在开始和结束坐标间画一条线
                        canvas.drawLine(startX, startY, endX, endY, paint);
                        // 刷新开始坐标
                        startX = (int) motionEvent.getX();
                        startY = (int) motionEvent.getY();
                        image.setImageBitmap(mBitmap);
                        break;
                }
                return true;
            }
        });

        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                cameraView.toggleFacing();
                canvas.drawColor(Color.BLACK);
                image.setImageBitmap(mBitmap);
            }
        });

        btnDetectObject.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Bitmap bitmap = mBitmap;

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                int[] data = new int[28 * 28];    //通过位图的大小创建像素点数组
                bitmap.getPixels(data, 0, 28, 0, 0, 28, 28);
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        int grey = data[28 * i + j];

                        grey = (grey >> 8) & 0xFF;

                        data[28 * i + j] = grey;
                    }
                }
                int w = 28;
                int h = 28;
                predictedClass = classificationFromCaffe2(h, w, data, run_HWC);
                result.setText(predictedClass);
            }
        });

        }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }
}
