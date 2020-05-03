package com.example.digitreader;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import views.DrawModel;
import views.DrawView;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, View.OnTouchListener {

    private static final int PIXEL_WIDTH = 28;

    // ui elements
    private Button clearBtn, classBtn;
    private TextView resText;

    // views
    private DrawModel drawModel;
    private DrawView drawView;
    private PointF mTmpPiont = new PointF();

    private float mLastX;
    private float mLastY;

    private Interpreter tfliteInterpreter;


    @Override
    // In the onCreate() method, you perform basic application startup logic that should happen
    //only once for the entire life of the activity.
    protected void onCreate(Bundle savedInstanceState) {
        //initialization
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //get drawing view from XML (where the finger writes the number)
        drawView = findViewById(R.id.draw);

        //get the model object
        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        //init the view with the model object
        drawView.setModel(drawModel);

        // give it a touch listener to activate when the user taps
        drawView.setOnTouchListener(this);

        //clear button
        //clear the drawing when the user taps
        clearBtn = findViewById(R.id.btn_clear);
        clearBtn.setOnClickListener(this);

        //class button
        //when tapped, this performs classification on the drawn image
        classBtn = findViewById(R.id.btn_class);
        classBtn.setOnClickListener(this);

        // res text
        //this is the text that shows the output of the classification
        resText = findViewById(R.id.tfRes);

        // tensorflowlite
        //load up our saved model to perform inference from local storage
        loadModel();

    }

    //the activity lifecycle

    @Override
    //OnResume() is called when the user resumes his Activity which he left a while ago,
    // //say he presses home button and then comes back to app, onResume() is called.
    protected void onResume() {
        drawView.onResume();
        super.onResume();
    }

    @Override
    //OnPause() is called when the user receives an event like a call or a text message,
    // //when onPause() is called the Activity may be partially or completely hidden.
    protected void onPause() {
        drawView.onPause();
        super.onPause();
    }

    private void loadModel() {

        try{
            tfliteInterpreter = new Interpreter(loadModelFile());
            Log.e("xlr8","Model Loaded");
        } catch(Exception ex){
            ex.printStackTrace();
            Log.e("xlr8","Model Not Loaded: "+ex.getMessage());

        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        // Open the model using an input stream, and memory map it to load
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mnist_cnn.tflite");
        FileInputStream inputStream = new FileInputStream((fileDescriptor.getFileDescriptor()));
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onClick(View view) {
        //when the user clicks something
        if (view.getId() == R.id.btn_clear) {
            //if its the clear button
            //clear the drawing
            drawModel.clear();
            drawView.reset();
            drawView.invalidate();
            //empty the text view
            resText.setText("");
        } else if (view.getId() == R.id.btn_class) {
            //Toast.makeText(this, "Not Implemented yet", Toast.LENGTH_SHORT).show();

            float pixels[] = drawView.getPixelData();
            float inputValue[][][][] = new float[1][28][28][1];

            int i=0;
            for(int r=0; r<28; r++){
                for(int c=0; c<28; c++){
                    inputValue[0][r][c][0] = pixels[i];
                    i++;
                }
            }

            // Output shape is [1][1]
            float[][] outputValue = new float[1][10];

            // Run inference passing the input shape and getting the output shape
            tfliteInterpreter.run(inputValue,outputValue);

            String result = "";
            int predictionNo = -1;
            float predictionValue = -1.0f;
            int indx= 0;

            for(float p : outputValue[0]){
                result += p+"   ";
                if(p > predictionValue){
                    predictionValue = p;
                    predictionNo = indx;
                }
                indx++;
            }

            resText.setText("Predicted No. : "+predictionNo+"\nPrediction Probability: "+predictionValue);
        }

    }

    @Override
    public boolean onTouch(View view, MotionEvent event) {
        //get the action and store it as an int
        int action = event.getAction() & MotionEvent.ACTION_MASK;
        //actions have predefined ints, lets match
        //to detect, if the user has touched, which direction the users finger is
        //moving, and if they've stopped moving

        //if touched
        if (action == MotionEvent.ACTION_DOWN) {
            //begin drawing line
            processTouchDown(event);
            return true;
            //draw line in every direction the user moves
        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;
            //if finger is lifted, stop drawing
        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    //draw line down

    private void processTouchDown(MotionEvent event) {
        //calculate the x, y coordinates where the user has touched
        mLastX = event.getX();
        mLastY = event.getY();
        //user them to calcualte the position
        drawView.calcPos(mLastX, mLastY, mTmpPiont);
        //store them in memory to draw a line between the
        //difference in positions
        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        //and begin the line drawing
        drawModel.startLine(lastConvX, lastConvY);
    }

    //the main drawing function
    //it actually stores all the drawing positions
    //into the drawmodel object
    //we actually render the drawing from that object
    //in the drawrenderer class
    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        drawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        drawModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        drawView.invalidate();
    }

    private void processTouchUp() {
        drawModel.endLine();
    }
}
