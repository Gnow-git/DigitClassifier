package com.example.digitclassifier;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Pair;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class Classifier {

    Context context;

    public Classifier(Context context) {
        this.context = context;
    }

    private ByteBuffer loadModelFile(String modelName) throws IOException {
        AssetManager am = context.getAssets();  //AssetManager 획득, assets 폴더 저장 리소스 접근
        AssetFileDescriptor afd = am.openFd(modelName); //openFd 함수에 tflite 파일명 전달
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor()); //getFileDescriptor 함수로 File Descriptor 획득, 파일 읽기 쓰기 가능
        FileChannel fc = fis.getChannel(); //read 대신 성능을 위해 getChannel
        long startOffset = afd.getStartOffset(); // ByteBuffer 클래스를 상속한 객체 반환
        long declaredLength = afd.getDeclaredLength();

        return fc.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private static final String MODEL_NAME = "keras_model_cnn.tflite";

    Interpreter interpreter = null;

    public void init() throws IOException {
        ByteBuffer model = loadModelFile(MODEL_NAME);
        model.order(ByteOrder.nativeOrder());
        interpreter = new Interpreter(model);

        initModelShape();
    }

    int modelInputWidth, modelInputHeight, modelInputChannel;
    int modelOutputClasses;

    private void initModelShape() {
        Tensor inputTensor = interpreter.getInputTensor(0);
        int[] inputShape = inputTensor.shape();
        modelInputChannel = inputShape[0];
        modelInputWidth = inputShape[1];
        modelInputHeight = inputShape[2];

        Tensor outputTensor = interpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        modelOutputClasses = outputShape[1];
    }

    private  Bitmap resizeBitmap(Bitmap bitmap){
        return Bitmap.createScaledBitmap(bitmap, modelInputWidth, modelInputHeight,false);
    }

    private ByteBuffer convertBitmapToGrayByteBuffer(Bitmap bitmap) {
        ByteBuffer byteByffer = ByteBuffer.allocateDirect(bitmap.getByteCount());
        byteByffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[bitmap.getWidth() * bitmap.getHeight()];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(),
                bitmap.getHeight());

        for (int pixel : pixels) {
            int r = pixel >> 16 & 0xFF;
            int g = pixel >> 8 & 0xFF;
            int b = pixel & 0xFF;

            float avgPixelValue = (r + g + b) / 3.0f;
            float normalizedPixelValue = avgPixelValue / 255.0f;

            byteByffer.putFloat(normalizedPixelValue);
        }

        return byteByffer;
    }

    public Pair<Integer, Float> classify(Bitmap image){
        ByteBuffer buffer = convertBitmapToGrayByteBuffer(resizeBitmap(image));
        float[][] result = new float[1][modelOutputClasses];

        interpreter.run(buffer, result);

        return argmax(result[0]);
    }

    private Pair<Integer, Float> argmax(float[] array){
        int argmax = 0;
        float max = array[0];
        for(int i = 1; i < array.length; i++){
            float f = array[i];
            if(f > max){
                argmax = i;
                max = f;
            }
        }
        return new Pair<>(argmax, max);
    }

    public void finish() {
        if(interpreter != null)
            interpreter.close();
    }
}

