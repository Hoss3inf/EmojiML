package com.example.emojiml;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.metadata.MetadataExtractor;
import org.tensorflow.lite.task.text.nlclassifier.NLClassifier;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity {
    private static final String MODEL_PATH = "converted_model.tflite";
    private Interpreter tflite;
    private static final String TAG = "Interpreter";

    private static final int SENTENCE_LEN = 256; // The maximum length of an input sentence.
    // Simple delimiter to split words.
    private static final String SIMPLE_SPACE_OR_PUNCTUATION = " |\\,|\\.|\\!|\\?|\n";
    private static final String START = "<START>";
    private static final String PAD = "<PAD>";
    private static final String UNKNOWN = "<UNKNOWN>";
    private final Map<String, Integer> dic = new HashMap<>();
    private final List<String> labels = new ArrayList<>();
    /** Number of results to show in the UI. */
    private static final int MAX_RESULTS = 3;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ByteBuffer buffer = null;
        try {
            buffer = loadModelFile(this.getAssets(), MODEL_PATH);
        tflite = new Interpreter(buffer);
        Log.v("TAG", "TFLite model loaded.");
        MetadataExtractor metadataExtractor = new MetadataExtractor(buffer);

        // Extract and load the dictionary file.
//        InputStream dictionaryFile = metadataExtractor.getAssociatedFile("vocab.txt");
//        loadDictionaryFile(dictionaryFile);
//        Log.v(TAG, "Dictionary loaded.");
//
//         Extract and load the label file.
//        InputStream labelFile = metadataExtractor.getAssociatedFile("labels.txt");
//        loadLabelFile(labelFile);
//        Log.v(TAG, "Labels loaded.");

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
            throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }
    public synchronized List<Result> classify(String text) {
        // Pre-prosessing.
        int[][] input = tokenizeInputText(text);

        // Run inference.
        Log.v("TAG", "Classifying text with TF Lite...");
        float[][] output = new float[1][labels.size()];
        tflite.run(input, output);

        // Find the best classifications.
        PriorityQueue<Result> pq =
                new PriorityQueue<>(
                        MAX_RESULTS, (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence()));
        for (int i = 0; i < labels.size(); i++) {
            pq.add(new Result("" + i, labels.get(i), output[0][i]));
        }
        final ArrayList<Result> results = new ArrayList<>();
        while (!pq.isEmpty()) {
            results.add(pq.poll());
        }

        Collections.sort(results);
        // Return the probability of each class.
        return results;
    }


    int[][] tokenizeInputText(String text) {
        int[] tmp = new int[SENTENCE_LEN];
        List<String> array = Arrays.asList(text.split(SIMPLE_SPACE_OR_PUNCTUATION));

        int index = 0;
        // Prepend <START> if it is in vocabulary file.
        if (dic.containsKey(START)) {
            tmp[index++] = dic.get(START);
        }

        for (String word : array) {
            if (index >= SENTENCE_LEN) {
                break;
            }
            tmp[index++] = dic.containsKey(word) ? dic.get(word) : (int) dic.get(UNKNOWN);
        }
        // Padding and wrapping.
        Arrays.fill(tmp, index, SENTENCE_LEN - 1, (int) dic.get(PAD));
        int[][] ans = {tmp};
        return ans;
    }

    private void loadDictionaryFile(InputStream ins) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(ins));
        // Each line in the dictionary has two columns.
        // First column is a word, and the second is the index of this word.
        while (reader.ready()) {
            List<String> line = Arrays.asList(reader.readLine().split(" "));
            if (line.size() < 2) {
                continue;
            }
            dic.put(line.get(0), Integer.parseInt(line.get(1)));
        }}

        private void loadLabelFile(InputStream ins) throws IOException {
            BufferedReader reader = new BufferedReader(new InputStreamReader(ins));
            // Each line in the label file is a label.
            while (reader.ready()) {
                labels.add(reader.readLine());
            }
        }

    }