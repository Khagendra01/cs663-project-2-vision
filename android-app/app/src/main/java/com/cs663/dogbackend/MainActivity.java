package com.cs663.dogbackend;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

// CS663 lab app. Take or pick a dog photo, type a prompt, send the same
// photo and prompt to both backend endpoints, show both answers.
public class MainActivity extends Activity {

    // Backend URL. The text field on screen starts with this value and
    // can be edited, so a new deploy does not need a rebuild.
    public static final String BACKEND_URL = "https://YOUR-CLOUD-RUN-URL.run.app";

    private static final int REQ_CAMERA = 1;
    private static final int REQ_PICK = 2;

    private final ExecutorService pool = Executors.newFixedThreadPool(2);

    private EditText etUrl;
    private EditText etPrompt;
    private ImageView imgPreview;
    private TextView tvStatus;
    private TextView tvBase;
    private TextView tvTuned;
    private Button btnAsk;

    private Bitmap photo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etUrl = findViewById(R.id.etUrl);
        etPrompt = findViewById(R.id.etPrompt);
        imgPreview = findViewById(R.id.imgPreview);
        tvStatus = findViewById(R.id.tvStatus);
        tvBase = findViewById(R.id.tvBase);
        tvTuned = findViewById(R.id.tvTuned);
        btnAsk = findViewById(R.id.btnAsk);
        Button btnCamera = findViewById(R.id.btnCamera);
        Button btnPick = findViewById(R.id.btnPick);

        etUrl.setText(BACKEND_URL);
        etPrompt.setText("Look at the image of the dog. What is the dog doing? "
                + "Answer with exactly one of: sitting, standing, lying down, walking.");

        btnCamera.setOnClickListener(v -> {
            Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(i, REQ_CAMERA);
        });

        btnPick.setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_GET_CONTENT);
            i.setType("image/*");
            startActivityForResult(Intent.createChooser(i, "Pick a picture"), REQ_PICK);
        });

        btnAsk.setOnClickListener(v -> askBoth());
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null) return;
        try {
            if (requestCode == REQ_CAMERA && data.getExtras() != null) {
                photo = (Bitmap) data.getExtras().get("data");
            } else if (requestCode == REQ_PICK) {
                Uri uri = data.getData();
                InputStream in = getContentResolver().openInputStream(uri);
                photo = BitmapFactory.decodeStream(in);
                if (in != null) in.close();
            }
            if (photo != null) {
                imgPreview.setImageBitmap(photo);
                tvStatus.setText("Photo ready. Tap Ask both models.");
            }
        } catch (Exception e) {
            tvStatus.setText("Could not load that picture: " + e.getMessage());
        }
    }

    private void askBoth() {
        if (photo == null) {
            tvStatus.setText("Take or pick a picture first.");
            return;
        }
        String base = etUrl.getText().toString().trim();
        if (base.isEmpty() || base.contains("YOUR-CLOUD-RUN-URL")) {
            tvStatus.setText("Set the real backend URL first.");
            return;
        }
        if (base.endsWith("/")) base = base.substring(0, base.length() - 1);
        String prompt = etPrompt.getText().toString().trim();
        if (prompt.isEmpty()) {
            tvStatus.setText("Type a prompt first.");
            return;
        }

        btnAsk.setEnabled(false);
        tvStatus.setText("Calling both models...");
        tvBase.setText("Calling...");
        tvTuned.setText("Calling...");

        byte[] jpeg = toJpeg(photo);
        String imageBase64 = Base64.encodeToString(jpeg, Base64.NO_WRAP);
        String url = base;

        pool.execute(() -> {
            String r = callModel(url + "/api/gemini", prompt, imageBase64);
            runOnUiThread(() -> tvBase.setText(r));
        });
        pool.execute(() -> {
            String r = callModel(url + "/api/fine-tuned-gemini", prompt, imageBase64);
            runOnUiThread(() -> {
                tvTuned.setText(r);
                tvStatus.setText("Done. Compare the two results above and below.");
                btnAsk.setEnabled(true);
            });
        });
    }

    // POST {prompt, imageBase64, mimeType} and return the "text" field.
    private String callModel(String endpoint, String prompt, String imageBase64) {
        HttpURLConnection conn = null;
        try {
            JSONObject payload = new JSONObject();
            payload.put("prompt", prompt);
            payload.put("mimeType", "image/jpeg");
            payload.put("imageBase64", imageBase64);

            conn = (HttpURLConnection) new URL(endpoint).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(60000);
            conn.setReadTimeout(60000);
            conn.setDoOutput(true);
            OutputStream out = conn.getOutputStream();
            out.write(payload.toString().getBytes(StandardCharsets.UTF_8));
            out.close();

            int code = conn.getResponseCode();
            InputStream in = code >= 200 && code < 300
                    ? conn.getInputStream() : conn.getErrorStream();
            String raw = readAll(in);
            JSONObject json = new JSONObject(raw);
            if (code < 200 || code >= 300) {
                return "Error " + code + ": " + json.optString("error", raw);
            }
            return json.optString("text", raw);
        } catch (Exception e) {
            return "Request failed: " + e.getMessage();
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private String readAll(InputStream in) throws Exception {
        if (in == null) return "{}";
        BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = r.readLine()) != null) sb.append(line);
        r.close();
        return sb.toString();
    }

    // Shrink big camera photos so the JSON body stays small.
    private byte[] toJpeg(Bitmap bmp) {
        int w = bmp.getWidth();
        int h = bmp.getHeight();
        int max = Math.max(w, h);
        Bitmap use = bmp;
        if (max > 1024) {
            float scale = 1024f / max;
            use = Bitmap.createScaledBitmap(bmp, Math.round(w * scale), Math.round(h * scale), true);
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        use.compress(Bitmap.CompressFormat.JPEG, 80, baos);
        return baos.toByteArray();
    }
}
