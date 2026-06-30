import os
import json
import io
import logging
from pathlib import Path
from urllib.parse import urlparse

import azure.functions as func
from azure.storage.blob import BlobClient, ContentSettings

import numpy as np
import librosa
import joblib

app = func.FunctionApp()

# ----------------------------
# Model cache (load once)
# ----------------------------
clf = None
le = None

def init_model():
    global clf, le
    if clf is None or le is None:
        here = Path(__file__).parent
        clf = joblib.load(here / "emotion_svc.pkl")       # pipeline: StandardScaler + SVC
        le  = joblib.load(here / "label_encoder.pkl")     # LabelEncoder
        logging.info("Loaded model + label encoder.")


# ----------------------------
# Feature extraction (240 features)
# ----------------------------
def extract_mfcc_features_from_bytes(audio_bytes, sr=16000, n_mfcc=40, top_db=25):
    audio_buffer = io.BytesIO(audio_bytes)
    y, _ = librosa.load(audio_buffer, sr=sr, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # 1) Reject very short clips after trimming
    if len(y) < sr * 0.5:   # shorter than 0.5 seconds
        return None, "no_voice_detected"

    # 2) Reject extremely low-energy clips
    if np.mean(np.abs(y)) < 0.005:
        return None, "no_voice_detected"

    # 3) Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)

    feat = np.concatenate([
        m.mean(axis=1),  m.std(axis=1),
        d1.mean(axis=1), d1.std(axis=1),
        d2.mean(axis=1), d2.std(axis=1),
    ]).astype(np.float32)

    return feat, None


# ----------------------------
# 1) Event Grid trigger (main pipeline on Flex)
# Upload WAV -> Event Grid event -> this runs -> writes JSON to ser-results
# ----------------------------
@app.event_grid_trigger(arg_name="event")
def process_audio_eventgrid(event: func.EventGridEvent):
    logging.warning("=== EVENT GRID TRIGGER HIT ===")

    try:
        init_model()

        data = event.get_json()

        # Blob event should include 'url' like:
        # https://<account>.blob.core.windows.net/<container>/<blobname>
        blob_url = data.get("url")
        if not blob_url:
            raise ValueError(f"No 'url' found in event data: {data}")

        parsed = urlparse(blob_url)
        # parsed.path: /ser-audio/trigger_test_03.wav
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"Unexpected blob URL path format: {parsed.path}")

        container_name, blob_name = path_parts[0], path_parts[1]
        logging.warning(f"Container: {container_name}, Blob: {blob_name}")

        # Download blob bytes using the Function App's storage connection string
        conn_str = os.environ["AzureWebJobsStorage"]
        in_client = BlobClient.from_connection_string(
            conn_str=conn_str,
            container_name=container_name,
            blob_name=blob_name
        )
        audio_bytes = in_client.download_blob().readall()

        # Validate WAV header (RIFF)
        if len(audio_bytes) < 12 or audio_bytes[0:4] != b"RIFF":
            raise ValueError(f"Downloaded bytes are not WAV (missing RIFF). First bytes: {audio_bytes[:60]!r}")

        # Predict
        feat, status = extract_mfcc_features_from_bytes(audio_bytes)
     

        if status is not None:
            result = {
                "source_container": container_name,
                "source_blob": blob_name,
                "status": status
            }
        else:
            X = feat.reshape(1, -1)
            y_pred = clf.predict(X)
            emotion = le.inverse_transform(y_pred)[0]

            result = {
                "source_container": container_name,
                "source_blob": blob_name,
                "predicted_emotion": str(emotion)
            }

        # Write result JSON to ser-results/<original>.json
        out_client = BlobClient.from_connection_string(
            conn_str=conn_str,
            container_name="ser-results",
            blob_name=f"{blob_name}.json"
        )
        out_client.upload_blob(
            json.dumps(result),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
        )

        logging.warning(f"✅ Wrote result: ser-results/{blob_name}.json -> {emotion}")

    except Exception as e:
        logging.exception("❌ Error in process_audio_eventgrid")
        # Write an error blob too (helps debugging without log-hunting)
        try:
            conn_str = os.environ["AzureWebJobsStorage"]
            out_client = BlobClient.from_connection_string(
                conn_str=conn_str,
                container_name="ser-results",
                blob_name=f"ERROR_{event.id}.json"
            )
            out_client.upload_blob(
                json.dumps({"error": str(e)}),
                overwrite=True,
                content_settings=ContentSettings(content_type="application/json")
            )
        except Exception:
            pass
        raise


# ----------------------------
# 2) HTTP bridge for the Agent (small payload)
# Agent calls: GET /api/get_emotion?file=<wavfilename>
# Returns the JSON stored in ser-results/<wavfilename>.json
# ----------------------------
from azure.core.exceptions import ResourceNotFoundError
@app.route(route="get_emotion", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def get_emotion(req: func.HttpRequest) -> func.HttpResponse:
    try:
        file_name = req.params.get("file")
        if not file_name:
            return func.HttpResponse(
                json.dumps({"error": "Missing query param: file"}),
                status_code=400,
                mimetype="application/json"
            )

        # Clean up possible quotes / spaces from agent input
        file_name = file_name.strip().strip('"').strip("'")

        conn_str = os.environ["AzureWebJobsStorage"]
        result_blob_name = f"{file_name}.json"

        bc = BlobClient.from_connection_string(
            conn_str=conn_str,
            container_name="ser-results",
            blob_name=result_blob_name
        )

        payload = bc.download_blob().readall().decode("utf-8")
        return func.HttpResponse(payload, status_code=200, mimetype="application/json")


    except ResourceNotFoundError:
        return func.HttpResponse(
            json.dumps({"status": "processing"}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="predict_live_emotion", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def predict_live_emotion(req: func.HttpRequest) -> func.HttpResponse:
    try:
        init_model()
        # ✅ Read raw audio bytes from request
        audio_bytes = req.get_body()

        # ✅ Extract features (you already have this function!)
        feat, status = extract_mfcc_features_from_bytes(audio_bytes)

        if status is not None:
            return func.HttpResponse(
                json.dumps({"status": status}),
                mimetype="application/json",
                status_code=200
            )

        # ✅ Predict
        X = feat.reshape(1, -1)
        y_pred = clf.predict(X)
        emotion = le.inverse_transform(y_pred)[0]

        return func.HttpResponse(
            json.dumps({"predicted_emotion": str(emotion)}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )