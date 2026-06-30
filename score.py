from pathlib import Path
import numpy as np
import librosa
import joblib
import base64
import io

# ---------- Feature extraction (240-D) ----------
def extract_mfcc_features_from_bytes(audio_bytes, sr=16000, n_mfcc=40, top_db=25):
    audio_buffer = io.BytesIO(audio_bytes) # Converts bytes (audio) into a “file-like object”
    y, _ = librosa.load(audio_buffer, sr=sr, mono=True) # y becomes the waveform (array of audio samples)
    y, _ = librosa.effects.trim(y, top_db=top_db) # Removes quite parts of the audio

    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)

    feat = np.concatenate([
        m.mean(axis=1),  m.std(axis=1),
        d1.mean(axis=1), d1.std(axis=1),
        d2.mean(axis=1), d2.std(axis=1),
    ]).astype(np.float32)

    return feat


def load_artifacts():
    here = Path(__file__).parent
    clf = joblib.load(here / "emotion_svc.pkl")     # pipeline
    le  = joblib.load(here / "label_encoder.pkl")
    return clf, le


# ---------- Azure ML hooks ----------
def init():
    global clf, le
    clf, le = load_artifacts()


def run(data):
    """
    Expected input:
    {
        "audio_base64": "..."
    }
    """
    try:
        audio_base64 = data["audio_base64"]
        audio_bytes = base64.b64decode(audio_base64)

        feat = extract_mfcc_features_from_bytes(audio_bytes)
        if feat.shape[0] != 240:
            return {"error": f"Expected 240 features, got {feat.shape[0]}"}

        X = feat.reshape(1, -1)
        y_pred = clf.predict(X)
        emotion = le.inverse_transform(y_pred)[0]

        return {"emotion": str(emotion)}

    except Exception as e:
        return {"error": str(e)}