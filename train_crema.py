import argparse
from email import parser
import time
import zipfile
from pathlib import Path
import numpy as np
import librosa
import warnings
import re
import shutil

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
np.random.seed(42)

# ---------- CREMA-D label parsing ----------
# Example filename: 1001_DFA_ANG_XX.wav
CREMA_RE = re.compile(r"^(?P<actor>\d{4})_[A-Z]{3}_(?P<emo>[A-Z]{3})_[A-Z]{2}$")

CREMA_EMO_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

def parse_crema_filename(path: Path):
    m = CREMA_RE.match(path.stem)
    if not m:
        raise ValueError(f"Not a CREMA-D filename: {path.name}")
    actor = int(m.group("actor"))
    emo_code = m.group("emo")
    if emo_code not in CREMA_EMO_MAP:
        raise ValueError(f"Unknown emotion code {emo_code} in file {path.name}")
    return CREMA_EMO_MAP[emo_code], actor

def parse_my_recording_filename(path: Path):
    name = path.stem.lower()

    # Original manual recordings
    if name.startswith("record_h"):
        return "happy", -1
    elif name.startswith("record_s"):
        return "sad", -1
    elif name.startswith("record_n"):
        return "neutral", -1
    elif name.startswith("record_a"):
        return "angry", -1
    elif name.startswith("record_f"):
        return "fearful", -1
    elif name.startswith("record_d"):
        return "disgust", -1

    # Browser/live recordings
    elif name.startswith("live_h"):
        return "happy", -1
    elif name.startswith("live_s"):
        return "sad", -1
    elif name.startswith("live_n"):
        return "neutral", -1
    elif name.startswith("live_a"):
        return "angry", -1
    elif name.startswith("live_f"):
        return "fearful", -1
    elif name.startswith("live_d"):
        return "disgust", -1

    # Extra speakers
    elif name.startswith("speaker2_h") or name.startswith("speaker3_h"):
        return "happy", -1
    elif name.startswith("speaker2_s") or name.startswith("speaker3_s"):
        return "sad", -1
    elif name.startswith("speaker2_n") or name.startswith("speaker3_n"):
        return "neutral", -1
    elif name.startswith("speaker2_a") or name.startswith("speaker3_a"):
        return "angry", -1
    elif name.startswith("speaker2_f") or name.startswith("speaker3_f"):
        return "fearful", -1
    elif name.startswith("speaker2_d") or name.startswith("speaker3_d"):
        return "disgust", -1

    else:
        raise ValueError(f"Could not infer label from personal recording filename: {path.name}")

# ---------- Feature extraction (240-D) ----------
def extract_mfcc_features(path: Path, sr=16000, n_mfcc=40, top_db=25):
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Reject very short audio after trimming
    if len(y) < sr * 0.5:
        raise ValueError(f"Audio too short after trimming: {path.name}")

    # Normalize amplitude to reduce loudness / spike bias
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

    return feat

def unzip_dataset(zip_path: Path, extract_dir: Path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir

def resolve_zip_path(p: Path) -> Path:
    """Azure inputs can come as a directory. If so, find the zip inside."""
    p = Path(p)
    if p.is_dir():
        zips = list(p.glob("*.zip"))
        if len(zips) != 1:
            raise FileNotFoundError(f"Expected exactly one .zip inside {p}, found: {zips}")
        return zips[0]
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_zip", type=str, default="", help="Path to crema_d_subset.zip (Azure input or local path)")
    parser.add_argument("--my_recordings", type=str, default="", help="Path to folder with your personal WAV recordings")
    parser.add_argument("--test_actors", type=str, default="", help="Comma-separated 4-digit actor IDs for test split. If empty, auto-picks one actor.")
    args = parser.parse_args()

    import json

    config_path = Path("config.json")
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    # If not provided via CLI, fall back to config.json
    if not args.data_zip:
        args.data_zip = config.get("data_zip", "")
    if not args.my_recordings:
        args.my_recordings = config.get("my_recordings", "")
    if not args.test_actors:
        args.test_actors = config.get("test_actors", "")

    data_zip = resolve_zip_path(Path(args.data_zip))
    assert data_zip.exists(), f"Zip not found at: {data_zip}"

    # Azure ML best practice: write outputs to ./outputs
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # 1) Unzip fresh each run (avoids mixing old extracted files)
    import tempfile
    extract_root = Path(tempfile.gettempdir()) / "ser_data_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    unzip_dataset(data_zip, extract_root)

    # 2) Find wav files
    wavs = sorted(extract_root.rglob("*.wav"))
    assert len(wavs) > 0, f"No .wav files found after unzip at: {extract_root}"

    print(f"Found WAV files: {len(wavs)}")
    print(f"Extract root: {extract_root}")
    print(f"Example file: {wavs[0].name}")

    # Optional personal recordings
    my_wavs = []
    if args.my_recordings.strip():
        my_recordings_dir = Path(args.my_recordings)
        if not my_recordings_dir.exists():
            raise FileNotFoundError(f"My recordings folder not found: {my_recordings_dir}")
        my_wavs = sorted(my_recordings_dir.rglob("*.wav"))
        print(f"Found personal WAV files: {len(my_wavs)}")
        if len(my_wavs) > 0:
            print(f"Example personal file: {my_wavs[0].name}")

    # 3) Feature extraction timing
    t0 = time.time()

    X, y, actors = [], [], []

    # CREMA-D subset files
    for p in wavs:
        label, actor = parse_crema_filename(p)
        feat = extract_mfcc_features(p)

        if feat.shape[0] != 240:
            raise ValueError(f"Unexpected feature size {feat.shape[0]} for file {p.name}. Expected 240.")

        X.append(feat)
        y.append(label)
        actors.append(actor)

    # Personal recordings
    for p in my_wavs:
        label, actor = parse_my_recording_filename(p)
        feat = extract_mfcc_features(p)

        if feat.shape[0] != 240:
            raise ValueError(f"Unexpected feature size {feat.shape[0]} for personal file {p.name}. Expected 240.")

        X.append(feat)
        y.append(label)
        actors.append(actor)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    actors = np.array(actors)

    feat_time = time.time() - t0

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Unique emotions:", np.unique(y))
    print("Unique actors (count):", len(np.unique(actors)))
    print("Unique actors (sample):", sorted(np.unique(actors).tolist())[:20])
    print(f"Feature extraction time (seconds): {feat_time:.2f}")

    # 4) Label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # 5) Speaker-independent split
    if args.test_actors.strip() == "":
        # auto-pick one actor as test to avoid empty test set
        unique_actors = sorted(np.unique(actors).tolist())
        test_actors = np.array([unique_actors[-1]])  # pick last one deterministically
        print(f"No --test_actors provided. Auto-picked test actor: {test_actors[0]}")
    else:
        test_actors = np.array([int(x.strip()) for x in args.test_actors.split(",") if x.strip() != ""])

    is_personal = actors == -1
    is_test = np.isin(actors, test_actors) & (~is_personal)
    is_train = ~is_test

    X_train, y_train = X[is_train], y_enc[is_train]
    X_test,  y_test  = X[is_test],  y_enc[is_test]

    print("Train samples:", X_train.shape[0], "| Test samples:", X_test.shape[0])
    print("Train actors (sample):", sorted(np.unique(actors[is_train]).tolist())[:20])
    print("Test actors :", sorted(np.unique(actors[is_test]).tolist()))

    if X_test.shape[0] == 0:
        raise ValueError(
            f"Test set is empty. test_actors={test_actors.tolist()} "
            f"Available actors sample: {sorted(np.unique(actors).tolist())[:30]}"
        )

    # 6) Train
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    )

    t2 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t2

    print(f"Training time (seconds): {train_time:.2f}")

    # 7) Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")

    labels_all = np.arange(len(le.classes_))  # 0..5 for 6 emotions
    print("Classification Report:\n")
    print(classification_report(
        y_test, y_pred,
        labels=labels_all,
        target_names=le.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=labels_all)
    print("Confusion Matrix (counts):\n", cm)


    # 8) Save artifacts
    joblib.dump(clf, outputs_dir / "emotion_svc.pkl")
    joblib.dump(le, outputs_dir / "label_encoder.pkl")

    with open(outputs_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Files: {len(wavs)}\n")
        f.write(f"Feature extraction time (s): {feat_time:.2f}\n")
        f.write(f"Training time (s): {train_time:.2f}\n")
        f.write(f"Test actors: {test_actors.tolist()}\n")
        f.write(f"Accuracy: {acc:.4f}\n")

    print("Saved artifacts to ./outputs")

    np.unique(y, return_counts=True)

if __name__ == "__main__":
    main()