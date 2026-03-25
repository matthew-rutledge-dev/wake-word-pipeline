#!/usr/bin/env python3
"""
Train openWakeWord model from generated audio samples.

Pipeline: augment clips → compute embeddings → train DNN → save checkpoints.
GPU used for: embedding computation (melspec+Google model) and DNN training.

Usage: python 02_train_oww.py <word_id>
"""
import _compat  # noqa: F401 — must be first
import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
PIPER_DIR = Path(os.environ.get("PIPER_SAMPLE_GENERATOR_PATH",
                                 str(REPO_DIR / "piper-sample-generator")))
OWW_DIR = Path(os.environ.get("OWW_PATH", str(REPO_DIR / "openWakeWord")))


def detect_device():
    """Return 'gpu' if CUDA available, else 'cpu'. OWW uses 'gpu'/'cpu' not 'cuda'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info("GPU: %s (%.1f GB) — using CUDA for training", name, mem)
        return "gpu"
    log.warning("No CUDA GPU — training on CPU (slower)")
    return "cpu"


def load_config(word_id: str) -> dict:
    cfg_path = REPO_DIR / "words" / word_id / "config.yaml"
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_oww_training_config(cfg: dict, artifact_dir: Path) -> dict:
    """Build an openWakeWord-compatible training config dict."""
    word_id = cfg["word_id"]
    target_phrase = cfg["display_name"].lower()
    oww = cfg["oww"]
    samples = cfg["samples"]
    aug = cfg["augmentation"]

    return {
        "model_name": word_id,
        "target_phrase": [target_phrase],
        "custom_negative_phrases": cfg.get("custom_negative_phrases", []),
        "n_samples": samples["positive_train"],
        "n_samples_val": samples["positive_val"],
        "tts_batch_size": cfg["piper"].get("batch_size", 50),
        "augmentation_batch_size": aug["batch_size"],
        "piper_sample_generator_path": str(PIPER_DIR),
        "output_dir": str(artifact_dir),
        "rir_paths": [str(artifact_dir / "rir_data")],
        "background_paths": [str(artifact_dir / "background_data")],
        "background_paths_duplication_rate": [1],
        "false_positive_validation_data_path": str(
            artifact_dir / "validation_set_features.npy"
        ),
        "augmentation_rounds": aug["rounds"],
        "feature_data_files": {},
        "batch_n_per_class": {
            "adversarial_negative": 50,
            "positive": 50,
        },
        "model_type": oww["model_type"],
        "layer_dim": oww["layer_size"],
        "steps": oww["steps"],
        "max_negative_weight": oww["max_negative_weight"],
        "target_false_positives_per_hour": oww["target_fp_per_hour"],
        "total_length": 32000,
    }


def _resample_dir_to_16k(directory: Path):
    """Resample all WAV files in directory (recursively) to 16kHz mono."""
    import torchaudio
    import soundfile as sf

    wavs = list(directory.rglob("*.wav"))
    if not wavs:
        return
    data0, sr0 = torchaudio.load(str(wavs[0]))
    if sr0 == 16000:
        return
    log.info("Resampling %d files in %s from %dHz to 16kHz", len(wavs), directory.name, sr0)
    resampler = torchaudio.transforms.Resample(orig_freq=sr0, new_freq=16000)
    for wav_path in wavs:
        data, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            data = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(data)
        if data.shape[0] > 1:
            data = data.mean(dim=0, keepdim=True)
        sf.write(str(wav_path), data.squeeze().numpy(), 16000, subtype="PCM_16")


def download_data_assets(artifact_dir: Path):
    """Download background noise, RIR data, validation features, and OWW feature models if missing."""
    import subprocess

    # Download OWW feature extraction models (melspectrogram.onnx, embedding_model.onnx)
    from openwakeword.utils import download_models
    download_models(model_names=[])

    rir_dir = artifact_dir / "rir_data"
    bg_dir = artifact_dir / "background_data"
    val_path = artifact_dir / "validation_set_features.npy"

    # Download MIT RIR dataset if missing
    if not rir_dir.exists() or not list(rir_dir.glob("*.wav")):
        log.info("Downloading room impulse response data...")
        rir_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["wget", "-q", "-O", str(rir_dir / "mit_rirs.zip"),
                 "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"],
                check=True, timeout=300,
            )
            subprocess.run(
                ["unzip", "-q", "-o", str(rir_dir / "mit_rirs.zip"),
                 "-d", str(rir_dir)],
                check=True,
            )
            (rir_dir / "mit_rirs.zip").unlink(missing_ok=True)
            # Resample RIR files to 16kHz (MIT RIRs are 32kHz)
            _resample_dir_to_16k(rir_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("RIR download failed: %s — augmentation will skip RIR", e)

    # Create placeholder background dir if missing
    if not bg_dir.exists():
        bg_dir.mkdir(parents=True, exist_ok=True)
        log.warning(
            "No background audio found at %s. "
            "For best results, add WAV files of background noise, music, and speech. "
            "Consider: AudioSet, FMA, or ACAV100M segments.",
            bg_dir,
        )

    # Download validation features if missing
    if not val_path.exists():
        log.info("Downloading pre-computed OWW validation features (~200MB)...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="davidscripka/openwakeword_features",
                filename="openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                local_dir=str(artifact_dir),
                repo_type="dataset",
            )
            os.rename(downloaded, str(val_path))
        except Exception as e:
            log.warning("Validation features download failed: %s", e)
            log.info("Creating minimal placeholder validation features")
            np.save(str(val_path), np.zeros((100, 96), dtype=np.float32))


def augment_and_compute_features(oww_config: dict, device: str):
    """Augment generated clips and compute openWakeWord features."""
    from openwakeword.data import augment_clips
    from openwakeword.utils import AudioFeatures, compute_features_from_generator

    model_name = oww_config["model_name"]
    output_dir = Path(oww_config["output_dir"])
    feature_save_dir = output_dir / model_name
    feature_save_dir.mkdir(parents=True, exist_ok=True)

    # Gather paths
    rir_paths = []
    for rp in oww_config["rir_paths"]:
        if Path(rp).exists():
            rir_paths.extend([str(f) for f in Path(rp).rglob("*.wav")])

    background_paths = []
    for bp in oww_config["background_paths"]:
        if Path(bp).exists():
            background_paths.extend([str(f) for f in Path(bp).rglob("*.wav")])

    total_length = oww_config.get("total_length", 32000)
    aug_rounds = oww_config["augmentation_rounds"]
    aug_batch = oww_config["augmentation_batch_size"]

    # Process each split
    for split_name, dir_suffix in [
        ("positive_features_train", "positive_train"),
        ("positive_features_val", "positive_val"),
        ("negative_features_train", "negative_train"),
        ("negative_features_val", "negative_val"),
    ]:
        feature_path = feature_save_dir / f"{split_name}.npy"
        clip_dir = output_dir / dir_suffix

        if feature_path.exists():
            log.info("Features already exist: %s", feature_path)
            continue

        if not clip_dir.exists() or not list(clip_dir.glob("*.wav")):
            log.warning("No clips found in %s — skipping feature computation", clip_dir)
            continue

        clips = [str(f) for f in clip_dir.glob("*.wav")] * aug_rounds
        n_clips = len(clips)
        log.info("Augmenting + computing features: %s (%d clips, device=%s)",
                 split_name, n_clips, device)

        clip_generator = augment_clips(
            clips,
            total_length=total_length,
            batch_size=aug_batch,
            background_clip_paths=background_paths,
            RIR_paths=rir_paths,
        )

        compute_features_from_generator(
            generator=clip_generator,
            n_total=n_clips,
            clip_duration=total_length,
            output_file=str(feature_path),
            device=device,
        )
        log.info("Saved features: %s", feature_path)


def train_model(oww_config: dict, device: str, artifact_dir: Path):
    """Train the openWakeWord DNN model."""
    from openwakeword.train import Model as OWWTrainModel
    from openwakeword.data import mmap_batch_generator

    model_name = oww_config["model_name"]
    feature_dir = artifact_dir / model_name

    # Load feature files
    pos_train = feature_dir / "positive_features_train.npy"
    neg_train = feature_dir / "negative_features_train.npy"
    pos_val = feature_dir / "positive_features_val.npy"
    neg_val = feature_dir / "negative_features_val.npy"
    fp_val_path = oww_config["false_positive_validation_data_path"]

    for required in [pos_train, pos_val]:
        if not Path(required).exists():
            log.error("Required feature file missing: %s", required)
            sys.exit(1)

    # Build data files dict for mmap_batch_generator
    data_files = {"1": str(pos_train)}
    if neg_train.exists():
        data_files["0"] = str(neg_train)

    # Add any additional feature data files
    for label, path in oww_config.get("feature_data_files", {}).items():
        if Path(path).exists():
            data_files[label] = path

    batch_n = dict(oww_config.get("batch_n_per_class", {}))
    # Map string keys to match data_files
    if "positive" in batch_n:
        batch_n["1"] = batch_n.pop("positive")
    if "adversarial_negative" in batch_n:
        batch_n["0"] = batch_n.pop("adversarial_negative")

    log.info("Building training data generator...")
    X_train = mmap_batch_generator(
        data_files=data_files,
        n_per_class=batch_n if batch_n else {},
    )

    # Validation data
    val_data_files = {}
    if pos_val.exists():
        val_data_files["1"] = str(pos_val)
    if neg_val.exists():
        val_data_files["0"] = str(neg_val)

    # Validation data - must be FINITE (mmap_batch_generator is infinite)
    X_val = None
    if val_data_files:
        _val_arrays = []
        _val_labels = []
        for lbl, fpath in val_data_files.items():
            arr = np.array(np.load(fpath, mmap_mode="r"))
            _val_arrays.append(arr)
            _val_labels.append(np.full(arr.shape[0], float(lbl)))
        _x_val = np.concatenate(_val_arrays)
        _y_val = np.concatenate(_val_labels)
        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(_x_val).float(),
                torch.from_numpy(_y_val).float(),
            ),
            batch_size=len(_y_val),
        )
        log.info("Validation DataLoader: %d samples", len(_y_val))

    # False positive validation data - chunked to avoid OOM on large files
    X_val_fp = None
    if Path(fp_val_path).exists():
        class _ChunkedFPDataset(torch.utils.data.IterableDataset):
            def __init__(self, path, chunk_size=8192):
                self._data = np.load(path, mmap_mode="r")
                self._n = self._data.shape[0]
                self._chunk = chunk_size
            def __iter__(self):
                for start in range(0, self._n, self._chunk):
                    chunk = np.array(self._data[start:start+self._chunk])
                    x = torch.from_numpy(chunk).float()
                    y = torch.zeros(x.shape[0])
                    yield x, y
        fp_ds = _ChunkedFPDataset(fp_val_path, chunk_size=8192)
        X_val_fp = torch.utils.data.DataLoader(fp_ds, batch_size=None, num_workers=0)

    # Determine input shape from feature files
    sample_features = np.load(str(pos_train), mmap_mode="r")
    n_features = sample_features.shape[-1]
    n_timesteps = sample_features.shape[1] if sample_features.ndim == 3 else 1

    # Create model
    log.info("Creating OWW model (type=%s, layer_dim=%d, device=%s)",
             oww_config["model_type"], oww_config["layer_dim"], device)

    oww_model = OWWTrainModel(
        n_classes=1,
        input_shape=(n_timesteps, n_features),
        model_type=oww_config["model_type"],
        layer_dim=oww_config["layer_dim"],
    )

    # Train
    log.info("Starting auto_train (%d steps, target FP/hr=%.2f)...",
             oww_config["steps"], oww_config["target_false_positives_per_hour"])

    best_model = oww_model.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=oww_config["steps"],
        max_negative_weight=oww_config["max_negative_weight"],
        target_fp_per_hour=oww_config["target_false_positives_per_hour"],
    )

    # Export to ONNX
    oww_model.export_model(
        model=best_model,
        model_name=model_name,
        output_dir=str(artifact_dir / "oww"),
    )
    log.info("OWW model exported to %s", artifact_dir / "oww")

    return best_model


def main():
    parser = argparse.ArgumentParser(description="Train openWakeWord model")
    parser.add_argument("word_id", help="Wake word identifier")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip augmentation if features already computed")
    args = parser.parse_args()

    cfg = load_config(args.word_id)
    device = detect_device()
    artifact_dir = REPO_DIR / "artifacts" / cfg["word_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "oww").mkdir(parents=True, exist_ok=True)

    oww_config = build_oww_training_config(cfg, artifact_dir)
    mc = MetricsCollector(word_id=cfg["word_id"], phase="02_train")
    mc.start()

    # Download training data assets
    download_data_assets(artifact_dir)

    # Augment clips and compute features
    if not args.skip_augment:
        augment_and_compute_features(oww_config, device)
    else:
        log.info("Skipping augmentation (--skip-augment)")

    # Train model
    train_model(oww_config, device, artifact_dir)

    mc.stop()
    log.info("OWW training complete for '%s'", cfg["display_name"])


if __name__ == "__main__":
    main()
