"""
Config-hash based cache invalidation for the OWW training pipeline.

Stores a hash of config parameters that affect each cache layer.
When config changes, stale caches are automatically purged so the
pipeline regenerates with the new settings.

Cache layers:
  - samples:  positive_train/, negative_train/, positive_val/, negative_val/
  - features: <word>/{positive,negative}_features_{train,val}.npy
  - model:    oww/<word>.onnx, oww/<word>.tflite

Import and call check_and_invalidate_caches() before pipeline steps.
"""
import hashlib
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def _hash_dict(d: dict) -> str:
    """Deterministic SHA-256 of a JSON-serializable dict."""
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _sample_config_keys(cfg: dict) -> dict:
    """Extract config values that affect sample generation."""
    return {
        "custom_negative_phrases": sorted(cfg.get("custom_negative_phrases", [])),
        "positive_train": cfg["samples"]["positive_train"],
        "positive_val": cfg["samples"]["positive_val"],
        "negative_train": cfg["samples"]["negative_train"],
        "negative_val": cfg["samples"]["negative_val"],
        "piper_model": cfg["piper"]["model"],
        "noise_scales": cfg["piper"].get("noise_scales"),
        "noise_scale_ws": cfg["piper"].get("noise_scale_ws"),
        "length_scales": cfg["piper"].get("length_scales"),
        "espeak_voices": cfg.get("espeak", {}).get("voices"),
    }


def _feature_config_keys(cfg: dict) -> dict:
    """Extract config values that affect feature computation."""
    return {
        **_sample_config_keys(cfg),
        "augmentation_rounds": cfg["augmentation"]["rounds"],
        "augmentation_batch_size": cfg["augmentation"]["batch_size"],
    }


def _model_config_keys(cfg: dict) -> dict:
    """Extract config values that affect model training."""
    return {
        **_feature_config_keys(cfg),
        "model_type": cfg["oww"]["model_type"],
        "layer_size": cfg["oww"]["layer_size"],
        "steps": cfg["oww"]["steps"],
        "max_negative_weight": cfg["oww"]["max_negative_weight"],
        "target_fp_per_hour": cfg["oww"]["target_fp_per_hour"],
    }


def check_and_invalidate_caches(cfg: dict, artifact_dir: Path) -> dict:
    """Compare stored config hashes with current config.

    Returns a dict of which layers were invalidated:
      {"samples": bool, "features": bool, "model": bool}
    """
    word_id = cfg["word_id"]
    hash_file = artifact_dir / ".config_hashes.json"

    stored = {}
    if hash_file.exists():
        try:
            stored = json.loads(hash_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    current_sample_hash = _hash_dict(_sample_config_keys(cfg))
    current_feature_hash = _hash_dict(_feature_config_keys(cfg))
    current_model_hash = _hash_dict(_model_config_keys(cfg))

    invalidated = {"samples": False, "features": False, "model": False}

    # Check samples (cascades to features and model)
    if stored.get("samples") != current_sample_hash:
        log.info("[%s] Config change detected affecting SAMPLES — purging sample + feature + model caches", word_id)
        _purge_samples(artifact_dir)
        _purge_features(artifact_dir, word_id)
        _purge_model(artifact_dir, word_id)
        invalidated = {"samples": True, "features": True, "model": True}

    # Check features (cascades to model)
    elif stored.get("features") != current_feature_hash:
        log.info("[%s] Config change detected affecting FEATURES — purging feature + model caches", word_id)
        _purge_features(artifact_dir, word_id)
        _purge_model(artifact_dir, word_id)
        invalidated["features"] = True
        invalidated["model"] = True

    # Check model only
    elif stored.get("model") != current_model_hash:
        log.info("[%s] Config change detected affecting MODEL — purging model cache", word_id)
        _purge_model(artifact_dir, word_id)
        invalidated["model"] = True

    else:
        log.info("[%s] Config hashes match — caches are valid", word_id)

    # Save current hashes
    hash_file.write_text(json.dumps({
        "samples": current_sample_hash,
        "features": current_feature_hash,
        "model": current_model_hash,
    }, indent=2))

    return invalidated


def _purge_samples(artifact_dir: Path):
    """Remove generated WAV samples (negative only — positives are reusable if
    piper config unchanged, but we purge all for safety when sample config changes)."""
    for subdir in ["negative_train", "negative_val"]:
        d = artifact_dir / subdir
        if d.exists():
            n = len(list(d.glob("*.wav")))
            shutil.rmtree(d)
            log.info("  Purged %s (%d files)", subdir, n)


def _purge_features(artifact_dir: Path, word_id: str):
    """Remove precomputed .npy feature files."""
    feature_dir = artifact_dir / word_id
    if not feature_dir.exists():
        return
    for npy in feature_dir.glob("*_features_*.npy"):
        npy.unlink()
        log.info("  Purged %s", npy.name)


def _purge_model(artifact_dir: Path, word_id: str):
    """Remove trained model files (ONNX + TFLite)."""
    oww_dir = artifact_dir / "oww"
    if not oww_dir.exists():
        return
    for ext in ("*.onnx", "*.onnx.data", "*.tflite"):
        for f in oww_dir.glob(ext):
            f.unlink()
            log.info("  Purged %s", f.name)
