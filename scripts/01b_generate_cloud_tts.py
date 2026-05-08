#!/usr/bin/env python3
"""
Generate positive WAV samples using cloud TTS APIs (xAI, OpenAI, Google).

Adds voice diversity to the Piper-only positive sample pool. Outputs 16kHz
16-bit mono WAV files into the existing positive_train/ directory alongside
Piper samples.

API keys are loaded from environment variables:
  XAI_API_KEY     - xAI TTS (dev key preferred)
  OPENAI_API_KEY  - OpenAI TTS (dev key preferred)
  GOOGLE_API_KEY  - Google Cloud TTS (prod, no dev key)

Or from a .env file at /opt/ai/wakeword-train/.env

Usage:
  python 01b_generate_cloud_tts.py hey_ara
  python 01b_generate_cloud_tts.py hey_ara --providers xai openai
  python 01b_generate_cloud_tts.py --words hey_ara ok_anya hey_knight
  python 01b_generate_cloud_tts.py --words all --providers xai
"""
import argparse
import base64
import io
import json
import logging
import os
import struct
import sys
import time
import uuid
import wave
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
TARGET_SR = 16000
ENV_FILE = Path("/opt/ai/wakeword-train/.env")

MAX_RETRIES = 3  # Max retries per sample on rate limit


def load_env():
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and val:
                    os.environ[key] = val  # .env overrides shell env


def get_api_key(provider):
    env_map = {"xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
    return os.environ.get(env_map.get(provider, ""))


# --- xAI TTS ---
XAI_VOICES = [
    "ara", "eve", "leo", "rex", "sal",
    "79f3a8b96d43", "78a495fdbb39", "96819d0bd28d", "8a8b3d7dc1e8",
    "6a41d324", "f15c6a6a", "d11249e6", "a7b78b05", "3b312632",
    "bedd6226", "93bea908", "355dca53", "01a7edae",
    "4c7f16ff", "57700f39", "524f4cb1", "3a7889066fa2", "1ebfec36",
    "43423dee",
]


def generate_xai_sample(text, voice_id, api_key):
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.x.ai/v1/tts",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"text": text, "voice_id": voice_id, "language": "en",
                      "output_format": {"codec": "wav", "sample_rate": TARGET_SR}},
                timeout=30,
            )
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    log.warning("xAI rate limited (attempt %d/%d) - sleeping %ds",
                                attempt + 1, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                log.warning("xAI rate limited after %d retries - skipping", MAX_RETRIES)
                return None
            if resp.status_code != 200:
                log.warning("xAI error %d voice=%s: %s", resp.status_code, voice_id, resp.text[:200])
                return None
            return resp.content
        except requests.RequestException as e:
            log.warning("xAI request failed: %s", e)
            return None
    return None


# --- OpenAI TTS ---
OPENAI_VOICES = ["alloy", "ash", "coral", "echo", "fable",
                  "nova", "onyx", "sage", "shimmer"]


def resample_wav_24k_to_16k(wav_bytes):
    try:
        with io.BytesIO(wav_bytes) as inp:
            with wave.open(inp, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                src_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
        if src_rate == TARGET_SR:
            return wav_bytes
        if sampwidth != 2:
            return _resample_ffmpeg(wav_bytes)
        fmt = f"<{n_frames * n_channels}h"
        samples = list(struct.unpack(fmt, raw))
        if n_channels > 1:
            mono = []
            for i in range(0, len(samples), n_channels):
                mono.append(sum(samples[i:i + n_channels]) // n_channels)
            samples = mono
        ratio = TARGET_SR / src_rate
        new_len = int(len(samples) * ratio)
        resampled = []
        for i in range(new_len):
            src_pos = i / ratio
            idx = int(src_pos)
            frac = src_pos - idx
            if idx + 1 < len(samples):
                val = samples[idx] * (1 - frac) + samples[idx + 1] * frac
            else:
                val = samples[idx] if idx < len(samples) else 0
            resampled.append(int(val))
        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(struct.pack(f"<{len(resampled)}h", *resampled))
        return out.getvalue()
    except Exception as e:
        log.warning("WAV resample failed: %s - trying ffmpeg", e)
        return _resample_ffmpeg(wav_bytes)


def _resample_ffmpeg(wav_bytes):
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(wav_bytes)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path.replace(".wav", "_16k.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", tmp_in_path, "-ar", str(TARGET_SR),
                        "-ac", "1", "-sample_fmt", "s16", tmp_out_path],
                       capture_output=True, check=True)
        with open(tmp_out_path, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def generate_openai_sample(text, voice, api_key):
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "tts-1", "voice": voice, "input": text, "response_format": "wav"},
                timeout=30,
            )
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 5 * (2 ** attempt)
                    log.warning("OpenAI rate limited (attempt %d/%d) - sleeping %ds",
                                attempt + 1, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                log.warning("OpenAI rate limited after %d retries - skipping", MAX_RETRIES)
                return None
            if resp.status_code != 200:
                log.warning("OpenAI error %d voice=%s: %s", resp.status_code, voice, resp.text[:200])
                return None
            return resample_wav_24k_to_16k(resp.content)
        except requests.RequestException as e:
            log.warning("OpenAI request failed: %s", e)
            return None
    return None


# --- Google Cloud TTS ---
GOOGLE_VOICES = [
    {"name": "en-US-Wavenet-A", "ssmlGender": "MALE"},
    {"name": "en-US-Wavenet-B", "ssmlGender": "MALE"},
    {"name": "en-US-Wavenet-C", "ssmlGender": "FEMALE"},
    {"name": "en-US-Wavenet-D", "ssmlGender": "MALE"},
    {"name": "en-US-Wavenet-E", "ssmlGender": "FEMALE"},
    {"name": "en-US-Wavenet-F", "ssmlGender": "FEMALE"},
    {"name": "en-US-Neural2-A", "ssmlGender": "MALE"},
    {"name": "en-US-Neural2-C", "ssmlGender": "FEMALE"},
    {"name": "en-US-Neural2-D", "ssmlGender": "MALE"},
    {"name": "en-US-Neural2-E", "ssmlGender": "FEMALE"},
    {"name": "en-US-Neural2-F", "ssmlGender": "FEMALE"},
    {"name": "en-GB-Wavenet-A", "ssmlGender": "FEMALE"},
    {"name": "en-GB-Wavenet-B", "ssmlGender": "MALE"},
    {"name": "en-GB-Wavenet-C", "ssmlGender": "FEMALE"},
    {"name": "en-GB-Wavenet-D", "ssmlGender": "MALE"},
    {"name": "en-AU-Wavenet-A", "ssmlGender": "FEMALE"},
    {"name": "en-AU-Wavenet-B", "ssmlGender": "MALE"},
    {"name": "en-AU-Wavenet-C", "ssmlGender": "FEMALE"},
    {"name": "en-AU-Wavenet-D", "ssmlGender": "MALE"},
]


def generate_google_sample(text, voice, api_key):
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "input": {"text": text},
                    "voice": {"languageCode": voice["name"][:5], "name": voice["name"],
                              "ssmlGender": voice["ssmlGender"]},
                    "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": TARGET_SR},
                },
                timeout=30,
            )
            if resp.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 10 * (2 ** attempt)
                    log.warning("Google rate limited (attempt %d/%d) - sleeping %ds",
                                attempt + 1, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                log.warning("Google rate limited after %d retries - skipping", MAX_RETRIES)
                return None
            if resp.status_code != 200:
                log.warning("Google error %d voice=%s: %s", resp.status_code, voice["name"], resp.text[:200])
                return None
            audio_b64 = resp.json().get("audioContent")
            if not audio_b64:
                log.warning("Google TTS: no audioContent in response")
                return None
            # Google returns raw LINEAR16 PCM — wrap in WAV header
            pcm_data = base64.b64decode(audio_b64)
            out = io.BytesIO()
            with wave.open(out, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SR)
                wf.writeframes(pcm_data)
            return out.getvalue()
        except requests.RequestException as e:
            log.warning("Google request failed: %s", e)
            return None
    return None


# --- Core ---

def load_config(word_id):
    cfg_path = REPO_DIR / "words" / word_id / "config.yaml"
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_all_word_ids():
    words_dir = REPO_DIR / "words"
    return sorted(d.name for d in words_dir.iterdir()
                  if d.is_dir() and (d / "config.yaml").exists())


def count_cloud_samples(output_dir, provider):
    return len(list(output_dir.glob(f"cloud_{provider}_*.wav")))


def generate_for_word(word_id, providers, samples_per_provider, dry_run=False):
    cfg = load_config(word_id)
    target_phrase = cfg["display_name"]
    artifact_dir = REPO_DIR / "artifacts" / word_id / "positive_train"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cloud_cfg = cfg.get("cloud_tts", {})
    if samples_per_provider is None:
        samples_per_provider = cloud_cfg.get("samples_per_provider", 500)

    log.info("=" * 60)
    log.info("Word: %s  Phrase: '%s'  Target: %d samples/provider",
             word_id, target_phrase, samples_per_provider)

    total_generated = 0

    for provider in providers:
        api_key = get_api_key(provider)
        if not api_key:
            log.warning("Skipping %s - no API key (set %s_API_KEY)", provider, provider.upper())
            continue

        existing = count_cloud_samples(artifact_dir, provider)
        if existing >= samples_per_provider:
            log.info("  %s: %d/%d exist - skipping", provider, existing, samples_per_provider)
            continue

        needed = samples_per_provider - existing
        log.info("  %s: generating %d samples (%d existing)", provider, needed, existing)

        if dry_run:
            continue

        if provider == "xai":
            voices = cloud_cfg.get("xai", {}).get("voices", XAI_VOICES)
            gen_func = lambda text, voice, _key=api_key: generate_xai_sample(text, voice, _key)
        elif provider == "openai":
            voices = cloud_cfg.get("openai", {}).get("voices", OPENAI_VOICES)
            gen_func = lambda text, voice, _key=api_key: generate_openai_sample(text, voice, _key)
        elif provider == "google":
            voices = cloud_cfg.get("google", {}).get("voices", GOOGLE_VOICES)
            gen_func = lambda text, voice, _key=api_key: generate_google_sample(text, voice, _key)
        else:
            continue

        generated = 0
        consecutive_errors = 0
        max_consecutive = 10

        for i in range(needed):
            voice = voices[i % len(voices)]
            wav_data = gen_func(target_phrase, voice)
            if wav_data is None:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive:
                    log.error("  %s: %d consecutive errors - aborting provider", provider, consecutive_errors)
                    break
                continue
            consecutive_errors = 0
            if isinstance(voice, dict):
                voice_label = voice["name"].replace("-", "_")
            else:
                voice_label = voice
            out_path = artifact_dir / f"cloud_{provider}_{voice_label}_{uuid.uuid4().hex[:8]}.wav"
            with open(out_path, "wb") as f:
                f.write(wav_data)
            generated += 1
            if generated % 50 == 0:
                log.info("  %s: %d/%d generated", provider, generated, needed)

        total_generated += generated
        log.info("  %s: done - %d new (total: %d)", provider, generated, existing + generated)

    log.info("Word %s: %d total new cloud samples", word_id, total_generated)
    return total_generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate cloud TTS samples for wake word training")
    parser.add_argument("word_id", nargs="?", help="Single word ID")
    parser.add_argument("--words", nargs="+",
                        help="Word IDs to process (use 'all' for all words)")
    parser.add_argument("--providers", nargs="+", choices=["xai", "openai", "google"],
                        default=["xai", "openai", "google"],
                        help="TTS providers to use (default: all)")
    parser.add_argument("--samples", type=int, default=None,
                        help="Samples per provider per word (default: from config or 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without making API calls")
    args = parser.parse_args()

    load_env()

    if args.words:
        word_ids = get_all_word_ids() if args.words == ["all"] else args.words
    elif args.word_id:
        word_ids = [args.word_id]
    else:
        parser.error("Provide a word_id or use --words")

    log.info("Cloud TTS: %d words, providers=%s, samples/provider=%s",
             len(word_ids), args.providers, args.samples or "config/500")
    for prov in args.providers:
        key = get_api_key(prov)
        if key:
            log.info("  %s: key loaded (%s...%s)", prov, key[:8], key[-4:])
        else:
            log.warning("  %s: NO KEY - will skip", prov)

    grand_total = 0
    for word_id in word_ids:
        try:
            n = generate_for_word(word_id, args.providers, args.samples, args.dry_run)
            grand_total += n
        except Exception as e:
            log.error("Error processing %s: %s", word_id, e, exc_info=True)

    log.info("=" * 60)
    log.info("COMPLETE: %d total new samples across %d words", grand_total, len(word_ids))


if __name__ == "__main__":
    main()
