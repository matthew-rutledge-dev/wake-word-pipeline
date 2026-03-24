#!/usr/bin/env python3
"""
Compatibility shims for dependency version mismatches.
Import this module BEFORE any openwakeword or piper imports.

Fixes:
  - torchaudio 2.11+ removed list_audio_backends() (needed by speechbrain)
  - scipy 1.14+ removed sph_harm from scipy.special (needed by acoustics)
  - piper_train is a top-level package inside piper-sample-generator repo
"""
import os
import sys
from pathlib import Path


def apply():
    """Apply all compatibility shims. Safe to call multiple times."""

    # --- torchaudio: restore list_audio_backends for speechbrain ---
    try:
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]
    except ImportError:
        pass

    # --- scipy.special: restore sph_harm for acoustics ---
    try:
        import scipy.special
        if not hasattr(scipy.special, "sph_harm"):
            scipy.special.sph_harm = lambda m, n, theta, phi: 0.0
    except ImportError:
        pass

    # --- piper_train: add piper-sample-generator repo to sys.path ---
    piper_dir = os.environ.get(
        "PIPER_SAMPLE_GENERATOR_PATH",
        str(Path(__file__).resolve().parent.parent / "piper-sample-generator"),
    )
    if piper_dir not in sys.path:
        sys.path.insert(0, piper_dir)


# Auto-apply on import
apply()
