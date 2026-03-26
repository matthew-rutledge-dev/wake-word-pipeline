#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/opt/ai/wakeword-train/mww-venv"
MWW_REPO="/opt/ai/wakeword-train/microWakeWord"

echo "=== microWakeWord Environment Setup ==="

# 1. Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/6] Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/6] Venv already exists."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# 2. Install TensorFlow with CUDA
echo "[2/6] Installing TensorFlow with CUDA support..."
pip install 'tensorflow[and-cuda]>=2.16'

# 3. Clone repo
if [ ! -d "$MWW_REPO/.git" ]; then
    echo "[3/6] Cloning OHF-Voice/micro-wake-word..."
    git clone https://github.com/OHF-Voice/micro-wake-word.git "$MWW_REPO"
else
    echo "[3/6] Repo already cloned."
fi

# 4. Install microwakeword editable
echo "[4/6] Installing microwakeword (editable)..."
pip install -e "$MWW_REPO"

# 5. Install additional dependencies
echo "[5/6] Installing additional dependencies..."
pip install pymicro-features mmap-ninja audiomentations ai-edge-litert scipy


# 5b. Patch venv activate with CUDA library paths for TF GPU
if ! grep -q "CUDA library paths" "$VENV_DIR/bin/activate"; then
    echo "[5b/6] Patching activate with CUDA library paths..."
    python3 -c "
d = chr(36)
libs = ':'.join([f'{d}_NVDIR/{p}/lib' for p in ['cublas','cuda_runtime','cudnn','cufft','curand','cusolver','cusparse','nccl','nvjitlink']])
patch = chr(10) + '# CUDA library paths for TF GPU (nvidia pip packages)' + chr(10)
patch += f'_NVDIR=\"{d}VIRTUAL_ENV/lib/python3.13/site-packages/nvidia\"' + chr(10)
patch += f'if [ -d \"{d}_NVDIR\" ]; then' + chr(10)
patch += f'    _CP=\"{libs}\"' + chr(10)
patch += f'    export LD_LIBRARY_PATH=\"{d}_CP{d}{{LD_LIBRARY_PATH:+:{d}LD_LIBRARY_PATH}}\"' + chr(10)
patch += 'fi' + chr(10)
with open('$VENV_DIR/bin/activate', 'a') as f:
    f.write(patch)
print('Activate script patched.')
"
    # Re-source to pick up new paths
    source "$VENV_DIR/bin/activate"
else
    echo "[5b/6] Activate already patched."
fi

# 6. Verify
echo "[6/6] Verifying installations..."
python3 -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpus)
if not gpus:
    print('WARNING: No GPU detected by TensorFlow')
"
python3 -c "import microwakeword; print('microwakeword: imported OK')"
python3 -c "import mmap_ninja; print('mmap_ninja: imported OK')"
python3 -c "import audiomentations; print('audiomentations: imported OK')"
python3 -c "import ai_edge_litert; print('ai_edge_litert: imported OK')"

echo ""
echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
