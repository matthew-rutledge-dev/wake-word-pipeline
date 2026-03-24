#!/usr/bin/env python3
"""Export trained OWW model to TFLite + ONNX. Must run with CUDA_VISIBLE_DEVICES=-1."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
WORD_ID = sys.argv[1]
print(f"TODO: Export OWW model for {WORD_ID}")
