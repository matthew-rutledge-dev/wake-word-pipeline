#!/usr/bin/env python3
"""Validate model shape and run basic detection smoke test."""
import sys
WORD_ID = sys.argv[1]
MODEL_TYPE = sys.argv[2] if len(sys.argv) > 2 else "oww"
print(f"TODO: Validate {MODEL_TYPE} model for {WORD_ID}")
