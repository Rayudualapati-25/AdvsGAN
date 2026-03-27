#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m biofake.cli demo build-assets --config configs/experiments/robust_hybrid.yaml

