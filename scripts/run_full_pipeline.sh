#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m biofake.cli pipeline run --config configs/experiments/full_report.yaml

