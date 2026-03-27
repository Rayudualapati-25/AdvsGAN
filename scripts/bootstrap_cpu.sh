#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -e .[dev]
echo "Environment bootstrapped for biofake."

