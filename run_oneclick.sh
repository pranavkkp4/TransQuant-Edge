#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv || true
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install pyyaml pandas matplotlib
python oneclick/run_all.py
