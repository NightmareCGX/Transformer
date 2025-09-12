#!/usr/bin/env bash
set -euo pipefail
pre-commit install
pre-commit run --all-files