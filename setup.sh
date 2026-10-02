#!/usr/bin/env bash

if [ -n "${BASH_SOURCE[0]}" ]; then
    _SETUP_SRC="${BASH_SOURCE[0]}"
else
    _SETUP_SRC="$0"
fi
SETUP_DIR="$(cd "$(dirname "$_SETUP_SRC")" && pwd)"

export PROJECT_ROOT="$SETUP_DIR"

export DATA_DIR="$PROJECT_ROOT/data"
export MODEL_DIR="$PROJECT_ROOT/model"
export PYTHON_DIR="$PROJECT_ROOT/python"
export UTILS_DIR="$PROJECT_ROOT/utils"
export C_DIR="$PROJECT_ROOT/c"