#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install all dependencies in a folder called python
RESOURCE_DIR="$SCRIPT_DIR/resources"

if [ -d "$RESOURCE_DIR" ]; then rm -Rf "$RESOURCE_DIR"; fi
mkdir -p "$RESOURCE_DIR"

DEPENDENCY_DIR="python"
pip install -r "$SCRIPT_DIR/requirements.txt" -t "$RESOURCE_DIR/$DEPENDENCY_DIR"
cd "$RESOURCE_DIR" && zip -r dependency_layer.zip "$DEPENDENCY_DIR"
rm -rf "$DEPENDENCY_DIR"