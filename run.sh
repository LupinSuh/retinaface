#!/bin/bash

# Check if target directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <target_directory> [--gpu <gpu_id>]"
  exit 1
fi

TARGET_DIR=$1
shift # Remove the first argument, so we can pass the rest to main.py

# Run the HEIC convertor
echo "Running HEIC convertor..."
bash heic_convertor.sh "$TARGET_DIR"

# Run the main python script
echo "Running main script..."
python main.py "$TARGET_DIR" "$@"
