#!/bin/bash
# Extract frames from video for AnySplat processing

VIDEO_PATH=$1
OUTPUT_DIR=$2
FPS=${3:-1}  # Default: 1 frame per second

if [ -z "$VIDEO_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <video_path> <output_dir> [fps]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Extract frames using ffmpeg
ffmpeg -i "$VIDEO_PATH" -vf "fps=$FPS" "$OUTPUT_DIR/frame_%04d.jpg"

echo "Extracted frames to $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | head -10
