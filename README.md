# Video Clipper

A Python tool that finds and clips segments of a video where a reference image appears, using OpenAI's CLIP model for semantic image similarity.

## Features

- Process local video files or YouTube URLs
- Sample video frames at 1 FPS
- Use CLIP (ViT-B/32) for semantic image similarity
- Automatically clip 30-second segments around matching frames
- CPU-only operation
- Command-line interface

## Requirements

- Python 3.7+
- FFmpeg installed on your system
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd video-clipper
```

2. Install FFmpeg:
- Windows: Download from https://ffmpeg.org/download.html
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python video_clipper.py <video_source> <reference_image> <output_folder> [--threshold THRESHOLD] [--duration DURATION]
```

Arguments:
- `video_source`: Path to video file or YouTube URL
- `reference_image`: Path to reference image file
- `output_folder`: Folder to save output clips
- `--threshold`: Similarity threshold (0-1, default: 0.3)
- `--duration`: Duration of output clips in seconds (default: 30)

Example:
```bash
# Process a local video file
python video_clipper.py input.mp4 reference.jpg output_clips

# Process a YouTube video
python video_clipper.py "https://www.youtube.com/watch?v=example" reference.jpg output_clips

# Custom threshold and duration
python video_clipper.py input.mp4 reference.jpg output_clips --threshold 0.4 --duration 45
```

## Notes

- The script runs entirely on CPU
- Processing time depends on video length and system performance
- Higher similarity thresholds (e.g., 0.4-0.5) will result in fewer but more precise matches
- Lower thresholds (e.g., 0.2-0.3) will catch more potential matches but may include false positives 