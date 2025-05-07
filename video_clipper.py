import os
import cv2
import torch
import clip
import numpy as np
from pathlib import Path
import ffmpeg
import yt_dlp
from typing import List, Tuple, Union
import logging
from datetime import timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoClipper:
    def __init__(self, similarity_threshold: float = 0.3, clip_duration: int = 30):
        """
        Initialize the VideoClipper with CLIP model and parameters.
        
        Args:
            similarity_threshold: Threshold for considering frames similar (0-1)
            clip_duration: Duration of output clips in seconds
        """
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.similarity_threshold = similarity_threshold
        self.clip_duration = clip_duration
        
    def download_youtube_video(self, url: str, output_path: str) -> str:
        """Download video from YouTube URL using yt-dlp."""
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path

    def get_video_path(self, video_source: str) -> str:
        """Handle both local files and YouTube URLs."""
        if video_source.startswith(('http://', 'https://')):
            output_path = f"temp_video_{hash(video_source)}.mp4"
            return self.download_youtube_video(video_source, output_path)
        return video_source

    def sample_frames(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """
        Sample frames from video at 1 FPS.
        
        Returns:
            List of tuples containing (timestamp, frame)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Sample every second
        
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((timestamp, frame_rgb))
                
            frame_count += 1
            
        cap.release()
        return frames

    def get_clip_embedding(self, image: np.ndarray) -> torch.Tensor:
        """Get CLIP embedding for an image."""
        # Preprocess image for CLIP
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def find_similar_segments(self, video_path: str, reference_image_path: str) -> List[Tuple[float, float]]:
        """
        Find segments where reference image appears in video.
        
        Returns:
            List of (start_time, end_time) tuples
        """
        # Load and process reference image
        ref_image = cv2.imread(reference_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_embedding = self.get_clip_embedding(ref_image)
        
        # Sample frames from video
        frames = self.sample_frames(video_path)
        
        # Find similar segments
        similar_segments = []
        current_segment = None
        
        for timestamp, frame in frames:
            frame_embedding = self.get_clip_embedding(frame)
            similarity = float(torch.nn.functional.cosine_similarity(ref_embedding, frame_embedding))
            
            if similarity > self.similarity_threshold:
                if current_segment is None:
                    current_segment = (timestamp, timestamp)
                else:
                    current_segment = (current_segment[0], timestamp)
            elif current_segment is not None:
                similar_segments.append(current_segment)
                current_segment = None
                
        if current_segment is not None:
            similar_segments.append(current_segment)
            
        return similar_segments

    def clip_video_segments(self, video_path: str, segments: List[Tuple[float, float]], output_folder: str):
        """Create video clips for each similar segment."""
        os.makedirs(output_folder, exist_ok=True)
        
        for i, (start_time, end_time) in enumerate(segments):
            # Add padding around the segment
            start_time = max(0, start_time - self.clip_duration/2)
            duration = self.clip_duration
            
            output_path = os.path.join(output_folder, f"clip_{i+1}.mp4")
            
            try:
                stream = ffmpeg.input(video_path, ss=start_time, t=duration)
                stream = ffmpeg.output(stream, output_path)
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                logger.info(f"Created clip {i+1} from {timedelta(seconds=start_time)} to {timedelta(seconds=start_time+duration)}")
            except ffmpeg.Error as e:
                logger.error(f"Error creating clip {i+1}: {e.stderr.decode()}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find and clip video segments containing a reference image")
    parser.add_argument("video_source", help="Path to video file or YouTube URL")
    parser.add_argument("reference_image", help="Path to reference image file")
    parser.add_argument("output_folder", help="Folder to save output clips")
    parser.add_argument("--threshold", type=float, default=0.3, help="Similarity threshold (0-1)")
    parser.add_argument("--duration", type=int, default=30, help="Duration of output clips in seconds")
    
    args = parser.parse_args()
    
    clipper = VideoClipper(similarity_threshold=args.threshold, clip_duration=args.duration)
    
    # Get video path (download if YouTube URL)
    video_path = clipper.get_video_path(args.video_source)
    
    try:
        # Find similar segments
        segments = clipper.find_similar_segments(video_path, args.reference_image)
        
        if not segments:
            logger.info("No similar segments found")
            return
            
        # Create clips
        clipper.clip_video_segments(video_path, segments, args.output_folder)
        
    finally:
        # Clean up temporary video file if it was downloaded
        if video_path != args.video_source and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    main() 