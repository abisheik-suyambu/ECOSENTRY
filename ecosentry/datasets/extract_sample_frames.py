"""
Script to extract thermal frames from RTPV-YOLO demonstration GIF.
Saves frames into ecosentry/datasets/sample_thermal/ as PNG files.
"""

import os
from PIL import Image, ImageSequence

def extract_frames(gif_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isfile(gif_path):
        raise FileNotFoundError(f"Source GIF not found: {gif_path}")
        
    print(f"Opening source GIF: {gif_path}")
    with Image.open(gif_path) as im:
        total_frames = getattr(im, "n_frames", 0)
        print(f"Detected {total_frames} frames in GIF.")
        
        extracted_files = []
        for idx, frame in enumerate(ImageSequence.Iterator(im), start=1):
            frame_rgb = frame.convert("RGB")
            out_filename = f"thermal_{idx:04d}.png"
            out_path = os.path.join(output_dir, out_filename)
            frame_rgb.save(out_path, format="PNG")
            file_size_kb = os.path.getsize(out_path) / 1024
            extracted_files.append((out_filename, frame_rgb.size, file_size_kb))
            print(f"Saved {out_filename} | Resolution: {frame_rgb.size} | Size: {file_size_kb:.1f} KB")

    print(f"\nExtraction complete: {len(extracted_files)} frames saved to {output_dir}")
    return extracted_files

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_gif = os.path.join(base_dir, "RTPV-YOLO", "examples", "hotspot-detect.gif")
    target_dir = os.path.join(base_dir, "ecosentry", "datasets", "sample_thermal")
    extract_frames(src_gif, target_dir)
