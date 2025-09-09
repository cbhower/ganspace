#!/usr/bin/env python3
"""
Latent Space Morphing for GANSpace
Creates smooth interpolation videos through GAN latent spaces.

Usage:
    python latent_morph.py --model=StyleGAN2 --class=ffhq --path=circle --frames=120 --output=morph_test
    python latent_morph.py --model=BigGAN-256 --class=husky --path=line --frames=60 --keypoints=4
"""

import torch
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import datetime

from models import get_model
from config import Config


def parse_args():
    """Parse command line arguments for latent morphing."""
    parser = argparse.ArgumentParser(description='Generate latent space morphing videos')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='BigGAN-256', 
                       help='Model name (BigGAN-256, StyleGAN2, etc.)')
    parser.add_argument('--class', dest='output_class', type=str, default='husky',
                       help='Output class for conditional models')
    parser.add_argument('--use_w', action='store_true', 
                       help='Use W space for StyleGAN models')
    parser.add_argument('--truncation', type=float, default=0.9,
                       help='Truncation value for sampling')
    
    # Path parameters
    parser.add_argument('--path', type=str, choices=['line', 'circle', 'custom'], default='circle',
                       help='Type of interpolation path')
    parser.add_argument('--keypoints', type=int, default=4,
                       help='Number of keypoints for interpolation')
    parser.add_argument('--frames', type=int, default=120,
                       help='Total number of frames to generate')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='morph_output',
                       help='Output directory name')
    parser.add_argument('--resolution', type=int, default=None,
                       help='Output image resolution (None = model default)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')
    parser.add_argument('--create_video', action='store_true',
                       help='Automatically create MP4 video after generating frames')
    parser.add_argument('--framerate', type=int, default=30,
                       help='Video framerate (fps)')
    
    # Seeds and reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for keypoint generation')
    parser.add_argument('--keypoint_seeds', type=int, nargs='*', default=None,
                       help='Specific seeds for keypoints (overrides random generation)')
    
    return parser.parse_args()


def slerp(val, low, high):
    """Spherical linear interpolation between two vectors."""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high  # Linear interpolation fallback
    return (np.sin((1.0-val)*omega) / so) * low + (np.sin(val*omega) / so) * high


def lerp(val, low, high):
    """Linear interpolation between two vectors."""
    return (1.0 - val) * low + val * high


def generate_keypoints(model, num_keypoints, seeds=None, truncation=0.9):
    """Generate keypoints in latent space."""
    keypoints = []
    
    if seeds is None:
        # Generate random seeds
        np.random.seed(42)  # For reproducibility
        seeds = np.random.randint(0, 100000, num_keypoints)
    
    for seed in seeds:
        latent = model.sample_latent(n_samples=1, seed=seed, truncation=truncation)
        keypoints.append(latent.cpu().numpy())
    
    return keypoints, seeds


def create_circular_path(keypoints, num_frames):
    """Create a circular interpolation path through keypoints."""
    keypoints = np.array(keypoints)
    num_keypoints = len(keypoints)
    
    # Add the first keypoint at the end to close the loop
    extended_keypoints = np.concatenate([keypoints, keypoints[0:1]], axis=0)
    
    interpolated_points = []
    
    for i in range(num_frames):
        # Current position along the path (0 to num_keypoints)
        t = (i / num_frames) * num_keypoints
        
        # Find which segment we're in
        segment_idx = int(np.floor(t)) % num_keypoints
        next_idx = (segment_idx + 1) % len(extended_keypoints)
        
        # Local interpolation parameter within the segment
        local_t = t - np.floor(t)
        
        # Spherical interpolation for better quality
        point = slerp(local_t, 
                     extended_keypoints[segment_idx].squeeze(),
                     extended_keypoints[next_idx].squeeze())
        
        interpolated_points.append(point)
    
    return interpolated_points


def create_linear_path(keypoints, num_frames):
    """Create a linear interpolation path through keypoints."""
    keypoints = np.array(keypoints)
    num_keypoints = len(keypoints)
    
    if num_keypoints < 2:
        raise ValueError("Need at least 2 keypoints for linear path")
    
    interpolated_points = []
    
    # Distribute frames across segments
    frames_per_segment = num_frames // (num_keypoints - 1)
    remaining_frames = num_frames % (num_keypoints - 1)
    
    for segment in range(num_keypoints - 1):
        # Number of frames for this segment
        segment_frames = frames_per_segment
        if segment < remaining_frames:
            segment_frames += 1
        
        for i in range(segment_frames):
            if segment == num_keypoints - 2 and i == segment_frames - 1:
                # Last frame should be exactly the last keypoint
                t = 1.0
            else:
                t = i / segment_frames
            
            point = slerp(t, 
                         keypoints[segment].squeeze(),
                         keypoints[segment + 1].squeeze())
            
            interpolated_points.append(point)
    
    return interpolated_points


def generate_frames(model, interpolated_points, output_dir, batch_size=4, resolution=None):
    """Generate images for all interpolated points."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    num_points = len(interpolated_points)
    
    print(f"Generating {num_points} frames...")
    
    # Process in batches
    for batch_start in tqdm(range(0, num_points, batch_size), desc="Generating frames"):
        batch_end = min(batch_start + batch_size, num_points)
        batch_points = interpolated_points[batch_start:batch_end]
        
        # Convert to tensor and move to device
        batch_latents = []
        for point in batch_points:
            latent_tensor = torch.from_numpy(point).float().to(device)
            if len(latent_tensor.shape) == 1:
                latent_tensor = latent_tensor.unsqueeze(0)
            batch_latents.append(latent_tensor)
        
        # Stack into batch
        if len(batch_latents) > 1:
            batch_tensor = torch.cat(batch_latents, dim=0)
        else:
            batch_tensor = batch_latents[0]
        
        # Generate images
        with torch.no_grad():
            images = model.forward(batch_tensor)
        
        # Save individual frames
        for i, img in enumerate(images):
            frame_idx = batch_start + i
            
            # Convert to PIL Image
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            if resolution is not None:
                img_pil = Image.fromarray(img_np).resize((resolution, resolution), Image.LANCZOS)
            else:
                img_pil = Image.fromarray(img_np)
            
            # Save with zero-padded frame number
            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            img_pil.save(frame_path)
    
    print(f"Frames saved to: {output_dir}")
    return output_dir


def create_video(frame_dir, output_path=None, framerate=30):
    """Create MP4 video from frame directory using ffmpeg."""
    import subprocess
    import shutil
    
    frame_dir = Path(frame_dir)
    if output_path is None:
        output_path = frame_dir / 'morph.mp4'
    else:
        output_path = Path(output_path)
    
    # Check if ffmpeg is available
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("Error: ffmpeg not found. Please install ffmpeg to create videos.")
        print("On macOS: brew install ffmpeg")
        print("On Ubuntu: sudo apt install ffmpeg")
        return False
    
    # Build ffmpeg command
    cmd = [
        ffmpeg_bin,
        '-y',  # Overwrite output file
        '-framerate', str(framerate),
        '-i', str(frame_dir / 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        str(output_path)
    ]
    
    try:
        print(f"Creating video: {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video created successfully: {output_path}")
            return True
        else:
            print(f"Error creating video: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False


def main():
    args = parse_args()
    
    print(f"Starting latent morphing with {args.model}")
    print(f"Path type: {args.path}, Keypoints: {args.keypoints}, Frames: {args.frames}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(args.model, args.output_class, device)
    if hasattr(model, 'use_w') and args.use_w:
        model.use_w()
    
    model.eval()
    
    # Generate keypoints
    seeds = args.keypoint_seeds
    if seeds and len(seeds) != args.keypoints:
        print(f"Warning: Got {len(seeds)} seeds but need {args.keypoints} keypoints")
        seeds = None
    
    keypoints, used_seeds = generate_keypoints(
        model, args.keypoints, seeds, args.truncation
    )
    
    print(f"Generated keypoints with seeds: {used_seeds}")
    
    # Create interpolation path
    if args.path == 'circle':
        interpolated_points = create_circular_path(keypoints, args.frames)
    elif args.path == 'line':
        interpolated_points = create_linear_path(keypoints, args.frames)
    else:
        raise ValueError(f"Unsupported path type: {args.path}")
    
    print(f"Created {len(interpolated_points)} interpolated points")
    
    # Generate output directory name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{args.output}_{args.model}_{args.path}_{timestamp}"
    
    # Generate frames
    output_dir = generate_frames(
        model, interpolated_points, f"out/morphing/{output_name}",
        args.batch_size, args.resolution
    )
    
    # Save metadata
    metadata = {
        'model': args.model,
        'output_class': args.output_class,
        'use_w': args.use_w,
        'truncation': args.truncation,
        'path_type': args.path,
        'keypoints': args.keypoints,
        'frames': args.frames,
        'keypoint_seeds': used_seeds.tolist(),
        'resolution': args.resolution,
        'generated_at': timestamp
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create video if requested
    if args.create_video:
        create_video(output_dir, framerate=args.framerate)
    
    print(f"\nMorphing complete!")
    print(f"Output directory: {output_dir}")
    if not args.create_video:
        print(f"To create video, run:")
        print(f"ffmpeg -framerate {args.framerate} -i {output_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/morph.mp4")


if __name__ == '__main__':
    main()
