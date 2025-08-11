#!/usr/bin/env python3
"""
AI Image Upscaling and Inpainting - Command Line Interface
"""

import argparse
import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.upscaling import UpscalingProcessor
from models.inpainting import InpaintingProcessor
from utils.image_utils import load_image, save_image, display_comparison, calculate_psnr, calculate_ssim
from utils.model_utils import get_device_info, create_model_summary

def main():
    parser = argparse.ArgumentParser(
        description="AI Image Upscaling and Inpainting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale an image
  python main.py --input image.jpg --output upscaled.jpg --mode upscale --scale 4 --model esrgan
  
  # Inpaint an image
  python main.py --input image.jpg --mask mask.png --output inpainted.jpg --mode inpainting --model lama
  
  # Batch upscaling
  python main.py --input-dir ./images --output-dir ./upscaled --mode upscale --scale 2
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--input-dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output', '-o', type=str, help='Output image path')
    parser.add_argument('--output-dir', type=str, help='Output directory for batch processing')
    
    # Processing mode
    parser.add_argument('--mode', '-m', type=str, choices=['upscale', 'inpainting'], 
                       required=True, help='Processing mode')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='esrgan', 
                       choices=['esrgan', 'realesrgan', 'edsr', 'lama'],
                       help='AI model to use')
    
    # Upscaling arguments
    parser.add_argument('--scale', '-s', type=int, default=4, 
                       choices=[2, 4, 8], help='Upscaling factor')
    
    # Inpainting arguments
    parser.add_argument('--mask', type=str, help='Mask image path for inpainting')
    
    # General arguments
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--quality', type=int, default=95, 
                       choices=range(1, 101), help='Output image quality (1-100)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--compare', action='store_true', help='Show comparison')
    parser.add_argument('--metrics', action='store_true', help='Calculate quality metrics')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    if args.mode == 'inpainting' and not args.mask:
        parser.error("--mask is required for inpainting mode")
    
    # Display device information
    if args.verbose:
        device_info = get_device_info()
        print("Device Information:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        print()
    
    # Process images
    if args.input_dir:
        process_batch(args)
    else:
        process_single(args)

def process_single(args):
    """Process a single image"""
    print(f"Processing image: {args.input}")
    
    if args.mode == 'upscale':
        process_upscaling(args)
    elif args.mode == 'inpainting':
        process_inpainting(args)

def process_batch(args):
    """Process multiple images in a directory"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'processed'
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        # Update args for this image
        args.input = str(image_file)
        args.output = str(output_dir / f"{image_file.stem}_processed{image_file.suffix}")
        
        try:
            if args.mode == 'upscale':
                process_upscaling(args)
            elif args.mode == 'inpainting':
                process_inpainting(args)
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue

def process_upscaling(args):
    """Process image upscaling"""
    start_time = time.time()
    
    # Initialize upscaling processor
    processor = UpscalingProcessor(
        model_type=args.model,
        scale_factor=args.scale,
        device=args.device
    )
    
    if args.verbose:
        print(f"Model: {args.model}")
        print(f"Scale factor: {args.scale}x")
        print(f"Device: {processor.device}")
        print(create_model_summary(processor.model))
        print()
    
    # Process image
    try:
        result = processor.upscale_image(args.input, args.output)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Display results
        print(f"Upscaling completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {args.output}")
        
        # Show comparison if requested
        if args.compare:
            original = load_image(args.input)
            display_comparison(original, result, f"Upscaling: {args.model} {args.scale}x")
        
        # Calculate metrics if requested
        if args.metrics:
            original = load_image(args.input)
            # Resize original to match result size for fair comparison
            from utils.image_utils import resize_image
            original_resized = resize_image(original, (result.shape[1], result.shape[0]))
            
            psnr = calculate_psnr(original_resized, result)
            ssim = calculate_ssim(original_resized, result)
            
            print(f"Quality Metrics:")
            print(f"  PSNR: {psnr:.2f} dB")
            print(f"  SSIM: {ssim:.4f}")
        
    except Exception as e:
        print(f"Error during upscaling: {e}")
        sys.exit(1)

def process_inpainting(args):
    """Process image inpainting"""
    start_time = time.time()
    
    # Initialize inpainting processor
    processor = InpaintingProcessor(
        model_type=args.model,
        device=args.device
    )
    
    if args.verbose:
        print(f"Model: {args.model}")
        print(f"Device: {processor.device}")
        print(create_model_summary(processor.model))
        print()
    
    # Process image
    try:
        result = processor.inpaint_image(args.input, args.mask, args.output)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Display results
        print(f"Inpainting completed in {processing_time:.2f} seconds")
        print(f"Output saved to: {args.output}")
        
        # Show comparison if requested
        if args.compare:
            original = load_image(args.input)
            mask = load_image(args.mask)
            display_comparison(original, result, f"Inpainting: {args.model}")
        
    except Exception as e:
        print(f"Error during inpainting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 