#!/usr/bin/env python3
"""
AI Image Upscaling and Inpainting - Enhanced Command Line Interface
Latest version with improved models, better performance, and enhanced features
"""

import argparse
import os
import sys
from pathlib import Path
import time
import warnings
from typing import Optional, List

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.upscaling import UpscalingProcessor
from models.inpainting import InpaintingProcessor
from utils.image_utils import (
    load_image, save_image, display_comparison, calculate_psnr, 
    calculate_ssim, enhance_image, apply_filters, create_image_grid,
    get_image_info, validate_image
)
from utils.model_utils import (
    get_device_info, create_model_summary, get_optimal_device,
    optimize_for_inference, benchmark_model, print_model_summary
)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced AI Image Upscaling and Inpainting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale an image with enhanced ESRGAN
  python main.py --input image.jpg --output upscaled.jpg --mode upscale --scale 4 --model esrgan --quality 95
  
  # Inpaint an image with LaMa model
  python main.py --input image.jpg --mask mask.png --output inpainted.jpg --mode inpainting --model lama
  
  # Batch upscaling with tiling for large images
  python main.py --input-dir ./images --output-dir ./upscaled --mode upscale --scale 2 --tile-size 512
  
  # Enhanced inpainting with brush tool simulation
  python main.py --input image.jpg --output inpainted.jpg --mode inpainting --model lama --brush-size 30
  
  # Process with quality metrics and comparison
  python main.py --input image.jpg --output upscaled.jpg --mode upscale --metrics --compare --verbose
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
                       choices=['esrgan', 'realesrgan', 'edsr', 'lama', 'transformer'],
                       help='AI model to use')
    
    # Upscaling arguments
    parser.add_argument('--scale', '-s', type=int, default=4, 
                       choices=[2, 4, 8], help='Upscaling factor')
    parser.add_argument('--tile-size', type=int, default=512, 
                       help='Tile size for processing large images')
    
    # Inpainting arguments
    parser.add_argument('--mask', type=str, help='Mask image path for inpainting')
    parser.add_argument('--brush-size', type=int, default=20, 
                       help='Brush size for interactive inpainting')
    
    # General arguments
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--quality', type=int, default=95, 
                       choices=range(1, 101), help='Output image quality (1-100)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--compare', action='store_true', help='Show comparison')
    parser.add_argument('--metrics', action='store_true', help='Calculate quality metrics')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark model performance')
    parser.add_argument('--enhance', action='store_true', help='Apply image enhancement')
    parser.add_argument('--filter', type=str, choices=['gaussian', 'median', 'unsharp'], 
                       help='Apply image filter')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")
    
    if args.mode == 'inpainting' and not args.mask and not args.brush_size:
        parser.error("Either --mask or --brush-size must be specified for inpainting mode")
    
    # Display device information
    if args.verbose:
        device_info = get_device_info()
        print("Device Information:")
        for key, value in device_info.items():
            if key == 'gpus':
                print(f"  {key}:")
                for gpu_id, gpu_data in value.items():
                    for gpu_key, gpu_value in gpu_data.items():
                        print(f"    {gpu_key}: {gpu_value}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Process images
    try:
        if args.input_dir:
            process_batch(args)
        else:
            process_single(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def process_single(args):
    """Process a single image with enhanced functionality"""
    print(f"Processing image: {args.input}")
    
    # Validate input image
    if not validate_image(args.input):
        raise ValueError(f"Invalid or corrupted image file: {args.input}")
    
    # Get image info
    if args.verbose:
        image_info = get_image_info(args.input)
        print("Image Information:")
        for key, value in image_info.items():
            print(f"  {key}: {value}")
        print()
    
    if args.mode == 'upscale':
        process_upscaling(args)
    elif args.mode == 'inpainting':
        process_inpainting(args)

def process_batch(args):
    """Process multiple images in batch mode"""
    print(f"Processing batch from directory: {args.input_dir}")
    
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for file in os.listdir(args.input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        raise ValueError(f"No image files found in {args.input_dir}")
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(args.input_dir, filename)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(args.output_dir, output_filename)
        
        print(f"Processing {i}/{len(image_files)}: {filename}")
        
        try:
            if args.mode == 'upscale':
                process_upscaling_single(input_path, output_path, args)
            elif args.mode == 'inpainting':
                process_inpainting_single(input_path, output_path, args)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Batch processing completed. {len(image_files)} images processed.")

def process_upscaling(args):
    """Process upscaling with enhanced features"""
    print("Initializing upscaling processor...")
    
    # Initialize processor
    processor = UpscalingProcessor(
        model_type=args.model,
        device=args.device,
        scale_factor=args.scale
    )
    
    # Display model information
    if args.verbose:
        model_info = processor.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()
    
    # Benchmark if requested
    if args.benchmark:
        print("Benchmarking model performance...")
        benchmark_results = benchmark_model(processor.model, device=processor.device)
        print("Benchmark Results:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value}")
        print()
    
    # Process image
    print("Processing image...")
    start_time = time.time()
    
    result = processor.upscale_image(
        args.input, 
        args.output, 
        tile_size=args.tile_size
    )
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Apply enhancements if requested
    if args.enhance:
        print("Applying image enhancements...")
        enhanced_result = enhance_image(
            Image.fromarray(result),
            brightness=1.1,
            contrast=1.05,
            sharpness=1.2,
            saturation=1.1
        )
        save_image(enhanced_result, args.output, quality=args.quality)
        result = np.array(enhanced_result)
    
    # Apply filters if requested
    if args.filter:
        print(f"Applying {args.filter} filter...")
        filtered_result = apply_filters(
            Image.fromarray(result),
            filter_type=args.filter
        )
        save_image(filtered_result, args.output, quality=args.quality)
        result = np.array(filtered_result)
    
    # Calculate metrics if requested
    if args.metrics:
        print("Calculating quality metrics...")
        original = load_image(args.input)
        psnr_value = calculate_psnr(original, result)
        ssim_value = calculate_ssim(original, result)
        
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
    
    # Show comparison if requested
    if args.compare:
        print("Displaying comparison...")
        display_comparison(args.input, result, ('Original', 'Upscaled'))
    
    print(f"Upscaling completed successfully. Output saved to: {args.output}")

def process_inpainting(args):
    """Process inpainting with enhanced features"""
    print("Initializing inpainting processor...")
    
    # Initialize processor
    processor = InpaintingProcessor(
        model_type=args.model,
        device=args.device
    )
    
    # Display model information
    if args.verbose:
        model_info = processor.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()
    
    # Benchmark if requested
    if args.benchmark:
        print("Benchmarking model performance...")
        benchmark_results = benchmark_model(processor.model, device=processor.device)
        print("Benchmark Results:")
        for key, value in benchmark_results.items():
            print(f"  {key}: {value}")
        print()
    
    # Process image
    print("Processing image...")
    start_time = time.time()
    
    if args.mask:
        # Use provided mask
        result = processor.inpaint_image(args.input, args.mask, args.output)
    else:
        # Use brush tool simulation
        result, mask = processor.inpaint_with_brush(
            args.input, 
            brush_size=args.brush_size, 
            output_path=args.output
        )
        
        # Save mask for reference
        mask_path = args.output.replace('.', '_mask.')
        save_image(mask, mask_path)
        print(f"Mask saved to: {mask_path}")
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Apply enhancements if requested
    if args.enhance:
        print("Applying image enhancements...")
        enhanced_result = enhance_image(
            Image.fromarray(result),
            brightness=1.05,
            contrast=1.1,
            sharpness=1.1
        )
        save_image(enhanced_result, args.output, quality=args.quality)
        result = np.array(enhanced_result)
    
    # Calculate metrics if requested
    if args.metrics and args.mask:
        print("Calculating quality metrics...")
        original = load_image(args.input)
        psnr_value = calculate_psnr(original, result)
        ssim_value = calculate_ssim(original, result)
        
        print(f"PSNR: {psnr_value:.2f} dB")
        print(f"SSIM: {ssim_value:.4f}")
    
    # Show comparison if requested
    if args.compare:
        print("Displaying comparison...")
        if args.mask:
            display_comparison(args.input, result, ('Original', 'Inpainted'))
        else:
            # Show original, mask, and result
            create_image_grid(
                [args.input, mask, result],
                ['Original', 'Mask', 'Inpainted'],
                cols=3
            )
    
    print(f"Inpainting completed successfully. Output saved to: {args.output}")

def process_upscaling_single(input_path: str, output_path: str, args):
    """Process single image for batch upscaling"""
    processor = UpscalingProcessor(
        model_type=args.model,
        device=args.device,
        scale_factor=args.scale
    )
    
    processor.upscale_image(input_path, output_path, tile_size=args.tile_size)

def process_inpainting_single(input_path: str, output_path: str, args):
    """Process single image for batch inpainting"""
    processor = InpaintingProcessor(
        model_type=args.model,
        device=args.device
    )
    
    if args.mask:
        processor.inpaint_image(input_path, args.mask, output_path)
    else:
        result, _ = processor.inpaint_with_brush(input_path, brush_size=args.brush_size)
        save_image(Image.fromarray(result), output_path, quality=args.quality)

if __name__ == "__main__":
    main() 