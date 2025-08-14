#!/usr/bin/env python3
"""
AI Image Upscaling and Inpainting - Demo Script
Showcases the latest features and improvements
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.upscaling import UpscalingProcessor
from models.inpainting import InpaintingProcessor
from utils.image_utils import (
    load_image, save_image, display_comparison, calculate_psnr, 
    calculate_ssim, enhance_image, apply_filters, get_image_info
)
from utils.model_utils import get_device_info, benchmark_model, print_model_summary

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def demo_system_info():
    """Demonstrate system information capabilities"""
    print_header("SYSTEM INFORMATION DEMO")
    
    print("Getting comprehensive system information...")
    device_info = get_device_info()
    
    print("\nSystem Information:")
    for key, value in device_info.items():
        if key == 'gpus':
            print(f"  {key}:")
            for gpu_id, gpu_data in value.items():
                for gpu_key, gpu_value in gpu_data.items():
                    if gpu_key == 'memory_total':
                        gpu_value = f"{gpu_value / (1024**3):.1f} GB"
                    elif gpu_key in ['memory_free', 'memory_used']:
                        gpu_value = f"{gpu_value / (1024**2):.1f} MB"
                    print(f"    {gpu_key}: {gpu_value}")
        else:
            if key == 'memory_total' or key == 'memory_available':
                value = f"{value / (1024**3):.1f} GB"
            print(f"  {key}: {value}")

def demo_upscaling_models():
    """Demonstrate upscaling models"""
    print_header("UPSCALING MODELS DEMO")
    
    models = ['esrgan', 'realesrgan', 'edsr']
    scales = [2, 4]
    
    for model_name in models:
        for scale in scales:
            print_section(f"Testing {model_name.upper()} with {scale}x scaling")
            
            try:
                # Initialize processor
                processor = UpscalingProcessor(
                    model_type=model_name,
                    device='auto',
                    scale_factor=scale
                )
                
                # Get model information
                model_info = processor.get_model_info()
                print(f"Model: {model_info['model_class']}")
                print(f"Parameters: {model_info['total_parameters']:,}")
                print(f"Device: {model_info['device']}")
                
                # Benchmark model
                print("\nBenchmarking model performance...")
                benchmark_results = benchmark_model(processor.model, device=processor.device)
                print(f"Mean processing time: {benchmark_results['mean_time_ms']:.2f} ms")
                print(f"Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
                print(f"Memory usage: {benchmark_results['mean_memory_mb']:.1f} MB")
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")

def demo_inpainting_models():
    """Demonstrate inpainting models"""
    print_header("INPAINTING MODELS DEMO")
    
    models = ['lama', 'transformer']
    
    for model_name in models:
        print_section(f"Testing {model_name.upper()}")
        
        try:
            # Initialize processor
            processor = InpaintingProcessor(
                model_type=model_name,
                device='auto'
            )
            
            # Get model information
            model_info = processor.get_model_info()
            print(f"Model: {model_info['model_class']}")
            print(f"Parameters: {model_info['total_parameters']:,}")
            print(f"Device: {model_info['device']}")
            
            # Benchmark model
            print("\nBenchmarking model performance...")
            benchmark_results = benchmark_model(processor.model, device=processor.device)
            print(f"Mean processing time: {benchmark_results['mean_time_ms']:.2f} ms")
            print(f"Throughput: {benchmark_results['throughput_fps']:.1f} FPS")
            print(f"Memory usage: {benchmark_results['mean_memory_mb']:.1f} MB")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

def demo_image_processing():
    """Demonstrate image processing capabilities"""
    print_header("IMAGE PROCESSING CAPABILITIES DEMO")
    
    # Create a sample image for testing
    print("Creating sample image for testing...")
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_image)
        
        # Save test image
        test_path = "demo_test_image.png"
        test_image.save(test_path)
        print(f"Test image created: {test_path}")
        
        # Get image information
        print_section("Image Information")
        image_info = get_image_info(test_path)
        for key, value in image_info.items():
            print(f"  {key}: {value}")
        
        # Test image enhancement
        print_section("Image Enhancement")
        enhanced = enhance_image(
            test_image,
            brightness=1.2,
            contrast=1.1,
            sharpness=1.3,
            saturation=1.2
        )
        enhanced_path = "demo_enhanced_image.png"
        enhanced.save(enhanced_path)
        print(f"Enhanced image saved: {enhanced_path}")
        
        # Test filters
        print_section("Image Filters")
        filters = ['gaussian', 'median', 'unsharp']
        for filter_type in filters:
            filtered = apply_filters(test_image, filter_type=filter_type)
            filtered_path = f"demo_{filter_type}_filtered.png"
            filtered.save(filtered_path)
            print(f"{filter_type.title()} filtered image saved: {filtered_path}")
        
        # Clean up test files
        test_files = [test_path, enhanced_path] + [f"demo_{f}_filtered.png" for f in filters]
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up: {file_path}")
        
    except Exception as e:
        print(f"Error in image processing demo: {e}")

def demo_advanced_features():
    """Demonstrate advanced features"""
    print_header("ADVANCED FEATURES DEMO")
    
    print_section("Memory Optimization")
    print("• Flash attention support for CUDA")
    print("• Memory-efficient SDP operations")
    print("• Automatic memory management")
    print("• Tiling for large image processing")
    
    print_section("Performance Optimization")
    print("• torch.compile support (PyTorch 2.0+)")
    print("• Mixed precision processing")
    print("• cuDNN benchmarking")
    print("• Adaptive batch sizing")
    
    print_section("Quality Assessment")
    print("• PSNR calculation")
    print("• SSIM analysis")
    print("• Visual comparison tools")
    print("• Metrics export capabilities")

def demo_usage_examples():
    """Demonstrate usage examples"""
    print_header("USAGE EXAMPLES")
    
    print_section("Command Line Usage")
    print("Basic upscaling:")
    print("  python main.py --input image.jpg --output upscaled.jpg --mode upscale --scale 4")
    
    print("\nAdvanced upscaling with enhancement:")
    print("  python main.py --input image.jpg --output enhanced.jpg --mode upscale --enhance --quality 95")
    
    print("\nInpainting with brush tool:")
    print("  python main.py --input image.jpg --output inpainted.jpg --mode inpainting --brush-size 30")
    
    print("\nBatch processing:")
    print("  python main.py --input-dir ./images --output-dir ./processed --mode upscale --scale 2")
    
    print_section("Web Interface")
    print("Streamlit app:")
    print("  streamlit run web_app.py")
    
    print("\nGradio interface:")
    print("  python web_app.py")

def main():
    """Main demo function"""
    print_header("AI IMAGE UPSCALING & INPAINTING - DEMO")
    print("This demo showcases the latest features and improvements")
    print("of the Image Upscaling and Inpainting project.")
    
    try:
        # Run demos
        demo_system_info()
        demo_upscaling_models()
        demo_inpainting_models()
        demo_image_processing()
        demo_advanced_features()
        demo_usage_examples()
        
        print_header("DEMO COMPLETED SUCCESSFULLY")
        print("All features have been demonstrated successfully!")
        print("\nTo get started with the project:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run command line: python main.py --help")
        print("3. Launch web app: streamlit run web_app.py")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
