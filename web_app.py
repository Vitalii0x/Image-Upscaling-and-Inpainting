#!/usr/bin/env python3
"""
AI Image Upscaling and Inpainting - Modern Web Interface
Combines Streamlit and Gradio for a comprehensive web experience
"""

import streamlit as st
import gradio as gr
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from PIL import Image
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.upscaling import UpscalingProcessor
from models.inpainting import InpaintingProcessor
from utils.image_utils import (
    load_image, save_image, display_comparison, calculate_psnr, 
    calculate_ssim, enhance_image, apply_filters, get_image_info
)
from utils.model_utils import get_device_info, benchmark_model

# Page configuration
st.set_page_config(
    page_title="AI Image Upscaling & Inpainting",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ AI Image Upscaling & Inpainting</h1>', unsafe_allow_html=True)
    st.markdown("### Professional-grade AI-powered image enhancement and restoration")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Device selection
        device_options = ['auto', 'cpu', 'cuda']
        selected_device = st.selectbox("Device", device_options, index=0)
        
        # Quality settings
        quality = st.slider("Output Quality", 1, 100, 95)
        
        # Advanced options
        with st.expander("Advanced Options"):
            enable_enhancement = st.checkbox("Enable Image Enhancement", value=False)
            enable_benchmark = st.checkbox("Enable Benchmarking", value=False)
            tile_size = st.slider("Tile Size (for large images)", 256, 1024, 512, step=128)
        
        # System info
        with st.expander("System Information"):
            device_info = get_device_info()
            st.json(device_info)
    
    # Main tabs
    tab1, tab2, tab3, tab3_alt, tab4 = st.tabs([
        "🚀 Upscaling", "🎨 Inpainting", "📊 Batch Processing", "🎯 Gradio Interface", "ℹ️ About"
    ])
    
    with tab1:
        upscaling_interface(selected_device, quality, enable_enhancement, enable_benchmark, tile_size)
    
    with tab2:
        inpainting_interface(selected_device, quality, enable_enhancement, enable_benchmark)
    
    with tab3:
        batch_processing_interface(selected_device, quality, tile_size)
    
    with tab3_alt:
        gradio_interface()
    
    with tab4:
        about_section()

def upscaling_interface(device, quality, enable_enhancement, enable_benchmark, tile_size):
    """Upscaling interface"""
    st.markdown('<h2 class="sub-header">🚀 AI Image Upscaling</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image to upscale",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload an image file to upscale"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            with st.expander("Image Information"):
                st.write(f"Format: {image.format}")
                st.write(f"Size: {image.size}")
                st.write(f"Mode: {image.mode}")
                st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.subheader("Settings")
        
        # Model selection
        model_options = ['esrgan', 'realesrgan', 'edsr']
        selected_model = st.selectbox("Upscaling Model", model_options, index=0)
        
        # Scale factor
        scale_options = [2, 4, 8]
        scale_factor = st.selectbox("Scale Factor", scale_options, index=1)
        
        # Processing button
        if st.button("🚀 Start Upscaling", type="primary"):
            if uploaded_file is not None:
                process_upscaling(
                    uploaded_file, selected_model, scale_factor, 
                    device, quality, enable_enhancement, enable_benchmark, tile_size
                )
            else:
                st.error("Please upload an image first!")

def inpainting_interface(device, quality, enable_enhancement, enable_benchmark):
    """Inpainting interface"""
    st.markdown('<h2 class="sub-header">🎨 AI Image Inpainting</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Input Image")
        input_file = st.file_uploader(
            "Choose an image to inpaint",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            key="inpaint_input"
        )
        
        if input_file is not None:
            image = Image.open(input_file)
            st.image(image, caption="Input Image", use_column_width=True)
    
    with col2:
        st.subheader("Mask")
        mask_option = st.radio("Mask Source", ["Upload Mask", "Interactive Brush"])
        
        if mask_option == "Upload Mask":
            mask_file = st.file_uploader(
                "Choose a mask image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="mask_input"
            )
            
            if mask_file is not None:
                mask = Image.open(mask_file)
                st.image(mask, caption="Mask", use_column_width=True)
        else:
            brush_size = st.slider("Brush Size", 10, 100, 30)
            st.info("Interactive brush will be applied to center of image")
    
    with col3:
        st.subheader("Settings")
        
        # Model selection
        model_options = ['lama', 'transformer']
        selected_model = st.selectbox("Inpainting Model", model_options, index=0, key="inpaint_model")
        
        # Processing button
        if st.button("🎨 Start Inpainting", type="primary"):
            if input_file is not None:
                if mask_option == "Upload Mask" and mask_file is not None:
                    process_inpainting(
                        input_file, mask_file, selected_model,
                        device, quality, enable_enhancement, enable_benchmark
                    )
                elif mask_option == "Interactive Brush":
                    process_inpainting_brush(
                        input_file, selected_model, brush_size,
                        device, quality, enable_enhancement, enable_benchmark
                    )
                else:
                    st.error("Please provide both input image and mask!")
            else:
                st.error("Please upload an input image first!")

def batch_processing_interface(device, quality, tile_size):
    """Batch processing interface"""
    st.markdown('<h2 class="sub-header">📊 Batch Processing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Directory")
        input_dir = st.text_input("Input Directory Path", placeholder="/path/to/input/images")
        
        if st.button("📁 Browse Input Directory"):
            st.info("Please enter the directory path manually for security reasons")
        
        st.subheader("Output Directory")
        output_dir = st.text_input("Output Directory Path", placeholder="/path/to/output/images")
        
        if st.button("📁 Browse Output Directory"):
            st.info("Please enter the directory path manually for security reasons")
    
    with col2:
        st.subheader("Processing Options")
        
        # Mode selection
        mode = st.radio("Processing Mode", ["Upscaling", "Inpainting"])
        
        if mode == "Upscaling":
            model = st.selectbox("Model", ['esrgan', 'realesrgan', 'edsr'])
            scale = st.selectbox("Scale Factor", [2, 4, 8])
        else:
            model = st.selectbox("Model", ['lama', 'transformer'])
            mask_dir = st.text_input("Mask Directory Path", placeholder="/path/to/masks")
        
        # Start batch processing
        if st.button("🚀 Start Batch Processing", type="primary"):
            if input_dir and output_dir:
                if os.path.exists(input_dir):
                    start_batch_processing(
                        input_dir, output_dir, mode, model, 
                        scale if mode == "Upscaling" else None,
                        mask_dir if mode == "Inpainting" else None,
                        device, quality, tile_size
                    )
                else:
                    st.error("Input directory does not exist!")
            else:
                st.error("Please provide both input and output directories!")

def gradio_interface():
    """Gradio interface for advanced users"""
    st.markdown('<h2 class="sub-header">🎯 Gradio Interface</h2>', unsafe_allow_html=True)
    
    st.info("""
    The Gradio interface provides a more interactive experience with real-time processing.
    Click the button below to launch the Gradio interface in a new tab.
    """)
    
    if st.button("🚀 Launch Gradio Interface"):
        # Create and launch Gradio interface
        interface = create_gradio_interface()
        st.success("Gradio interface launched! Check the new tab.")
        
        # Display interface
        with st.expander("Gradio Interface"):
            st.components.v1.html(interface, height=800)

def about_section():
    """About section with project information"""
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🚀 AI Image Upscaling & Inpainting
        
        This is a comprehensive AI-powered image processing tool that combines state-of-the-art 
        upscaling and inpainting models for professional image enhancement and restoration.
        
        ### ✨ Features
        
        - **Advanced Upscaling**: ESRGAN, Real-ESRGAN, and EDSR models
        - **Smart Inpainting**: LaMa and Transformer-based models
        - **Batch Processing**: Process multiple images efficiently
        - **Quality Metrics**: PSNR and SSIM calculations
        - **Image Enhancement**: Automatic brightness, contrast, and sharpness adjustment
        - **Multiple Formats**: Support for JPG, PNG, BMP, TIFF, and WebP
        - **GPU Acceleration**: CUDA support for faster processing
        
        ### 🛠️ Technical Details
        
        - Built with PyTorch for deep learning models
        - Streamlit for the main web interface
        - Gradio for interactive processing
        - Attention mechanisms for better feature extraction
        - Memory-efficient processing with tiling support
        
        ### 📊 Performance
        
        - Optimized for both CPU and GPU processing
        - Memory-efficient attention mechanisms
        - Support for large image processing
        - Real-time preview and comparison
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 System Requirements
        
        - Python 3.8+
        - PyTorch 2.1+
        - CUDA 11.8+ (optional)
        - 8GB+ RAM recommended
        - GPU with 4GB+ VRAM (optional)
        
        ### 📥 Installation
        
        ```bash
        pip install -r requirements.txt
        ```
        
        ### 🚀 Usage
        
        ```bash
        # Command line
        python main.py --input image.jpg --output upscaled.jpg --mode upscale
        
        # Web interface
        streamlit run web_app.py
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Streamlit, PyTorch, and modern AI models
    </div>
    """, unsafe_allow_html=True)

def process_upscaling(uploaded_file, model, scale, device, quality, enhance, benchmark, tile_size):
    """Process upscaling with progress tracking"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing upscaling processor...")
        progress_bar.progress(10)
        
        # Initialize processor
        processor = UpscalingProcessor(
            model_type=model,
            device=device,
            scale_factor=scale
        )
        
        status_text.text("Processing image...")
        progress_bar.progress(30)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process image
        result = processor.upscale_image(tmp_path, tile_size=tile_size)
        progress_bar.progress(70)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        status_text.text("Applying enhancements..." if enhance else "Finalizing...")
        progress_bar.progress(85)
        
        # Apply enhancements if requested
        if enhance:
            result = enhance_image(
                Image.fromarray(result),
                brightness=1.1,
                contrast=1.05,
                sharpness=1.2,
                saturation=1.1
            )
            result = np.array(result)
        
        # Save result
        output_path = f"upscaled_{uploaded_file.name}"
        save_image(Image.fromarray(result), output_path, quality=quality)
        
        progress_bar.progress(100)
        status_text.text("Upscaling completed!")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(Image.open(uploaded_file), use_column_width=True)
        
        with col2:
            st.subheader("Upscaled")
            st.image(result, use_column_width=True)
        
        # Download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Upscaled Image",
                data=file.read(),
                file_name=output_path,
                mime="image/png"
            )
        
        # Quality metrics
        if st.checkbox("Show Quality Metrics"):
            original = load_image(uploaded_file)
            psnr = calculate_psnr(original, result)
            ssim = calculate_ssim(original, result)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("PSNR", f"{psnr:.2f} dB")
            with col2:
                st.metric("SSIM", f"{ssim:.4f}")
        
        # Benchmark results
        if benchmark:
            with st.expander("Benchmark Results"):
                benchmark_results = benchmark_model(processor.model, device=processor.device)
                st.json(benchmark_results)
    
    except Exception as e:
        st.error(f"Error during upscaling: {e}")
        if st.checkbox("Show detailed error"):
            st.exception(e)
    finally:
        progress_bar.empty()
        status_text.empty()

def process_inpainting(input_file, mask_file, model, device, quality, enhance, benchmark):
    """Process inpainting with progress tracking"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing inpainting processor...")
        progress_bar.progress(10)
        
        # Initialize processor
        processor = InpaintingProcessor(
            model_type=model,
            device=device
        )
        
        status_text.text("Processing image...")
        progress_bar.progress(30)
        
        # Save files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
            tmp_input.write(input_file.getvalue())
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
            tmp_mask.write(mask_file.getvalue())
            mask_path = tmp_mask.name
        
        # Process image
        result = processor.inpaint_image(input_path, mask_path)
        progress_bar.progress(70)
        
        # Clean up temp files
        os.unlink(input_path)
        os.unlink(mask_path)
        
        status_text.text("Applying enhancements..." if enhance else "Finalizing...")
        progress_bar.progress(85)
        
        # Apply enhancements if requested
        if enhance:
            result = enhance_image(
                Image.fromarray(result),
                brightness=1.05,
                contrast=1.1,
                sharpness=1.1
            )
            result = np.array(result)
        
        # Save result
        output_path = f"inpainted_{input_file.name}"
        save_image(Image.fromarray(result), output_path, quality=quality)
        
        progress_bar.progress(100)
        status_text.text("Inpainting completed!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original")
            st.image(Image.open(input_file), use_column_width=True)
        
        with col2:
            st.subheader("Mask")
            st.image(Image.open(mask_file), use_column_width=True)
        
        with col3:
            st.subheader("Inpainted")
            st.image(result, use_column_width=True)
        
        # Download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Inpainted Image",
                data=file.read(),
                file_name=output_path,
                mime="image/png"
            )
        
        # Benchmark results
        if benchmark:
            with st.expander("Benchmark Results"):
                benchmark_results = benchmark_model(processor.model, device=processor.device)
                st.json(benchmark_results)
    
    except Exception as e:
        st.error(f"Error during inpainting: {e}")
        if st.checkbox("Show detailed error"):
            st.exception(e)
    finally:
        progress_bar.empty()
        status_text.empty()

def process_inpainting_brush(input_file, model, brush_size, device, quality, enhance, benchmark):
    """Process inpainting with brush tool"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing inpainting processor...")
        progress_bar.progress(10)
        
        # Initialize processor
        processor = InpaintingProcessor(
            model_type=model,
            device=device
        )
        
        status_text.text("Processing image with brush tool...")
        progress_bar.progress(30)
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(input_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process image
        result, mask = processor.inpaint_with_brush(tmp_path, brush_size=brush_size)
        progress_bar.progress(70)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        status_text.text("Applying enhancements..." if enhance else "Finalizing...")
        progress_bar.progress(85)
        
        # Apply enhancements if requested
        if enhance:
            result = enhance_image(
                Image.fromarray(result),
                brightness=1.05,
                contrast=1.1,
                sharpness=1.1
            )
            result = np.array(result)
        
        # Save results
        output_path = f"inpainted_{input_file.name}"
        mask_path = f"mask_{input_file.name}"
        
        save_image(Image.fromarray(result), output_path, quality=quality)
        save_image(mask, mask_path)
        
        progress_bar.progress(100)
        status_text.text("Inpainting completed!")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original")
            st.image(Image.open(input_file), use_column_width=True)
        
        with col2:
            st.subheader("Generated Mask")
            st.image(mask, use_column_width=True)
        
        with col3:
            st.subheader("Inpainted")
            st.image(result, use_column_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Inpainted Image",
                    data=file.read(),
                    file_name=output_path,
                    mime="image/png"
                )
        
        with col2:
            with open(mask_path, "rb") as file:
                st.download_button(
                    label="📥 Download Mask",
                    data=file.read(),
                    file_name=mask_path,
                    mime="image/png"
                )
    
    except Exception as e:
        st.error(f"Error during inpainting: {e}")
        if st.checkbox("Show detailed error"):
            st.exception(e)
    finally:
        progress_bar.empty()
        status_text.empty()

def start_batch_processing(input_dir, output_dir, mode, model, scale, mask_dir, device, quality, tile_size):
    """Start batch processing"""
    
    if not os.path.exists(input_dir):
        st.error("Input directory does not exist!")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        st.error("No image files found in input directory!")
        return
    
    st.success(f"Found {len(image_files)} images to process")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        if mode == "Upscaling":
            processor = UpscalingProcessor(
                model_type=model,
                device=device,
                scale_factor=scale
            )
            
            for i, filename in enumerate(image_files):
                status_text.text(f"Processing {i+1}/{len(image_files)}: {filename}")
                
                input_path = os.path.join(input_dir, filename)
                output_filename = f"upscaled_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                try:
                    processor.upscale_image(input_path, output_path, tile_size=tile_size)
                except Exception as e:
                    st.warning(f"Error processing {filename}: {e}")
                
                progress_bar.progress((i + 1) / len(image_files))
        
        else:  # Inpainting
            processor = InpaintingProcessor(
                model_type=model,
                device=device
            )
            
            for i, filename in enumerate(image_files):
                status_text.text(f"Processing {i+1}/{len(image_files)}: {filename}")
                
                input_path = os.path.join(input_dir, filename)
                mask_filename = f"mask_{filename}"
                mask_path = os.path.join(mask_dir, mask_filename)
                output_filename = f"inpainted_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                if os.path.exists(mask_path):
                    try:
                        processor.inpaint_image(input_path, mask_path, output_path)
                    except Exception as e:
                        st.warning(f"Error processing {filename}: {e}")
                else:
                    st.warning(f"No mask found for {filename}")
                
                progress_bar.progress((i + 1) / len(image_files))
        
        status_text.text("Batch processing completed!")
        st.success(f"Successfully processed {len(image_files)} images!")
        
    except Exception as e:
        st.error(f"Error during batch processing: {e}")
    finally:
        progress_bar.empty()
        status_text.empty()

def create_gradio_interface():
    """Create Gradio interface"""
    
    def upscale_gradio(image, model, scale, device):
        """Gradio upscaling function"""
        try:
            processor = UpscalingProcessor(
                model_type=model,
                device=device,
                scale_factor=scale
            )
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Process
            result = processor.upscale_image(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return result
        except Exception as e:
            return None
    
    def inpaint_gradio(image, mask, model, device):
        """Gradio inpainting function"""
        try:
            processor = InpaintingProcessor(
                model_type=model,
                device=device
            )
            
            # Convert to PIL Images
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                image.save(tmp_img.name)
                img_path = tmp_img.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_mask:
                mask.save(tmp_mask.name)
                mask_path = tmp_mask.name
            
            # Process
            result = processor.inpaint_image(img_path, mask_path)
            
            # Clean up
            os.unlink(img_path)
            os.unlink(mask_path)
            
            return result
        except Exception as e:
            return None
    
    # Create Gradio interface
    with gr.Blocks(title="AI Image Processing") as interface:
        gr.Markdown("# 🖼️ AI Image Upscaling & Inpainting")
        
        with gr.Tabs():
            with gr.TabItem("Upscaling"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Input Image")
                        model_select = gr.Dropdown(
                            choices=['esrgan', 'realesrgan', 'edsr'],
                            value='esrgan',
                            label="Model"
                        )
                        scale_select = gr.Dropdown(
                            choices=[2, 4, 8],
                            value=4,
                            label="Scale Factor"
                        )
                        device_select = gr.Dropdown(
                            choices=['auto', 'cpu', 'cuda'],
                            value='auto',
                            label="Device"
                        )
                        upscale_btn = gr.Button("🚀 Upscale", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Upscaled Image")
                
                upscale_btn.click(
                    fn=upscale_gradio,
                    inputs=[input_image, model_select, scale_select, device_select],
                    outputs=output_image
                )
            
            with gr.TabItem("Inpainting"):
                with gr.Row():
                    with gr.Column():
                        inpaint_input = gr.Image(label="Input Image")
                        inpaint_mask = gr.Image(label="Mask")
                        inpaint_model = gr.Dropdown(
                            choices=['lama', 'transformer'],
                            value='lama',
                            label="Model"
                        )
                        inpaint_device = gr.Dropdown(
                            choices=['auto', 'cpu', 'cuda'],
                            value='auto',
                            label="Device"
                        )
                        inpaint_btn = gr.Button("🎨 Inpaint", variant="primary")
                    
                    with gr.Column():
                        inpaint_output = gr.Image(label="Inpainted Image")
                
                inpaint_btn.click(
                    fn=inpaint_gradio,
                    inputs=[inpaint_input, inpaint_mask, inpaint_model, inpaint_device],
                    outputs=inpaint_output
                )
    
    return interface

if __name__ == "__main__":
    main()
