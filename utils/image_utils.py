import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List
import os
from pathlib import Path
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None, 
               mode: str = 'RGB') -> Image.Image:
    """Enhanced image loading with size adjustment and error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert(mode)
        
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        raise RuntimeError(f"Error loading image {image_path}: {e}")

def save_image(image: Union[Image.Image, np.ndarray], output_path: str, 
               quality: int = 95, optimize: bool = True) -> None:
    """Enhanced image saving with quality control and format detection"""
    try:
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine format from extension
        file_ext = Path(output_path).suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg']:
            image.save(output_path, 'JPEG', quality=quality, optimize=optimize)
        elif file_ext == '.png':
            image.save(output_path, 'PNG', optimize=optimize)
        elif file_ext == '.webp':
            image.save(output_path, 'WEBP', quality=quality, lossless=False)
        elif file_ext == '.tiff':
            image.save(output_path, 'TIFF', compression='tiff_lzw')
        else:
            # Default to PNG for unknown extensions
            image.save(output_path, 'PNG', optimize=optimize)
            
    except Exception as e:
        raise RuntimeError(f"Error saving image to {output_path}: {e}")

def display_comparison(original: Union[str, Image.Image, np.ndarray], 
                      processed: Union[str, Image.Image, np.ndarray], 
                      titles: Tuple[str, str] = ('Original', 'Processed'),
                      figsize: Tuple[int, int] = (12, 6)) -> None:
    """Enhanced comparison display with better visualization"""
    
    # Load images if they're paths
    if isinstance(original, str):
        original = load_image(original)
    if isinstance(processed, str):
        processed = load_image(processed)
    
    # Convert numpy arrays to PIL if needed
    if isinstance(original, np.ndarray):
        original = Image.fromarray(original)
    if isinstance(processed, np.ndarray):
        processed = Image.fromarray(processed)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Display images
    axes[0].imshow(original)
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_psnr(original: Union[str, Image.Image, np.ndarray], 
                  processed: Union[str, Image.Image, np.ndarray]) -> float:
    """Calculate Peak Signal-to-Noise Ratio with enhanced accuracy"""
    
    # Load and convert images
    if isinstance(original, str):
        original = load_image(original)
    if isinstance(processed, str):
        processed = load_image(processed)
    
    # Convert to numpy arrays
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(processed, Image.Image):
        processed = np.array(processed)
    
    # Ensure same size
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert to float and normalize
    original = original.astype(np.float64) / 255.0
    processed = processed.astype(np.float64) / 255.0
    
    # Calculate MSE
    mse = np.mean((original - processed) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr

def calculate_ssim(original: Union[str, Image.Image, np.ndarray], 
                  processed: Union[str, Image.Image, np.ndarray]) -> float:
    """Calculate Structural Similarity Index with enhanced implementation"""
    
    # Load and convert images
    if isinstance(original, str):
        original = load_image(original)
    if isinstance(processed, str):
        processed = load_image(processed)
    
    # Convert to numpy arrays
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(processed, Image.Image):
        processed = np.array(processed)
    
    # Ensure same size
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale if RGB
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    # Calculate SSIM using scikit-image
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(original_gray, processed_gray, data_range=255)
    except ImportError:
        # Fallback to OpenCV implementation
        return _calculate_ssim_opencv(original_gray, processed_gray)

def _calculate_ssim_opencv(img1: np.ndarray, img2: np.ndarray) -> float:
    """Fallback SSIM calculation using OpenCV"""
    # Simple SSIM approximation
    mu1 = cv2.GaussianBlur(img1.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2.astype(np.float64), (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur((img1.astype(np.float64) ** 2), (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur((img2.astype(np.float64) ** 2), (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur((img1.astype(np.float64) * img2.astype(np.float64)), (11, 11), 1.5) - mu1_mu2
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return np.mean(ssim_map)

def enhance_image(image: Union[str, Image.Image, np.ndarray], 
                 brightness: float = 1.0, contrast: float = 1.0, 
                 sharpness: float = 1.0, saturation: float = 1.0) -> Image.Image:
    """Enhanced image enhancement with multiple parameters"""
    
    # Load image if it's a path
    if isinstance(image, str):
        image = load_image(image)
    
    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    return image

def apply_filters(image: Union[str, Image.Image, np.ndarray], 
                 filter_type: str = 'gaussian', **kwargs) -> Image.Image:
    """Apply various image filters with enhanced options"""
    
    # Load image if it's a path
    if isinstance(image, str):
        image = load_image(image)
    
    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply filters
    if filter_type == 'gaussian':
        radius = kwargs.get('radius', 2)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif filter_type == 'median':
        size = kwargs.get('size', 3)
        image = image.filter(ImageFilter.MedianFilter(size=size))
    elif filter_type == 'min':
        size = kwargs.get('size', 3)
        image = image.filter(ImageFilter.MinFilter(size=size))
    elif filter_type == 'max':
        size = kwargs.get('size', 3)
        image = image.filter(ImageFilter.MaxFilter(size=size))
    elif filter_type == 'mode':
        size = kwargs.get('size', 3)
        image = image.filter(ImageFilter.ModeFilter(size=size))
    elif filter_type == 'unsharp':
        radius = kwargs.get('radius', 2)
        percent = kwargs.get('percent', 150)
        threshold = kwargs.get('threshold', 3)
        image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return image

def create_image_grid(images: List[Union[str, Image.Image, np.ndarray]], 
                     titles: Optional[List[str]] = None, 
                     cols: int = 3, figsize: Tuple[int, int] = (15, 10)) -> None:
    """Create a grid display of multiple images"""
    
    # Load images if they're paths
    loaded_images = []
    for img in images:
        if isinstance(img, str):
            loaded_images.append(load_image(img))
        elif isinstance(img, np.ndarray):
            loaded_images.append(Image.fromarray(img))
        else:
            loaded_images.append(img)
    
    # Calculate grid dimensions
    n_images = len(loaded_images)
    rows = (n_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Display images
    for i, (img, ax) in enumerate(zip(loaded_images, axes.flat)):
        ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        axes.flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def get_image_info(image_path: str) -> dict:
    """Get comprehensive information about an image"""
    try:
        image = Image.open(image_path)
        
        info = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height,
            'file_size': os.path.getsize(image_path),
            'dpi': image.info.get('dpi', 'Unknown'),
            'compression': image.info.get('compression', 'Unknown'),
            'progressive': image.info.get('progressive', False),
            'transparency': image.info.get('transparency', 'Unknown')
        }
        
        return info
    except Exception as e:
        raise RuntimeError(f"Error getting image info for {image_path}: {e}")

def validate_image(image_path: str) -> bool:
    """Validate if an image file is valid and can be opened"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_supported_formats() -> List[str]:
    """Get list of supported image formats"""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'] 