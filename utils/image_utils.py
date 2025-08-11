import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
from typing import Tuple, Union, List
import matplotlib.pyplot as plt

def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Try PIL first
    try:
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    except:
        # Fallback to OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(image: Union[np.ndarray, Image.Image], output_path: str) -> None:
    """Save an image to file"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if isinstance(image, np.ndarray):
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
    elif isinstance(image, Image.Image):
        image.save(output_path)
    else:
        raise TypeError("Image must be numpy array or PIL Image")

def resize_image(image: Union[np.ndarray, Image.Image], 
                target_size: Tuple[int, int], 
                method: str = 'bilinear') -> np.ndarray:
    """Resize image to target size"""
    if isinstance(image, np.ndarray):
        # OpenCV resize
        if method == 'bilinear':
            method_cv = cv2.INTER_LINEAR
        elif method == 'bicubic':
            method_cv = cv2.INTER_CUBIC
        elif method == 'nearest':
            method_cv = cv2.INTER_NEAREST
        else:
            method_cv = cv2.INTER_LINEAR
            
        resized = cv2.resize(image, target_size, interpolation=method_cv)
        return resized
    elif isinstance(image, Image.Image):
        # PIL resize
        if method == 'bilinear':
            method_pil = Image.BILINEAR
        elif method == 'bicubic':
            method_pil = Image.BICUBIC
        elif method == 'nearest':
            method_pil = Image.NEAREST
        else:
            method_pil = Image.BILINEAR
            
        resized = image.resize(target_size, method_pil)
        return np.array(resized)
    else:
        raise TypeError("Image must be numpy array or PIL Image")

def create_mask(image_size: Tuple[int, int], 
               mask_type: str = 'rectangle',
               **kwargs) -> np.ndarray:
    """Create different types of masks"""
    
    if mask_type == 'rectangle':
        x = kwargs.get('x', 100)
        y = kwargs.get('y', 100)
        width = kwargs.get('width', 200)
        height = kwargs.get('height', 200)
        
        mask = np.zeros(image_size, dtype=np.uint8)
        mask[y:y+height, x:x+width] = 255
        return mask
        
    elif mask_type == 'circle':
        center_x = kwargs.get('center_x', image_size[1] // 2)
        center_y = kwargs.get('center_y', image_size[0] // 2)
        radius = kwargs.get('radius', 100)
        
        mask = np.zeros(image_size, dtype=np.uint8)
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_area] = 255
        return mask
        
    elif mask_type == 'polygon':
        points = kwargs.get('points', [(100, 100), (200, 100), (150, 200)])
        
        mask = np.zeros(image_size, dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        return mask
        
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

def create_mask_from_coordinates(image_size: Tuple[int, int], 
                               coordinates: List[Tuple[int, int]]) -> np.ndarray:
    """Create mask from polygon coordinates"""
    mask = np.zeros(image_size, dtype=np.uint8)
    points_array = np.array(coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    return mask

def display_comparison(original: np.ndarray, 
                      processed: np.ndarray, 
                      title: str = "Image Comparison") -> None:
    """Display original and processed images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(processed)
    axes[1].set_title("Processed")
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Structural Similarity Index"""
    from skimage.metrics import structural_similarity as ssim
    
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    return ssim(original_gray, processed_gray)

def apply_gaussian_noise(image: np.ndarray, std: float = 25.0) -> np.ndarray:
    """Apply Gaussian noise to image for testing"""
    noise = np.random.normal(0, std, image.shape).astype(np.float64)
    noisy_image = image.astype(np.float64) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to image for testing"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) 