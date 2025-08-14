import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import os
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AttentionBlock(nn.Module):
    """Self-attention block for better feature extraction"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate Q, K, V
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention mechanism
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Enhanced residual block with attention and better normalization"""
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.GroupNorm(8, channels)  # Using GroupNorm instead of BatchNorm
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.GroupNorm(8, channels)
        self.attention = AttentionBlock(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out * 0.1 + residual
        return out

class UpsampleBlock(nn.Module):
    """Enhanced upsampling block with better interpolation"""
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
        self.conv_after = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        x = self.conv_after(x)
        return x

class ESRGANModel(nn.Module):
    """Enhanced Super-Resolution Generative Adversarial Network with latest improvements"""
    
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=23):
        super(ESRGANModel, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution with better initialization
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        
        # Residual blocks with attention
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        
        # Upsampling blocks
        self.upsampling_layers = nn.Sequential(
            UpsampleBlock(num_features, 2),
            UpsampleBlock(num_features, 2)
        )
        
        # Output convolution
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using modern initialization methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        residual = x
        out = self.conv_input(x)
        out = self.prelu(out)
        out = self.residual_layers(out)
        out = self.upsampling_layers(out)
        out = self.conv_output(out)
        return out + F.interpolate(residual, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

class RealESRGANModel(nn.Module):
    """Enhanced Real-ESRGAN for general image restoration"""
    
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=23):
        super(RealESRGANModel, self).__init__()
        self.scale_factor = scale_factor
        
        # Enhanced architecture with attention
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        
        # Residual blocks with attention
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        
        # Upsampling blocks
        self.upsampling_layers = nn.Sequential(
            UpsampleBlock(num_features, 2),
            UpsampleBlock(num_features, 2)
        )
        
        # Output convolution
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using modern initialization methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        residual = x
        out = self.conv_input(x)
        out = self.prelu(out)
        out = self.residual_layers(out)
        out = self.upsampling_layers(out)
        out = self.conv_output(out)
        return out + F.interpolate(residual, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

class EDSRModel(nn.Module):
    """Enhanced Deep Super-Resolution Network"""
    
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=16):
        super(EDSRModel, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )
        
        # Output convolution
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using modern initialization methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.residual_layers(out)
        out = out + residual
        out = self.upsampling(out)
        out = self.conv_output(out)
        return out

class UpscalingProcessor:
    """Enhanced processor for image upscaling with multiple models"""
    
    def __init__(self, model_type='esrgan', device='auto', scale_factor=4):
        self.model_type = model_type.lower()
        self.scale_factor = scale_factor
        
        # Device selection with better error handling
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # Enable memory efficient attention if available
                if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model based on type
        if self.model_type == 'esrgan':
            self.model = ESRGANModel(scale_factor=scale_factor)
        elif self.model_type == 'realesrgan':
            self.model = RealESRGANModel(scale_factor=scale_factor)
        elif self.model_type == 'edsr':
            self.model = EDSRModel(scale_factor=scale_factor)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Enhanced image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
        ])
    
    def upscale_image(self, image_path: str, output_path: str = None, 
                     tile_size: int = 512) -> np.ndarray:
        """Upscale image with tiling for large images"""
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Handle large images with tiling
        if max(image.size) > tile_size:
            return self._upscale_with_tiling(image, output_path, tile_size)
        else:
            return self._upscale_single(image, output_path)
    
    def _upscale_single(self, image: Image.Image, output_path: str = None) -> np.ndarray:
        """Upscale a single image"""
        with torch.no_grad():
            # Prepare input
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Process
            output_tensor = self.model(input_tensor)
            
            # Convert back to PIL
            output_tensor = output_tensor.squeeze(0).cpu()
            output_image = self.reverse_transform(output_tensor)
            
            # Save if output path provided
            if output_path:
                output_image.save(output_path, quality=95, optimize=True)
            
            return np.array(output_image)
    
    def _upscale_with_tiling(self, image: Image.Image, output_path: str = None, 
                            tile_size: int = 512) -> np.ndarray:
        """Upscale large image using tiling approach"""
        width, height = image.size
        new_width = width * self.scale_factor
        new_height = height * self.scale_factor
        
        # Create output image
        output_image = Image.new('RGB', (new_width, new_height))
        
        # Process tiles
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Extract tile
                tile = image.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
                
                # Upscale tile
                upscaled_tile = self._upscale_single(tile)
                upscaled_tile = Image.fromarray(upscaled_tile)
                
                # Paste into output
                output_x = x * self.scale_factor
                output_y = y * self.scale_factor
                output_image.paste(upscaled_tile, (output_x, output_y))
        
        # Save if output path provided
        if output_path:
            output_image.save(output_path, quality=95, optimize=True)
        
        return np.array(output_image)
    
    def batch_upscale(self, input_dir: str, output_dir: str, 
                      file_extensions: list = None) -> None:
        """Batch upscale images in a directory"""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process files
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in file_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"upscaled_{filename}")
                
                try:
                    self.upscale_image(input_path, output_path)
                    print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'scale_factor': self.scale_factor,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'model_class': self.model.__class__.__name__
        } 