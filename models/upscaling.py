import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional
import os

class ResidualBlock(nn.Module):
    """Residual block for ESRGAN"""
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * 0.1 + residual
        return out

class UpsampleBlock(nn.Module):
    """Upsampling block for ESRGAN"""
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class ESRGANModel(nn.Module):
    """Enhanced Super-Resolution Generative Adversarial Network"""
    
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=23):
        super(ESRGANModel, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        
        # Residual blocks
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        
        # Upsampling blocks
        self.upsampling_layers = nn.Sequential(
            UpsampleBlock(num_features, 2),
            UpsampleBlock(num_features, 2)
        )
        
        # Output convolution
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.conv_input(x)
        out = self.prelu(out)
        out = self.residual_layers(out)
        out = self.upsampling_layers(out)
        out = self.conv_output(out)
        return out + F.interpolate(residual, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

class RealESRGANModel(nn.Module):
    """Real-ESRGAN for general image restoration"""
    
    def __init__(self, scale_factor=4, num_channels=3, num_features=64, num_blocks=23):
        super(RealESRGANModel, self).__init__()
        self.scale_factor = scale_factor
        
        # Similar architecture to ESRGAN but with additional improvements
        self.conv_input = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        
        # Residual blocks with improved normalization
        self.residual_layers = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        
        # Upsampling blocks
        self.upsampling_layers = nn.Sequential(
            UpsampleBlock(num_features, 2),
            UpsampleBlock(num_features, 2)
        )
        
        # Output convolution
        self.conv_output = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.conv_input(x)
        out = self.prelu(out)
        out = self.residual_layers(out)
        out = self.upsampling_layers(out)
        out = self.conv_output(out)
        return out + F.interpolate(residual, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

class EDSRModel(nn.Module):
    """Enhanced Deep Residual Networks for Super Resolution"""
    
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
        
    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.residual_layers(out)
        out = out + residual
        out = self.upsampling(out)
        out = self.conv_output(out)
        return out

class UpscalingProcessor:
    """Main processor for image upscaling"""
    
    def __init__(self, model_type='esrgan', scale_factor=4, device='auto'):
        self.model_type = model_type.lower()
        self.scale_factor = scale_factor
        
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.reverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.ToPILImage()
        ])
    
    def _create_model(self):
        """Create the specified upscaling model"""
        if self.model_type == 'esrgan':
            return ESRGANModel(scale_factor=self.scale_factor)
        elif self.model_type == 'realesrgan':
            return RealESRGANModel(scale_factor=self.scale_factor)
        elif self.model_type == 'edsr':
            return EDSRModel(scale_factor=self.scale_factor)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def upscale_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """Upscale a single image"""
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Postprocess
        output_tensor = output_tensor.squeeze(0).cpu()
        output_image = self.reverse_transform(output_tensor)
        
        # Convert to numpy array
        output_array = np.array(output_image)
        
        # Save if output path provided
        if output_path:
            output_image.save(output_path)
            
        return output_array
    
    def upscale_batch(self, image_paths: list, output_dir: str = None) -> list:
        """Upscale multiple images"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            if output_dir:
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_upscaled{ext}")
            else:
                output_path = None
                
            result = self.upscale_image(image_path, output_path)
            results.append(result)
            
        return results 