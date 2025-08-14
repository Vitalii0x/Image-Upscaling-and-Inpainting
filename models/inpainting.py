import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from typing import Tuple, Optional, Union
import warnings
import os

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AttentionBlock(nn.Module):
    """Self-attention block for better feature extraction in inpainting"""
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
    """Enhanced residual block with attention for inpainting"""
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.gn1 = nn.GroupNorm(8, channels)  # Using GroupNorm instead of BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.gn2 = nn.GroupNorm(8, channels)
        self.attention = AttentionBlock(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.attention(out)
        out = out + residual
        return out

class LaMaInpainting(nn.Module):
    """Enhanced Large Mask Inpainting (LaMa) model with attention mechanisms"""
    
    def __init__(self, in_channels=4, out_channels=3, num_features=128):
        super(LaMaInpainting, self).__init__()
        
        # Encoder with attention
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, num_features, 3, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, num_features, 3, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features)
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.ConvTranspose2d(num_features, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.ConvTranspose2d(num_features, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, out_channels, 7, 1, 3),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using modern initialization methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class TransformerInpainting(nn.Module):
    """Transformer-based inpainting model for better long-range dependencies"""
    
    def __init__(self, in_channels=4, out_channels=3, num_features=128, num_heads=8):
        super(TransformerInpainting, self).__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=num_features,
                nhead=num_heads,
                dim_feedforward=num_features * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])
        
        # Output convolution
        self.conv_out = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using modern initialization methods"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Reshape for transformer (batch, height*width, channels)
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Reshape back (batch, channels, height, width)
        x = x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Output convolution
        x = self.conv_out(x)
        x = self.sigmoid(x)
        
        return x

class InpaintingProcessor:
    """Enhanced processor for image inpainting with multiple models"""
    
    def __init__(self, model_type='lama', device='auto'):
        self.model_type = model_type.lower()
        
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
        if self.model_type == 'lama':
            self.model = LaMaInpainting()
        elif self.model_type == 'transformer':
            self.model = TransformerInpainting()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Enhanced image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.reverse_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
    
    def create_mask_from_coordinates(self, image_size: Tuple[int, int], 
                                   mask_coords: list) -> Image.Image:
        """Create a mask from polygon coordinates"""
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(mask_coords, fill=255)
        return mask
    
    def create_rectangular_mask(self, image_size: Tuple[int, int], 
                               bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Create a rectangular mask from bounding box coordinates"""
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill=255)
        return mask
    
    def inpaint_image(self, image: Union[str, Image.Image], 
                     mask: Union[str, Image.Image, list, Tuple[int, int, int, int]], 
                     output_path: str = None) -> np.ndarray:
        """Enhanced inpainting with multiple mask input types"""
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Handle different mask types
        if isinstance(mask, list):
            mask = self.create_mask_from_coordinates(image.size, mask)
        elif isinstance(mask, tuple) and len(mask) == 4:
            mask = self.create_rectangular_mask(image.size, mask)
        elif isinstance(mask, str):
            mask = Image.open(mask).convert('L')
        
        # Ensure mask is the same size as image
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        
        # Prepare input (image + mask)
        image_tensor = self.transform(image).unsqueeze(0)
        mask_tensor = self.transform(mask).unsqueeze(0)
        
        # Combine image and mask
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=1).to(self.device)
        
        # Process with model
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Convert back to PIL
        output_tensor = output_tensor.squeeze(0).cpu()
        output_image = self.reverse_transform(output_tensor)
        
        # Save if output path provided
        if output_path:
            output_image.save(output_path, quality=95, optimize=True)
        
        return np.array(output_image)
    
    def inpaint_with_brush(self, image: Union[str, Image.Image], 
                          brush_size: int = 20, 
                          output_path: str = None) -> Tuple[np.ndarray, Image.Image]:
        """Interactive inpainting with brush tool simulation"""
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Create a simple brush mask (center of image)
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        
        # Create circular brush mask
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw circle for brush
        left = center_x - brush_size // 2
        top = center_y - brush_size // 2
        right = center_x + brush_size // 2
        bottom = center_y + brush_size // 2
        
        draw.ellipse([left, top, right, bottom], fill=255)
        
        # Perform inpainting
        result = self.inpaint_image(image, mask, output_path)
        
        return result, mask
    
    def batch_inpaint(self, input_dir: str, mask_dir: str, output_dir: str,
                      file_extensions: list = None) -> None:
        """Batch inpainting for multiple images"""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process files
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in file_extensions):
                input_path = os.path.join(input_dir, filename)
                
                # Look for corresponding mask
                mask_filename = f"mask_{filename}"
                mask_path = os.path.join(mask_dir, mask_filename)
                
                if os.path.exists(mask_path):
                    output_path = os.path.join(output_dir, f"inpainted_{filename}")
                    
                    try:
                        self.inpaint_image(input_path, mask_path, output_path)
                        print(f"Processed: {filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                else:
                    print(f"No mask found for {filename}")
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'model_class': self.model.__class__.__name__
        } 