import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from typing import Tuple, Optional, Union
import os

class LaMaInpainting(nn.Module):
    """Large Mask Inpainting (LaMa) model for irregular masks"""
    
    def __init__(self, in_channels=4, out_channels=3, num_features=128):
        super(LaMaInpainting, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features, num_features, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 7, 1, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class InpaintingProcessor:
    """Main processor for image inpainting"""
    
    def __init__(self, model_type='lama', device='auto'):
        self.model_type = model_type.lower()
        
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
            
        # Initialize model
        self.model = LaMaInpainting()
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
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
    
    def inpaint_image(self, image: Union[str, Image.Image], 
                     mask: Union[str, Image.Image, list], 
                     output_path: str = None) -> np.ndarray:
        """Inpaint an image using LaMa model"""
        
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Handle mask creation
        if isinstance(mask, list):
            mask = self.create_mask_from_coordinates(image.size, mask)
        elif isinstance(mask, str):
            mask = Image.open(mask).convert('L')
        
        # Prepare input (image + mask)
        image_tensor = self.transform(image).unsqueeze(0)
        mask_tensor = self.transform(mask).unsqueeze(0)
        
        # Combine image and mask
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=1).to(self.device)
        
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