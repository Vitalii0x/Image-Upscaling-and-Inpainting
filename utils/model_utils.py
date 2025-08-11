import torch
import os
import requests
from typing import Optional
import zipfile
import shutil

def get_device_info() -> dict:
    """Get information about available devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': 'cpu',
        'device_names': []
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = f'cuda:{torch.cuda.current_device()}'
        info['device_names'] = [torch.cuda.get_device_name(i) for i in range(info['device_count'])]
    
    return info

def select_device(device: str = 'auto') -> torch.device:
    """Select the best available device"""
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device)

def download_model(model_url: str, save_path: str, chunk_size: int = 8192) -> bool:
    """Download a model file from URL"""
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return False

def get_model_path(model_name: str, models_dir: str = 'models') -> Optional[str]:
    """Get the path to a specific model"""
    model_path = os.path.join(models_dir, model_name)
    if os.path.exists(model_path):
        return model_path
    return None

def save_model_state(model: torch.nn.Module, save_path: str) -> bool:
    """Save model state dictionary"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model_state(model: torch.nn.Module, load_path: str) -> bool:
    """Load model state dictionary"""
    try:
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location='cpu')
            model.load_state_dict(state_dict)
            return True
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def set_model_to_eval(model: torch.nn.Module) -> None:
    """Set model to evaluation mode and disable gradients"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

def enable_mixed_precision(model: torch.nn.Module) -> bool:
    """Enable mixed precision training if supported"""
    try:
        if hasattr(torch, 'amp') and torch.cuda.is_available():
            from torch.cuda.amp import autocast
            return True
        return False
    except:
        return False

def create_model_summary(model: torch.nn.Module) -> str:
    """Create a summary of the model architecture"""
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Parameters: {count_parameters(model):,}")
    summary.append(f"Size: {get_model_size_mb(model):.2f} MB")
    
    if torch.cuda.is_available():
        summary.append(f"CUDA Devices: {torch.cuda.device_count()}")
        summary.append(f"Current Device: {torch.cuda.current_device()}")
        summary.append(f"Device Name: {torch.cuda.get_device_name()}")
    
    return "\n".join(summary) 