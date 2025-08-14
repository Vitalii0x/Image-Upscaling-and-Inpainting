import torch
import torch.nn as nn
import psutil
import os
import platform
from typing import Dict, Any, Optional, List
import warnings
import time

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information for AI processing"""
    info = {
        'system': platform.system(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'memory_percent': psutil.virtual_memory().percent
    }
    
    # PyTorch specific info
    info.update({
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
    })
    
    # GPU info if available
    if torch.cuda.is_available():
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_info[f'gpu_{i}'] = {
                'name': gpu_name,
                'memory_total': gpu_memory,
                'memory_free': torch.cuda.memory_reserved(i),
                'memory_used': torch.cuda.memory_allocated(i)
            }
        info['gpus'] = gpu_info
    
    return info

def get_optimal_device(device_preference: str = 'auto') -> torch.device:
    """Get the optimal device for AI processing with smart fallback"""
    
    if device_preference == 'auto':
        if torch.cuda.is_available():
            # Check if CUDA memory is sufficient
            try:
                # Try to allocate a small tensor to test memory
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                return torch.device('cuda')
            except RuntimeError:
                print("CUDA memory insufficient, falling back to CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device('cpu')
    
    elif device_preference == 'cpu':
        return torch.device('cpu')
    
    elif device_preference == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    
    else:
        print(f"Device {device_preference} not available, falling back to CPU")
        return torch.device('cpu')

def optimize_for_inference(model: nn.Module, device: torch.device) -> nn.Module:
    """Optimize model for inference with various techniques"""
    
    # Move to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    # Enable optimizations if available
    if device.type == 'cuda':
        # Enable cuDNN benchmarking for optimal performance
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Use torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    
    return model

def create_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    """Create a comprehensive model summary with memory usage estimation"""
    
    def register_hook(module):
        def hook(module, input, output):
            module.register_buffer('output_shape', torch.tensor(output.shape))
        hooks.append(module.register_forward_hook(hook))
    
    hooks = []
    model.apply(register_hook)
    
    # Create dummy input
    try:
        dummy_input = torch.randn(input_size)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"Warning: Could not run forward pass: {e}")
        return {}
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate parameters and memory
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate activation memory (rough approximation)
    activation_size = 0
    for module in model.modules():
        if hasattr(module, 'output_shape'):
            activation_size += module.output_shape.numel() * 4  # Assuming float32
    
    summary = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'parameter_memory_mb': param_size / (1024 * 1024),
        'buffer_memory_mb': buffer_size / (1024 * 1024),
        'estimated_activation_memory_mb': activation_size / (1024 * 1024),
        'total_memory_mb': (param_size + buffer_size + activation_size) / (1024 * 1024),
        'model_depth': len(list(model.modules())),
        'input_size': input_size,
        'output_size': tuple(model.output_shape.tolist()) if hasattr(model, 'output_shape') else None
    }
    
    return summary

def save_model_checkpoint(model: nn.Module, filepath: str, 
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         epoch: int = 0, loss: float = 0.0,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint with comprehensive metadata"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {}),
        'timestamp': torch.tensor(torch.cuda.Event() if torch.cuda.is_available() else time.time())
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to {filepath}")

def load_model_checkpoint(model: nn.Module, filepath: str,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load model checkpoint with error handling and device management"""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model state loaded successfully from {filepath}")
    except Exception as e:
        print(f"Warning: Could not load model state: {e}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    return checkpoint

def get_model_memory_usage(model: nn.Module, device: torch.device) -> Dict[str, float]:
    """Get detailed memory usage information for a model"""
    
    if device.type == 'cuda':
        # GPU memory usage
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)    # MB
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated,
            'free_mb': torch.cuda.get_device_properties(device).total_memory / (1024 * 1024) - reserved
        }
    else:
        # CPU memory usage (rough estimation)
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }

def benchmark_model(model: nn.Module, input_size: tuple = (1, 3, 224, 224), 
                   num_runs: int = 100, device: torch.device = None) -> Dict[str, float]:
    """Benchmark model performance with timing and memory measurements"""
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark runs
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = model(dummy_input)
                end_time.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)  # milliseconds
            else:
                start_time = torch.cuda.Event() if torch.cuda.is_available() else time.time()
                _ = model(dummy_input)
                end_time = torch.cuda.Event() if torch.cuda.is_available() else time.time()
                
                if torch.cuda.is_available():
                    elapsed_time = start_time.elapsed_time(end_time)
                else:
                    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            times.append(elapsed_time)
            
            # Memory usage
            memory = get_model_memory_usage(model, device)
            memory_usage.append(memory.get('allocated_mb', 0))
    
    # Calculate statistics
    times = torch.tensor(times)
    memory_usage = torch.tensor(memory_usage)
    
    benchmark_results = {
        'mean_time_ms': float(times.mean()),
        'std_time_ms': float(times.std()),
        'min_time_ms': float(times.min()),
        'max_time_ms': float(times.max()),
        'mean_memory_mb': float(memory_usage.mean()),
        'max_memory_mb': float(memory_usage.max()),
        'throughput_fps': 1000.0 / float(times.mean()) if float(times.mean()) > 0 else 0
    }
    
    return benchmark_results

def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> None:
    """Print a formatted model summary"""
    
    summary = create_model_summary(model, input_size)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {summary['total_parameters'] - summary['trainable_parameters']:,}")
    print(f"Model Depth: {summary['model_depth']} layers")
    print(f"Input Size: {summary['input_size']}")
    if summary['output_size']:
        print(f"Output Size: {summary['output_size']}")
    print("-" * 60)
    print(f"Parameter Memory: {summary['parameter_memory_mb']:.2f} MB")
    print(f"Buffer Memory: {summary['buffer_memory_mb']:.2f} MB")
    print(f"Estimated Activation Memory: {summary['estimated_activation_memory_mb']:.2f} MB")
    print(f"Total Estimated Memory: {summary['total_memory_mb']:.2f} MB")
    print("=" * 60) 