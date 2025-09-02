import psutil
import gc
import torch
import logging
import os
from functools import wraps
import numpy as np


class MemoryMonitor:
    """Utility class for monitoring memory usage throughout the pipeline"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.logger = logging.getLogger(__name__)
        
    def get_memory_info(self):
        """Get current memory usage information"""
        # System memory
        vm = psutil.virtual_memory()
        
        # Process memory
        process_mem = self.process.memory_info()
        
        # GPU memory (if available)
        gpu_mem = {}
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem[f'gpu_{i}'] = {
                        'allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                        'reserved': torch.cuda.memory_reserved(i) / 1024**3,    # GB
                        'max_allocated': torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                    }
        except Exception as e:
            self.logger.warning(f"Could not get GPU memory info: {e}")
            
        return {
            'system': {
                'total': vm.total / 1024**3,      # GB
                'available': vm.available / 1024**3,  # GB
                'percent': vm.percent,
                'used': vm.used / 1024**3         # GB
            },
            'process': {
                'rss': process_mem.rss / 1024**3,     # GB (Physical memory)
                'vms': process_mem.vms / 1024**3,     # GB (Virtual memory)
            },
            'gpu': gpu_mem
        }
    
    def log_memory_usage(self, stage_name=""):
        """Log current memory usage"""
        mem_info = self.get_memory_info()
        
        # Update peak memory
        current_rss = mem_info['process']['rss']
        if current_rss > self.peak_memory:
            self.peak_memory = current_rss
            
        self.logger.info(f"=== Memory Usage - {stage_name} ===")
        self.logger.info(f"System: {mem_info['system']['used']:.2f}GB / {mem_info['system']['total']:.2f}GB ({mem_info['system']['percent']:.1f}%)")
        self.logger.info(f"Process RSS: {mem_info['process']['rss']:.2f}GB")
        self.logger.info(f"Process VMS: {mem_info['process']['vms']:.2f}GB")
        self.logger.info(f"Peak Memory: {self.peak_memory:.2f}GB")
        
        for gpu_id, gpu_info in mem_info['gpu'].items():
            self.logger.info(f"GPU {gpu_id}: {gpu_info['allocated']:.2f}GB allocated, {gpu_info['reserved']:.2f}GB reserved")
            
        return mem_info
    
    def estimate_array_memory(self, shape, dtype):
        """Estimate memory usage for an array"""
        dtype_sizes = {
            'uint8': 1, 'uint16': 2, 'uint32': 4, 'uint64': 8,
            'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
            'float32': 4, 'float64': 8
        }
        
        if isinstance(dtype, np.dtype):
            dtype = dtype.name
            
        bytes_per_element = dtype_sizes.get(dtype, 4)  # Default to 4 bytes
        total_elements = np.prod(shape)
        total_bytes = total_elements * bytes_per_element
        total_gb = total_bytes / 1024**3
        
        return total_gb
    
    def log_array_info(self, array, array_name="array"):
        """Log information about a specific array"""
        if array is not None:
            memory_gb = self.estimate_array_memory(array.shape, array.dtype)
            self.logger.info(f"{array_name}: shape={array.shape}, dtype={array.dtype}, memory≈{memory_gb:.2f}GB")
        else:
            self.logger.info(f"{array_name}: None")


def memory_profile(stage_name=""):
    """Decorator to profile memory usage of a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            monitor.log_memory_usage(f"Before {stage_name}")
            
            # Force garbage collection before
            gc.collect()
            
            result = func(*args, **kwargs)
            
            # Force garbage collection after
            gc.collect()
            
            monitor.log_memory_usage(f"After {stage_name}")
            return result
        return wrapper
    return decorator


# Global instance
memory_monitor = MemoryMonitor() 