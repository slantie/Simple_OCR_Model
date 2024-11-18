import torch
import sys
import platform
from datetime import datetime

def test_gpu():
    """
    Test GPU availability and perform basic operations to verify functionality.
    Returns detailed information about the system and GPU configuration.
    """
    # System information
    system_info = {
        'Python Version': sys.version,
        'PyTorch Version': torch.__version__,
        'Operating System': platform.platform(),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # GPU availability check
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        # Get device properties
        device_properties = {
            'Device Count': device_count,
            'Current Device ID': current_device,
            'Device Name': device_name,
            'Device Capability': torch.cuda.get_device_capability(current_device),
            'Total Memory (GB)': round(torch.cuda.get_device_properties(current_device).total_memory / (1024**3), 2)
        }
        
        # Perform basic GPU operations to verify functionality
        try:
            # Create tensors on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Record start time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            # Perform matrix multiplication
            z = torch.matmul(x, y)
            end_time.record()
            
            # Synchronize CUDA events
            torch.cuda.synchronize()
            
            # Calculate elapsed time
            elapsed_time = start_time.elapsed_time(end_time)
            
            operation_results = {
                'Status': 'Success',
                'Operation': 'Matrix Multiplication (1000x1000)',
                'Execution Time (ms)': round(elapsed_time, 2)
            }
            
        except Exception as e:
            operation_results = {
                'Status': 'Failed',
                'Error': str(e)
            }
            
    else:
        device_properties = None
        operation_results = None
    
    return {
        'System Information': system_info,
        'CUDA Available': cuda_available,
        'Device Properties': device_properties,
        'Operation Test': operation_results
    }

def print_results(results):
    """
    Print the test results in a formatted manner.
    """
    print("\n===== GPU Test Results =====")
    print("\nSystem Information:")
    for key, value in results['System Information'].items():
        print(f"{key}: {value}")
    
    print(f"\nCUDA Available: {results['CUDA Available']}")
    
    if results['Device Properties']:
        print("\nDevice Properties:")
        for key, value in results['Device Properties'].items():
            print(f"{key}: {value}")
        
        print("\nOperation Test:")
        for key, value in results['Operation Test'].items():
            print(f"{key}: {value}")
    else:
        print("\nNo GPU detected or CUDA is not available")

if __name__ == "__main__":
    results = test_gpu()
    print_results(results)