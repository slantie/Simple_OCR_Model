import torch
import torchvision
import numpy as np

def test_cuda_compatibility():
    print("=== System Information ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA computation
        print("\n=== CUDA Computation Test ===")
        try:
            # Create tensors on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            # Perform matrix multiplication
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            z = torch.matmul(x, y)
            end.record()
            
            # Synchronize CUDA
            torch.cuda.synchronize()
            
            print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
            print("CUDA computation test passed successfully!")
            
        except Exception as e:
            print(f"CUDA computation test failed: {str(e)}")
    
    # Test NumPy interoperability
    print("\n=== NumPy Interoperability Test ===")
    try:
        # Create NumPy array and convert to tensor
        np_array = np.random.randn(100, 100)
        torch_tensor = torch.from_numpy(np_array)
        if torch.cuda.is_available():
            torch_tensor = torch_tensor.cuda()
        print("NumPy-PyTorch interoperability test passed!")
    except Exception as e:
        print(f"NumPy interoperability test failed: {str(e)}")

if __name__ == "__main__":
    test_cuda_compatibility()