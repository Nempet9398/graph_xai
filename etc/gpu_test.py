import torch

def check_gpu():
    if torch.cuda.is_available():
        print("CUDA is available. GPU is ready to use.")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Please check your GPU setup.")

if __name__ == "__main__":
    check_gpu()