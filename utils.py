import torch
import time

def measure_inference_time(model, loader, device, num_runs=1000):
    model.eval()
    batch = next(iter(loader))
    image = batch["image"].to(device)
    batch_size = image.shape[0]
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(image)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(image)
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_batch_time = total_time / num_runs
    avg_sample_time = avg_batch_time / batch_size
    
    print(f"Average batch processing time: {avg_batch_time*1000:.2f} ms")
    print(f"Average sample processing time: {avg_sample_time*1000:.2f} ms")
    print(f"Samples per second: {1.0/avg_sample_time:.2f}")
    
    return avg_sample_time


def print_trainable_parameters(model):
    # Print total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # # Print parameters by layer
    # print("\nTrainable parameters by layer:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.numel():,}")