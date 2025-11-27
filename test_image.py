import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from asp import angular_spectrum_propagation

def load_image(path, size=(512, 512), is_phase=False, device='cuda'):
    img = Image.open(path).convert('L').resize(size, Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if is_phase:
        arr *= 2 * np.pi
    return torch.from_numpy(arr).to(device, dtype=torch.float32)

def calculate_rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true)**2)).item()

def calculate_psnr(pred, true):
    mse = torch.mean((pred - true)**2)
    max_val = torch.max(true)
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()

def calculate_ssim(pred, true, window_size=11, data_range=1.0):
    pred = pred.unsqueeze(0).unsqueeze(0)
    true = true.unsqueeze(0).unsqueeze(0)
    
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2.0 * 1.5**2))
    window = (gauss / gauss.sum()).view(1, 1, -1, 1) * (gauss / gauss.sum()).view(1, 1, 1, -1)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(true, window, padding=window_size//2, groups=1)
    
    sigma1_sq = F.conv2d(pred**2, window, padding=window_size//2, groups=1) - mu1**2
    sigma2_sq = F.conv2d(true**2, window, padding=window_size//2, groups=1) - mu2**2
    sigma12 = F.conv2d(pred * true, window, padding=window_size//2, groups=1) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_intensity = load_image('dog=19.9mm.tif', device=device)
    target_intensity = load_image('dog=20mm.tif', device=device)
    phase = load_image('dogPhi=20mm.tif', is_phase=True, device=device)
    
    predicted_intensity = angular_spectrum_propagation(
        phase=phase,
        intensity=input_intensity,
        distance_mm=0.1,
        wavelength_m=632.8e-9,
        pixel_size_m=8e-6,
        pad_factor=1.25
    )
    
    if isinstance(predicted_intensity, np.ndarray):
        predicted_intensity = torch.from_numpy(predicted_intensity).to(device)
    
    rmse = calculate_rmse(predicted_intensity, target_intensity)
    psnr = calculate_psnr(predicted_intensity, target_intensity)
    ssim_val = calculate_ssim(predicted_intensity, target_intensity)
    
    print(f"RMSE: {rmse:.6f}")
    print(f"PSNR: {psnr:.2f}")
    print(f"SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    main()