import torch
import numpy as np
from asp import angular_spectrum_propagation
from tabulate import tabulate


def gaussian_beam(H, W, w0_pixels, center=None):
    """生成高斯光束的光强 (I = |E|^2)"""
    y = torch.arange(H) - (H // 2 if center is None else center[0])
    x = torch.arange(W) - (W // 2 if center is None else center[1])
    Y, X = torch.meshgrid(y, x, indexing='ij')
    intensity = torch.exp(-2 * (X**2 + Y**2) / w0_pixels**2)
    return intensity


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}")
    
    # 参数设置
    H, W = 1024, 1024
    pixel_size_m = 8e-6
    wavelength_m = 632.8e-9
    w0_pixels = 16
    w0_m = w0_pixels * pixel_size_m
    z_R_mm = np.pi * (w0_m**2) / wavelength_m * 1e3
    
    print("参数设置:")
    print(f"  - 图像尺寸: {H}x{W}")
    print(f"  - 波长: {wavelength_m*1e9:.1f} nm")
    print(f"  - 腰斑半径 w0: {w0_m*1e3:.3f} mm ({w0_pixels} pixels)")
    print(f"  - 瑞利长度 z_R: {z_R_mm:.2f} mm")
    
    # 输入数据
    phase_in = torch.zeros((H, W), dtype=torch.float32)
    intensity_in = gaussian_beam(H, W, w0_pixels)
    phase_in = phase_in.to(device)
    intensity_in = intensity_in.to(device)
    
    distances_mm = np.array([0.1, 0.25, 0.5, 0.75, 1.0]) * z_R_mm
    
    # 精度测试
    records = []

    for distance in distances_mm:
        output_intensity = angular_spectrum_propagation(
            phase_in, intensity_in, distance, wavelength_m,
            pixel_size_m, pad_factor=1.5
        )

        energy_ratio = output_intensity.sum().item() / intensity_in.sum().item()
        actual_peak   = output_intensity.max().item() / intensity_in.max().item()
        theory_peak   = 1.0 / (1.0 + (distance / z_R_mm) ** 2)
        peak_error    = abs(actual_peak - theory_peak) / theory_peak * 100

        records.append((distance/z_R_mm,          
                        distance,
                        energy_ratio,
                        actual_peak,
                        theory_peak,
                        peak_error))
        
    headers = ["距离(z/z_R)", "距离(mm)", "能量比", 
           "峰值比(实际)", "峰值比(理论)", "峰值误差(%)"]
    print(tabulate(records, headers=headers, floatfmt=(".2f", ".2f", ".5f", ".4f", ".4f", ".2f")))


if __name__ == "__main__":
    main()