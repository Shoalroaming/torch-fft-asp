from typing import Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

def to_tensor(x: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=device)
    else:
        x = x.to(device)
    return x

 
def validate_inputs(
    phase: Union[torch.Tensor, np.ndarray],
    intensity: Union[torch.Tensor, np.ndarray],
    distance_mm: float,
    wavelength_m: float,
    pixel_size_m: float,
    pad_factor: float,
) -> None:
    """
    角谱法传播参数的安全性检查
    
    异常:
        ValueError: 当任何参数不满足要求时
    """
    
    if not (isinstance(distance_mm, (int, float)) and distance_mm >= 0):
        raise ValueError(f"传播距离必须是非负数字: {distance_mm}")   
    if wavelength_m <= 0:
        raise ValueError(f"波长必须为正数: {wavelength_m}")
    if pixel_size_m <= 0:
        raise ValueError(f"像素尺寸必须为正数: {pixel_size_m}")
    if not (isinstance(pad_factor, (int, float)) and pad_factor >= 1.0):
        raise ValueError(f"填充因子必须≥1.0: {pad_factor}")
    if phase.shape != intensity.shape:
        raise ValueError(
            f"相位图和光强图尺寸不匹配:\n"
            f"  相位图: {phase.shape}\n"
            f"  光强图: {intensity.shape}"
        )
        
    H, W = phase.shape[-2:]
    if H != W:
        raise ValueError(f"输入为正方形，但当前空间尺寸为 {H}x{W}")

    ndim = phase.ndim
    if ndim not in [2, 3, 4]:
        raise ValueError(
            f"输入维度必须是2D、3D或4D: {ndim}D {phase.shape}"
        )

 
def angular_spectrum_propagation(
    phase: Union[torch.Tensor, np.ndarray],
    intensity: Union[torch.Tensor, np.ndarray],
    distance_mm: float = 20.0,
    wavelength_m: float = 632.8e-9,
    pixel_size_m: float = 8e-6,
    pad_factor: float = 1.5,
) -> torch.Tensor:
    """
    角谱法衍射传播
    
    参数:
        phase: 相位图 [rad], (H,W) 或 (C,H,W) 或 (B,C,H,W)
        intensity: 光强图 [任意正数单位], 维度必须与相位图匹配
        distance_mm: 传播距离 [mm]
        wavelength_m: 波长 [m]
        pixel_size_m: 像素尺寸 [m]
        pad_factor: 填充倍数（1=不填充，1.5=填充后边长变为1.5倍）
    
    返回:
        传播后的光强分布 [B,C,H,W] 或根据输入降维
    """
    
    validate_inputs(
        phase, intensity, distance_mm, wavelength_m, pixel_size_m, pad_factor
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    phase_t = to_tensor(phase, device)
    intensity_t = to_tensor(intensity, device)
 
    
    # 统一为4D
    if phase_t.dim() == 2:
        phase_t = phase_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        intensity_t = intensity_t.unsqueeze(0).unsqueeze(0)
    elif phase_t.dim() == 3:
        phase_t = phase_t.unsqueeze(0)  # [1,C,H,W]
        intensity_t = intensity_t.unsqueeze(0)
    
    B, C, H, W = phase_t.shape
    amplitude = intensity_t.sqrt()
    complex_field = amplitude * torch.exp(1j * phase_t)

    # 填充
    if pad_factor > 1.0:
        pad_pixels = int(H * (pad_factor - 1) / 2) 
        complex_field = F.pad(complex_field, (pad_pixels,)*4, mode='constant', value=0)
        N = complex_field.shape[-1]
    else:
        N = H
        pad_pixels = 0

    f = fftshift(fftfreq(N, pixel_size_m, device=device))
    f_sq = f**2
    freq_sq = f_sq[:, None] + f_sq[None, :]  # f_x² + f_y²

    k = 2 * torch.pi / wavelength_m
    z = distance_mm * 1e-3

    inside = 1 - (wavelength_m**2) * freq_sq
    h_filter = torch.exp(1j * k * z * torch.sqrt(inside))

    field_fft = fftshift(fft2(complex_field, norm='ortho'))
    propagated = ifft2(ifftshift(field_fft * h_filter), norm='ortho')

    # 裁剪
    if pad_pixels > 0:
        propagated = propagated[..., pad_pixels:-pad_pixels, pad_pixels:-pad_pixels]

    output_intensity = propagated.abs().square()

    # 恢复形状
    if phase.ndim == 2:
        output_intensity = output_intensity.squeeze(0).squeeze(0)
    elif phase.ndim == 3:
        output_intensity = output_intensity.squeeze(0)


    return output_intensity
