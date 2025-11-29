# 角谱法衍射传播

基于PyTorch的角谱法衍射传播实现，支持GPU加速。

## 功能特性

- 角谱法衍射传播计算
- 支持2D/3D/4D张量输入
- 自动填充
- GPU加速支持

## 文件结构
```
  .
├─ asp.py              # 角谱法
├─ test_gaussian.py    # 高斯光波传播测试
├─ test_image.py       # 实际图像传播测试
├─ dog=20mm.tif        # 20mm处灰度图
├─ dog=19.9mm.tif      # 19.1mm处灰度图
├─ dog=20.1mm.tif      # 20.1mm处灰度图
 dogPhi=20mm.tif     # 20mm处相位图
├─
└─
```

## 核心函数

### `angular_spectrum_propagation`

```python
angular_spectrum_propagation(
    phase,              # 相位图 [rad]
    intensity,          # 光强图
    distance_mm=20.0,   # 传播距离 [mm]
    wavelength_m=632.8e-9,  # 波长 [m]
    pixel_size_m=8e-6,  # 像素尺寸 [m]
    pad_factor=1.5      # 填充因子
)
```

## 使用示例

高斯光束传播测试

```bash
python test_gaussian.py
```

验证角谱法在瑞利长度范围内的传播精度。

实际图像传播测试

```bash
python test_real_image.py  
```

使用真实图像测试传播效果，计算RMSE、PSNR、SSIM指标。

## 输入输出

- 输入: 相位图 + 光强图 (H,W) 或 (C,H,W) 或 (B,C,H,W)  
- 输出: 传播后的光强分布 (保持输入维度)

## 依赖

- PyTorch  
- NumPy  
- Pillow (仅测试脚本)  
- tabulate (仅测试脚本)  

