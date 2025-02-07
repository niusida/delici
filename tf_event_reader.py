import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import nibabel as nib
import os

def analyze_intensity_distribution(nii_data):
    """
    分析图像强度分布来帮助确定灰质阈值
    
    Args:
        nii_data (np.ndarray): 原始nii图像数据,3D numpy数组
        
    Returns:
        tuple: (lower_thresh, upper_thresh) 建议的下阈值和上阈值
        
    Raises:
        ValueError: 当输入数据为空或包含无效值时
    """
    if nii_data is None or nii_data.size == 0:
        raise ValueError("输入数据不能为空")
    
    # 打印原始数据的基本信息
    print("\n原始数据统计信息:")
    print(f"数据形状: {nii_data.shape}")
    print(f"最小值: {nii_data.min()}")
    print(f"最大值: {nii_data.max()}")
    print(f"平均值: {nii_data.mean()}")
    print(f"非零元素数量: {np.count_nonzero(nii_data)}")
    print(f"nan值的数量: {np.count_nonzero(np.isnan(nii_data))}")
    
    # 处理nan值
    nii_data = np.nan_to_num(nii_data, nan=0.0)
    
    # 再次打印处理后的统计信息
    print("\n处理nan后的统计信息:")
    print(f"最小值: {nii_data.min()}")
    print(f"最大值: {nii_data.max()}")
    print(f"平均值: {nii_data.mean()}")
    print(f"非零元素数量: {np.count_nonzero(nii_data)}")
    
    # 归一化数据到0-1范围
    if nii_data.max() == nii_data.min():
        raise ValueError("数据没有变化范围（最大值等于最小值）")
        
    normalized_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min())
    
    # 调整非零阈值，使用更小的值
    non_zero_mask = normalized_data > 0.001  # 降低阈值到0.001
    valid_intensities = normalized_data[non_zero_mask]
    
    print("\n归一化后的数据统计信息:")
    print(f"有效像素数量: {valid_intensities.size}")
    print(f"最小有效值: {valid_intensities.min() if valid_intensities.size > 0 else 'N/A'}")
    print(f"最大有效值: {valid_intensities.max() if valid_intensities.size > 0 else 'N/A'}")
    
    if valid_intensities.size == 0:
        raise ValueError("没有有效的非零像素")
    
    # 计算基本统计量
    mean_val = np.mean(valid_intensities)
    std_val = np.std(valid_intensities)
    
    # 使用KDE估计密度分布
    kde = stats.gaussian_kde(valid_intensities)
    x_range = np.linspace(0, 1, 200)
    density = kde(x_range)
    
    # 找到局部极大值
    peaks, _ = find_peaks(density)
    peak_values = x_range[peaks]
    
    # 可视化分布
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图和KDE
    plt.subplot(121)
    plt.hist(valid_intensities, bins=50, density=True, alpha=0.6)
    plt.plot(x_range, density, 'r-', lw=2, label='KDE')
    plt.plot(x_range[peaks], density[peaks], "x", label='Peaks')
    plt.axvline(mean_val, color='g', linestyle='--', label='Mean')
    plt.axvline(mean_val - std_val, color='y', linestyle='--', label='Mean ± Std')
    plt.axvline(mean_val + std_val, color='y', linestyle='--')
    plt.xlabel('Normalized Intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Intensity Distribution')
    
    # 绘制中间切片
    plt.subplot(122)
    middle_slice = normalized_data[:, :, normalized_data.shape[2]//2]
    plt.imshow(middle_slice, cmap='gray')
    plt.colorbar(label='Normalized Intensity')
    plt.title('Middle Slice')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    # 使用Otsu方法获取初始阈值
    otsu_thresh = threshold_otsu(valid_intensities)
    
    # 使用更严格的阈值范围，专注于灰质区域
    lower_thresh = 0.27  # 提高下阈值
    upper_thresh = 0.32  # 降低上阈值
    
    # 打印详细统计信息
    print(f"\n统计信息:")
    print(f"平均值: {mean_val:.3f}")
    print(f"标准差: {std_val:.3f}")
    print(f"Otsu阈值: {otsu_thresh:.3f}")
    print(f"峰值位置: {peak_values}")
    print(f"\n建议的阈值范围:")
    print(f"下阈值: {lower_thresh:.3f}")
    print(f"上阈值: {upper_thresh:.3f}")
    
    # 在返回阈值之前添加体素统计
    print("\n体素统计:")
    total_voxels = nii_data.size
    non_nan_voxels = np.count_nonzero(~np.isnan(nii_data))
    normalized_mask = (normalized_data >= lower_thresh) & (normalized_data <= upper_thresh)
    selected_voxels = np.count_nonzero(normalized_mask)
    
    print(f"总体素数: {total_voxels}")
    print(f"非nan体素数: {non_nan_voxels}")
    print(f"阈值范围内的体素数: {selected_voxels}")
    print(f"占非nan体素的百分比: {(selected_voxels/non_nan_voxels)*100:.2f}%")
    print(f"占总体素的百分比: {(selected_voxels/total_voxels)*100:.2f}%")
    
    return lower_thresh, upper_thresh

def validate_threshold(nii_data, lower_thresh, upper_thresh):
    """
    验证所选阈值的效果
    
    Args:
        nii_data (np.ndarray): 原始nii图像数据,3D numpy数组
        lower_thresh (float): 下阈值,范围[0,1]
        upper_thresh (float): 上阈值,范围[0,1]
        
    Raises:
        ValueError: 当输入数据无效或阈值范围错误时
    """
    if nii_data is None or nii_data.size == 0:
        raise ValueError("输入数据不能为空")
        
    if not 0 <= lower_thresh <= upper_thresh <= 1:
        raise ValueError("阈值范围无效")
    
    # 归一化数据
    normalized_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min())
    
    # 创建掩码
    mask = (normalized_data >= lower_thresh) & (normalized_data <= upper_thresh)
    
    # 选择三个正交平面的中间切片
    z_slice = normalized_data.shape[2] // 2
    y_slice = normalized_data.shape[1] // 2
    x_slice = normalized_data.shape[0] // 2
    
    # 创建图像网格
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 轴向视图
    axes[0, 0].imshow(normalized_data[:, :, z_slice], cmap='gray')
    axes[0, 0].set_title('Original Axial')
    axes[0, 1].imshow(mask[:, :, z_slice], cmap='hot')
    axes[0, 1].set_title('Mask Axial')
    
    # 冠状视图
    axes[1, 0].imshow(normalized_data[:, y_slice, :], cmap='gray')
    axes[1, 0].set_title('Original Coronal')
    axes[1, 1].imshow(mask[:, y_slice, :], cmap='hot')
    axes[1, 1].set_title('Mask Coronal')
    
    # 矢状视图
    axes[2, 0].imshow(normalized_data[x_slice, :, :], cmap='gray')
    axes[2, 0].set_title('Original Sagittal')
    axes[2, 1].imshow(mask[x_slice, :, :], cmap='hot')
    axes[2, 1].set_title('Mask Sagittal')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def main():
    """测试代码的主函数"""
    # 读取NIfTI文件
    nii_path = r"E:\py test\austim\datasets\Image\train\14590.nii"
    
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"找不到文件: {nii_path}")
        
    try:
        nii_img = nib.load(nii_path)
        test_data = nii_img.get_fdata()
        
        # 检查数据是否正确加载
        if test_data.size == 0:
            raise ValueError("读取的数据为空")
            
        if np.all(test_data == 0):
            raise ValueError("数据全为0")
            
        # 分析强度分布
        lower_thresh, upper_thresh = analyze_intensity_distribution(test_data)
        
        # 验证阈值
        validate_threshold(test_data, lower_thresh, upper_thresh)
        
        # 保持图像窗口打开
        plt.show()
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()