import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

# 加载模板脑图像
template_brain_path = "datasets/Image/train/14590.nii"
brain_img = nib.load(template_brain_path)
brain_data = brain_img.get_fdata()
brain_mask = brain_data > 0

# 获取图像尺寸
img_shape = brain_data.shape
print(f"图像尺寸: {img_shape}")

def world_to_voxel(world_coords, img_shape):
    """将世界坐标转换为体素坐标"""
    x, y, z = world_coords
    
    # 直接使用原始坐标，但确保在图像范围内
    vox_x = int(np.clip(x, 0, img_shape[0]-1))
    vox_y = int(np.clip(y, 0, img_shape[1]-1))
    vox_z = int(np.clip(z, 0, img_shape[2]-1))
    
    return vox_x, vox_y, vox_z

# 修正比较名称
comparisons = ['0_vs_1', '0_vs_2']  # 修改为实际的格式
activation_volumes = {comp: np.zeros_like(brain_data) for comp in comparisons}

# 添加调试信息
print(f"Brain data shape: {brain_data.shape}")
print(f"Brain mask non-zero voxels: {np.sum(brain_mask)}")

def extract_coordinates(coord_str):
    """从坐标字符串中提取坐标值"""
    try:
        # 移除所有空格并清理字符串
        coord_str = coord_str.strip().strip('()').replace(' ', '')
        x, y, z = map(float, coord_str.split(','))
        return x, y, z
    except Exception as e:
        print(f"坐标提取错误: {coord_str} - {e}")
        return None

# 读取所有fold的数据并累积激活
for fold in range(1, 6):
    file_path = f'analysis/fold_{fold}_summary.csv'
    print(f"处理 {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Fold {fold} 数据形状: {df.shape}")
    
    # 分别处理每种比较
    for comparison in comparisons:
        # 筛选当前比较的数据
        comp_df = df[df['Comparison'] == comparison]
        print(f"\n处理比较 {comparison}, 数据量: {len(comp_df)}")
        
        # 使用90%分位数作为阈值
        if not comp_df.empty:
            importance_threshold = comp_df['Channel Importance'].quantile(0.90)
            important_df = comp_df[comp_df['Channel Importance'] > importance_threshold]
            print(f"重要特征数量: {len(important_df)}")
            
            for _, row in important_df.iterrows():
                # 打印原始数据以便调试
                print(f"\n处理特征通道 {row['Feature Channel']}")
                print(f"Top Voxels: {row['Top Voxels']}")
                print(f"Voxel Activations: {row['Voxel Activations']}")
                
                voxels = [v.strip() for v in row['Top Voxels'].split(';')]
                activations = [float(a.strip()) for a in row['Voxel Activations'].split(';')]
                importance = row['Channel Importance']
                
                for voxel, activation in zip(voxels, activations):
                    coords = extract_coordinates(voxel)
                    if coords is None:
                        continue
                    
                    # 使用新的坐标转换函数
                    x, y, z = world_to_voxel(coords, img_shape)
                    
                    print(f"\n原始坐标: {coords}")
                    print(f"转换后坐标: ({x}, {y}, {z})")
                    print(f"该点是否在大脑掩模内: {brain_mask[x, y, z]}")
                    
                    if brain_mask[x, y, z]:  # 如果点在大脑区域内
                        sigma = 2
                        kernel_size = int(3 * sigma)
                        x_grid, y_grid, z_grid = np.meshgrid(
                            np.arange(-kernel_size, kernel_size+1),
                            np.arange(-kernel_size, kernel_size+1),
                            np.arange(-kernel_size, kernel_size+1)
                        )
                        gaussian = np.exp(-(x_grid**2 + y_grid**2 + z_grid**2)/(2*sigma**2))
                        gaussian = gaussian / gaussian.sum() * activation * importance * 5
                        
                        for i in range(gaussian.shape[0]):
                            for j in range(gaussian.shape[1]):
                                for k in range(gaussian.shape[2]):
                                    x_idx = x + i - kernel_size
                                    y_idx = y + j - kernel_size
                                    z_idx = z + k - kernel_size
                                    
                                    if (0 <= x_idx < img_shape[0] and 
                                        0 <= y_idx < img_shape[1] and
                                        0 <= z_idx < img_shape[2] and
                                        brain_mask[x_idx, y_idx, z_idx]):
                                        
                                        activation_volumes[comparison][x_idx, y_idx, z_idx] += gaussian[i, j, k]

# 最后，确保所有激活都在大脑区域内
for comparison in comparisons:
    # 将大脑区域外的激活设为0
    activation_volumes[comparison] = activation_volumes[comparison] * brain_mask

# 在最终可视化之前打印激活体积的信息
for comparison in comparisons:
    print(f"\n{comparison} 最终激活信息:")
    print(f"激活值范围: {np.min(activation_volumes[comparison])} to {np.max(activation_volumes[comparison])}")
    print(f"非零值数量: {np.sum(activation_volumes[comparison] > 0)}")
    print(f"总激活值: {np.sum(activation_volumes[comparison])}")

# 为每种比较创建单独的可视化
for comparison in comparisons:
    activation_volume = activation_volumes[comparison]
    print(f"\n处理比较 {comparison} 的可视化")
    print(f"激活值范围: {np.min(activation_volume)} to {np.max(activation_volume)}")
    print(f"非零激活值数量: {np.sum(activation_volume > 0)}")
    
    if np.sum(activation_volume > 0) > 0:
        # 标准化激活值到[0,1]范围
        activation_volume = (activation_volume - np.min(activation_volume)) / (np.max(activation_volume) - np.min(activation_volume))
        
        # 创建激活图的nifti对象
        activation_img = nib.Nifti1Image(activation_volume, brain_img.affine)
        
        # 使用较低的阈值以显示更多细节
        threshold = np.percentile(activation_volume[activation_volume > 0], 25)
        
        # 创建多视图可视化
        fig = plt.figure(figsize=(20, 5))
        plt.suptitle(f'Brain Activation for {comparison}', fontsize=16)
        
        views = ['x', 'y', 'z']
        for i, view in enumerate(views, 1):
            plt.subplot(1, 4, i)
            display = plotting.plot_stat_map(
                activation_img,
                bg_img=brain_img,
                display_mode=view,
                cut_coords=5,
                colorbar=True,
                title=f'{view.upper()} view',
                threshold=threshold,
                cmap='hot'
            )
        
        # 添加3D渲染视图
        plt.subplot(1, 4, 4)
        display = plotting.plot_glass_brain(
            activation_img,
            display_mode='ortho',
            colorbar=True,
            title='3D Glass Brain',
            threshold=threshold,
            cmap='hot'
        )
        
        plt.tight_layout()
        plt.savefig(f'brain_activation_multiview_{comparison}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建交互式3D视图
        html_view = plotting.view_img(
            activation_img,
            bg_img=brain_img,
            threshold=threshold,
            cmap='hot',
            title=f'Interactive 3D Brain Activation Map - {comparison}',
            symmetric_cmap=False,
            opacity=0.7
        )
        
        html_view.save_as_html(f'brain_activation_3d_interactive_{comparison}.html')
    else:
        print(f"警告: {comparison} 没有有效的激活值")