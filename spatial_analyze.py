# spatial_attention_analysis.py
import os
import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from model import MedicalCNNTransformer
from data_preprocessing import MRIPreprocessor
import csv

def load_model(model_path, device):
    """加载模型"""
    model = MedicalCNNTransformer(
        input_shape=(121, 145, 121),
        num_classes=3,
        tabular_dim=1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def get_attention_map(model, nii_path, device):
    """获取空间注意力图"""
    # 加载nii文件
    nii_img = nib.load(nii_path)
    volume = nii_img.get_fdata()
    original_shape = volume.shape
    
    # 打印原始数据信息
    print("\n原始数据详细信息:")
    print(f"原始数据形状: {original_shape}")
    print(f"原始数据类型: {volume.dtype}")
    print(f"原始数据值范围: {volume.min()} - {volume.max()}")
    print(f"原始数据非零元素数量: {np.count_nonzero(volume)}")
    
    # 预处理
    preprocessor = MRIPreprocessor(target_shape=(121, 145, 121))
    preprocessed_volume = preprocessor(volume.copy())
    
    # 转换为tensor
    volume_tensor = torch.from_numpy(preprocessed_volume).float().unsqueeze(0).unsqueeze(0)
    volume_tensor = volume_tensor.to(device)
    
    # 前向传播获取注意力图
    with torch.no_grad():
        _ = model(volume_tensor, torch.zeros(1, 1).to(device))
        attention_map = model.get_spatial_attention_map()
    
    print("\n注意力图详细信息:")
    attention_data = attention_map[0,0].cpu().numpy()
    print(f"注意力图形状: {attention_data.shape}")
    print(f"注意力图数据类型: {attention_data.dtype}")
    print(f"注意力图值范围: {attention_data.min()} - {attention_data.max()}")
    print(f"注意力图非零元素数量: {np.count_nonzero(attention_data)}")
    
    return attention_data, original_shape, volume, nii_img.affine

def create_brain_mask(original_data):
    """创建灰质掩码"""
    # 归一化数据到0-1范围
    norm_data = (original_data - original_data.min()) / (original_data.max() - original_data.min())
    
    # 使用更宽松的阈值
    lower_thresh = 0.2  # 原来是0.27
    upper_thresh = 0.4  # 原来是0.32
    gray_matter_mask = (norm_data >= lower_thresh) & (norm_data <= upper_thresh)
    
    return gray_matter_mask.astype(np.float32)

def analyze_aal_regions(attention_map, aal_template_path, aal_names_dir, original_shape, original_data, save_dir, percentile=99):
    """基于AAL模板分析脑区分布"""
    # 加载AAL模板
    print("\n正在加载AAL模板文件:", aal_template_path)
    aal_img = nib.load(aal_template_path)
    aal_data = aal_img.get_fdata()
    
    # 打印详细的形状信息
    print("\nAAL模板详细信息:")
    print(f"AAL模板形状: {aal_data.shape}")
    print(f"AAL模板数据类型: {aal_data.dtype}")
    print(f"AAL模板值范围: {aal_data.min()} - {aal_data.max()}")
    print(f"AAL模板非零元素数量: {np.count_nonzero(aal_data)}")
    
    print("\n原始数据详细信息:")
    print(f"原始数据形状: {original_shape}")
    print(f"原始数据类型: {original_data.dtype}")
    print(f"原始数据值范围: {original_data.min()} - {original_data.max()}")
    print(f"原始数据非零元素数量: {np.count_nonzero(original_data)}")
    
    print("\n注意力图详细信息:")
    print(f"注意力图形状: {attention_map.shape}")
    print(f"注意力图数据类型: {attention_map.dtype}")
    print(f"注意力图值范围: {attention_map.min()} - {attention_map.max()}")
    
    # 打印AAL数据的基本信息
    print("\nAAL标签信息:")
    unique_labels = np.unique(aal_data)
    print(f"唯一标签数量: {len(unique_labels)}")
    print(f"唯一标签值: {unique_labels}")
    
    # 加载区域名称
    region_names = load_aal_names(aal_names_dir)
    
    # 首先将注意力图缩放到AAL大小
    print("\n将注意力图缩放到AAL大小:")
    feature_factors = [s/t for s, t in zip(aal_data.shape, attention_map.shape)]
    print(f"注意力图缩放因子: {feature_factors}")
    resized_map = zoom(attention_map, feature_factors, order=1)
    print(f"缩放后的注意力图形状: {resized_map.shape}")
    
    # 将原始数据缩小到AAL模板大小
    if original_data.shape != aal_data.shape:
        print(f"\n缩小原始数据:")
        print(f"从 {original_data.shape} 到 {aal_data.shape}")
        factors = [s/t for s, t in zip(aal_data.shape, original_data.shape)]
        print(f"原始数据缩放因子: {factors}")
        resized_data = zoom(original_data, factors, order=1)
        print(f"缩小后的原始数据形状: {resized_data.shape}")
    else:
        resized_data = original_data
    
    # 创建灰质掩码
    gray_matter_mask = create_brain_mask(resized_data)
    print(f"\n灰质掩码信息:")
    print(f"灰质掩码形状: {gray_matter_mask.shape}")
    print(f"灰质体素数量: {np.sum(gray_matter_mask > 0)}")
    print(f"占总体素比例: {np.sum(gray_matter_mask > 0)/np.prod(gray_matter_mask.shape)*100:.2f}%")
    
    # 计算阈值
    threshold = np.percentile(resized_map, percentile)
    print(f"注意力图阈值 (percentile={percentile}): {threshold}")
    
    # 获取重要体素
    print("\n计算重要体素:")
    print(f"灰质掩码形状: {gray_matter_mask.shape}")
    print(f"注意力图形状: {resized_map.shape}")
    important_voxels = (gray_matter_mask > 0) & (resized_map > threshold)
    print(f"重要体素掩码形状: {important_voxels.shape}")
    print(f"重要体素数量: {np.sum(important_voxels)}")
    print(f"占灰质体素比例: {np.sum(important_voxels)/np.sum(gray_matter_mask > 0)*100:.2f}%")
    
    # 初始化区域统计
    region_stats = {}
    
    # 获取唯一的区域标签
    unique_regions = np.unique(aal_data)
    
    print("\nAAL脑区分析结果:")
    print("-" * 100)
    print(f"{'脑区名称':<40} {'总体素数':>12} {'重要体素数':>12} {'占比':>8} {'平均激活值':>12}")
    print("-" * 100)
    
    # 统计每个区域的体素
    for region_id in unique_regions:
        if region_id == 0:  # 跳过背景
            continue
            
        # 获取区域名称
        region_name = region_names.get(int(region_id))
        if region_name is None:
            print(f"警告: 未找到区域ID {region_id} 的名称")
            continue
            
        # 获取当前区域的掩码
        region_mask = (aal_data == region_id)
        
        # 计算区域内的重要体素
        region_important_voxels = important_voxels & region_mask
        
        # 统计信息
        total_voxels = np.sum(region_mask)
        important_count = np.sum(region_important_voxels)
        
        # 只处理有体素的区域
        if total_voxels > 0:
            mean_activation = np.mean(resized_map[region_important_voxels]) if important_count > 0 else 0
            percentage = (important_count / total_voxels * 100)
            
            # 存储统计信息
            region_stats[region_id] = {
                'name': region_name,
                'total_voxels': total_voxels,
                'important_voxels': important_count,
                'percentage': percentage,
                'mean_activation': mean_activation
            }
            
            # 打印所有有体素的区域
            print(f"{region_name:<40} {total_voxels:>12} {important_count:>12} "
                  f"{percentage:>7.2f}% {mean_activation:>11.4f}")
    
    # 保存结果到CSV文件
    save_region_stats(region_stats, save_dir)
    
    # 创建top区域的mask
    create_top_regions_mask(aal_data, region_stats, aal_img.affine, save_dir)
    
    return region_stats

def get_important_voxels(attention_map, original_shape, original_data, save_dir=None, 
                        aal_template_path=None, aal_names_dir=None, percentile=95):
    """基于百分位数选择重要体素
    
    Args:
        attention_map: 注意力图数据
        original_shape: 原始数据形状
        original_data: 原始数据
        save_dir: 保存目录
        aal_template_path: AAL模板路径
        aal_names_dir: AAL名称目录
        percentile: 百分位数阈值,默认95(选择激活值最高的5%体素)
    """
    # 如果提供了AAL模板，获取目标大小
    target_shape = None
    if aal_template_path and os.path.exists(aal_template_path):
        aal_img = nib.load(aal_template_path)
        target_shape = aal_img.get_fdata().shape
        print(f"\n使用AAL模板大小作为目标: {target_shape}")
        
        # 首先将注意力图缩放到AAL大小
        print("\n将注意力图缩放到AAL大小:")
        feature_factors = [s/t for s, t in zip(target_shape, attention_map.shape)]
        print(f"注意力图缩放因子: {feature_factors}")
        resized_map = zoom(attention_map, feature_factors, order=1)
        print(f"缩放后的注意力图形状: {resized_map.shape}")
        
        # 将原始数据缩小到AAL模板大小
        print(f"\n缩小原始数据:")
        print(f"从 {original_data.shape} 到 {target_shape}")
        factors = [s/t for s, t in zip(target_shape, original_data.shape)]
        print(f"原始数据缩放因子: {factors}")
        resized_data = zoom(original_data, factors, order=1)
        print(f"缩小后的原始数据形状: {resized_data.shape}")
    else:
        # 如果没有AAL模板，保持原来的大小调整逻辑
        factors = [t/c for t, c in zip(original_shape, attention_map.shape)]
        resized_map = zoom(attention_map, factors, order=1)
        resized_data = original_data
    
    # 创建灰质掩码
    gray_matter_mask = create_brain_mask(resized_data)
    
    # 计算整个注意力图的阈值
    threshold = np.percentile(resized_map, percentile)
    
    # 打印统计信息
    total_voxels = np.prod(resized_data.shape)
    gray_matter_voxels = np.sum(gray_matter_mask > 0)
    print(f"\n体素统计信息:")
    print(f"总体素数: {total_voxels}")
    print(f"灰质体素数: {gray_matter_voxels}")
    print(f"灰质占比: {gray_matter_voxels/total_voxels*100:.2f}%")
    print(f"激活阈值 (percentile={percentile}): {threshold:.4f}")
    
    # 同时满足灰质掩码和阈值条件
    print("\n计算重要体素:")
    print(f"灰质掩码形状: {gray_matter_mask.shape}")
    print(f"注意力图形状: {resized_map.shape}")
    selected_mask = (gray_matter_mask > 0) & (resized_map > threshold)
    print(f"重要体素掩码形状: {selected_mask.shape}")
    selected_indices = np.where(selected_mask.flatten())[0]
    
    if len(selected_indices) == 0:
        print("警告：没有找到满足条件的体素")
        return [], []
        
    selected_values = resized_map.flatten()[selected_indices]
    
    # 打印选择的体素信息
    print(f"选中的体素数: {len(selected_indices)}")
    print(f"占总体素比例: {len(selected_indices)/total_voxels*100:.2f}%")
    print(f"占灰质比例: {len(selected_indices)/gray_matter_voxels*100:.2f}%")
    print(f"选中体素的激活值范围: {selected_values.min():.4f} - {selected_values.max():.4f}")
    
    # 转换回3D坐标
    shape = resized_data.shape
    x = selected_indices // (shape[1] * shape[2])
    y = (selected_indices % (shape[1] * shape[2])) // shape[2]
    z = selected_indices % shape[2]
    
    coords_list = list(zip(x, y, z))
    
    # 如果提供了AAL模板,进行区域分析
    if aal_template_path and os.path.exists(aal_template_path):
        print("\n执行AAL区域分析...")
        if save_dir:
            region_stats = analyze_aal_regions(
                attention_map,
                aal_template_path,
                aal_names_dir,
                shape,  # 使用缩小后的形状
                resized_data,  # 使用缩小后的数据
                save_dir,  # 直接使用主保存目录
                percentile
            )
    
    return coords_list, selected_values

def create_marked_nii(original_data, coordinates, values, affine, save_path, aal_template_path=None):
    """创建标记后的nii文件"""
    # 创建标记掩码
    marker_mask = np.zeros_like(original_data)
    
    # 如果有AAL模板，计算坐标转换因子
    if aal_template_path and os.path.exists(aal_template_path):
        aal_img = nib.load(aal_template_path)
        aal_shape = aal_img.get_fdata().shape
        scale_factors = [t/s for t, s in zip(original_data.shape, aal_shape)]
        print(f"\n将坐标从AAL大小 {aal_shape} 转换到原始大小 {original_data.shape}")
        print(f"缩放因子: {scale_factors}")
    else:
        scale_factors = [1, 1, 1]
    
    # 归一化注意力值到0-1范围
    norm_values = (values - values.min()) / (values.max() - values.min())
    
    # 在重要位置添加标记
    for coord, value in zip(coordinates, norm_values):
        # 转换坐标到原始大小
        x = int(coord[0] * scale_factors[0])
        y = int(coord[1] * scale_factors[1])
        z = int(coord[2] * scale_factors[2])
        
        # 确保坐标在有效范围内
        if (0 <= x < original_data.shape[0] and 
            0 <= y < original_data.shape[1] and 
            0 <= z < original_data.shape[2]):
            marker_mask[x, y, z] = value
    
    # 保存mask文件
    mask_nii = nib.Nifti1Image(marker_mask, affine)
    mask_path = save_path.replace('.nii.gz', '_mask.nii.gz')
    nib.save(mask_nii, mask_path)
    print(f"已保存标记掩码文件: {mask_path}")
    
    # 保存原始数据
    orig_nii = nib.Nifti1Image(original_data, affine)
    nib.save(orig_nii, save_path)

def save_coordinates(coordinates, values, save_path):
    """保存体素坐标和注意力值"""
    with open(save_path, 'w') as f:
        f.write("x,y,z,attention_value\n")
        for coord, value in zip(coordinates, values):
            x, y, z = coord
            f.write(f"{x},{y},{z},{value:.4f}\n")

def analyze_spatial_attention(nii_path, model_path, save_dir, aal_template_path=None, aal_names_dir=None):
    """分析空间注意力"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    print(f"\n分析空间注意力...")
    
    # 获取注意力图和原始数据
    attention_map, original_shape, original_data, affine = get_attention_map(
        model, nii_path, device
    )
    
    # 获取重要体素,并过滤非脑区
    coords, values = get_important_voxels(
        attention_map, 
        original_shape,
        original_data,
        save_dir,
        aal_template_path,
        aal_names_dir,
        percentile=95  # 调整为95,选择激活值最高的5%体素
    )
    
    if len(coords) == 0:
        print("警告：没有找到重要体素")
        return
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存标记后的nii文件
    nii_save_path = os.path.join(save_dir, 'marked.nii.gz')
    create_marked_nii(original_data, coords, values, affine, nii_save_path, aal_template_path)
    print(f"已保存标记后的nii文件: {nii_save_path}")
    
    # 保存坐标信息
    coords_save_path = os.path.join(save_dir, 'coordinates.csv')
    save_coordinates(coords, values, coords_save_path)
    print(f"已保存坐标信息: {coords_save_path}")

def main():
    # 设置路径
    model_dir = "models"
    nii_path = "E:/py test/austim/datasets/Image/train/28858.nii"
    save_dir = "E:/py test/austim/spatial_attention_results"
    
    # AAL文件路径
    aal_template_path = "E:/py test/austim/aal/AAL_61x73x61.nii"
    aal_names_dir = "E:/py test/austim/aal/AAL_NIFTI_61_73_61_333_90_reslice"
    
    # 设置模型路径
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        for fold_idx in range(5):
            fold_path = os.path.join(model_dir, f'fold_{fold_idx}_best.pt')
            if os.path.exists(fold_path):
                model_path = fold_path
                print(f"找到fold模型: {model_path}")
                break
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    try:
        # 直接分析空间注意力,不再区分类别
        analyze_spatial_attention(
            nii_path, 
            model_path, 
            save_dir,
            aal_template_path,
            aal_names_dir
        )
            
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

def load_aal_names(aal_dir):
    """从AAL目录加载区域名称
    
    Args:
        aal_dir: AAL文件夹路径
        
    Returns:
        dict: 区域ID到名称的映射
    """
    region_names = {}
    try:
        # 获取所有.img文件
        img_files = [f for f in os.listdir(aal_dir) if f.endswith('.img')]
        
        # 从文件名提取区域名称
        for i, img_file in enumerate(img_files, start=1):
            region_name = img_file[:-4]  # 移除.img后缀
            region_names[i] = region_name
        
        print(f"成功加载了 {len(region_names)} 个脑区名称")
        print("示例区域名称:")
        sample_ids = list(region_names.keys())[:3]
        for id in sample_ids:
            print(f"ID {id}: {region_names[id]}")
            
    except Exception as e:
        print(f"加载AAL名称时出错: {str(e)}")
        return {}
        
    return region_names

def save_region_stats(region_stats, save_dir):
    """保存区域统计结果到CSV文件
    
    Args:
        region_stats: 区域统计信息字典
        save_dir: 保存目录
    """
    csv_path = os.path.join(save_dir, 'region_statistics.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '脑区名称',
            '总体素数',
            '重要体素数',
            '占比(%)',
            '平均激活值'
        ])
        
        # 按重要体素数量排序
        sorted_regions = sorted(
            region_stats.items(),
            key=lambda x: x[1]['important_voxels'],
            reverse=True
        )
        
        # 只保存有重要体素的区域
        for region_id, stats in sorted_regions:
            if stats['important_voxels'] > 0:
                writer.writerow([
                    stats['name'],
                    stats['total_voxels'],
                    stats['important_voxels'],
                    f"{stats['percentage']:.2f}",
                    f"{stats['mean_activation']:.4f}"
                ])
    
    print(f"\n区域统计结果已保存到: {csv_path}")

def create_top_regions_mask(aal_data, region_stats, affine, save_dir, num_regions=2):
    """创建包含重要体素最多的脑区的mask文件
    
    Args:
        aal_data: AAL模板数据
        region_stats: 区域统计信息字典
        affine: 仿射矩阵
        save_dir: 保存目录
        num_regions: 要标记的top区域数量
    """
    # 按重要体素数量排序
    sorted_regions = sorted(
        region_stats.items(),
        key=lambda x: x[1]['important_voxels'],
        reverse=True
    )
    
    # 获取top N个区域的ID
    top_region_ids = [region_id for region_id, _ in sorted_regions[:num_regions]]
    
    # 创建mask
    mask = np.zeros_like(aal_data)
    
    # 标记top区域
    for region_id in top_region_ids:
        mask[aal_data == region_id] = 1
        
    # 打印信息
    print(f"\n标记了以下{num_regions}个最重要的脑区:")
    for region_id, stats in sorted_regions[:num_regions]:
        print(f"区域: {stats['name']}")
        print(f"  重要体素数: {stats['important_voxels']}")
        print(f"  总体素数: {stats['total_voxels']}")
        print(f"  占比: {stats['percentage']:.2f}%")
    
    # 保存mask文件
    mask_nii = nib.Nifti1Image(mask, affine)
    mask_path = os.path.join(save_dir, 'top_regions_mask.nii.gz')
    nib.save(mask_nii, mask_path)
    print(f"\n已保存top区域mask文件: {mask_path}")

if __name__ == "__main__":
    main()