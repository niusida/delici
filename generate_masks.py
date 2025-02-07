import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def load_aal_template(aal_template_path):
    """加载AAL模板"""
    print(f"加载AAL模板: {aal_template_path}")
    aal_img = nib.load(aal_template_path)
    return aal_img.get_fdata(), aal_img.affine

def load_aal_names(aal_names_dir):
    """加载AAL区域名称"""
    print(f"加载AAL区域名称: {aal_names_dir}")
    region_names = {}
    img_files = [f for f in os.listdir(aal_names_dir) if f.endswith('.img')]
    for i, img_file in enumerate(img_files, start=1):
        region_name = img_file[:-4]
        region_names[i] = region_name
    return region_names

def resize_aal_template(aal_data, target_shape):
    """将AAL模板调整到目标大小，保持空间对应关系
    
    Args:
        aal_data: 原始AAL数据
        target_shape: 目标形状，如(121,145,121)
        
    Returns:
        调整大小后的AAL数据
    """
    # 计算缩放因子
    zoom_factors = [t/s for t, s in zip(target_shape, aal_data.shape)]
    print(f"缩放因子: {zoom_factors}")
    
    # 使用最近邻插值进行缩放，保持区域ID
    resized_data = zoom(aal_data, zoom_factors, order=0)
    
    # 验证调整后的数据
    original_regions = set(np.unique(aal_data))
    resized_regions = set(np.unique(resized_data))
    
    if original_regions != resized_regions:
        print("警告: 调整大小后的区域ID可能有变化")
        print(f"原始区域ID: {original_regions}")
        print(f"调整后区域ID: {resized_regions}")
        
        # 检查丢失的区域
        lost_regions = original_regions - resized_regions
        if lost_regions:
            print(f"丢失的区域ID: {lost_regions}")
            
        # 检查新增的区域
        new_regions = resized_regions - original_regions
        if new_regions:
            print(f"新增的区域ID: {new_regions}")
    
    # 验证每个区域的体素数量变化
    for region_id in original_regions:
        if region_id == 0:  # 跳过背景
            continue
        original_count = np.sum(aal_data == region_id)
        resized_count = np.sum(resized_data == region_id)
        volume_ratio = resized_count / original_count
        expected_ratio = np.prod(zoom_factors)
        
        # 检查体素数量变化是否合理
        if not (0.5 * expected_ratio <= volume_ratio <= 1.5 * expected_ratio):
            print(f"警告: 区域 {region_id} 的体素数量变化异常")
            print(f"原始体素数: {original_count}")
            print(f"调整后体素数: {resized_count}")
            print(f"实际比例: {volume_ratio:.2f}")
            print(f"期望比例: {expected_ratio:.2f}")
    
    return resized_data

def generate_region_mask(aal_data, region_id, affine, output_path, region_name):
    """为指定脑区生成mask并保存为nii文件"""
    print(f"\n生成区域 {region_name} (ID: {region_id}) 的mask...")
    
    # 1. 验证输入数据
    print("验证AAL数据:")
    print(f"AAL数据形状: {aal_data.shape}")
    print(f"AAL数据值范围: [{aal_data.min()}, {aal_data.max()}]")
    print(f"AAL中的唯一区域ID: {np.unique(aal_data)}")
    
    if region_id not in np.unique(aal_data):
        print(f"错误: AAL数据中不存在区域ID {region_id}")
        return False
    
    # 2. 创建二值化mask
    mask = (aal_data == region_id).astype(np.float32)
    
    # 3. 验证mask的值
    unique_values = np.unique(mask)
    print("\nMask验证:")
    print(f"Mask中的唯一值: {unique_values}")
    print(f"Mask中1的数量: {np.sum(mask == 1)}")
    print(f"Mask中0的数量: {np.sum(mask == 0)}")
    
    # 4. 验证mask的有效性
    total_voxels = np.sum(mask)
    if total_voxels == 0:
        print(f"错误: 区域 {region_name} 没有有效体素")
        return False
    
    # 5. 检查mask的连通性
    from scipy import ndimage
    labeled_array, num_features = ndimage.label(mask)
    if num_features > 1:
        print(f"警告: 区域 {region_name} 包含 {num_features} 个不连通的组件")
        # 打印每个组件的大小
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_array == i)
            print(f"组件 {i} 的体素数量: {component_size}")
    
    # 6. 计算mask的空间特征
    center_of_mass = ndimage.center_of_mass(mask)
    print(f"\n空间特征:")
    print(f"重心坐标: {center_of_mass}")
    
    # 计算mask的边界框
    nonzero = np.nonzero(mask)
    bbox_min = np.min(nonzero, axis=1)
    bbox_max = np.max(nonzero, axis=1)
    print(f"边界框: min={bbox_min}, max={bbox_max}")
    
    # 7. 创建并保存nii文件
    mask_img = nib.Nifti1Image(mask, affine)
    
    # 添加详细的header信息
    mask_img.header['descrip'] = f'Mask for {region_name}'
    mask_img.header['intent_name'] = f'ROI_{region_name}'
    
    # 保存文件
    nib.save(mask_img, output_path)
    print(f"\n保存结果:")
    print(f"Mask文件已保存: {output_path}")
    print(f"总体素数量: {total_voxels}")
    print(f"占总体积比例: {(total_voxels / mask.size) * 100:.2f}%")
    
    return True

def main():
    """主函数"""
    try:
        # 设置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        aal_template_path = os.path.join(current_dir, "aal/AAL_61x73x61.nii")
        aal_names_dir = os.path.join(current_dir, "aal/AAL_NIFTI_61_73_61_333_90_reslice")
        
        # 目标形状
        target_shapes = [
            (121, 145, 121),
            (91, 109, 91)
        ]
        
        # 需要生成mask的区域名称
        target_regions = [
          "Frontal_Sup_Orb_L" , 
          "Frontal_Sup_Medial_L",
          "Frontal_Sup_Orb_R",
          "Amygdala_L",
          "Amygdala_R",
 

    
        ]
        
        # 加载原始AAL数据
        aal_data, affine = load_aal_template(aal_template_path)
        region_names = load_aal_names(aal_names_dir)
        
        # 为每个目标大小生成mask
        for target_shape in target_shapes:
            shape_str = f"{target_shape[0]}_{target_shape[1]}_{target_shape[2]}"
            print(f"\n处理目标大小: {shape_str}")
            
            # 创建对应大小的输出目录
            output_dir = os.path.join(current_dir, f"region_masks_{shape_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 调整AAL模板大小
            resized_aal = resize_aal_template(aal_data, target_shape)
            
            # 调整affine矩阵以反映新的体素大小
            scale_factors = [t/s for t, s in zip(target_shape, aal_data.shape)]
            new_affine = affine.copy()
            for i in range(3):
                new_affine[i,i] = affine[i,i] / scale_factors[i]
            
            # 生成每个区域的mask
            for region_name in target_regions:
                # 查找区域ID
                region_id = None
                for id, name in region_names.items():
                    if name == region_name:
                        region_id = id
                        break
                
                if region_id is None:
                    print(f"警告: 未找到区域 {region_name} 的ID")
                    continue
                
                # 生成mask文件
                output_path = os.path.join(output_dir, f"{region_name}_mask.nii")
                success = generate_region_mask(resized_aal, region_id, new_affine, 
                                            output_path, region_name)
                
                if not success:
                    print(f"警告: 区域 {region_name} 的mask可能无效")
        
        print("\n所有mask文件已生成完成")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 