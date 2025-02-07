# important_region_mask.py
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import pandas as pd

class ImportantRegionMaskGenerator:
    """基于AAL模板分析脑区信号强度"""
    def __init__(self, target_shape=(91, 109, 91)):
        self.target_shape = target_shape
        self.region_names = {}  # 序号(1-116)到名称的映射
        self.id_to_index = {}   # AAL四位数ID到序号(1-116)的映射
        
    def load_aal_template(self, aal_template_path):
        """加载AAL模板"""
        print(f"加载AAL模板: {aal_template_path}")
        aal_img = nib.load(aal_template_path)
        aal_data = aal_img.get_fdata()
        
        # 获取所有非零区域ID并排序
        region_ids = sorted(list(set(np.unique(aal_data)) - {0}))
        
        # 创建四位数ID到序号的映射
        self.id_to_index = {int(id): idx+1 for idx, id in enumerate(region_ids)}
        print("\nAAL区域ID映射:")
        print(f"区域ID范围: {min(region_ids)} - {max(region_ids)}")
        print(f"映射示例(前5个): {dict(list(self.id_to_index.items())[:5])}")
        
        return aal_data, aal_img.affine
        
    def load_aal_names(self, aal_names_path):
        """从txt文件加载AAL区域名称,按序号1-116对应
        
        Args:
            aal_names_path: aal116NodeNames.txt的路径
        """
        print(f"\n加载AAL区域名称: {aal_names_path}")
        with open(aal_names_path, 'r') as f:
            lines = f.readlines()
            
        # 清空现有的region_names字典
        self.region_names.clear()
            
        # 按行读取,每行对应一个序号(1-116)
        for i, line in enumerate(lines, start=1):
            region_name = line.strip()
            self.region_names[i] = region_name
            
        print(f"加载了 {len(self.region_names)} 个脑区名称")
        print("示例区域名称:")
        for i in range(1, min(6, len(self.region_names) + 1)):
            print(f"序号 {i}: {self.region_names[i]}")
            
    def load_label_1_data(self, excel_path, image_dir):
        """加载所有label为1的数据
        
        Args:
            excel_path: train.xlsx路径
            image_dir: 图像目录路径
            
        Returns:
            list: 包含(patient_id, image_path)的列表
        """
        print(f"\n加载label为1的样本信息...")
        df = pd.read_excel(excel_path)
        label_1_samples = df[df['label'] == 1]
        
        data_list = []
        for _, row in label_1_samples.iterrows():
            patient_id = str(int(row['patient_id']))
            image_path = os.path.join(image_dir, f"{patient_id}.nii")
            if os.path.exists(image_path):
                data_list.append((patient_id, image_path))
            else:
                print(f"警告: 找不到图像文件 {image_path}")
                
        print(f"找到 {len(data_list)} 个label为1的样本")
        return data_list
        
    def load_nifti(self, nifti_path):
        """加载原始nifti文件"""
        print(f"加载nifti文件: {nifti_path}")
        img = nib.load(nifti_path)
        data = img.get_fdata()
        return data, img.affine
        
    def preprocess_data(self, data):
        """预处理数据"""
        # 处理NaN和Inf
        data = np.nan_to_num(data)
        
        # 标准化到[0,1]
        if data.max() != data.min():
            data = (data - data.min()) / (data.max() - data.min())
            
        return data
        
    def resize_volume(self, volume, target_shape):
        """调整数据大小"""
        if volume.shape == target_shape:
            return volume
            
        scale_factors = [t/s for t, s in zip(target_shape, volume.shape)]
        return zoom(volume, scale_factors, order=1)
        
    def resize_aal_template(self, aal_data, target_shape):
        """调整AAL模板大小,保持区域标签"""
        if aal_data.shape == target_shape:
            return aal_data
            
        scale_factors = [t/s for t, s in zip(target_shape, aal_data.shape)]
        return zoom(aal_data, scale_factors, order=0)
        
    def analyze_region_intensities(self, brain_data, aal_data):
        """分析每个脑区的信号强度"""
        region_intensities = []
        
        # 获取所有非零区域ID
        region_ids = sorted(list(set(np.unique(aal_data)) - {0}))
        
        for rid in region_ids:
            # 获取当前区域的mask
            region_mask = (aal_data == rid)
            
            # 提取区域内的信号值
            region_signals = brain_data[region_mask]
            
            if len(region_signals) > 0:
                # 计算统计量
                mean_intensity = np.mean(region_signals)
                std_intensity = np.std(region_signals)
                max_intensity = np.max(region_signals)
                min_intensity = np.min(region_signals)
                
                # 将四位数ID转换为序号(1-116)
                region_index = self.id_to_index.get(int(rid))
                
                # 获取区域名称
                region_name = self.region_names.get(region_index, f"Unknown-{rid}")
                
                # 如果是Unknown,打印调试信息
                if region_name.startswith("Unknown"):
                    print(f"警告: 找不到区域ID {rid} (序号 {region_index}) 的名称")
                
                region_intensities.append({
                    'id': rid,
                    'index': region_index,
                    'name': region_name,
                    'mean': mean_intensity,
                    'std': std_intensity,
                    'max': max_intensity,
                    'min': min_intensity,
                    'voxel_count': len(region_signals)
                })
        
        return region_intensities
        
    def aggregate_results(self, all_intensities):
        """汇总多个样本的结果
        
        Args:
            all_intensities: 列表,包含每个样本的region_intensities
            
        Returns:
            list: 汇总后的区域强度信息
        """
        # 创建用于存储汇总结果的字典
        aggregated = {}
        
        # 遍历每个样本的结果
        for sample_intensities in all_intensities:
            for region in sample_intensities:
                rid = region['id']
                if rid not in aggregated:
                    aggregated[rid] = {
                        'id': rid,
                        'index': region['index'],
                        'name': region['name'],
                        'means': [],
                        'stds': [],
                        'maxs': [],
                        'mins': [],
                        'voxel_count': region['voxel_count']
                    }
                
                aggregated[rid]['means'].append(region['mean'])
                aggregated[rid]['stds'].append(region['std'])
                aggregated[rid]['maxs'].append(region['max'])
                aggregated[rid]['mins'].append(region['min'])
        
        # 计算最终统计量
        final_results = []
        for rid, stats in aggregated.items():
            final_results.append({
                'id': rid,
                'index': stats['index'],
                'name': stats['name'],
                'mean': np.mean(stats['means']),
                'std': np.std(stats['means']),
                'max': np.max(stats['maxs']),
                'min': np.min(stats['mins']),
                'voxel_count': stats['voxel_count']
            })
        
        # 按平均信号强度排序
        final_results.sort(key=lambda x: x['mean'], reverse=True)
        return final_results
        
    def save_results(self, region_intensities, output_dir):
        """保存分析结果"""
        # 保存CSV文件
        csv_path = os.path.join(output_dir, 'region_intensities.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['Region ID', 'Region Index', 'Region Name', 'Mean Intensity', 
                           'Std Intensity', 'Max Intensity', 'Min Intensity', 
                           'Voxel Count'])
            
            for region in region_intensities:
                writer.writerow([
                    region['id'],
                    region['index'],
                    region['name'],
                    f"{region['mean']:.4f}",
                    f"{region['std']:.4f}",
                    f"{region['max']:.4f}",
                    f"{region['min']:.4f}",
                    region['voxel_count']
                ])
        
        print(f"\n结果已保存到: {csv_path}")
        
        # 生成条形图
        plt.figure(figsize=(15, 8))
        top_n = 20  # 显示前20个区域
        
        regions = region_intensities[:top_n]
        names = [f"{r['index']}-{r['name']}" for r in regions]  # 添加序号到名称
        means = [r['mean'] for r in regions]
        stds = [r['std'] for r in regions]
        
        plt.bar(range(len(names)), means, yerr=stds, capsize=5)
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.xlabel('Brain Regions')
        plt.ylabel('Mean Signal Intensity')
        plt.title('Top 20 Brain Regions by Signal Intensity (Label 1 Samples)')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, 'top_regions_intensity.png'))
        plt.close()
        
    def visualize_top_regions(self, brain_data, aal_data, region_intensities, 
                            output_dir, top_n=5):
        """可视化信号最强的脑区"""
        # 创建mask
        top_regions_mask = np.zeros_like(aal_data)
        
        # 标记前N个区域
        for region in region_intensities[:top_n]:
            region_id = region['id']
            top_regions_mask[aal_data == region_id] = 1
        
        # 选择中心切片
        z_center = brain_data.shape[2] // 2
        
        plt.figure(figsize=(15, 5))
        
        # 原始数据
        plt.subplot(131)
        plt.imshow(brain_data[:,:,z_center], cmap='gray')
        plt.title('Original Data')
        plt.axis('off')
        
        # Mask
        plt.subplot(132)
        plt.imshow(top_regions_mask[:,:,z_center], cmap='hot')
        plt.title(f'Top {top_n} Regions')
        plt.axis('off')
        
        # 叠加显示
        plt.subplot(133)
        plt.imshow(brain_data[:,:,z_center], cmap='gray')
        plt.imshow(top_regions_mask[:,:,z_center], cmap='hot', alpha=0.3)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.savefig(os.path.join(output_dir, 'top_regions_visualization.png'))
        plt.close()
        
    def check_aal_template(self, aal_data):
        """检查AAL模板的区域ID分布
        
        Args:
            aal_data: AAL模板数据
        """
        # 获取所有非零区域ID
        region_ids = sorted(list(set(np.unique(aal_data)) - {0}))
        
        print("\nAAL模板检查结果:")
        print(f"模板形状: {aal_data.shape}")
        print(f"数据类型: {aal_data.dtype}")
        print(f"值范围: [{aal_data.min()}, {aal_data.max()}]")
        print(f"非零区域数量: {len(region_ids)}")
        print(f"区域ID列表: {region_ids[:10]}...")
        
        # 检查是否有浮点数ID
        has_float = any(not float(x).is_integer() for x in region_ids)
        if has_float:
            print("警告: 检测到浮点数区域ID")
            
        # 检查ID是否连续
        expected_ids = set(range(1, len(region_ids) + 1))
        actual_ids = set(int(x) for x in region_ids)
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        
        if missing_ids:
            print(f"缺失的区域ID: {sorted(missing_ids)}")
        if extra_ids:
            print(f"额外的区域ID: {sorted(extra_ids)}")
            
        return region_ids
        
    def generate_individual_masks(self, aal_data, affine, region_intensities, output_dir, top_n=10):
        """为前N个最强信号的脑区生成单独的mask文件
        
        Args:
            aal_data: AAL模板数据
            affine: 仿射矩阵
            region_intensities: 区域强度信息列表
            output_dir: 输出目录
            top_n: 生成前N个区域的mask
        """
        print(f"\n生成前{top_n}个脑区的mask文件...")
        
        # 创建mask保存目录
        mask_dir = os.path.join(output_dir, 'individual_masks')
        os.makedirs(mask_dir, exist_ok=True)
        
        # 为每个区域生成mask
        for i, region in enumerate(region_intensities[:top_n], 1):
            region_id = region['id']
            region_name = region['name']
            region_index = region['index']
            
            # 创建mask
            mask = (aal_data == region_id).astype(np.float32)
            
            # 统计mask信息
            voxel_count = np.sum(mask)
            print(f"\n区域 {i}: {region_name}")
            print(f"区域ID: {region_id}")
            print(f"序号: {region_index}")
            print(f"体素数量: {voxel_count}")
            
            # 保存为nii.gz文件
            mask_name = f"{i:02d}_{region_index:03d}_{region_name}_mask.nii.gz"
            mask_path = os.path.join(mask_dir, mask_name)
            
            mask_img = nib.Nifti1Image(mask, affine)
            nib.save(mask_img, mask_path)
            print(f"Mask已保存: {mask_path}")
            
        print(f"\n所有mask文件已保存到: {mask_dir}")

def main():
    """主函数"""
    try:
        # 设置路径
        base_dir = "E:/py test/austim"
        image_dir = os.path.join(base_dir, "datasets/Image/train")
        excel_path = os.path.join(base_dir, "datasets/Table/train.xlsx")
        aal_template_path = os.path.join(base_dir, "aal/aal116MNI.nii.gz")
        aal_names_path = os.path.join(base_dir, "aal/aal116NodeNames.txt")
        output_dir = os.path.join(base_dir, "region_intensities_label1")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建生成器
        generator = ImportantRegionMaskGenerator()
        
        # 1. 加载AAL数据
        print("\n1. 加载AAL数据...")
        aal_data, aal_affine = generator.load_aal_template(aal_template_path)
        
        # 检查AAL模板
        generator.check_aal_template(aal_data)
        
        # 加载区域名称
        generator.load_aal_names(aal_names_path)
        
        # 2. 加载label为1的样本列表
        data_list = generator.load_label_1_data(excel_path, image_dir)
        
        # 3. 处理每个样本
        print("\n3. 处理每个样本...")
        all_intensities = []
        
        for patient_id, nifti_path in data_list:
            print(f"\n处理样本 {patient_id}...")
            
            # 加载和预处理数据
            orig_data, orig_affine = generator.load_nifti(nifti_path)
            processed_data = generator.preprocess_data(orig_data)
            
            # 调整数据大小
            target_shape = generator.target_shape
            resized_data = generator.resize_volume(processed_data, target_shape)
            resized_aal = generator.resize_aal_template(aal_data, target_shape)
            
            # 分析区域强度
            intensities = generator.analyze_region_intensities(resized_data, resized_aal)
            all_intensities.append(intensities)
        
        # 4. 汇总结果
        print("\n4. 汇总所有样本结果...")
        final_results = generator.aggregate_results(all_intensities)
        
        # 5. 打印结果
        print("\n信号强度最高的前10个脑区:")
        print("-" * 100)
        print(f"{'排名':^6}{'区域名称':<40}{'平均强度':^12}{'标准差':^12}{'体素数':^10}")
        print("-" * 100)
        for i, region in enumerate(final_results[:10], 1):
            print(f"{i:^6}{region['name']:<40}{region['mean']:^12.4f}"
                  f"{region['std']:^12.4f}{region['voxel_count']:^10}")
        
        # 6. 保存结果
        print("\n6. 保存分析结果...")
        generator.save_results(final_results, output_dir)
        
        # 7. 生成单独的mask文件
        print("\n7. 生成单独的mask文件...")
        generator.generate_individual_masks(aal_data, aal_affine, final_results, output_dir)
        
        # 8. 可视化结果
        print("\n8. 生成可视化结果...")
        # 使用第一个样本的数据进行可视化
        generator.visualize_top_regions(resized_data, resized_aal, 
                                      final_results, output_dir)
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()