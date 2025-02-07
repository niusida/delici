import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class FeatureAnalyzer:
    """特征分析器"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
    
    def extract_important_voxels(self, attention_map):
        """提取最后一层的空间注意力图信息
        
        Args:
            attention_map: 空间注意力图 [B, C, D, H, W]
        """
        # 将5D注意力图转换为3D
        # 首先对batch维度取平均
        attention_map = attention_map.mean(0)  # [C, D, H, W]
        # 然后对通道维度取平均
        attention_map = attention_map.mean(0)  # [D, H, W]
        
        attention_map = attention_map.cpu().numpy()
        logging.info(f"注意力图形状: {attention_map.shape}")
        logging.info(f"注意力值范围: [{attention_map.min():.4f}, {attention_map.max():.4f}]")
        
        # 将下采样空间的坐标映射回原始图像空间
        original_shape = (121, 145, 121)  # 原始图像尺寸
        feature_shape = attention_map.shape  # 特征图尺寸
        
        # 计算每个维度的步长
        x_step = original_shape[0] / feature_shape[0]
        y_step = original_shape[1] / feature_shape[1]
        z_step = original_shape[2] / feature_shape[2]
        
        logging.info(f"步长: x={x_step:.2f}, y={y_step:.2f}, z={z_step:.2f}")
        
        # 获取所有体素点的坐标和激活值
        coordinates = []
        activations = []
        
        # 对注意力图进行高斯平滑以减少噪声
        attention_map = gaussian_filter(attention_map, sigma=1.0)
        
        # 标准化到[0,1]范围
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        for x in range(feature_shape[0]):
            for y in range(feature_shape[1]):
                for z in range(feature_shape[2]):
                    # 计算映射后的坐标
                    mapped_x = int((x + 0.5) * x_step)
                    mapped_y = int((y + 0.5) * y_step)
                    mapped_z = int((z + 0.5) * z_step)
                    
                    # 确保坐标在有效范围内
                    if (0 <= mapped_x < original_shape[0] and 
                        0 <= mapped_y < original_shape[1] and 
                        0 <= mapped_z < original_shape[2]):
                        coordinates.append((mapped_x, mapped_y, mapped_z))
                        activations.append(float(attention_map[x, y, z]))
        
        logging.info(f"映射后的体素点数量: {len(coordinates)}")
        
        # 打印坐标分布信息
        if coordinates:
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]
            z_coords = [c[2] for c in coordinates]
            
            logging.info("映射后的坐标范围:")
            logging.info(f"X: [{min(x_coords)}, {max(x_coords)}], 分布: {np.percentile(x_coords, [25,50,75])}")
            logging.info(f"Y: [{min(y_coords)}, {max(y_coords)}], 分布: {np.percentile(y_coords, [25,50,75])}")
            logging.info(f"Z: [{min(z_coords)}, {max(z_coords)}], 分布: {np.percentile(z_coords, [25,50,75])}")
        
        # 计算统计信息
        activations = np.array(activations)
        stats = {
            'mean_activation': float(np.mean(activations)),
            'max_activation': float(np.max(activations)),
            'min_activation': float(np.min(activations)),
            'std_activation': float(np.std(activations))
        }
        
        return {
            'coordinates': coordinates,
            'activations': activations.tolist(),
            'stats': stats
        }
    
    def analyze_batch(self, batch_data, batch_labels):
        """分析一个batch的数据,只处理最后一层的空间注意力图"""
        images, tabular = batch_data[0].to(self.device), batch_data[1].to(self.device)
        labels = batch_labels.to(self.device)
        
        self.model.enable_feature_storing(True)
        
        with torch.no_grad():
            outputs = self.model(images, tabular)
            
        feature_maps = self.model.get_feature_maps()
        layer_results = {}
        
        # 获取最后一层的空间注意力图
        if 'spatial_attention' in feature_maps:
            attention_map = feature_maps['spatial_attention']
            logging.info(f"注意力图形状: {attention_map.shape}")
            logging.info(f"注意力图范围: [{attention_map.min():.4f}, {attention_map.max():.4f}]")
            
            voxel_info = self.extract_important_voxels(attention_map)
            
            layer_results['spatial_attention'] = {
                'voxel_info': voxel_info
            }
        
        return layer_results
    
    def analyze_dataset(self, dataloader):
        """分析整个数据集"""
        self.model.eval()
        all_results = []
        
        # 收集所有batch的结果
        for batch_data, labels, tabular in tqdm(dataloader, desc="分析特征"):
            batch_results = self.analyze_batch((batch_data, tabular), labels)
            all_results.append(batch_results)
        
        # 合并所有结果
        merged_results = {}
        for layer_name in all_results[0].keys():
            if layer_name == 'spatial_attention':
                merged_results[layer_name] = {
                    'voxel_info': self._merge_voxel_info([r[layer_name]['voxel_info'] for r in all_results])
                }
        
        self.results = merged_results
        return merged_results
    
    def _merge_voxel_info(self, voxel_info_list):
        """合并��个batch的体素信息"""
        all_coordinates = []
        all_activations = []
        
        for info in voxel_info_list:
            all_coordinates.extend(info['coordinates'])
            all_activations.extend(info['activations'])
        
        all_activations = np.array(all_activations)
        
        # 计算合并后的统计信息
        stats = {
            'mean_activation': float(np.mean(all_activations)),
            'max_activation': float(np.max(all_activations)),
            'min_activation': float(np.min(all_activations)),
            'std_activation': float(np.std(all_activations))
        }
        
        return {
            'coordinates': all_coordinates,
            'activations': all_activations.tolist(),
            'stats': stats
        }
    
    def save_results(self, save_path):
        """保存分析结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.results, save_path, pickle_protocol=4)
        logging.info(f"分析结果已保存到: {save_path}")
        
        # 保存可读的文本摘要
        summary_path = os.path.join(os.path.dirname(save_path), 'analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== 特征分析摘要 ===\n\n")
            
            for layer_name, layer_data in self.results.items():
                f.write(f"{layer_name} 层分析结果:\n")
                if layer_name == 'spatial_attention':
                    voxel_info = layer_data['voxel_info']
                    f.write(f"体素点数量: {len(voxel_info['coordinates'])}\n")
                    f.write(f"体素激活统计:\n")
                    for stat_name, value in voxel_info['stats'].items():
                        f.write(f"  {stat_name}: {value:.4f}\n")
                f.write("\n") 