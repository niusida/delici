import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_preprocessing import MRIDataset

class FeatureAnalyzer:
    """特征分析器类"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def analyze_channel_importance(self, save_dir):
        """分析通道重要性
        
        Args:
            save_dir: 结果保存目录
        
        Returns:
            dict: 包含各层特征重要性的字典
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取每层的SE注意力权重
        se_layers = {
            'layer1': self.model.se1,
            'layer2': self.model.se2,
            'layer3': self.model.se3
        }
        
        results = {}
        print("正在分析通道重要性...")
        
        for layer_name, se_layer in se_layers.items():
            # 获取SE层的fc层权重
            fc_weights = se_layer.fc[0].weight.data.cpu().numpy()  # 第一个全连接层的权重
            channel_importance = np.mean(np.abs(fc_weights), axis=0)  # 计算每个通道的重要性
            
            # 获取最重要的通道
            top_channels = np.argsort(channel_importance)[::-1]
            top_weights = channel_importance[top_channels]
            
            results[layer_name] = {
                'top_channels': top_channels,
                'weights': channel_importance,
                'statistics': {
                    'mean': np.mean(channel_importance),
                    'std': np.std(channel_importance),
                    'max': np.max(channel_importance),
                    'min': np.min(channel_importance)
                }
            }
            
            # 绘制权重分布
            self._plot_weight_distribution(channel_importance, layer_name, save_dir)
            
            # 保存top-10通道信息
            self._save_top_channels(top_channels[:10], top_weights[:10], layer_name, save_dir)
        
        return results
    
    def analyze_channel_locations(self, dummy_input, top_k=5):
        """分析重要通道对应的体素位置
        
        Args:
            dummy_input: 示例输入，形状为(1, 1, D, H, W)
            top_k: 每层分析的前k个重要通道
            
        Returns:
            dict: 包含每层重要通道的体素位置信息
        """
        print("\n正在分析重要通道的体素位置...")
        
        # 启用特征图存储
        self.model.enable_feature_storing(True)
        
        # 前向传播获取特征图
        with torch.no_grad():
            _ = self.model(dummy_input.to(self.device), torch.zeros(1, 1).to(self.device))
        
        # 获取特征图
        feature_maps = self.model.get_feature_maps()
        
        # 获取每层的重要通道
        channel_locations = {}
        layer_names = ['layer1', 'layer2', 'layer3']
        
        for layer_name in layer_names:
            print(f"\n分析 {layer_name} 的重要通道位置:")
            features = feature_maps[layer_name]['features'][0]  # (C, D, H, W)
            
            # 获取该层的top channels
            top_channels = self.get_top_channels(layer_name)[:top_k]
            
            channel_info = []
            for channel_idx in top_channels:
                # 获取该通道的特征图
                channel_map = features[channel_idx].cpu().numpy()
                
                # 找到最大激活位置
                max_activation = np.max(channel_map)
                max_pos = np.unravel_index(np.argmax(channel_map), channel_map.shape)
                
                # 计算原始图像坐标
                original_coords = self.feature_to_original_coords(
                    max_pos, 
                    layer_name, 
                    dummy_input.shape[2:]
                )
                
                info = {
                    'channel_idx': channel_idx,
                    'max_activation': float(max_activation),
                    'feature_coords': max_pos,
                    'original_coords': original_coords
                }
                channel_info.append(info)
                
                print(f"Channel {channel_idx}:")
                print(f"  特征图坐标: {max_pos}")
                print(f"  原始图像坐标: {original_coords}")
                print(f"  最大激活值: {max_activation:.4f}")
            
            channel_locations[layer_name] = channel_info
        
        # 关闭特征图存储
        self.model.enable_feature_storing(False)
        
        return channel_locations
    
    def get_top_channels(self, layer_name):
        """获取指定层的重要通道"""
        se_layer = getattr(self.model, f'se{layer_name[-1]}')
        fc_weights = se_layer.fc[0].weight.data.cpu().numpy()
        channel_importance = np.mean(np.abs(fc_weights), axis=0)
        return np.argsort(channel_importance)[::-1]
    
    def feature_to_original_coords(self, feature_coords, layer_name, original_shape):
        """将特征图坐标映射到原始图像坐标
        
        Args:
            feature_coords: 特征图中的坐标(d, h, w)
            layer_name: 层名称
            original_shape: 原始图像形状(D, H, W)
            
        Returns:
            tuple: 原始图像中的坐标(d, h, w)
        """
        # 计算每层的总步长
        if layer_name == 'layer1':
            stride = 1
        elif layer_name == 'layer2':
            stride = 1
        else:  # layer3
            stride = 2
            
        # 考虑padding的影响
        padding = 3  # 初始卷积的padding
        
        # 计算原始坐标
        d = feature_coords[0] * stride + padding
        h = feature_coords[1] * stride + padding
        w = feature_coords[2] * stride + padding
        
        # 确保坐标在有效范围内
        d = min(max(d, 0), original_shape[0] - 1)
        h = min(max(h, 0), original_shape[1] - 1)
        w = min(max(w, 0), original_shape[2] - 1)
        
        return (int(d), int(h), int(w))
    
    def _plot_weight_distribution(self, weights, layer_name, save_dir):
        """绘制权重分布图"""
        plt.figure(figsize=(10, 6))
        sns.histplot(weights, kde=True)
        plt.title(f'{layer_name} Channel Importance Distribution')
        plt.xlabel('Importance Value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, f'{layer_name}_importance_dist.png'))
        plt.close()
    
    def _save_top_channels(self, channels, weights, layer_name, save_dir):
        """保存最重要通道的信息"""
        df = pd.DataFrame({
            'Channel': channels,
            'Importance': weights
        })
        df.to_csv(os.path.join(save_dir, f'{layer_name}_top_channels.csv'), index=False)

def analyze_feature_importance(model, save_dir):
    """分析模型特征重要性
    
    Args:
        model: 训练好的模型
        save_dir: 保存分析结果的目录
        
    Returns:
        dict: 包含各层特征分析结果的字典
    """
    # 准备数据加载器
    image_dir = "E:/py test/austim/datasets/Image"  # 移除多余的train
    table_dir = "E:/py test/austim/datasets/Table"  # 只传入目录路径
    
    # 创建数据集和数据加载器
    dataset = MRIDataset(
        image_dir=image_dir,
        table_path=table_dir,  # 直接传入目录路径
        label_column='label',
        target_shape=(121, 145, 121),
        transform=None  # 不需要数据增强
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 分析通道重要性
    print("开始分析特征重要性...")
    results = {}
    
    # 获取每个SE层的注意力权重
    se_layers = {
        'layer1': model.se1,
        'layer2': model.se2,
        'layer3': model.se3
    }
    
    # 分析每个层的特征重要性
    for layer_name, se_layer in se_layers.items():
        print(f"\n分析 {layer_name} 的特征重要性...")
        
        # 收集该层的权重
        weights = []
        for batch in dataloader:
            # 解包batch元组
            images, labels, features = batch
            
            # 将数据移到设备上
            images = images.to(device)
            features = features.to(device)
            
            # 前向传播并获取注意力权重
            with torch.no_grad():
                _ = model(images, features)
                layer_weights = se_layer.get_attention_weights()
                if layer_weights is not None:
                    # 确保权重维度正确 [batch_size, channels]
                    if layer_weights.dim() == 2:
                        weights.append(layer_weights.cpu())
                    else:
                        print(f"警告: {layer_name} 权重维度不正确: {layer_weights.shape}")
        
        if not weights:
            print(f"警告: {layer_name} 没有收集到权重")
            continue
            
        # 计算平均权重
        weights = torch.cat(weights, dim=0)  # [total_samples, channels]
        mean_weights = weights.mean(dim=0)   # [channels]
        
        # 计算统计信息
        stats = {
            'mean': mean_weights.mean().item(),
            'std': mean_weights.std().item(),
            'max': mean_weights.max().item(),
            'min': mean_weights.min().item()
        }
        
        # 获取最重要的通道
        num_channels = mean_weights.size(0)
        top_k = min(10, num_channels)  # 取前10个或全部
        top_indices = mean_weights.argsort(descending=True)[:top_k]
        
        # 保存结果
        results[layer_name] = {
            'weights': mean_weights,
            'statistics': stats,
            'top_channels': top_indices.tolist()
        }
        
        # 保存权重分布图
        plt.figure(figsize=(10, 6))
        plt.hist(mean_weights.numpy(), bins=30)
        plt.title(f'{layer_name} Channel Attention Weights Distribution')
        plt.xlabel('Weight')
        plt.ylabel('Count')
        plt.savefig(os.path.join(save_dir, f'{layer_name}_weight_dist.png'))
        plt.close()
        
        # 保存top channels信息
        df = pd.DataFrame({
            'channel': top_indices.tolist(),
            'weight': mean_weights[top_indices].tolist()
        })
        df.to_csv(os.path.join(save_dir, f'{layer_name}_top_channels.csv'), index=False)
        
        # 在forward传播后
        print(f"Batch weights shape: {layer_weights.shape if layer_weights is not None else None}")
        print(f"Batch weights range: {layer_weights.min().item():.4f} to {layer_weights.max().item():.4f}")
        
        # 在累积权重后
        print(f"Accumulated weights: {len(weights)} batches")
        print(f"Mean weights shape: {mean_weights.shape}")
        
    return results

if __name__ == "__main__":
    from model import MedicalCNNTransformer
    
    # 设置路径
    model_dir = "models"      # 模型保存目录
    log_dir = "logs"         # 日志保存目录
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 设置模型路径
        model_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            # 尝试寻找fold模型
            for fold_idx in range(5):  # 假设最多5个fold
                fold_path = os.path.join(model_dir, f'fold_{fold_idx}_best.pt')
                if os.path.exists(fold_path):
                    model_path = fold_path
                    print(f"找到fold模型: {model_path}")
                    break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        # 加载模型
        print("正在加载模型...")
        model = MedicalCNNTransformer(
            input_shape=(121, 145, 121),
            num_classes=3,
            tabular_dim=1
        )
        
        # 加载预训练权重
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        
        # 设置保存目录
        save_dir = os.path.join(log_dir, 'feature_analysis')
        os.makedirs(save_dir, exist_ok=True)
        
        # 执行特征分析
        print("开始特征分析...")
        results = analyze_feature_importance(model, save_dir)
        
        # 打印分析结果
        for layer_name, layer_results in results.items():
            print(f"\n{layer_name} 分析结果:")
            print("统计信息:")
            for stat_name, value in layer_results['statistics'].items():
                print(f"  {stat_name}: {value:.4f}")
            print("\nTop-5 最重要通道:")
            top_channels = layer_results['top_channels'][:5]
            top_weights = layer_results['weights'][top_channels]
            for channel, weight in zip(top_channels, top_weights):
                print(f"  Channel {channel}: {weight:.4f}")
                
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
