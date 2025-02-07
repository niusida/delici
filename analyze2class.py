import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_preprocessing import MRIDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import csv

class ModelAnalyzer:
    """模型特征分析器"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def analyze_features(self, dataloader, save_dir="analysis_results"):
        """分析模型特征"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 修改特征收集结构
        features = {
            'layer1': [],
            'layer2': [],
            'layer3': [],
            'spatial': []
        }
        
        print("收集特征激活值...")
        for batch_idx, batch in enumerate(dataloader):
            images, labels, tabular = batch
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            
            try:
                with torch.no_grad():
                    # 前向传播
                    outputs = self.model(images, tabular)
                    
                    # 收集SE层激活
                    for layer_name in ['layer1', 'layer2', 'layer3']:
                        se_layer = getattr(self.model, f'se{layer_name[-1]}')
                        if hasattr(se_layer, 'get_attention_weights'):
                            activations = se_layer.get_attention_weights()
                            features[layer_name].append(activations.cpu())
                        else:
                            print(f"Warning: {layer_name} missing get_attention_weights method")
                    
                    # 收集空间注意力激活
                    if hasattr(self.model.spatial_att, 'get_attention_map'):
                        spatial_attention = self.model.spatial_att.get_attention_map()
                        features['spatial'].append(spatial_attention.cpu())
                    else:
                        print("Warning: spatial_att missing get_attention_map method")
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 分析每一层的特征
        results = {}
        for layer_name in ['layer1', 'layer2', 'layer3']:
            print(f"\n分析 {layer_name} 的特征...")
            
            # 直接连接该层的所有激活值
            layer_activations = torch.cat(features[layer_name], dim=0)
            
            # 计算每个通道的统计量
            mean_activation = torch.mean(layer_activations, dim=0)  # 平均权重
            std_activation = torch.std(layer_activations, dim=0)    # 权重标准差
            max_activation = torch.max(layer_activations, dim=0)[0] # 最大权重
            min_activation = torch.min(layer_activations, dim=0)[0] # 最小权重
            
            # 保存通道分析结果
            channel_info = pd.DataFrame({
                'channel': range(len(mean_activation)),
                'weight': mean_activation.numpy(),           # 重命名为weight更直观
                'std': std_activation.numpy(),
                'max_weight': max_activation.numpy(),
                'min_weight': min_activation.numpy()
            })
            
            # 按权重降序排序
            channel_info = channel_info.sort_values('weight', ascending=False)
            
            # 保存到CSV
            csv_path = os.path.join(save_dir, f'{layer_name}_detailed_analysis.csv')
            channel_info.to_csv(csv_path, index=False)
            print(f"已保存{layer_name}的分析结果到: {csv_path}")
            
            # 打印top-10通道
            print(f"\n{layer_name} Top-10 通道 (按权重排序):")
            for _, row in channel_info.head(10).iterrows():
                print(f"Channel {int(row['channel'])}:")
                print(f"  权重: {row['weight']:.4f}")
                print(f"  标准差: {row['std']:.4f}")
                print(f"  最大权重: {row['max_weight']:.4f}")
                print(f"  最小权重: {row['min_weight']:.4f}")
            
            # 保存权重分布图
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.hist(mean_activation.numpy(), bins=30)
            plt.title(f'{layer_name} Channel Weights Distribution')
            plt.xlabel('Weight')
            
            plt.subplot(1, 3, 2)
            plt.hist(std_activation.numpy(), bins=30)
            plt.title('Weight Standard Deviation')
            plt.xlabel('Std')
            
            plt.subplot(1, 3, 3)
            plt.hist(max_activation.numpy(), bins=30)
            plt.title('Max Weight Distribution')
            plt.xlabel('Max Weight')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{layer_name}_weight_distributions.png'))
            plt.close()
            
            results[layer_name] = {
                'channel_info': channel_info,
                'mean_weight': mean_activation,
                'std': std_activation,
                'max_weight': max_activation,
                'min_weight': min_activation
            }
        
        return results

def main():
    """主函数"""
    # 设置路径
    model_dir = "models"
    save_dir = "analysis_results"
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        from model import MedicalCNNTransformer
        model = MedicalCNNTransformer(
            input_shape=(121, 145, 121),
            num_classes=3,
            tabular_dim=1
        )
        
        # 查找模型文件
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
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        
        # 准备数据加载器
        image_dir = "E:/py test/austim/datasets/Image"
        table_dir = "E:/py test/austim/datasets/Table"
        
        dataset = MRIDataset(
            image_dir=image_dir,
            table_path=table_dir,
            label_column='label',
            target_shape=(121, 145, 121),
            transform=None
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )
        
        # 创建分析器并执行分析
        analyzer = ModelAnalyzer(model, device)
        results = analyzer.analyze_features(
            dataloader=dataloader,
            save_dir=save_dir
        )
        
        print("\n分析完成!")
        print(f"结果保存在: {save_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
