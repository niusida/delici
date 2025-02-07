import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

def extract_feature_maps(model, data_loader, device, class_pairs=[(0,1), (0,2)]):
    """提取特征图并进行注意力映射"""
    model.eval()
    feature_maps = {f'class_{i}_{j}': [] for i,j in class_pairs}
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            
            # 获取特征和注意力权重
            output = model(images)
            features = output['features']  # [B, C, H, W]
            
            # 获取channel attention权重
            channel_weights = model.feature_extractor[0].channel_att(features)  # 使用CBAM的channel attention
            
            # 对每对类别分别处理
            for class_i, class_j in class_pairs:
                # 选择相关类别的样本
                mask_i = labels == class_i
                mask_j = labels == class_j
                if not (mask_i.any() and mask_j.any()):
                    continue
                    
                # 提取这两个类别的特征
                features_i = features[mask_i]
                features_j = features[mask_j]
                weights_i = channel_weights[mask_i]
                weights_j = channel_weights[mask_j]
                
                # 计算类别间的差异特征
                mean_features_i = (features_i * weights_i).mean(0)
                mean_features_j = (features_j * weights_j).mean(0)
                diff_features = mean_features_i - mean_features_j
                
                # 保存结果
                feature_maps[f'class_{class_i}_{class_j}'].append(diff_features.cpu().numpy())
    
    # 对每对类别的结果取平均
    for key in feature_maps:
        feature_maps[key] = np.mean(feature_maps[key], axis=0)
        
    return feature_maps

def map_to_voxels(feature_maps, original_shape):
    """将特征图映射回原始体素空间"""
    voxel_maps = {}
    
    for key, feat_map in feature_maps.items():
        # 使用双线性插值调整大小
        feat_map_tensor = torch.from_numpy(feat_map).unsqueeze(0)  # [1, C, H, W]
        resized_map = nn.functional.interpolate(
            feat_map_tensor,
            size=original_shape[-2:],  # 调整到原始高宽
            mode='bilinear',
            align_corners=False
        )
        
        # 转换回numpy并保存
        voxel_maps[key] = resized_map.squeeze(0).numpy()
    
    return voxel_maps

def get_data_loader(data_dir=r'E:\py test\austim\data', batch_size=8):
    """加载数据集"""
    from data_preprocessing import AlzheimerDataset  # 使用已有的数据集类
    from torch.utils.data import DataLoader
    import pandas as pd
    
    # 加载数据
    train_excel = os.path.join(data_dir, 'train.xlsx')
    filepaths, labels, tabular_features, _, _, _ = load_data(data_dir, train_excel, split='train')
    
    # 创建数据集实例
    dataset = AlzheimerDataset(
        filepaths=filepaths,
        labels=labels,
        tabular_features=tabular_features,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return data_loader

def get_original_shape():
    """获取原始图像形状"""
    return (3, 224, 224)  # 从代码中可以看到图像被调整为这个尺寸

def main():
    # 设置路径
    model_dir = r'E:\py test\austim\models'
    output_dir = r'E:\py test\austim\analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(os.path.join(model_dir, 'best_model.pth'))
    model = model.to(device)
    
    # 加载数据
    data_loader = get_data_loader()
    
    # 提取特征图
    feature_maps = extract_feature_maps(model, data_loader, device)
    
    # 获取原始图像形状
    original_shape = get_original_shape()
    
    # 映射到体素空间
    voxel_maps = map_to_voxels(feature_maps, original_shape)
    
    # 保存结果
    for key, vmap in voxel_maps.items():
        np.save(os.path.join(output_dir, f'voxel_map_{key}.npy'), vmap)
        
if __name__ == '__main__':
    main()
