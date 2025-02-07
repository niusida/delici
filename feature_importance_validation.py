import torch
import torch.nn as nn
from model import AlzheimerCNN
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from data_preprocessing import AlzheimerDataset, load_data  # 确保导入数据处理相关的类
import os
from sklearn.model_selection import KFold
import seaborn as sns

def validate_top_features(model, val_loader, device, top_feature_indices, fold):
    """
    使用只保留前N个重要特征的方式验证模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    def hook_fn(module, input, output):
        # 只保留top N个特征通道
        mask = torch.zeros_like(output)
        for idx in top_feature_indices:
            if idx < output.shape[1]:  # 确保索引在有效范围内
                mask[:, idx, :, :] = 1
        return output * mask  # 将其他特征通道置为0
    
    # 注册hook到layer4层
    hook = model.backbone[7].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for inputs, tabular, labels in val_loader:
            inputs = inputs.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            
            # 前向传播（会经过我们的hook函数）
            outputs = model(inputs, tabular)
            
            # 获取预测结果
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 移除hook
    hook.remove()
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    print(f"\n[Fold {fold}] 验证结果:")
    print(f"准确率: {accuracy:.4f}")
    print("详细分类报告:")
    print(report)
    
    return accuracy

def visualize_features(features, top_indices, save_path):
    """可视化特征图"""
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(top_indices):
        plt.subplot(1, 5, i+1)
        feature_map = features[idx].cpu().numpy()
        plt.imshow(feature_map, cmap='viridis')
        plt.title(f'Feature {idx}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_feature_correlations(model, val_loader, device, top_feature_indices):
    """分析特征之间的相关性"""
    feature_correlations = []
    
    def collect_features(module, input, output):
        feature_correlations.append(output.detach())
    
    hook = model.backbone[7].register_forward_hook(collect_features)
    
    with torch.no_grad():
        for inputs, tabular, labels in val_loader:
            inputs = inputs.to(device)
            tabular = tabular.to(device)
            
            feature_correlations.clear()
            _ = model(inputs, tabular)
            features = feature_correlations[0]
            
            # 计算特征图之间的相关性
            feature_maps = features[0]  # 取第一个样本
            correlation_matrix = np.corrcoef([f.flatten().cpu().numpy() for f in feature_maps])
            
            # 可视化相关性矩阵
            plt.figure(figsize=(10, 10))
            sns.heatmap(correlation_matrix[top_feature_indices][:, top_feature_indices],
                       cmap='coolwarm', center=0)
            plt.title('Top Features Correlation Matrix')
            plt.savefig('feature_correlations.png')
            plt.close()
            break
    
    hook.remove()

def main():
    # 数据预处理转换
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    image_dir = 'E:/py test/austim/datasets/Image'  # 修改为您的图像目录路径
    table_dir = 'E:/py test/austim/datasets/Table'        # 修改为您的表格目录路径
    train_excel = os.path.join(table_dir, 'train.xlsx')
    
    # 加载数据和特征
    filepaths, labels, tabular_features, selected_features, scaler_obj, selector = load_data(
        image_dir, 
        train_excel,
        split='train', 
        k=10
    )
    
    # 修改这里：加载前100个特征
    feature_ranking = pd.read_csv('feature_weights_comprehensive.csv')
    top_100_features = feature_ranking['特征索引'].head(100).values  # 改为100个特征
    
    # 使用K折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(filepaths)):
        print(f'\nProcessing Fold {fold + 1}')
        
        # 创建验证集数据集
        val_dataset = AlzheimerDataset(
            [filepaths[i] for i in val_ids],
            [labels[i] for i in val_ids],
            tabular_features[val_ids],
            transform=val_transform
        )
        
        # 创建数据加载器
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        # 加载模型
        model_path = f'E:/py test/austim/models/best_model_fold_{fold+1}.pth'  # 修改模型路径
        model = AlzheimerCNN(num_classes=3)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to('cuda')
        
        # 验证top100特征的重要性
        acc = validate_top_features(model, val_loader, 'cuda', top_100_features, fold+1)
        accuracies.append(acc)
    
    # 输出总体结果
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print("\n总体结果:")
    print(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 6), accuracies)
    plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'平均准确率: {mean_acc:.4f}')
    plt.fill_between(range(1, 6), mean_acc-std_acc, mean_acc+std_acc, 
                     color='r', alpha=0.2, label=f'标准差: {std_acc:.4f}')
    plt.xlabel('Fold')
    plt.ylabel('准确率')
    plt.title('使用前100个重要特征的验证准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('top_100_features_validation.png')
    plt.close()

if __name__ == "__main__":
    import os
    from sklearn.model_selection import KFold
    main() 