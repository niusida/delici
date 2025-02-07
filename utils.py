# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nibabel as nib
import logging
from datetime import datetime
import os

def setup_logging(save_dir):
    """设置日志"""
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{datetime.now():%Y%m%d_%H%M%S}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def visualize_attention(volume, attention_weights, save_path=None):
    """可视化注意力权重"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 选择中心切片
    slice_x = volume[volume.shape[0]//2, :, :]
    slice_y = volume[:, volume.shape[1]//2, :]
    slice_z = volume[:, :, volume.shape[2]//2]
    
    att_x = attention_weights[attention_weights.shape[0]//2, :, :]
    att_y = attention_weights[:, attention_weights.shape[1]//2, :]
    att_z = attention_weights[:, :, attention_weights.shape[2]//2]
    
    # 绘制三个视图
    ax1.imshow(slice_x, cmap='gray')
    ax1.imshow(att_x, cmap='jet', alpha=0.5)
    ax1.set_title('Sagittal View')
    
    ax2.imshow(slice_y, cmap='gray')
    ax2.imshow(att_y, cmap='jet', alpha=0.5)
    ax2.set_title('Coronal View')
    
    ax3.imshow(slice_z, cmap='gray')
    ax3.imshow(att_z, cmap='jet', alpha=0.5)
    ax3.set_title('Axial View')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_nifti(volume, affine, save_path):
    """保存为NIFTI格式"""
    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, save_path)

class MetricTracker:
    """指标跟踪器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_name, value, count=1):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0
            self.counts[metric_name] = 0
        self.metrics[metric_name] += value * count
        self.counts[metric_name] += count
    
    def avg(self, metric_name):
        return self.metrics[metric_name] / self.counts[metric_name]
    
    def result(self):
        return {k: self.avg(k) for k in self.metrics}

def analyze_feature_importance(model, data_loader, save_dir):
    """分析特征重要性"""
    feature_importance = {}
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, target, tabular) in enumerate(data_loader):
            # 获取注意力权重
            attention_weights = model(data.cuda(), tabular.cuda())
            
            # 转换为字典格式
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = {
                    'overall': attention_weights.cpu().numpy()
                }
            
            # 存储结果
            for key, value in attention_weights.items():
                if key not in feature_importance:
                    feature_importance[key] = []
                feature_importance[key].append(value)
    
    # 计算平均值
    for key in feature_importance:
        feature_importance[key] = np.mean(feature_importance[key], axis=0)
    
    return feature_importance

def visualize_feature_importance(feature_importance, save_dir):
    """可视化特征重要性"""
    os.makedirs(save_dir, exist_ok=True)
    
    for key, importance in feature_importance.items():
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.title(f'Feature Importance: {key}')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{key}_importance.png'))
        plt.close()