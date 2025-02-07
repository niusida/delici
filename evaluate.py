# evaluate.py
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, average_precision_score,
    accuracy_score
)
from utils import visualize_3d_volume, visualize_3d_attention
import pandas as pd
from scipy import stats

class ModelEvaluator:
    """模型评估器"""
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def evaluate_full(self, data_loader, class_names=None):
        """完整评估"""
        results = {}
        
        # 获取预测结果
        preds, labels, features = self._get_predictions(data_loader)
        
        # 基础指标
        results['accuracy'] = accuracy_score(labels, preds)
        results['confusion_matrix'] = confusion_matrix(labels, preds)
        results['classification_report'] = classification_report(
            labels, preds, 
            target_names=class_names if class_names else None,
            digits=4
        )
        
        # ROC和PR曲线
        results['roc_auc'] = self._compute_roc_curves(labels, preds, class_names)
        results['pr_auc'] = self._compute_pr_curves(labels, preds, class_names)
        
        # 特征分析
        results['feature_analysis'] = self._analyze_features(features, labels)
        
        # 保存可视化结果
        self._save_visualizations(results, class_names)
        
        return results
    
    def _get_predictions(self, data_loader):
        """获取模型预测结果"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for inputs, tabular, labels in data_loader:
                inputs = inputs.float().to(self.device)
                tabular = tabular.float().to(self.device)
                
                outputs = self.model(inputs, tabular)
                _, preds = outputs['logits'].max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_features.extend(outputs['features'].cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_features)
    
    def _compute_roc_curves(self, labels, preds, class_names):
        """计算ROC曲线"""
        n_classes = len(np.unique(labels))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # 为每个类别计算ROC曲线
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels == i, preds == i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(
                fpr[i], tpr[i],
                label=f'{class_names[i] if class_names else i} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'roc_curves.png'))
        plt.close()
        
        return roc_auc
    
    def _compute_pr_curves(self, labels, preds, class_names):
        """计算PR曲线"""
        n_classes = len(np.unique(labels))
        precision = dict()
        recall = dict()
        pr_auc = dict()
        
        # 为每个类别计算PR曲线
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(labels == i, preds == i)
            pr_auc[i] = average_precision_score(labels == i, preds == i)
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(
                recall[i], precision[i],
                label=f'{class_names[i] if class_names else i} (AP = {pr_auc[i]:.2f})'
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'pr_curves.png'))
        plt.close()
        
        return pr_auc
    
    def _analyze_features(self, features, labels):
        """分析特征分布"""
        results = {}
        
        # t-SNE降维可视化
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        for i in np.unique(labels):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                label=f'Class {i}'
            )
        plt.title('t-SNE Feature Visualization')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'tsne_features.png'))
        plt.close()
        
        # 特征统计分析
        results['feature_stats'] = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'p_values': []
        }
        
        # 计算每个特征的类别区分能力
        for i in range(features.shape[1]):
            f_stat, p_val = stats.f_oneway(
                *[features[labels == j, i] for j in np.unique(labels)]
            )
            results['feature_stats']['p_values'].append(p_val)
        
        return results
    
    def _save_visualizations(self, results, class_names):
        """保存评估结果可视化"""
        # 混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names if class_names else 'auto',
            yticklabels=class_names if class_names else 'auto'
        )
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 保存分类报告
        with open(os.path.join(self.save_dir, 'classification_report.txt'), 'w') as f:
            f.write(results['classification_report'])

def visualize_results(results_dict, save_dir):
    """可视化训练结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results_dict['train_loss'], label='Train')
    plt.plot(results_dict['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(results_dict['train_acc'], label='Train')
    plt.plot(results_dict['val_acc'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()