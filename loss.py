import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.eps = 1e-8
        
    def forward(self, inputs, targets):
        # 从字典中获取logits
        if isinstance(inputs, dict):
            inputs = inputs['logits']
            
        # 添加数值稳定性
        inputs = torch.clamp(inputs, min=-100, max=100)
        
        # 计算每个类别的样本数量
        label_counts = torch.bincount(targets, minlength=self.num_classes)
        total_samples = label_counts.sum()
        
        # 计算权重
        weights = total_samples / (label_counts + self.eps)
        weights = weights / weights.sum()
        weights = weights.to(inputs.device)
        
        return F.cross_entropy(inputs, targets, weight=weights, reduction='mean')

class FocalLoss(nn.Module):
    def __init__(self, num_classes=3, alpha=None, gamma=2.0):
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        # 从字典中获取logits
        if isinstance(inputs, dict):
            inputs = inputs['logits']
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        self.alpha = self.alpha.to(inputs.device)
        at = self.alpha.gather(0, targets.data.view(-1))
        
        focal_loss = at * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.75):
        super().__init__()
        self.alpha = alpha
        self.ce = WeightedCrossEntropyLoss(num_classes=num_classes)
        self.focal = FocalLoss(num_classes=num_classes, gamma=2.0)
        
    def forward(self, inputs, targets):
        # 从字典中获取logits
        if isinstance(inputs, dict):
            inputs = inputs['logits']
            
        ce_loss = self.ce(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return self.alpha * ce_loss + (1 - self.alpha) * focal_loss

def get_loss_function(loss_type='ce', num_classes=3, **kwargs):
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型 ('ce' 或 'focal')
        num_classes: 类别数量
        **kwargs: 其他参数
    """
    if loss_type == 'ce':
        return WeightedCrossEntropyLoss(num_classes=num_classes)
    elif loss_type == 'focal':
        return FocalLoss(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")