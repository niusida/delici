# train.py
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import copy
import os
from torchvision import transforms
from loss import CombinedLoss
from torch.utils.data import DataLoader

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

class Trainer:
    def __init__(self, model, optimizer, criterion, device, config, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger
        
        # 使用AMP - 更新GradScaler初始化
        self.use_amp = config['training']['use_amp']
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            
        # 只在训练模式（有optimizer）时设置学习率调度器
        if self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=5
            )
            
            # 初始化最佳模型和性能
            self.best_valid_loss = float('inf')
            self.best_model = None
            
            # 初始化早停相关参数
            self.patience = config['training'].get('patience', 15)  # 从配置中获取patience，默认15
            self.early_stop_counter = 0
            self.should_stop = False
        
    def train(self, train_loader, valid_loader, epochs=None):
        """训练模型
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            epochs: 训练轮数，如果为None则使用config中的设置
            
        Returns:
            tuple: (best_model_state_dict, metrics_dict)
                - best_model_state_dict: 最佳模型的状态字典
                - metrics_dict: 包含训练和验证指标的字典
        """
        if epochs is None:
            epochs = self.config['training']['num_epochs']
            
        best_metrics = None
            
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # 训练阶段
            train_metrics = self._train_epoch(train_loader)
            print(f'Training Loss: {train_metrics["loss"]:.4f}')
            print(f'Training Accuracy: {train_metrics["accuracy"]*100:.2f}%')
            
            # 验证阶段
            valid_metrics = self._validate(valid_loader)
            print(f'Validation Loss: {valid_metrics["loss"]:.4f}')
            print(f'Validation Accuracy: {valid_metrics["accuracy"]*100:.2f}%')
            
            # 更新学习率
            self.scheduler.step(valid_metrics['loss'])
            
            # 检查是否需要保存最佳模型
            if valid_metrics['loss'] < self.best_valid_loss:
                self.best_valid_loss = valid_metrics['loss']
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0  # 重置早停计数器
                best_metrics = {
                    'train': train_metrics,
                    'valid': valid_metrics,
                    'epoch': epoch + 1
                }
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.best_valid_loss,
                }, os.path.join(self.config['model_dir'], 'best_model.pth'))
            else:
                self.early_stop_counter += 1
                
            # 检查是否应该早停
            if self.early_stop_counter >= self.patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                if self.logger:
                    self.logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
                
            # 记录到wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'valid_loss': valid_metrics['loss'],
                    'valid_accuracy': valid_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'early_stop_counter': self.early_stop_counter  # 添加早停计数器到日志
                })
        
        # 确保返回最佳模型和对应的指标
        return self.best_model, best_metrics
    
    def _train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        metrics = MetricTracker()
        
        # 添加进度条
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target, tabular) in enumerate(pbar):
            # 使用non_blocking加速数据传输
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            tabular = tabular.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            if self.use_amp:
                with autocast():
                    output = self.model(data, tabular)
                    loss = self.criterion(output['logits'], target)
                    
                    # 计算准确率
                    pred = output['logits'].argmax(dim=1)
                    correct = pred.eq(target).sum().item()
                    total = target.size(0)
                    accuracy = correct / total
                    
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data, tabular)
                loss = self.criterion(output['logits'], target)
                
                # 计算准确率
                pred = output['logits'].argmax(dim=1)
                correct = pred.eq(target).sum().item()
                total = target.size(0)
                accuracy = correct / total
                
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 更新指标
            metrics.update('loss', loss.item(), 1)
            metrics.update('accuracy', accuracy, 1)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': metrics.avg('loss'),
                'acc': f"{100. * metrics.avg('accuracy'):.2f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # 清理特征图释放内存    
        self.model.clear_feature_maps()
        torch.cuda.empty_cache()  # 清理GPU缓存
        
        return metrics.result()
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        metrics = MetricTracker()
        
        with torch.no_grad():
            for data, target, tabular in val_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                tabular = tabular.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        output = self.model(data, tabular)
                        loss = self.criterion(output['logits'], target)
                else:
                    output = self.model(data, tabular)
                    loss = self.criterion(output['logits'], target)
                
                # 计算准确率
                pred = output['logits'].argmax(dim=1)
                correct = pred.eq(target).sum().item()
                total = target.size(0)
                accuracy = correct / total
                
                # 更新指标
                metrics.update('loss', loss.item(), 1)
                metrics.update('accuracy', accuracy, 1)
        
        return metrics.result()