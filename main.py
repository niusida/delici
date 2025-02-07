import copy
import random
import torch
import numpy as np
import os
import argparse
import wandb
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import classification_report

from model import MedicalCNNTransformer
from data_preprocessing import MRIDataset, MRIPreprocessor
from train import Trainer
from utils import setup_logging, plot_confusion_matrix, visualize_attention, analyze_feature_importance, visualize_feature_importance
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='图像数据目录')
    parser.add_argument('--table_dir', type=str, required=True, help='表格数据目录')
    parser.add_argument('--model_dir', type=str, required=True, help='模型保存目录')
    parser.add_argument('--log_dir', type=str, required=True, help='日志保存目录')
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度训练')
    
    parser.add_argument('--config', type=str, help='配置文件路径(可选)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--fold', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--mode', type=str, default='train', help='运行模式: train/test')
    parser.add_argument('--checkpoint', type=str, help='测试模式下的模型检查点路径')
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_loaders(dataset, fold_idx, fold_splits, batch_size, num_workers):
    """获取训练和验证数据加载器"""
    train_idx, val_idx = fold_splits[fold_idx]
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    
    return train_loader, val_loader

def train_fold(fold_idx, model, dataset, config, logger, args):
    """训练一个fold"""
    # 更新配置，添加必要的键
    config['fold'] = fold_idx
    config['model_dir'] = args.model_dir  # 从命令行参数获取
    
    # 创建数据划分
    kf = KFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=42)
    
    # 获取当前fold的训练集和验证集索引
    train_indices = []
    val_indices = []
    for i, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        if i == fold_idx:
            train_indices = train_idx
            val_indices = val_idx
            break
    
    # 创建数据加载器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn
    )
    
    # 初始化优化���和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 创建训练器 - 使用关键字参数
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config['device'],
        config=config,
        logger=logger
    )
    
    # 训练模型
    best_model_state_dict, metrics = trainer.train(
        train_loader,
        val_loader,
        epochs=config['training']['num_epochs']
    )
    
    # 保存模型
    model_save_path = os.path.join(args.model_dir, f'fold_{fold_idx}_best.pt')
    torch.save(best_model_state_dict, model_save_path)  # 直接保存state_dict
    
    return best_model_state_dict, metrics

def test(model, test_loader, config, logger):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for data, target, tabular in test_loader:
            data = data.to(config['device'])
            target = target.to(config['device'])
            tabular = tabular.to(config['device'])
            
            outputs = model(data, tabular)
            # 从输出字典中获取logits
            logits = outputs['logits']
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    # 记录结果
    logger.info(f"\n测试结果:")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\n分类报告:")
    logger.info(classification_report(all_targets, all_preds))
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets
    }

def analyze_results(model, dataset, config, logger):
    """分析模型结果"""
    # 准备数据加载器
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'])
    
    # 分析特征重要性
    feature_importance = analyze_feature_importance(model, loader)
    
    # 记录结果
    logger.info("\nFeature Importance Analysis:")
    for comparison, results in feature_importance.items():
        logger.info(f"\n{comparison}:")
        logger.info("Top 10 most important features:")
        for idx, (feature_idx, importance) in enumerate(zip(*results['top_features'])):
            logger.info(f"{idx+1}. Feature {feature_idx}: {importance:.4f}")
    
    # 保存可视化结果
    visualize_feature_importance(feature_importance, 
                               save_dir=os.path.join(config['log_dir'], 'feature_importance'))

def collate_fn(batch):
    """处理包含None的batch"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    # 解析参数
    args = parse_args()
    
    # 配置设置
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        # 使用命令行参数创���默认配置
        config = {
            'data': {
                'image_dir': args.image_dir,
                'table_path': args.table_dir,
                'label_column': 'label'
            },
            'training': {
                'batch_size': 16,
                'num_workers': 4,
                'lr': 1e-4,
                'weight_decay': 0.01,
                'num_epochs': 100,
                'patience': 15,
                'n_folds': 5,
                'use_amp': args.use_amp
            },
            'model': {
                'input_shape': (121, 145, 121),
                'num_classes': 3
            },
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'max_grad_norm': 1.0  # 梯度裁剪阈值
        }
    
    # 设置设置
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志目录
    save_dir = os.path.join(
        args.log_dir,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logging(save_dir)
    
    # 修正模型文件路径检查
    model_dir = os.path.join(os.path.dirname(args.log_dir), 'models')
    found_model = False
    all_accuracies = []
    
    if os.path.exists(model_dir):
        for fold_idx in range(config['training']['n_folds']):
            model_path = os.path.join(model_dir, f'fold_{fold_idx}_best.pt')
            if os.path.exists(model_path):
                found_model = True
                logger.info(f"找到已保存的模型: {model_path}")
                
                # 创建模型
                model = MedicalCNNTransformer(
                    input_shape=config['model']['input_shape'],
                    num_classes=config['model']['num_classes']
                )
                
                # 加载模型
                model.load_state_dict(torch.load(model_path))
                model = model.to(config['device'])
                
                # 准备数据集
                dataset = MRIDataset(
                    config['data']['image_dir'],
                    config['data']['table_path'],
                    config['data']['label_column'],
                    target_shape=(121, 145, 121)
                )
                
                # 准备测试数据加载器
                test_loader = DataLoader(
                    dataset,
                    batch_size=config['training']['batch_size'],
                    shuffle=False,
                    num_workers=config['training']['num_workers']
                )
                
                # 评估模型
                model.eval()
                correct = 0
                total = 0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for data, target, tabular in test_loader:
                        data = data.to(config['device'])
                        target = target.to(config['device'])
                        tabular = tabular.to(config['device'])
                        
                        outputs = model(data, tabular)
                        logits = outputs['logits']
                        _, predicted = torch.max(logits, 1)
                        
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                
                accuracy = correct / total
                all_accuracies.append(accuracy)
                
                logger.info(f"Fold {fold_idx} 评估结果:")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info("\n分类报告:")
                logger.info(classification_report(all_targets, all_preds))
        
        if found_model:
            logger.info(f"\n所有Fold的平均准确率: {np.mean(all_accuracies):.4f}")
            return
    
    # 如果没有找到模型，继续原有的训练流程
    if not found_model:
        logger.info("未找到已保存的模型，开始训练...")
        
        # 创建数据集
        dataset = MRIDataset(
            config['data']['image_dir'],
            config['data']['table_path'],
            config['data']['label_column'],
            target_shape=(121, 145, 121)
        )
        
        # 对每个fold进行训练
        for fold_idx in range(config['training']['n_folds']):
            logger.info(f"\nTraining Fold {fold_idx+1}/{config['training']['n_folds']}")
            
            # 创建新的模型实例
            model = MedicalCNNTransformer(
                input_shape=config['model']['input_shape'],
                num_classes=config['model']['num_classes']
            )
            model = model.to(config['device'])
            
            # 训练当前fold
            model_state_dict, fold_metrics = train_fold(fold_idx, model, dataset, config, logger, args)
            
            # 保存模型
            model_save_path = os.path.join(args.model_dir, f'fold_{fold_idx}_best.pt')
            torch.save(model_state_dict, model_save_path)  # 直接保存state_dict
            
            # 记录准确率
            if fold_metrics and 'valid' in fold_metrics:
                all_accuracies.append(fold_metrics['valid']['accuracy'])
    
    # 初始化wandb
    if config.get('use_wandb', False):
        wandb.init(project=config['project_name'], config=config)
    
    # 准备测试数据加载器
    test_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    import torchvision.transforms as transforms




    metrics = test(model, test_loader, config, logger)
    logger.info(f"Test Results: {metrics}")
    
    # 视化结果
    if config.get('visualize', False):
        vis_save_dir = os.path.join(args.log_dir, 'visualizations')
        os.makedirs(vis_save_dir, exist_ok=True)
        visualize_attention(
            model,
            test_loader,
            save_dir=vis_save_dir
        )

if __name__ == '__main__':
    main()