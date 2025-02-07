# data_preprocessing.py
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate
import elasticdeform
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader, random_split
import logging

class MRIPreprocessor:
    """MRI图像预处理器"""
    def __init__(self, target_shape=(121, 145, 121)):
        self.target_shape = target_shape
        
    def __call__(self, volume):
        try:
            # 1. 处理NaN和Inf值
            volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 2. 调整大小
            volume = self.resize_volume(volume)
            
            # 3. 标准化
            if volume.max() != volume.min():
                volume = (volume - volume.min()) / (volume.max() - volume.min())
            else:
                volume = np.zeros_like(volume)
            
            return volume
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def resize_volume(self, volume):
        """调整体积大小到目标尺寸"""
        try:
            current_shape = volume.shape
            scale_factors = [t/c for t, c in zip(self.target_shape, current_shape)]
            resized_volume = zoom(volume, scale_factors, order=1)  # 使用线性插值
            return np.ascontiguousarray(resized_volume)
        except Exception as e:
            print(f"Error in resize_volume: {str(e)}")
            raise
    
    def skull_stripping(self, volume):
        """头骨剥离"""
        # 实现头骨剥离算法
        return volume
    
    def bias_field_correction(self, volume):
        """偏置场校正"""
        # 实现N4偏置场校正
        return volume
    
    def intensity_normalization(self, volume):
        """强度标准化"""
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume

class MRIDataset(Dataset):
    """MRI数据集"""
    def __init__(self, image_dir, table_path, label_column, target_shape=(121, 145, 121), transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.preprocessor = MRIPreprocessor(target_shape=target_shape)
        
        # 加载数据
        self.filepaths, self.labels, self.features = self._load_dataset(table_path, label_column)
        
        # 记录无效样本
        self.invalid_samples = set()
        
    def __getitem__(self, idx):
        try:
            # 如果是已知的无效样本，跳过
            if idx in self.invalid_samples:
                return None
                
            # 加载图像
            img_path = self.filepaths[idx]
            nifti_img = nib.load(img_path)
            volume = np.ascontiguousarray(nifti_img.get_fdata())
            
            # 预处理
            volume = self.preprocessor(volume)
            
            # 确保数据类型和内存布局正确
            volume = np.ascontiguousarray(volume, dtype=np.float32)
            
            # 检查并修复异常值
            if np.isnan(volume).any() or np.isinf(volume).any():
                # 替换NaN和Inf值为0或平均值
                volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=-1.0)
                print(f"Warning: Fixed NaN/Inf values in sample {idx}")
            
            # 标准化
            if volume.max() != volume.min():
                volume = (volume - volume.min()) / (volume.max() - volume.min())
            else:
                volume = np.zeros_like(volume)
            
            # 转换为张量
            volume = torch.from_numpy(volume).float().unsqueeze(0)
            
            # 获取标签和特征
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            feature = torch.tensor([self.features[idx]], dtype=torch.float32)
            
            # 最终检查
            if torch.isnan(volume).any() or torch.isinf(volume).any():
                print(f"Warning: Sample {idx} still contains NaN/Inf after preprocessing")
                self.invalid_samples.add(idx)
                return None
                
            return volume, label, feature
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            print(f"File path: {self.filepaths[idx]}")
            self.invalid_samples.add(idx)
            return None
    
    def __len__(self):
        return len(self.filepaths)
    
    def _load_dataset(self, table_path, label_column):
        """加载数据集"""
        try:
            # 直接读取train.xlsx
            excel_file = os.path.join(table_path, 'train.xlsx')
            if not os.path.exists(excel_file):
                raise ValueError(f"找不到文件: {excel_file}")
            
            df = pd.read_excel(excel_file)
            
            valid_filepaths = []
            valid_labels = []
            valid_features = []
            
            for idx, row in df.iterrows():
                patient_id = str(int(row['patient_id']))
                img_path = os.path.join(self.image_dir, 'train', f"{patient_id}.nii")
                
                if os.path.exists(img_path):
                    valid_filepaths.append(img_path)
                    valid_labels.append(row[label_column])
                    # 提取meanFD作为额��特征
                    valid_features.append(row['meanFD'])
                else:
                    print(f"警告: 找不到图像文件 {img_path}")
            
            if not valid_filepaths:
                raise ValueError("没有找到匹配的图像文件")
            
            return valid_filepaths, np.array(valid_labels), np.array(valid_features)
            
        except Exception as e:
            print(f"加载数据集时发生错误: {str(e)}")
            if 'df' in locals():
                print(f"可用的列名: {df.columns.tolist()}")
            raise

class Transform3D:
    """3D数据增强"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, volume):
        if np.random.random() < self.p:
            transforms = [
                self.random_rotate,
                self.random_flip,
                self.random_noise,
                self.random_brightness,
                self.elastic_deform,
                self.random_zoom,
                self.random_shift,
                self.random_contrast,
                self.random_gamma
            ]
            
            # 随机应用3-4个变换
            num_transforms = np.random.randint(3, 5)
            for transform in np.random.choice(transforms, num_transforms, replace=False):
                volume = transform(volume)
                
        return volume
    
    def random_rotate(self, volume):
        """随机旋转"""
        angle = np.random.uniform(-10, 10)
        return rotate(volume, angle, axes=(1,2), reshape=False)
    
    def random_flip(self, volume):
        """随机翻转"""
        axis = np.random.choice([0, 1, 2])
        return np.flip(volume, axis=axis)
    
    def random_noise(self, volume):
        """随机噪声"""
        noise = np.random.normal(0, 0.01, volume.shape)
        return volume + noise
    
    def random_brightness(self, volume):
        """随机亮度调整"""
        factor = np.random.uniform(0.8, 1.2)
        return volume * factor
    
    def elastic_deform(self, volume):
        """弹性变形"""
        return elasticdeform.deform_random_grid(volume, sigma=2, points=3)
    
    def random_zoom(self, volume):
        """随机缩放"""
        scale = np.random.uniform(0.8, 1.2)
        return zoom(volume, scale)
        
    def random_shift(self, volume):
        """随机平移"""
        shift = np.random.randint(-10, 10, size=3)
        return shift_3d(volume, shift)
        
    def random_contrast(self, volume):
        """随机对比度调整"""
        factor = np.random.uniform(0.7, 1.3)
        mean = volume.mean()
        return (volume - mean) * factor + mean
        
    def random_gamma(self, volume):
        """随机gamma校正"""
        gamma = np.random.uniform(0.7, 1.3)
        return np.power(volume, gamma)

def get_data_loaders(data_root, batch_size=8, num_workers=4):
    """创建数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
    
    Returns:
        train_loader, val_loader, test_loader: 训练集、验证集和测试集的数据加载器
    """
    # 创建数据集
    image_dir = os.path.join(data_root, 'Image')  # 修改为实际的目录名
    table_path = os.path.join(data_root, 'Table')  # 修改为实际的目录名
    
    logging.info(f"图像目录: {image_dir}")
    logging.info(f"表格目录: {table_path}")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"找不到图像目录: {image_dir}")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"找不到表格目录: {table_path}")
    
    # 创建完整数据集
    full_dataset = MRIDataset(
        image_dir=image_dir,
        table_path=table_path,
        label_column='label'
    )
    
    logging.info(f"数据集大小: {len(full_dataset)}")
    
    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader