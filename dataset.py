import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pandas as pd

class BrainDataset(Dataset):
    def __init__(self, data_dir):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录，应包含images和tabular子目录
        """
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.tabular_file = os.path.join(data_dir, 'tabular', 'clinical_data.csv')
        
        # 加载图像文件列表
        self.image_files = []
        for file in os.listdir(self.image_dir):
            if file.endswith('.nii.gz'):
                self.image_files.append(file)
                
        # 加载表格数据
        if os.path.exists(self.tabular_file):
            self.tabular_data = pd.read_csv(self.tabular_file)
        else:
            # 如果没有表格数据，创建空的DataFrame
            self.tabular_data = pd.DataFrame(index=range(len(self.image_files)))
            self.tabular_data['age'] = 0
            self.tabular_data['sex'] = 0
            self.tabular_data['education'] = 0
            self.tabular_data['diagnosis'] = 0
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = nib.load(img_path)
        img_data = img.get_fdata()
        
        # 标准化图像数据
        img_data = (img_data - img_data.mean()) / (img_data.std() + 1e-8)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img_data).float()
        img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度
        
        # 获取表格数据
        tabular = self.tabular_data.iloc[idx][['age', 'sex', 'education']].values
        tabular = torch.from_numpy(tabular).float()
        
        # 获取标签
        label = self.tabular_data.iloc[idx]['diagnosis']
        label = torch.tensor(label).long()
        
        return img_tensor, label, tabular 