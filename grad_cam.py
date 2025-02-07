# grad_cam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import visualize_3d_volume, visualize_3d_attention


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成3D Grad-CAM
        
        Args:
            input_tensor: 输入张量 [1, 1, D, H, W]
            target_class: 目标类别
            
        Returns:
            cam: 3D注意力图 [D, H, W]
        """
        # 前向传播
        model_output = self.model(input_tensor, None)  # 假设不使用tabular数据
        
        if target_class is None:
            target_class = model_output['logits'].argmax(dim=1)
        
        # 清除之前的梯度
        self.model.zero_grad()
        
        # 反向传播
        target = model_output['logits'][0, target_class]
        target.backward()
        
        # 计算权重
        weights = F.adaptive_avg_pool3d(self.gradients, 1)
        
        # 生成CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU
        
        # 标准化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0, 0].detach().cpu().numpy()

def visualize_grad_cam_3d(model, input_tensor, target_class, save_dir):
    """
    可视化3D Grad-CAM结果
    
    Args:
        model: 模型
        input_tensor: 输入张量 [1, 1, D, H, W]
        target_class: 目标类别
        save_dir: 保存目录
    """
    # 获取目标层
    target_layer = model.feature_extractor[-2]  # 假设使用feature_extractor的倒数第二层
    
    # 初始化GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # 生成CAM
    cam = grad_cam.generate_cam(input_tensor, target_class)
    
    # 获取原始输入
    input_data = input_tensor[0, 0].detach().cpu().numpy()
    
    # 可视化原始输入和CAM的叠加结果
    visualize_3d_attention(
        input_data,
        cam,
        save_path=f"{save_dir}/grad_cam_class_{target_class}.png"
    )
    
    # 保存3D CAM数据
    np.save(f"{save_dir}/grad_cam_class_{target_class}.npy", cam)
    
    # 生成正交视图
    d, h, w = cam.shape
    plt.figure(figsize=(15, 5))
    
    # 显示三个正交平面的CAM
    plt.subplot(131)
    plt.imshow(input_data[d//2], cmap='gray')
    plt.imshow(cam[d//2], cmap='jet', alpha=0.5)
    plt.title('Sagittal View')
    
    plt.subplot(132)
    plt.imshow(input_data[:, h//2], cmap='gray')
    plt.imshow(cam[:, h//2], cmap='jet', alpha=0.5)
    plt.title('Coronal View')
    
    plt.subplot(133)
    plt.imshow(input_data[:, :, w//2], cmap='gray')
    plt.imshow(cam[:, :, w//2], cmap='jet', alpha=0.5)
    plt.title('Axial View')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/grad_cam_orthogonal_class_{target_class}.png")
    plt.close()