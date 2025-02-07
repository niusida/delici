# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SELayer3D(nn.Module):
    """SE注意力层,用于通道特征选择"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self._attention_weights = None

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        
        # 存储权重时保持维度
        self._attention_weights = y.detach().view(b, c)
            
        return x * y.expand_as(x)
    
    def get_attention_weights(self, x=None):
        """获取通道注意力权重"""
        if x is not None:
            with torch.no_grad():
                b, c, _, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                weights = self.fc(y)
                return weights
        return self._attention_weights

class SpatialAttention3D(nn.Module):
    """空间注意力,用于体素定位"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        self._attention_map = None

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        attention_map = self.sigmoid(y)
        
        # 不管是否在训练模式下都保存注意力图
        self._attention_map = attention_map.detach()
            
        return x * attention_map
    
    def get_attention_map(self):
        return self._attention_map

class ResBlock3D(nn.Module):
    """3D残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 第一个卷积使用stride
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        # 第二个卷积不使用stride
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # shortcut路径
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class TabularFeatureExtractor(nn.Module):
    """表格特征提取器"""
    def __init__(self, input_dim, hidden_dims=[64, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
            
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        # 处理输入维度
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        elif x.dim() == 2 and x.size(1) != self.net[0].in_features:
            x = x.view(x.size(0), -1)  # 确保维度匹配
        return self.net(x)

class MedicalCNNTransformer(nn.Module):
    """改进后的医学图像分析模型"""
    def __init__(self, input_shape=(121, 145, 121), num_classes=3, tabular_dim=1):
        super().__init__()
        
        # 初始卷积层 - 使用更大的stride来减小特征图
        self.conv_init = nn.Conv3d(1, 32, kernel_size=7, stride=4, padding=3)
        self.bn_init = nn.BatchNorm3d(32)
        
        # 残差块 - 增加stride以减小特征图
        self.layer1 = ResBlock3D(32, 64, stride=2)
        self.layer2 = ResBlock3D(64, 128, stride=2)
        self.layer3 = ResBlock3D(128, 256, stride=2)
        
        # SE注意力层
        self.se1 = SELayer3D(64)
        self.se2 = SELayer3D(128)
        self.se3 = SELayer3D(256)
        
        # 空间注意力层
        self.spatial_att = SpatialAttention3D()
        
        # 表格特征处理
        self.tabular_net = TabularFeatureExtractor(tabular_dim)
        
        # 计算卷积后特征图大小
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            dummy = self.get_conv_features(dummy)
            conv_features_dim = dummy.view(1, -1).size(1)
        
        # 特征融合和分类层
        fusion_dim = conv_features_dim + self.tabular_net.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 添加特征存储开关
        self.store_features = False
        self.feature_maps = {}
    
    def get_conv_features(self, x):
        """获取卷积特征
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 展平的卷积特征
        """
        # 初始特征提取
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        # 第一层特征
        x = self.layer1(x)
        x = self.se1(x)
        
        # 第二层特征
        x = self.layer2(x)
        x = self.se2(x)
        
        # 第三层特征 
        x = self.layer3(x)
        x = self.se3(x)
        
        # 空间注意力
        x = self.spatial_att(x)
        
        # 展平特征
        return x.view(x.size(0), -1)
        
    def forward(self, x, tabular):
        # 获取卷积特征
        x = self.get_conv_features(x)
        
        # 获取表格特征
        tabular_features = self.tabular_net(tabular)
        
        # 特征融合
        combined = torch.cat([x, tabular_features], dim=1)
        
        # 分类
        logits = self.classifier(combined)
        
        if self.store_features:
            return {
                'logits': logits,
                'features': self.feature_maps
            }
        return {
            'logits': logits
        }
    
    def enable_feature_storing(self, enable=True):
        """控制是否存储特征图"""
        self.store_features = enable
        if not enable:
            self.feature_maps.clear()
    
    def get_feature_maps(self):
        """获取所有层的特征图"""
        return self.feature_maps
    
    def get_channel_attention_weights(self):
        """获取所有SE层的通道注意力权重"""
        return {
            'layer1': self.se1.get_attention_weights,
            'layer2': self.se2.get_attention_weights,
            'layer3': self.se3.get_attention_weights
        }
    
    def get_spatial_attention_map(self):
        """获取空间注意力图"""
        return self.spatial_att.get_attention_map()
    
    def clear_feature_maps(self):
        """清除存储的特征图"""
        self.feature_maps.clear()
    
    def train(self, mode=True):
        """重写train方法,在切换到评估模式时清除特征图"""
        super().train(mode)
        if not mode:  # 切换到eval模式时
            self.clear_feature_maps()
        return self

if __name__ == "__main__":
    # 测试代码
    model = MedicalCNNTransformer(input_shape=(121, 145, 121), num_classes=2)
    dummy_image = torch.randn(4, 1, 121, 145, 121)
    dummy_tabular = torch.randn(4, 4)
    output = model(dummy_image, dummy_tabular)
    print("输出形状:", output['logits'].shape)

    # 需要分析特征时：
    model.enable_feature_storing(True)
    output = model(dummy_image, dummy_tabular)
    features = output['features']
    model.enable_feature_storing(False)