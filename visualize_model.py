import torch
from model import MedicalCNNTransformer
import os
import onnx
import torch.nn as nn

def save_model_for_netron(model_path, output_path):
    """
    保存模型为ONNX格式以便使用Netron可视化
    
    Args:
        model_path: 训练好的模型路径
        output_path: 输出的ONNX文件路径
    """
    try:
        # 加载模型
        print(f"正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 获取模型状态字典
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint
            
        # 创建模型实例 - 使用默认参数
        model = MedicalCNNTransformer(
            input_shape=(121, 145, 121),
            num_classes=3
        )
        
        # 加载状态字典
        model.load_state_dict(state_dict)
        model.eval()
        
        # 创建示例输入
        dummy_image = torch.randn(1, 1, 121, 145, 121)  # [B, C, D, H, W]
        
        # 导出为ONNX格式
        torch.onnx.export(
            model,
            dummy_image,
            output_path,
            input_names=['image'],
            output_names=['logits', 'loc_features'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
                'loc_features': {0: 'batch_size'}
            },
            do_constant_folding=True,
            opset_version=14,
            verbose=True,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
        
        # 验证导出的ONNX模型
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"\n模型已成功保存为ONNX格式: {output_path}")
        print("\n模型结构概览:")
        print_model_structure(model)
        print("\n您可以使用Netron打开ONNX文件来可视化完整的模型结构")
        print("Netron下载地址: https://netron.app/")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

def print_model_structure(model, indent=0):
    """
    打印模型的层次结构
    
    Args:
        model: PyTorch模型
        indent: 缩进级别
    """
    def get_layer_info(layer):
        """获取层的基本信息"""
        info = layer.__class__.__name__
        if isinstance(layer, (nn.Conv3d, nn.Linear)):
            params = sum(p.numel() for p in layer.parameters())
            info += f" (参数数量: {params:,})"
        return info
    
    # 打印当前层
    print("  " * indent + "└─ " + get_layer_info(model))
    
    # 递归打印子层
    if hasattr(model, "_modules"):
        for name, layer in model._modules.items():
            if layer is not None:
                print("  " * (indent + 1) + "├─ " + name + ":")
                print_model_structure(layer, indent + 2)

def analyze_model_complexity(model):
    """
    分析模型的复杂度
    
    Args:
        model: PyTorch模型
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n模型复杂度分析:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # 设置路径
    model_dir = r'E:\py test\austim\models'
    output_dir = r'E:\py test\austim\model_visualization'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有fold的模型
    for fold in range(1, 6):
        model_path = os.path.join(model_dir, f'fold_{fold-1}_best.pt')
        if os.path.exists(model_path):
            print(f"\n处理 fold {fold} 的模型...")
            
            # 设置输出路径
            output_path = os.path.join(output_dir, f'model_fold_{fold}.onnx')
            
            # 保存并可视化模型
            save_model_for_netron(model_path, output_path)
            
            # 创建模型实例用于分析
            model = MedicalCNNTransformer(
                input_shape=(121, 145, 121),
                num_classes=3
            )
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            
            # 分析模型复杂度
            analyze_model_complexity(model)