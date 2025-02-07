import torch

# 假设这是你的模型类
class AlzheimerCNN(torch.nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # ... 其他卷积层 ...
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 假设最后一层卷积
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            # 假设最后一层卷积之后是展平层
        )
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 56 * 56, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2)  # 输出两个类别
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def load_trained_model(model_path):
    model = AlzheimerCNN()
    # Load state dict
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # Load the new state dict
    model.load_state_dict(model_dict)
    return model


def get_classifier_image_weights_importance(model):
    # 获取最后一个卷积层的输出通道
    last_conv_layer = None
    for layer in model.conv_layers:
        if isinstance(layer, torch.nn.Conv2d):
            last_conv_layer = layer

    if last_conv_layer is not None:
        image_feature_dim = last_conv_layer.out_channels * 28 * 28  # 假设卷积层输出为 28x28
    else:
        raise ValueError("没有找到卷积层")

    # 这里添加计算图像特征重要性的代码
    # classifier_importance 的计算需要根据你的需求来实现
    classifier_importance = torch.randn(image_feature_dim)  # 示例，替换为实际计算
    sorted_indices = torch.argsort(classifier_importance, descending=True)
    return classifier_importance, sorted_indices

def main():
    model_path = 'E:/py test/austim/models/best_model_fold_1.pth'

    model = load_trained_model(model_path)
    classifier_importance, sorted_indices = get_classifier_image_weights_importance(model)
    print("图像特征重要性排序：", sorted_indices)

if __name__ == "__main__":
    main()
