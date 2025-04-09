import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# @save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

# Compose 其主要功能是将多个图像变换操作组合在一起
test_augs = torchvision.transforms.Compose([
    # 1. 原图像的尺寸大小不一致。 调整图像大小至256x256（保持长宽比）
    torchvision.transforms.Resize([256, 256]),
    
    # 2. 中心裁剪224x224区域（与训练输入尺寸一致）
    torchvision.transforms.CenterCrop(224),
    
    # 3. 将PIL图像转为PyTorch张量，并自动归一化到[0,1]范围
    torchvision.transforms.ToTensor(),
    
    # 4. 标准化处理（使用ImageNet的均值和标准差）
    #    - 输入应为[C,H,W]格式的张量
    #    - 对每个通道：(input - mean) / std
    normalize])

pretrained_net = torchvision.models.resnet18(pretrained=True)
print(pretrained_net.fc.in_features)

finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)


# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    # 创建训练数据加载器
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)  # 训练集需要打乱顺序
    
    # 创建测试数据加载器 
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)  # 测试集不需要打乱
    
    # 获取所有可用的GPU设备
    devices = d2l.try_all_gpus()
    # 使用交叉熵损失函数，reduction="none"表示不自动求平均或求和
    loss = nn.CrossEntropyLoss(reduction="none")
    
    # 参数分组设置：输出层使用更大学习率
    if param_group:
        # 获取除最后一层(fc层)外的所有参数
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        # 优化器设置：
        # - 其他层使用基础学习率
        # - 输出层使用10倍学习率
        # - 添加L2正则化(weight_decay)
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
            lr=learning_rate, weight_decay=0.001)
    else:
        # 不使用参数分组，所有参数使用相同学习率
        trainer = torch.optim.SGD(net.parameters(), 
                                 lr=learning_rate,
                                 weight_decay=0.001)
    
    # 调用训练函数，使用多GPU训练
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


if __name__ == '__main__':
    train_fine_tuning(finetune_net, 5e-5)
