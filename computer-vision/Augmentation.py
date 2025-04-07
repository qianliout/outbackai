import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('/Users/liuqianli/work/python/outbackai/img/cat1.jpg')
d2l.plt.imshow(img)

# def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
#     Y = [aug(img) for _ in range(num_rows * num_cols)]
#     d2l.show_images(Y, num_rows, num_cols, scale=scale)
#
# apply(img, torchvision.transforms.RandomHorizontalFlip())
#
# apply(img, torchvision.transforms.RandomVerticalFlip())
#
# shape_aug = torchvision.transforms.RandomResizedCrop(
#     (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)
# 
# apply(img, torchvision.transforms.ColorJitter(
#     brightness=0.5, contrast=0, saturation=0, hue=0))
#
# apply(img, torchvision.transforms.ColorJitter(
#     brightness=0, contrast=0, saturation=0, hue=0.5))
#
# color_aug = torchvision.transforms.ColorJitter(
#     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)
#
# augs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)
#
# all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
#                                           download=True)
# d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
#
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
    
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')



def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)

if __name__ == '__main__':
    batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
    net.apply(d2l.init_cnn)
    train_with_data_aug(train_augs, test_augs, net)