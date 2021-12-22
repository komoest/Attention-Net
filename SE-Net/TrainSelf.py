import torch, torchvision
from torch import nn, optim
from SenetSelf import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


# 设置数据集
def load_dataset():
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),  # 图片随机垂直翻转
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])  # 归一化（-1， 1）可加快收敛
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.226, 0.224, 0.225))])

    # cifar10数据集
    train_dataset = datasets.CIFAR10(root='../data_hub/cifar10/data_1', train=True, transform=transform_train,
                                     download=False)
    test_dataset = datasets.CIFAR10(root='../data_hub/cifar10/data_1', train=False, transform=transform_test,
                                    download=False)
    # 返回训练集和验证集
    return train_dataset, test_dataset


batch_size = 200
lr = 0.01
epoch_num = 10
lr_step = 3

train_dataset, test_dataset = load_dataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# gpu加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SeNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
schedule = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5, last_epoch=-1)

loss_list = []  # 统计每个epoch每个batch的loss
# 训练
time_start = time.time()
for epoch in range(epoch_num):
    total_train, correct_train, loss_train = 0, 0, 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        loss_list.append(loss.item())
        total_train += labels.size(0)
        correct_train += (outputs.argmax(dim=1) == labels).sum().item()
        train_acc = 100.0 * correct_train / total_train

        if ((i + 1) % 100 == 0):        # 100个train——loader打印一次
            print(
                "[%d epoch %d / %d step] | loss:%.6f | accuracy = %6.3f %%" % (
                    epoch + 1, i + 1, len(train_loader), loss_train / (i + 1), train_acc))  # 第几步，loss，accuracy
    lr_0 = optimizer.param_groups[0]["lr"]  # 打印当前epoch的学习率
    print("learn rate : %.15f" % lr_0)

time_end = time.time()
print("cost time : %s" % (time_end - time_start))  # 改动

model.eval()
correct_test, total_test = 0, 0

with torch.no_grad():
    print("========================test============================")
    for inputs, labels in test_loader:      # 不用epoch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        total_test += inputs.size(0)
        correct_test += (outputs.argmax(dim=1) == labels).sum().item()
print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct_test / total_test))

