import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class ConvNet_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 5, 2, 0)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 10, 7, 1, 1)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.conv3(x).squeeze()

        return x


def train(model, data_loader, optimizer):
    model.train()
    correct = 0
    total_loss = 0
    for _, (feature, label) in enumerate(data_loader):
        feature_g = feature.to(device)
        label_g = label.to(device)
        optimizer.zero_grad()
        output = model(feature_g)
        loss = F.cross_entropy(output, label_g, reduction='sum').to(device)
        total_loss += loss.item()
        pred = torch.argmax(output, 1)
        correct += pred.eq(label.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    correct /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    return correct, total_loss


def train_L1(model, data_loader, optimizer, factor):
    model.train()
    correct = 0
    total_loss = 0
    for _, (feature, label) in enumerate(data_loader):
        feature_g = feature.to(device)
        label_g = label.to(device)
        optimizer.zero_grad()
        output = model(feature_g)
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(abs(param))
        loss = F.cross_entropy(output, label_g, reduction='sum').to(device) + factor * regularization_loss
        total_loss += loss.item()
        pred = torch.argmax(output, 1)
        correct += pred.eq(label_g.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    correct /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    return correct, total_loss


def train_L2(model, data_loader, optimizer, factor):
    model.train()
    correct = 0
    total_loss = 0
    for _, (feature, label) in enumerate(data_loader):
        feature_g = feature.to(device)
        label_g = label.to(device)
        optimizer.zero_grad()
        output = model(feature_g)
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.sum(param ** 2)
        loss = F.cross_entropy(output, label_g, reduction='sum').to(device) + factor * regularization_loss
        total_loss += loss.item()
        pred = torch.argmax(output, 1)
        correct += pred.eq(label_g.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    correct /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    return correct, total_loss


def predict(model, data_loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for _, (feature, label) in enumerate(data_loader):
            feature_g = feature.to(device)
            label_g = label.to(device)
            output = model(feature_g)
            total_loss += F.cross_entropy(output, label_g, reduction='sum').item()
            pred = torch.argmax(output, 1)
            correct += pred.eq(label_g.view_as(pred)).sum().item()
    correct /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    return correct, total_loss


if __name__ == '__main__':
    save_dir = 'pic'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model = ConvNet()
    model.to(device)
    epoch = 10
    # lr = 0.0001
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1037,), (0.3081,))
                       ])),
        batch_size=600, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
        ])),
        batch_size=600, shuffle=True)
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for i in range(epoch):
        acc, loss = train_L1(model, train_loader, optimizer_adam, 0.01)
        test_acc, test_loss = predict(model, test_loader)
        train_acc_list.append(acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        test_loss_list.append(test_loss)
        print("Epoch {}".format(i + 1))
        print("train_acc: {:.4}%".format(acc * 100.0),
              '\t' "train_loss: {:.6}".format(loss))
        print("test_acc: {:.4}%".format(test_acc * 100.0),
              '\t' "test_loss: {:.6}".format(test_loss))

    plt.plot(range(1, epoch + 1), train_acc_list, 'g-', label='Train accuracy')
    plt.plot(range(1, epoch + 1), test_acc_list, 'b', label='Test accuracy')
    plt.title('Train and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(save_dir + '/accuracy.png')

    plt.clf()
    plt.plot(range(1, epoch + 1), train_loss_list, 'g-', label='Train loss')
    plt.plot(range(1, epoch + 1), test_loss_list, 'b', label='Test loss')
    plt.title('Train and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_dir + '/loss.png')
