import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'CNN_raw'
save_file = model_name + '/checkpoint_' + model_name + '.pth.tar'


class ConvNet(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 10, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x)).squeeze()

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def train(model, data_loader, optimizer):
    model.train()
    correct = 0
    total_loss = 0
    for _, (feature, label) in enumerate(tqdm(data_loader)):
        feature = feature.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(feature)
        loss = F.cross_entropy(output, label, reduction='sum')
        total_loss += loss.item()
        pred = torch.argmax(output, 1)
        correct += pred.eq(label.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
    correct /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    return correct, total_loss


def val(model, data_loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for _, (feature, label) in enumerate(tqdm(data_loader)):
            feature = feature.to(device)
            label = label.to(device)
            output = model(feature)
            total_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = torch.argmax(output, 1)
            correct += pred.eq(label.view_as(pred)).sum().item()
    correct /= len(data_loader.dataset)
    total_loss /= len(data_loader.dataset)
    return correct, total_loss


transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])


if __name__ == '__main__':
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    model = ConvNet().to(device)
    model.weight_init(0, 0.02)
    epoch = 35
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform), batch_size=32, shuffle=True)
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    start_epoch = 0
    if os.path.exists(save_file):
        print('Loading existing models....')
        checkpoint = torch.load(save_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        train_acc_list = checkpoint['train_acc_list']
        test_acc_list = checkpoint['test_acc_list']
        train_loss_list = checkpoint['train_loss_list']
        test_loss_list = checkpoint['test_loss_list']
    out = open(model_name + '/' + model_name + '_out.txt', 'w')

    for i in range(start_epoch, epoch):
        acc, loss = train(model, train_loader, optimizer)
        test_acc, test_loss = val(model, test_loader)
        train_acc_list.append(acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        test_loss_list.append(test_loss)
        print("Epoch {}".format(i + 1))
        print("train_acc: {:.4}%".format(acc * 100.0),
              '\t' "train_loss: {:.6}".format(loss))
        print("test_acc: {:.4}%".format(test_acc * 100.0),
              '\t' "test_loss: {:.6}".format(test_loss))
        out.write("Epoch {}".format(i + 1) + '\n')
        out.write("train_acc: {:.4}%".format(acc * 100.0) + '\t' + "train_loss: {:.6}".format(loss) + '\n')
        out.write("test_acc: {:.4}%".format(test_acc * 100.0) + '\t' + "test_loss: {:.6}".format(test_loss) + '\n')
    out.close()
    plt.plot(range(1, epoch + 1), train_acc_list, 'g-', label='Train accuracy')
    plt.plot(range(1, epoch + 1), test_acc_list, 'b', label='Test accuracy')
    plt.title('Train and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(model_name + '/' + model_name + '_accuracy.png')

    plt.clf()
    plt.plot(range(1, epoch + 1), train_loss_list, 'g-', label='Train loss')
    plt.plot(range(1, epoch + 1), test_loss_list, 'b', label='Test loss')
    plt.title('Train and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(model_name + '/' + model_name + '_loss.png')

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list,
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list
    }, save_file)
