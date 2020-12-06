import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'ResNet50_CBAM'
save_file = 'results/' + model_name + '/checkpoint_' + model_name + '.pth.tar'
save_dir = 'results/' + model_name + '/'
epoch = 50


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
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


def load_model(model_name):
    if model_name == 'CNN_raw':
        model = models.ConvNet()
        model.weight_init(0, 0.02)
    elif model_name == 'CNN_CBAM':
        model = models.ConvNet_CBAM()
        model.weight_init(0, 0.02)
    elif model_name == 'ResNet18_raw':
        model = models.resnet18()
    elif model_name == 'ResNet18_CBAM':
        model = models.resnet18_CBAM()
    elif model_name == 'ResNet34_raw':
        model = models.resnet34()
    elif model_name == 'ResNet34_CBAM':
        model = models.resnet34_CBAM()
    elif model_name == 'ResNet50_raw':
        model = models.resnet50()
    elif model_name == 'ResNet50_CBAM':
        model = models.resnet50_CBAM()
    else:
        raise RuntimeError('Unknown model type!')

    return model


if __name__ == '__main__':
    if not os.path.exists('results/' + model_name):
        os.mkdir('results/' + model_name)
    model = load_model(model_name).to(device)
    lr = 0.001
    if model_name == 'CNN_raw' or model_name == 'CNN_CBAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        schedular = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform), batch_size=128, shuffle=True)
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
        scheduler = checkpoint['scheduler']
    out = open(save_dir + model_name + '_out.txt', 'a+')

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
        scheduler.step()
    out.close()
    plt.plot(range(1, epoch + 1), train_acc_list, 'g-', label='Train accuracy')
    plt.plot(range(1, epoch + 1), test_acc_list, 'b', label='Test accuracy')
    plt.title('Train and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(save_dir + model_name + '_accuracy.png')

    plt.clf()
    plt.plot(range(1, epoch + 1), train_loss_list, 'g-', label='Train loss')
    plt.plot(range(1, epoch + 1), test_loss_list, 'b', label='Test loss')
    plt.title('Train and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_dir + model_name + '_loss.png')

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_acc_list': train_acc_list,
        'test_acc_list': test_acc_list,
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list,
        'scheduler': scheduler if scheduler is not None else None
    }, save_file)
