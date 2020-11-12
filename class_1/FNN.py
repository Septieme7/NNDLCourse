import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class FCC(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_width):
        super(FCC, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_width)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(hidden_width, out_dim)
        self.relu2 = nn.ReLU(inplace=True)
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        return x


def train(model, dataset_loader, loss_function, optimizer):
    model.training = True
    correct = 0
    for i, (feature, label) in enumerate(dataset_loader):
        output = model(feature)
        pred = torch.argmax(output, 1)
        res = pred == label
        for i in res:
            if i:
                correct += 1
        loss = loss_function(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return correct


def test(model, dataset_loader):
    model.training = False
    correct = 0
    with torch.no_grad():
        for _, (feature, label) in enumerate(dataset_loader):
            output = model(feature)
            # 计算预测准确率
            pred = torch.argmax(output, 1)
            res = pred == label
            for i in res:
                if i:
                    correct += 1
    return correct


def load_data(dataset_name, dataset_path):
    if dataset_name == 'iris' or dataset_name == 'robot':
        dataset = pd.read_csv(dataset_path).values
        feature = dataset[:, :-1]
        label2cvt = dataset[:, -1]
        label_dict = {}
        label = []
        code = 0
        for i in label2cvt:
            if i in label_dict:
                label.append(label_dict[i])
            else:
                label_dict[i] = code
                label.append(code)
                code += 1
        train_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 != 0]).astype(float),
                                     dtype=torch.float32)
        test_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 == 0]).astype(float),
                                    dtype=torch.float32)
        train_label = torch.tensor([label[i] for i in range(len(label)) if i % 3 != 0])
        test_label = torch.tensor([label[i] for i in range(len(label)) if i % 3 == 0])
    elif dataset_name == 'soybean':
        dataset = pd.read_csv(dataset_path).values
        feature = dataset[:, 1:]
        label2cvt = dataset[:, 0]
        for i in range(len(feature)):
            for j in range(len(feature[i])):
                if feature[i][j] == '?':
                    feature[i][j] = -1
        label_dict = {}
        label = []
        code = 0
        for i in label2cvt:
            if i in label_dict:
                label.append(label_dict[i])
            else:
                label_dict[i] = code
                label.append(code)
                code += 1
        train_feature = torch.tensor(np.array(feature[:307]).astype(float),
                                     dtype=torch.float32)
        test_feature = torch.tensor(np.array(feature[307:]).astype(float),
                                    dtype=torch.float32)
        train_label = torch.tensor(label[:307])
        test_label = torch.tensor(label[307:])

    train_loader = torch.utils.data.TensorDataset(train_feature, train_label)
    test_loader = torch.utils.data.TensorDataset(test_feature, test_label)

    return train_loader, test_loader


path_dict = {"iris": "/Users/septieme/School/NN&DL/class_1/Data/iris_multi/iris.data",
             "soybean": "/Users/septieme/School/NN&DL/class_1/Data/soybean_multi/soybean-large.data",
             "robot": "/Users/septieme/School/NN&DL/class_1/Data/robot_multi/sensor_readings_24.data"}

if __name__ == '__main__':
    dataset_name = 'robot'

    if dataset_name == 'iris':
        model = FCC(4, 3, 16)
        epoch = 100
        batch_size = 10
        learning_rate = 0.002
    elif dataset_name == 'robot':
        model = FCC(24, 4, 64)
        epoch = 100
        batch_size = 100
        learning_rate = 0.04
    elif dataset_name == 'soybean':
        model = FCC(35, 19, 64)
        epoch = 100
        batch_size = 50
        learning_rate = 0.5
    else:
        raise FileNotFoundError('Unknown dataset, the available datasets are wdbc and sonar.')

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_data, test_data = load_data(dataset_name, path_dict[dataset_name])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_acc = []
    test_acc = []
    for i in range(1, epoch + 1):
        acc = train(model, train_loader, loss_function, optimizer) * 100.0 / len(train_data)
        accuracy = test(model, test_loader) * 100.0 / len(test_data)
        train_acc.append(acc)
        test_acc.append(accuracy)
        print("epoch:", i, '\t', "train_acc: {:.4}%".format(acc),
              '\t' "test_acc: {:.4}%".format(accuracy))
    plt.plot(range(1, epoch + 1), train_acc, 'g-', label='Train accuracy')
    plt.plot(range(1, epoch + 1), test_acc, 'b', label='Test accuracy')
    plt.title('Train and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
