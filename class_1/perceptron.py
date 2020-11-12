import torch
from torch import nn
import pandas as pd
import numpy as np


class Perceptron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Perceptron, self).__init__()
        self.net = nn.Linear(in_dim, out_dim)
        for params in self.net.parameters():
            nn.init.normal_(params, mean=0, std=0.01)

    def forward(self, x):
        x = self.net(x)
        return x


def train(model, dataset_loader, loss_function, optimizer):
    model.training = True
    for _, (feature, label) in enumerate(dataset_loader):
        output = model(feature)
        loss = loss_function(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


def load_data(data_name, file_path):
    if data_name == 'sonar':
        dataset = pd.read_csv(file_path, header=None).values
        feature = dataset[:, :-1]
        label = dataset[:, -1]
        label = [0 if label[i] == 'R' else 1 for i in range(len(label))]
    elif data_name == 'wdbc':
        dataset = pd.read_csv(file_path, header=None).values
        feature = dataset[:, 2:]
        label = dataset[:, 1]
        label = [0 if label[i] == 'M' else 1 for i in range(len(label))]
    train_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 != 0]).astype(float),
                                 dtype=torch.float32)
    test_feature = torch.tensor(np.array([feature[i] for i in range(len(feature)) if i % 3 == 0]).astype(float),
                                dtype=torch.float32)
    train_label = torch.tensor([label[i] for i in range(len(label)) if i % 3 != 0])
    test_label = torch.tensor([label[i] for i in range(len(label)) if i % 3 == 0])
    train_data = torch.utils.data.TensorDataset(train_feature, train_label)
    test_data = torch.utils.data.TensorDataset(test_feature, test_label)

    return train_data, test_data


path_dict = {"sonar": "/Users/septieme/School/NN&DL/class_1/Data/sonar_binary/sonar.all-data",
             "wdbc": "/Users/septieme/School/NN&DL/class_1/Data/wdbc_binary/wdbc.data", }

if __name__ == '__main__':
    dataset_title = 'sonar'

    if dataset_title == 'sonar':
        model = Perceptron(in_dim=60, out_dim=2)
    elif dataset_title == 'wdbc':
        model = Perceptron(in_dim=30, out_dim=2)
    else:
        raise FileNotFoundError('Unknown dataset, the available datasets are wdbc and sonar.')
    epoch = 100
    batch_size = 50
    learning_rate = 0.25
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_data, test_data = load_data(dataset_title, path_dict[dataset_title])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    for i in range(1, epoch + 1):
        train(model, train_loader, loss_function, optimizer)
        accuracy = test(model, test_loader)
        print("epoch:", i, '\t', "acc:", accuracy * 100.0 / len(test_data))
