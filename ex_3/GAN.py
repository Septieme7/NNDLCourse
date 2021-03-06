import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 600
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def train_discriminator(model, image, criterion, batch_size):
    D_optimizer.zero_grad()

    x_real, y_real = image.view(-1, mnist_dim), torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = model(x_real)
    D_real_loss = criterion(D_output, y_real)

    z = Variable(torch.randn(batch_size, z_dim).to(device))
    x_fake, y_fake = generator(z), Variable(torch.zeros(batch_size, 1).to(device))

    D_output = model(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def train_generator(generator, discriminator, criterion, batch_size):
    G_optimizer.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    G_output = generator(z)
    D_output = discriminator(G_output)
    G_loss = criterion(D_output, y)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


if __name__ == "__main__":
    z_dim = 100
    lr = 0.0002
    epochs = 200
    save_dir = 'GAN_pic'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    generator = Generator(z_dim, mnist_dim).to(device)
    discriminator = Discriminator(mnist_dim).to(device)
    loss_function = nn.BCELoss()
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_generator = []
    loss_discriminator = []
    for epoch in range(1, epochs + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(train_discriminator(discriminator, x, loss_function, batch_size))
            G_losses.append(train_generator(generator, discriminator, loss_function, batch_size))
        loss_discriminator.append(torch.mean(torch.FloatTensor(D_losses)))
        loss_generator.append(torch.mean(torch.FloatTensor(G_losses)))
        print('Epoch %d: loss_d: %.3f, loss_g: %.3f' % (
            epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    plt.plot(range(1, epochs + 1), loss_discriminator, 'g-', label='Discriminator loss')
    plt.plot(range(1, epochs + 1), loss_generator, 'b--', label='Generator loss')
    plt.title('Gen and Dis loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir + '/GAN_loss.png')
    with torch.no_grad():
        test_z = Variable(torch.randn(100, z_dim).to(device))
        generated = generator(test_z)
        save_image(generated.view(generated.size(0), 1, 28, 28), save_dir + '/GAN_generated' + '.png', nrow=10)
