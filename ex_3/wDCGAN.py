import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch import autograd
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 600
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Generator(nn.Module):
    def __init__(self, z_dim=100, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(z_dim, d * 4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 3, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))

        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def compute_gradient_penalty(model, real, fake):
    alpha = torch.Tensor(np.random.random((real.size(0), 1, 28, 28))).to(device)
    interpolates = (alpha * real + (1 - alpha) * fake).to(device).requires_grad_(True)
    D_interpolates = model(interpolates).view(-1, 1)
    fake_out = Variable(torch.Tensor(real.shape[0], 1).fill_(1.0).to(device), requires_grad=False)
    gradients = autograd.grad(
        outputs=D_interpolates,
        inputs=interpolates,
        grad_outputs=fake_out,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_discriminator(model, image, criterion, batch_size, factor):
    D_optimizer.zero_grad()

    # 真样本训练
    x_real, y_real = image.view(-1, 1, 28, 28), torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = model(x_real).view(-1, 1)
    D_real_loss = criterion(D_output, y_real)

    # 生成样本训练
    z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))
    x_fake, y_fake = generator(z), Variable(torch.zeros(batch_size, 1).to(device))
    D_output = model(x_fake).view(-1, 1)
    D_fake_loss = criterion(D_output, y_fake)

    D_loss = D_real_loss + D_fake_loss + factor * compute_gradient_penalty(model, x_real, x_fake)
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def train_generator(generator, discriminator, criterion, batch_size):
    G_optimizer.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim, 1, 1).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    G_output = generator(z)
    D_output = discriminator(G_output).view(-1, 1)
    G_loss = criterion(D_output, y)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


if __name__ == "__main__":
    z_dim = 100
    lr = 0.0002
    epochs = 20
    save_dir = 'GAN_pic'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    generator = Generator(z_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.weight_init(0.0, 0.02)
    discriminator.weight_init(0.0, 0.02)
    loss_function = nn.BCELoss()
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    for epoch in range(1, epochs + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(train_discriminator(discriminator, x, loss_function, batch_size, 10))
            G_losses.append(train_generator(generator, discriminator, loss_function, batch_size))

        print('Epoch %d: loss_d: %.3f, loss_g: %.3f' % (
            epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    with torch.no_grad():
        test_z = Variable(torch.randn(100, z_dim, 1, 1).to(device))
        generated = generator(test_z)
        save_image(generated.view(generated.size(0), 1, 28, 28), save_dir + '/wDCGAN_generated' + '.png', nrow=10)
