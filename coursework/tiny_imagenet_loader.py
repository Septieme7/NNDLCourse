import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

labels_t = []
image_names = []
with open('./tiny-imagenet-200/wnids.txt') as wnid:
    for line in wnid:
        labels_t.append(line.strip('\n'))
for label in labels_t:
    txt_path = './tiny-imagenet-200/train/' + label + '/' + label + '_boxes.txt'
    image_name = []
    with open(txt_path) as txt:
        for line in txt:
            image_name.append(line.strip('\n').split('\t')[0])
    image_names.append(image_name)
labels = np.arange(200)

val_labels_t = []
val_labels = []
val_names = []
with open('./tiny-imagenet-200/val/val_annotations.txt') as txt:
    for line in txt:
        val_names.append(line.strip('\n').split('\t')[0])
        val_labels_t.append(line.strip('\n').split('\t')[1])
for i in range(len(val_labels_t)):
    for i_t in range(len(labels_t)):
        if val_labels_t[i] == labels_t[i_t]:
            val_labels.append(i_t)
val_labels = np.array(val_labels)


class data(Dataset):
    def __init__(self, type, transform):
        self.type = type
        if type == 'train':
            i = 0
            self.images = []
            for label in labels_t:
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join('./tiny-imagenet-200/train', label, 'images', image_name)
                    # image.append(cv2.imread(image_path))
                    fp = Image.open(image_path).convert('RGB')
                    image.append(np.array(fp))
                    fp.close()
                self.images.append(image)
                i = i + 1
            self.images = np.array(self.images)
            print(self.images.shape)
            self.images = self.images.reshape(-1, 64, 64, 3)
        elif type == 'val':
            self.val_images = []
            for val_image in val_names:
                val_image_path = os.path.join('./tiny-imagenet-200/val/images', val_image)
                # self.val_images.append(cv2.imread(val_image_path))
                fp = Image.open(val_image_path).convert('RGB')
                self.val_images.append(np.array(fp))
                fp.close()
            self.val_images = np.array(self.val_images)
        self.transform = transform

    def __getitem__(self, index):
        label = []
        image = []
        if self.type == 'train':
            label = index // 500
            image = self.images[index]
        if self.type == 'val':
            label = val_labels[index]
            image = self.val_images[index]
        return self.transform(image), label

    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.images.shape[0]
        if self.type == 'val':
            len = self.val_images.shape[0]
        return len


def get_loader():
    train_dataset = data('train', transform=transforms.Compose([transforms.ToTensor()]))
    print(train_dataset[0])
    val_dataset = data('val', transform=transforms.Compose([transforms.ToTensor()]))
    batch_size = 100
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
