from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import shutil
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config
import torch
from utils import lab_to_rgb, save_some_examples

IMG_SIZE = config.IMG_SIZE


class ColorizeDataset(Dataset):
    def __init__(self, path, split='train'):
        if split == 'train':
            self.paths = [os.path.join(path, split + '/' + f) for f in os.listdir(os.path.join(path, 'train')) if f.endswith('.jpg')]
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE), transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            self.paths = [os.path.join(path, split + '/' + f) for f in os.listdir(os.path.join(path, 'val')) if f.endswith('.jpg')]
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE), transforms.InterpolationMode.BICUBIC),
            ])
        else:
            self.paths = [path]
            self.transforms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE), transforms.InterpolationMode.BICUBIC),
            ])

        self.size = IMG_SIZE
        self.split = split

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        img = np.array(img)
        img_l = rgb2lab(img).astype(np.float32)     # Converting RGB to L * a * b
        img_l = transforms.ToTensor()(img_l)
        L = img_l[[0], ...] / 50. - 1.    # Normalize to [-1, 1]
        ab = img_l[[1, 2], ...] / 110.    # Normalize to [-1, 1]

        return L, ab

    def __len__(self):
        return len(self.paths)


def make_dataloaders(paths, split, shuffle=True, batch_size=16):
    dataset = ColorizeDataset(paths, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4) 

    return dataloader


if __name__ == '__main__':
    img_path = 'data/images/'

    loader = make_dataloaders(img_path, 'train', batch_size=1)

    for L, ab in loader:
        print(L.shape)  # batch x 1 x 256 x 256
        print(ab.shape)  # batch x 2 x 256 x 256

        img = lab_to_rgb(L, ab)
        print(np.max(img))
        print(np.min(img))
        fake_img = np.squeeze(img)
        fake_img = Image.fromarray((fake_img * 255))
        fake_img.save('fake_img.png')
        break
