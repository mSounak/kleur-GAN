import torch
from utils import save_checkpoint, load_checkpoint, visualize
import torch.nn as nn
import torch.optim as optim
import config
from dataset import ColorizeDataset, make_dataloaders
from generator_model import Generator
from discriminator_model import Discriminator
from tqdm import tqdm


def main():
    gen = Generator().to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint('models/checkpoints/gen.pth.tar', gen, opt_gen, lr=config.LEARNING_RATE)

    val_loader = make_dataloaders(config.PARENT_DIR, 'val')

    visualize(gen, val_loader)


if __name__ == '__main__':
    main()


    

        