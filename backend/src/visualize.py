from utils import load_checkpoint, visualize
import torch.optim as optim
import config
from dataset import make_dataloaders
from res_unet import res_unet


def main():
    gen = res_unet().to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint('models/checkpoints/gen.pth.tar', gen, opt_gen, lr=config.LEARNING_RATE)

    val_loader = make_dataloaders(config.PARENT_DIR, 'val')

    visualize(gen, val_loader)


if __name__ == '__main__':
    main()
