import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import ColorizeDataset, make_dataloaders
from generator_model import Generator
from discriminator_model import Discriminator
from tqdm import tqdm


def train_fn(disc, gen, loader, opt_disc, opt_gen, loss_L1, loss_BCE, g_scaler, d_scaler):

    for idx, (x, y) in enumerate(tqdm(loader, leave=False)):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake_color = gen(x)
            d_real = disc(x, y)
            d_fake = disc(x, fake_color.detach())

            d_real_loss = loss_BCE(d_real, torch.ones_like(d_real))
            d_fake_loss = loss_BCE(d_fake, torch.zeros_like(d_fake))

            d_loss = (d_real_loss + d_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train the generator
        with torch.cuda.amp.autocast():
            d_fake = disc(x, fake_color)
            g_fake_loss = loss_BCE(d_fake, torch.ones_like(d_fake))
            l1 = loss_L1(fake_color, y) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()





def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    loss_bce = nn.BCEWithLogitsLoss()
    loss_L1 = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN, disc, opt_disc, config.LEARNING_RATE)

    train_loader = make_dataloaders(config.PARENT_DIR, 'train')
    val_loader = make_dataloaders(config.PARENT_DIR, 'val', batch_size=1)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, loss_L1, loss_bce, g_scaler, d_scaler)


        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="generated_images")
        


if __name__ == "__main__":
    main()