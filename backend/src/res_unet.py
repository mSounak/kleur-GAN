from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
import torch.optim as optim
import torch.nn as nn
import config
from utils import AverageMeter
from tqdm import tqdm
from dataset import make_dataloaders
import torch


def res_unet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_g = DynamicUnet(body, n_output, img_size=(size, size)).to(config.DEVICE)
    return net_g


def pretrain_generator(net_g, train_dl, opt, criterion, epochs):
    for epoch in range(epochs):
        loss_meter = AverageMeter()
        for x, y in tqdm(train_dl):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            preds = net_g(x)
            loss = criterion(preds, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item())

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Loss: {loss_meter.avg:.4f}')


if __name__ == '__main__':
    train_dl = make_dataloaders(config.PARENT_DIR, 'train', batch_size=8)
    net_g = res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_g.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    pretrain_generator(net_g, train_dl, opt, criterion, epochs=20)
    torch.save(net_g.state_dict(), config.MODEL_PATH + 'res_unet.pt')
    print('=> Model saved')
