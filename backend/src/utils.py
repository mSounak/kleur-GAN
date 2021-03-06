import torch
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_d_fake = AverageMeter()
    loss_d_real = AverageMeter()
    loss_d = AverageMeter()
    loss_g_gan = AverageMeter()
    loss_g_l1 = AverageMeter()
    loss_g = AverageMeter()
    
    return {'loss_D_fake': loss_d_fake,
            'loss_D_real': loss_d_real,
            'loss_D': loss_d,
            'loss_G_GAN': loss_g_gan,
            'loss_G_L1': loss_g_l1,
            'loss_G': loss_g}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def image2lab(image):

    transform = transforms.Compose([
        transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
    ])

    image = Image.open(image).convert('RGB')
    img = transform(image)
    img = np.array(img)
    img_l = rgb2lab(img).astype(np.float32)     # Converting RGB to L * a * b
    img_l = transforms.ToTensor()(img_l)
    L = img_l[[0], ...] / 50. - 1.    # Normalize to [-1, 1]
    ab = img_l[[1, 2], ...] / 110.    # Normalize to [-1, 1]

    return L, ab


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img) * 255
        rgb_imgs.append(img_rgb.astype(np.uint8))
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):

    L, ab = next(iter(data))
    L = L.to(config.DEVICE)
    ab = ab.to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        fake_color = model(L)

    real_color = ab
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in tqdm(range(6)):
        ax = plt.subplot(3, 6, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 6, i + 1 + 6)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 6, i + 1 + 12)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"generated_images/grid_viz/colorization.png")


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        fake_img = lab_to_rgb(x, y_fake)
        fake_img = np.squeeze(fake_img)
        Image.fromarray((fake_img)).save(folder + f"y_gen_{epoch}.png")

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
