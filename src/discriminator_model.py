import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

# x, y <- concatenate these along the channels
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):        # 256 -> 30x30
        super(Discriminator, self).__init__()
        self.initital = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], 4, 2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, 4, 1, padding=1, padding_mode='reflect')
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initital(x)
        return self.model(x)




def test():
    x = torch.randn((1, 3, 286, 286))
    y = torch.randn((1, 3, 286, 286))

    model = Discriminator()
    out = model(x, y)
    print(out.shape)

if __name__ == '__main__':
    test()