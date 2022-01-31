import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu'):
        super(Block, self).__init__()
        self.conv = nn.Sequential()