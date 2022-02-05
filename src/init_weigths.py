import torch
import torch.nn as nn


def init_weights(model, init='norm', gain=0.02):
    """Initialize network weights.

    Parameters
    ----------
    model : torch.nn.Module
        Network model.
    init : str
        Initialization method.
    gain : float
        Initialization gain.

    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            if init == 'norm':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init)