import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


############################################################################
# Re-usable blocks
############################################################################


class TransposeBlock(nn.Module):
    def __init__(self, ic=4, oc=4, kernel_size=3, padding=1, stride=2, drop=.001):
        super(TransposeBlock, self).__init__()

        if padding is None:
            padding = int(kernel_size // 2 // stride)

        operations = []
        operations += [spectral_norm(nn.ConvTranspose2d(in_channels=ic,
                                                        out_channels=oc,
                                                        padding=padding,
                                                        output_padding=0,
                                                        kernel_size=kernel_size,
                                                        stride=stride,
                                                        bias=False))]

        operations += [nn.LeakyReLU(inplace=True), nn.BatchNorm2d(oc), nn.Dropout(drop)]
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        x = self.operations(x)
        return x


def disc_conv_block(ni, nf, kernel_size=3, stride=1):
    # conv_block with spectral normalization
    layers = []
    conv = spectral_norm(nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2, stride=stride))
    relu = nn.LeakyReLU(inplace=True)

    layers += [conv, relu]
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.query = spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.key = spectral_norm(nn.Conv1d(in_channel, in_channel // 8, 1))
        self.value = spectral_norm(nn.Conv1d(in_channel, in_channel, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)

        out = self.gamma * attn + x
        return out


############################################################################
# Generator and Discriminator
############################################################################

class Generator(nn.Module):
    # Generator to convert Z sized vector to image
    def __init__(self, layers=5, z_size=128, filts=1024, max_filts=512, min_filts=128, kernel_size=4, channels=3,
                 drop=.0, center_drop=.0, attention=True, res=True):
        super(Generator, self).__init__()
        operations = []

        filt_count = min_filts

        for a in range(layers):

            print('up-block')
            operations += [TransposeBlock(ic=int(min(max_filts, filt_count * 2)),
                                          oc=int(min(max_filts, filt_count)),
                                          kernel_size=kernel_size,
                                          padding=1,
                                          drop=drop,
                                          res=res)]
            if a == 1 and attention:
                print('attn-block')
                operations += [SelfAttention(int(min(max_filts, filt_count * 2)))]
            filt_count = int(filt_count * 2)

        operations += [
            TransposeBlock(ic=filts,
                           oc=int(min(max_filts, filt_count)),
                           kernel_size=kernel_size,
                           padding=1,
                           drop=center_drop, res=res),
            TransposeBlock(ic=z_size,
                           oc=filts,
                           kernel_size=kernel_size,
                           padding=0,
                           stride=1,
                           drop=center_drop,
                           res=res)
        ]

        operations.reverse()

        operations += [nn.ReflectionPad2d(3),
                       spectral_norm(nn.Conv2d(in_channels=min_filts,
                                               out_channels=channels,
                                               kernel_size=7,
                                               padding=0,
                                               stride=1))]

        self.model = nn.Sequential(*operations)

    def forward(self, x):
        x = self.model(x)
        return F.tanh(x)


class Discriminator(nn.Module):
    # Using reverse shuffling should reduce the repetitive shimmering patterns
    def __init__(self, channels=3, filts_min=128, filts=512, kernel_size=4, layers=3, attention=False):
        super(Discriminator, self).__init__()
        operations = []

        in_operations = [nn.ReflectionPad2d(3),
                         spectral_norm(nn.Conv2d(in_channels=channels,
                                                 out_channels=filts_min,
                                                 kernel_size=7,
                                                 stride=1))]

        filt_count = filts_min

        for a in range(layers):
            operations += [
                disc_conv_block(min(filt_count, filts),
                                min(filt_count * 2, filts),
                                kernel_size=kernel_size,
                                stride=2)]
            print('down-block')
            if a == 1 and attention:
                print('attn-block')
                operations += [SelfAttention(min(filt_count * 2, filts))]

            print(min(filt_count * 2, filts))
            filt_count = int(filt_count * 2)

        out_operations = [
            spectral_norm(nn.Conv2d(in_channels=min(filt_count, filts),
                                    out_channels=1,
                                    padding=1,
                                    kernel_size=kernel_size,
                                    stride=1))]

        operations = in_operations + operations + out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        x = self.operations(x)
        return x
