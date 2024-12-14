import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (bs, channel, height, width) --> (bs, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (bs, 128, height, width) --> (bs, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (bs, 128, height, width) --> (bs, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (bs, 128, height, width) --> (bs, 128, height / 2, width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (bs, 128, height / 2, width / 2)) --> (bs, 256, height / 2, width / 2)
            VAE_ResidualBlock(128, 256),

            # (bs, 256, height / 2, width / 2) --> (bs, 256, height / 2, width / 2)
            VAE_ResidualBlock(256, 256),

            # (bs, 256, height / 2, width / 2) --> (bs, 256, height / 4, width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (bs, 256, height / 4, width / 4) --> (bs, 512, height / 4, width / 4)
            VAE_ResidualBlock(256, 512),

            # (bs, 512, height / 4, width / 4) --> (bs, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512),

            # (bs, 512, height / 4, width / 4) --> (bs, 512, height / 8, width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            VAE_AttentionBlock(512),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            nn.GroupNorm(32, 512),

            # (bs, 512, height / 8, width / 8) --> (bs, 512, height / 8, width / 8)
            nn.SiLU(),

            # (bs, 512, height / 8, width / 8) --> (bs, 8, height / 8, width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (bs, 8, height / 8, width / 8) --> (bs, 8, height / 8, width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (bs, channel, h, w)
        # noise: (bs, out_channels, h/8, w/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding left, padding right, padding top, padding bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # (bs, 8, h/8, w/8) --> 2 tensors of (bs, 4, h/8, w/8) along dim=1
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (bs, 4, h/8, w/8) --> (bs, 4, h/8, w/8)
        log_variance = torch.clamp(log_variance, -32, 20)  # what this does is if any value is very small or big, it brings them in between this

        # (bs, 4, h/8, w/8) --> (bs, 4, h/8, w/8)
        variance = log_variance.exp()

        # (bs, 4, h/8, w/8) --> (bs, 4, h/8, w/8)
        stdev = variance.sqrt()

        # Z = N(0, 1) --> N(mean, variance) = X?
        # X = mean + stdev * Z
        x = mean + stdev * noise

        # scale output by a constant
        x *= 0.18215

        return x





