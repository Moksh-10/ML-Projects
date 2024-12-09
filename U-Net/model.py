import torch
from torch import nn
import torchvision.transforms.functional as TF


class Double_Conv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer_1(x)


class Unet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for f in features:
            self.downs.append(Double_Conv(in_ch, f))
            in_ch = f

        self.bottle_neck = Double_Conv(features[-1], features[-1]*2)
        self.last_layer = nn.Conv2d(features[0], out_ch, kernel_size=1)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(Double_Conv(f*2, f))


    def forward(self, x):
        skip_conn = []

        for l in self.downs:
            x = l(x)
            skip_conn.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)

        skip_conn = skip_conn[::-1]

        for l in range(0, len(self.ups), 2):
            x = self.ups[l](x)
            s = skip_conn[l//2]
            # print(f"x: {x.shape}")
            # print(f"s: {s.shape}")
            if x.shape != s.shape:
                x = TF.resize(x, size=s.shape[2:])

            y = torch.cat((s, x), dim=1)
            x = self.ups[l+1](y)

        return self.last_layer(x)


gg = torch.randn((3,3,256,256))
model1 = Unet(in_ch=3, out_ch=3)
ff = model1(gg)
print(ff.shape)
