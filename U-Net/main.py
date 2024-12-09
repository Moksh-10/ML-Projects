# import torch
# import torch.nn as nn
# import torchvision.transforms.functional as TF
#
#
# class Two_Conv(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(Two_Conv, self).__init__()
#         self.b1 = nn.Sequential(
#             nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.b1(x)
#
#
# class UNET(nn.Module):
#     def __init__(self, in_c=3, out_c=1, features=[64, 128, 256, 512]):
#         super(UNET, self).__init__()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         for f in features:
#             self.downs.append(Two_Conv(in_c, f))  # 3 --> 64 --> 128 --> 256 --> 512 2 times each
#             in_c = f
#
#         for f in reversed(features):
#             self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)) # 1024 --> 512
#             self.ups.append(Two_Conv(f*2, f)) # f*2 because f are coming from conv trans. 2d and f are coming from res_conn
#
#         self.bn = Two_Conv(features[-1], features[-1]*2) #bottle neck 512 --> 1024 --> 1024
#         self.fc = nn.Conv2d(features[0], out_c, kernel_size=1) # last layer at output 64 --> 1
#
#     def forward(self, x):
#         skip_conn = []
#         # print(x.shape)
#         for d in self.downs:
#             x = d(x)
#             skip_conn.append(x)
#             x = self.pool(x)
#             # print(x.shape)
#
#         x = self.bn(x)
#         # print(x.shape)
#         skip_conn = skip_conn[::-1] # reverse skip_conn
#
#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             sk = skip_conn[idx//2]
#             print(x.shape)
#             print(sk.shape)
#             print(sk.shape[2:])
#             if x.shape != sk.shape:
#                 x = TF.resize(x, size=sk.shape[2:])
#
#             y = torch.cat((sk, x), dim=1)
#             x = self.ups[idx+1](y)
#             # print(x.shape)
#         return self.fc(x)
#
# gg = torch.randn((3,1,161,161))
# model1 = UNET(in_c=1, out_c=1)
# ff = model1(gg)
# print(ff.shape)
