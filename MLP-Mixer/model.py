import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

in_c = 3
new_dim = 512
ps = 16
token_dim = 256
ch_dim = 2048
img_size = 224
dropout = 0.2
num_classes = 10
n_layer = 10


class mlp_layer(nn.Module):
    def __init__(self):
        super().__init__()

        num_patches = (img_size // ps) ** 2

        self.ln_1 = nn.LayerNorm(new_dim)

        # incoming x for tokens must be of (patches * channels)
        # (patches * c) --> (c * patches)
        self.t_layer = nn.Sequential(
            nn.Linear(num_patches, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_patches),
            nn.Dropout(dropout)
        )

        self.ln_2 = nn.LayerNorm(new_dim)

        self.c_layer = nn.Sequential(
            nn.Linear(new_dim, ch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ch_dim, new_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        res = x
        x = self.ln_1(x)
        x = x.transpose(1, 2)
        x = self.t_layer(x)
        x = x.transpose(1, 2)
        x += res

        res = x
        x = self.ln_2(x)
        x = self.c_layer(x)
        x += res
        return x


class final_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Sequential(
            nn.Conv2d(in_c, new_dim, kernel_size=ps, stride=ps),
            nn.Flatten(2)
        )

        self.layers = nn.ModuleList([mlp_layer() for _ in range(n_layer)])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Linear(new_dim, num_classes)

    def forward(self, x):
        x = self.pe(x)
        x = x.transpose(1, 2)
        for l in self.layers:
            x = l(x)
        x = x.transpose(1, 2)
        x = self.avg_pool(x)
        x = x.squeeze(2)
        x = self.cls(x)
        return x


# x = torch.randn(64, 3, 224, 224)
# y = final_model()
# z = y(x)
# print(z.shape)

tr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

data = datasets.CIFAR10("/dataset", train=True, transform=tr, download=True)
loader = DataLoader(data, batch_size=64, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = final_model().to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=3e-4)

for e in range(5):
    model.train()
    loop = tqdm(enumerate(loader))
    l = 0

    for b_id, (x, y) in loop:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        l += loss.item()

        loop.set_postfix(loss=loss.item())
    l /= len(loader)
    print(f'epoch {e + 1}/ loss: {l:.4f}')

for e in range(2):
    model.eval()
    c, t = 0, 0
    loop = tqdm(enumerate(loader))
    with torch.no_grad():
        for b, (x, y) in loop:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            _, just = torch.max(y_pred, dim=1)
            c += (just == y).sum().item()
            t += y.size(0)

    acc = c / t
    print(f'acc: {acc:.4f}%')

# for i in range(10):
#     r_id = random.randint(0, len(data)-1)
#     x, y = data[r_id]
#     x = x.unsqueeze(0).to(device)
#     with torch.no_grad():
#         y_pred = model(x)
#         _, pred = torch.max(y_pred, dim=1)
#     x = x.cpu().squeeze()
#     pred = pred.cpu().item()
#
#     plt.figure(figsize=(6, 6))
#     plt.imshow(x.squeeze())
#     plt.title(f'true: {y}, pred: {pred}')
#     plt.axis('off')
#     plt.show()

