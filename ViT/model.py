import torch
import torch.nn as nn
import math
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model_Args:
    emb_dim: int = 16
    patch_size: int = 16
    img_size: int = 224
    n_channels: int = 3
    # num_patches + 1 = seq_len (196 + 1 )
    seq_len: int = 197
    n_heads: int = 4
    eps: float = 1e-6
    dropout: float = 0.2


class attention(nn.Module):
    def __init__(self, args: Model_Args):
        super().__init__()
        self.args = args

        self.emd_dim = args.emb_dim
        self.seq_len = args.seq_len
        self.n_heads = args.n_heads
        self.head_dim = self.emd_dim // self.n_heads

        self.wq = nn.Linear(self.emd_dim, self.emd_dim)
        self.wk = nn.Linear(self.emd_dim, self.emd_dim)
        self.wv = nn.Linear(self.emd_dim, self.emd_dim)
        self.wo = nn.Linear(self.emd_dim, self.emd_dim)

    def forward(self, x: torch.Tensor):
        bs, seq_len, dim = x.shape

        # (bs, seq_len, emb_dim) --> (bs, seq_len, emb_dim)
        q = self.wq(x)
        v = self.wv(x)
        k = self.wk(x)

        # (bs, seq_len, dim) --> (bs, seq_len, n_h, h_dim) --> (bs, n_h, seq_len, h_dim)
        q = q.view(bs, seq_len, self.n_heads, self.head_dim).reshape(bs, self.n_heads, seq_len, self.head_dim)
        k = k.view(bs, seq_len, self.n_heads, self.head_dim).reshape(bs, self.n_heads, seq_len, self.head_dim)
        v = v.view(bs, seq_len, self.n_heads, self.head_dim).reshape(bs, self.n_heads, seq_len, self.head_dim)

        # (bs, n_h, seq_len, h_dim) * (bs, n_h, h_dim, seq_len) --> (bs, n_h, seq_len, seq_len)
        attn = (q @ k.transpose(2, 3) / math.sqrt(dim))
        # (bs, n_h, seq_len, seq_len) --> (bs, n_h, seq_len, seq_len)
        attn_sc = nn.functional.softmax(attn, dim=-1)
        # (bs, n_h, seq_len, seq_len) * (bs, n_h, seq_len, h_dim) --> (bs, n_h, seq_len, h_dim)
        attn_sc = attn_sc @ v

        # (bs, n_h, seq_len, h_dim) --> (bs, seq_len, n_h, h_dim) --> (bs, seq_len, n_h * h_dim)
        attn_sc = attn_sc.transpose(1, 2).contiguous().view(bs, seq_len, -1)

        # (bs, seq_len, n_h * h_dim) --> (bs, seq_len, emb_dim)
        attn_sc = self.wo(attn_sc)

        # (bs, seq_len, emb_dim)
        return attn_sc


class feed_forward(nn.Module):
    def __init__(self, args: Model_Args):
        super().__init__()
        self.l1 = nn.Linear(args.emb_dim, 4 * args.emb_dim)
        self.l2 = nn.Linear(4 * args.emb_dim, args.emb_dim)

    def forward(self, x: torch.Tensor):
        x = self.l1(x)
        x = nn.functional.gelu(x)
        x = self.l2(x)
        return x


class encoder(nn.Module):
    def __init__(self, args: Model_Args):
        super().__init__()
        self.args = args
        self.attn = attention(self.args)
        self.feed_for = feed_forward(self.args)
        self.ln1 = nn.LayerNorm(self.args.emb_dim, eps=self.args.eps)
        self.ln2 = nn.LayerNorm(self.args.emb_dim, eps=self.args.eps)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.ln1(x)
        x = self.attn(x)
        x += res
        res = x
        x = self.ln2(x)
        x = self.feed_for(x)
        x += res
        return x


class n_encoders(nn.Module):
    def __init__(self, args: Model_Args):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([encoder(self.args) for _ in range(6)])

    def forward(self, x: torch.Tensor):
        for l in self.layers:
            x = l(x)
        return x


class patch_emb(nn.Module):
    def __init__(self, args: Model_Args):
        super().__init__()
        self.args = args
        self.conv = nn.Conv2d(in_channels=self.args.n_channels,
                              out_channels=self.args.emb_dim,
                              kernel_size=self.args.patch_size,
                              stride=self.args.patch_size)
        self.flatten = nn.Flatten(2)
        self.num_patches = (self.args.img_size // self.args.patch_size) ** 2
        self.cls = nn.Parameter(torch.randn((1, 1, self.args.emb_dim)), requires_grad=True)
        self.pos_embd = nn.Parameter(torch.randn((1, self.num_patches + 1, self.args.emb_dim)), requires_grad=True)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, x: torch.Tensor):
        cls = self.cls.expand(x.shape[0], -1, -1)
        # (bs, in_c, h, w) --> (bs, emb_dim, new_h, new_w)
        x = self.conv(x)
        # (bs, emb_dim, new_h, new_w) --> (bs, emb_dim, new_h * new_w) --> (bs, new_h * new_w, emb_dim)
        x = self.flatten(x).permute(0, 2, 1)
        # (bs, new_h * new_w, emb_dim) --> (bs, num_patches + 1, emb_dim)
        x = torch.cat([x, cls], dim=1)
        # (bs, num_patches + 1, emb_dim) --> (bs, num_patches + 1, emb_dim)
        x += self.pos_embd
        # (bs, num_patches + 1, emb_dim)
        return x


class vit_model(nn.Module):
    def __init__(self, args: Model_Args, classes: int):
        super().__init__()
        self.args = args
        self.classes = classes
        self.encoders = n_encoders(args)
        self.patch_embd = patch_emb(args)
        self.mlp = nn.Linear(self.args.emb_dim, self.classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embd(x)
        x = self.encoders(x)
        x = self.mlp(x[:, 0, :])
        return x


if __name__=='__main__':

    y = Model_Args()
    a = torch.randn(512, y.n_channels, y.img_size, y.img_size)
    print(f'input: {a.shape}')
    # xd = attention(y)
    # xr = feed_forward(y)
    # x = xd(a)
    # xe = xr(x)
    # xde = n_encoders(y)
    # # xc = xd(a)
    # xd = patch_emb(y)
    # xc = xd(a)
    # xc = xde(a)
    xf = vit_model(y, 10)
    xc = xf(a)
    print(f'output: {xc.shape}')

# parameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='/data', train=True, transform=train_transform, download=True)
test_data = datasets.CIFAR10(root='/data', train=False, transform=test_transform, download=True)
classes = len(train_data.classes)
print(f'number of classes: {classes}')
print(f'size of train data: {len(train_data)}')

train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

args = Model_Args
m = vit_model(args, classes=classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)

train_r_loss, train_r_acc = [], []

for e in range(NUM_EPOCHS):
    m.train()
    running_loss = 0
    correct, total = 0, 0
    for batch_idx, (img, label) in tqdm(enumerate(train_dl)):
        img = img.to(device)
        label = label.type(torch.uint8).to(device)
        optimizer.zero_grad()
        y = m(img)
        y_pred = torch.argmax(y, dim=1)
        loss = loss_fn(y, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct += (y_pred == label).sum().item()
        total += label.size(0)

    train_loss = running_loss / (batch_idx + 1)
    train_r_loss.append(train_loss)
    train_acc = correct / total
    train_r_acc.append(train_acc)

    print(f"for epoch: {e+ 1}: ")
    print(f"train_loss: {train_loss}")
    print(f"train_acc: {train_acc}")


for e in range(1):
    m.eval()
    r_loss = 0
    test_loss, test_acc = 0, 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (img, label) in tqdm(enumerate(test_dl)):
            img = img.to(device)
            label = label.type(torch.uint8).to(device)
            y = m(img)
            y_pred = torch.argmax(y, dim=1)
            loss = loss_fn(y, label)
            r_loss += loss.item()
            correct += (y_pred == label).sum().item()
            total += label.size(0)

    test_loss = r_loss / (batch_idx + 1)
    test_acc = correct / total

    print(f"for epoch: {e + 1}: ")
    print(f"test_loss: {test_loss}")
    print(f"test_acc: {test_acc}")




