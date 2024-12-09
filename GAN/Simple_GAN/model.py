import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        in_f = img_dim
        self.disc = nn.Sequential(
            nn.Linear(in_f, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class gen(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 28 * 28 *1
bs = 32
ne = 50

disc = discriminator(img_dim).to(device)
gen = gen(z_dim, img_dim).to(device)
fixed_noise = torch.randn((bs, z_dim)).to(device)

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST(root="/dataset", transform=tr, download=True)
loader = DataLoader(dataset, bs, shuffle=True)
opt_disc = torch.optim.Adam(disc.parameters(), lr)
opt_gen = torch.optim.Adam(gen.parameters(), lr)
loss_fn = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for e in range(ne):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        bs = real.shape[0]

        noise = torch.randn(bs, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = loss_fn(disc_fake, torch.ones_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2

        opt_disc.zero_grad()
        lossD.backward()
        opt_disc.step()


