import torch
from einops import rearrange
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision
import cv2
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"


class simple_vae(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(28*28, 196),
            nn.Tanh(),
            nn.Linear(196, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        self.mean = nn.Sequential(
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.var = nn.Sequential(
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.dec = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 196),
            nn.Tanh(),
            nn.Linear(196, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        img = torch.flatten(x, start_dim=1)
        out = self.enc(img)
        mean = self.mean(out)
        log_var = self.var(out)

        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std)
        new_img = mean + z * std

        new = self.dec(new_img)
        new = new.reshape((new.size(0), 1, 28, 28))
        return mean, log_var, new

def train():
    train_data = torchvision.datasets.MNIST(root = "./data/train", train = True, download = True, transform=transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root="./data/test", train = False, download=True, transform=transforms.ToTensor())

    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = simple_vae().to(device)

    recon_losses = []
    kl_losses = []
    losses = []

    num_epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()

    for i in range(num_epochs):
        for x, y in tqdm(loader):
            # print(x.shape, y.shape)
            x = x.float().to(device)
            optimizer.zero_grad()
            mean, log_var, out = model(x)

            cv2.imwrite('input.jpeg', 255*((x+1)/2).detach().cpu().numpy()[0, 0])
            cv2.imwrite('output.jpeg', 255 * ((out + 1) / 2).detach().cpu().numpy()[0, 0])

            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var, dim=-1))
            recon_loss = mse_loss(out, x)
            loss = recon_loss + 0.00001 * kl_loss

            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print('finished epoch: {} | recon_loss: {:.4f} | kl_loss: {:.4f}'.format(i+1, np.mean(recon_losses), np.mean(kl_losses)))

    print('training done')

    idxs = torch.randint(0, len(test_data) - 1, (100, ))
    ims = torch.cat([test_data[idx][0][None, :] for idx in idxs]).float()

    _, _, gen_im = model(ims)

    ims = (ims + 1)/2
    gen_im = 1 - (gen_im + 1)/2
    out = torch.hstack([ims, gen_im])
    output = rearrange(out, 'b c h w -> b () h (c w)')
    grid = torchvision.utils.make_grid(output, nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('rec.png')
    print('done rec')




# a = torch.randn(5, 1, 28, 28)
# g = simple_vae()
# f = g(a)
# print(a.shape)
