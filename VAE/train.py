import torch
from torch import nn
import torchvision.datasets as datasets
from tqdm import tqdm
from model import vae
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = 784
h_dim = 200
z_dim = 20
ne = 10
bs = 32
lr = 3e-4

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, bs, shuffle=True)
model = vae(input_dim, h_dim, z_dim).to(device)
optim = torch.optim.Adam(model.parameters(), lr)
loss_fn = nn.BCELoss(reduction="sum")

for e in range(ne):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.to(device).view(x.shape[0], input_dim)
        x_reconstructed, mu, sigma = model(x)

        #loss
        reconstruction_loss = loss_fn(x_reconstructed, x) # push towards input img
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) # push towards being a gaussian distribution

        loss = reconstruction_loss + kl_div
        optim.zero_grad()
        loss.backward()
        optim.step()
        loop.set_postfix(loss=loss.item())


def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encoding_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encoding_digit.append((mu, sigma))

    mu, sigma = encoding_digit[digit]
    for ex in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.deocder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated{digit}_ex{ex}.png")

for i in range(10):
    inference(i, num_examples=4)
