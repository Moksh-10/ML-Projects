import torch
from torch import nn

# input img --> hidden dim --> mean, std --> parametarization trick --> deocder --> output img

class vae(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        #encoder
        self.img_2_hid = nn.Linear(input_dim, h_dim)
        self.hid_2_mu = nn.Linear(h_dim, z_dim)
        self.hid_2_sigma = nn.Linear(h_dim, z_dim)

        #deocder
        self.z_2_hid = nn.Linear(z_dim, h_dim)
        self.hid_2_img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi (z/x)
        h = self.relu(self.img_2_hid(x))
        mu, sigma = self.hid_2_mu(h), self.hid_2_sigma(h)
        return mu, sigma

    def deocder(self, x):
        # p_theta(x/z)
        h = self.relu(self.z_2_hid(x))
        return torch.sigmoid(self.hid_2_img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.deocder(z_reparameterized)
        return x_reconstructed, mu, sigma
        # mu, sigma is for kl divergence


# if __name__=="__main__" :
#     x = torch.randn(4, 28*28) # 28*28
#     vae = vae(input_dim=784)
#     x_recosntructed, mu, sigma = vae(x)
#     print(x_recosntructed.shape, mu.shape, sigma.shape)
#     # print(vae(x).shape)
