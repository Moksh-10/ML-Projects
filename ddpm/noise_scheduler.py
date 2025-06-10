import torch
from torch import nn


class linear_noise_sch:
    def __init__(self, num_timestpes, beta_start, beta_end):
        self.num_ts = num_timestpes
        self.bs = beta_start
        self.be = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timestpes)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        original_shape = original.shape
        bs = original_shape[0]

        # timestep = 1d tensor
        # sample = (b, c, h, w)

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(bs)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(bs)

        # (b) --> (b, 1, 1, 1)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_ts(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x0
        else:
            var = (1 - self.alpha_cum_prod[t-1]) / (1 - self.alpha_cum_prod[t])
            var = var * self.betas[t]
            sigma = var ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0

