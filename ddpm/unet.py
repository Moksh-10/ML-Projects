import torch
from torch import nn


def get_time_embd(time_steps, t_emb_dim):
    factor = 10000 ** ((torch.arange
                        (0, t_emb_dim//2, device=time_steps.device) / (t_emb_dim//2)
                        ))
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.ds = down_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.attn_norm = nn.GroupNorm(8, out_ch)
        self.attn = nn.MultiheadAttention(out_ch, num_heads, batch_first=True)
        self.res_input_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1) if self.ds else nn.Identity()

    def forward(self, x, t_emb):
        out = x

        # resnet
        res_in = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.res_input_conv(res_in)

        # attn
        bs, c, h, w = out.shape
        in_attn = out.reshape(bs, c, h*w)
        in_attn = self.attn_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attn(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(bs, c, h, w)
        out = out + out_attn

        out = self.down_sample_conv(out)
        return out


