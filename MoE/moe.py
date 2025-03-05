import math
import torch
from torch import nn
import torch.nn.functional as F

n_embd = 768
n_head = 12
n_layer = 12
head_size = 768 // 12
dropout = 0.2


class MultiheadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size):
        super().__init__()
        self.emb = n_embd
        self.nh = num_heads
        self.hd = head_size
        self.d = nn.Dropout(dropout)

        self.wq = nn.Linear(n_embd, n_embd)
        self.wk = nn.Linear(n_embd, n_embd)
        self.wv = nn.Linear(n_embd, n_embd)
        self.wo = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        bs, sl, dim = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bs, sl, self.nh, self.hd).transpose(1, 2)
        v = v.view(bs, sl, self.nh, self.hd).transpose(1, 2)
        k = k.view(bs, sl, self.nh, self.hd).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        attn = F.softmax(attn, dim=-1)
        attn = self.d(attn)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(bs, sl, self.emb)
        attn = self.wo(attn)
        return attn


class expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(n_embd, 4 * n_embd),
                               nn.ReLU(),
                               nn.Linear(4 * n_embd, n_embd),
                               nn.Dropout(dropout))

    def forward(self, x):
        x = self.a(x)
        return x


class noisy_top_k_router(nn.Module):
    def __init__(self, n_embd, num_exp, top_k):
        super().__init__()
        self.tk = top_k
        self.ln = nn.Linear(n_embd, num_exp)
        self.noise_ln = nn.Linear(n_embd, num_exp)

    def forward(self, mh_out):
        logits = self.ln(mh_out)
        nl = self.noise_ln(mh_out)
        noise = torch.randn_like(logits) * F.softplus(nl)
        noise_log = logits + noise
        logits = noise_log
        tk_log, ind = logits.topk(self.tk, dim=-1)
        z = torch.full_like(logits, float('-inf'))
        sl = z.scatter(-1, ind, tk_log)
        ro = F.softmax(sl, dim=-1)
        return ro, ind


class sparse_moe(nn.Module):
    def __init__(self, n_embd, num_exp, top_k):
        super().__init__()
        self.router = noisy_top_k_router(n_embd, num_exp, top_k)
        self.exp = nn.ModuleList([expert(n_embd) for _ in range(num_exp)])
        self.tk = top_k

    def forward(self, x):
        gate_out, ind = self.router(x)
        final_out = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gat_out = gate_out.view(-1, gate_out.size(-1))

        for i, ex in enumerate(self.exp):
            exp_mask = (ind == i).any(dim=-1)
            flat_mask = exp_mask.view(-1)

            if flat_mask.any():
                exp_inp = flat_x[flat_mask]
                exp_out = ex(exp_inp)
                gate_sc = flat_gat_out[flat_mask, i].unsqueeze(1)
                wei_out = exp_out * gate_sc
                final_out[exp_mask] += wei_out

        return final_out


class block(nn.Module):
    def __init__(self, n_embd, n_head, num_exp, top_k):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiheadAttention(n_embd, n_head, head_size)
        self.moe = sparse_moe(n_embd, num_exp, top_k)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


dev = 'cuda' if torch.cuda.is_available() else 'cpu'

x = torch.randn(5, 69, n_embd).to(dev)
a = block(n_embd, n_head, num_exp=8, top_k=2).to(dev)
b = a(x)
print(b.shape)



