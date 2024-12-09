import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
from torch.cuda import device


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers : int = 32
    n_heads: int = 32 # for q
    n_kv_heads: Optional[int] = 32 # for k and v
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float=10000.0):
    assert head_dim % 2 == 0, "dim must be even"
    # shape: (head_dim / 2)
    # theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2, ....dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # (head_dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # m params
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiplt each theta by each pos using outer product
    # takes 1st element of 1st vector and multiplies it by every element of 2nd vector, then takes 2nd elememnt of 1st vector and multiplies it with every element of 2nd vecotr and goes on
    # shape: (seq_len) outer prod by (head_dim / 2) --> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # polar form of c = R * exp(i * m * theta), R=1
    # (seq_len, head_dim / 2) --> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (bs, seq_len, h, head_dim) -> (bs, seq_len, h, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (bs, seq_len, h, head_dim/2) * (1, seq_len, 1, head_dim/2) --> (bs, seq_len, h, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (bs, seq_len, h, head_dim / 2) --> (bs, seq_len, h, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (bs, seq_len, h, head_dim/2, 2) --> (bs, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # gamma

    def _norm(self, x: torch.Tensor):
        # (b, seq_len, dim) * (b, seq_len, 1) --> (b, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (dim) * (b, seq_len, dim) --> (b, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (b, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :] # just adds new dimension
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # n_rep = n_heads_q // n_kv_heads
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round to nearest multiple of multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # hs = 7 , multiple_of = 5
        # (7 + 4) // 5 = 2 --> (2 * 5) = 10 first multiple of 5 bigger than 7
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # seq_len = 1 because at input we only give 1 token at a time
        batch_size, seq_len, _ = x.shape # (b, 1, dim)

        # (b, 1, dim) --> (b, 1, h_q * head_dim) its of same size
        xq = self.wq(x)
        # (b, 1, dim) --> (b, 1, h_kv * head_dim) it may be smaller
        xk = self.wk(x)
        # (b, 1, dim) --> (b, 1, h_kv * head_dim) it also may be smaller
        xv = self.wv(x)

        # (b, 1, h_q * head_dim) --> (b, 1, h_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (b, 1, h_kv * head_dim) --> (b, 1, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # won't change the shapes
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # in kv cache we append the incoming value in key and value
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # get all cached keys and values till this
        # (bs, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # repeat the heads of k and v to reach the number oh heads of queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (b, 1, h_q, head_dim) --> (b, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # (b, h_q, 1, head_dim) @ (b, h_q, head_dim, seq_len_kv) --> (b, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(x)

        # (b, h_q, 1, seq_len_kv) @ (b, h_q, seq_len_kv, head_dim) --> (b, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (b, h_q, 1, head_dim) --> (b, 1, h_q, head_dim) --> (b, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (b, 1, dim) --> (b, 1, dim)
        return self.wo(output)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (b, seq_len, dim) + (b, seq_len, dim) --> (b, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.feed_forward_norm(h))
        return out 


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "vocab size must be defined"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # as input we will always give a single token as all others will always be in the cache
        # (b, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "only one token at a time as input"

        # (b, seq_len) --> (b, seq_ken, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for l in self.layers:
            h = l(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
