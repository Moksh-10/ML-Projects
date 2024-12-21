import torch
import torch.nn as nn

class lf(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads, k):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.k = k
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.proj_matrix = nn.Parameter(torch.randn(seq_len, k))

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.shape

        Q = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.key_proj(x)    # (batch_size, seq_len, embed_dim)
        V = self.value_proj(x)  # (batch_size, seq_len, embed_dim)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        K_proj = torch.matmul(self.proj_matrix.T, K.transpose(2, 3))  # (batch_size, num_heads, k, head_dim)
        V_proj = torch.matmul(self.proj_matrix.T, V.transpose(2, 3))  # (batch_size, num_heads, k, head_dim)

        attention_scores = torch.matmul(Q, K_proj.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, k)
        attention_weights = torch.softmax(attention_scores, dim=-1)          # (batch_size, num_heads, seq_len, k)

        attention_output = torch.matmul(attention_weights, V_proj)          # (batch_size, num_heads, seq_len, head_dim)

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        output = self.out_proj(attention_output)  # (batch_size, seq_len, embed_dim)

        return output

batch_size = 2
seq_len = 16
embed_dim = 64
num_heads = 4
k = 5

x = torch.rand(batch_size, seq_len, embed_dim)
y = lf(embed_dim, seq_len, num_heads, k)
output = y(x)
print(output.shape)
