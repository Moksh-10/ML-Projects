import torch

def sw(hs, overlap):
    chunk_size = [hs.size(0), torch.div(hs.size(1), overlap, rounding_mode="trunc") -1, overlap*2, hs.size(2)]
    overlap_chunks = torch.empty(chunk_size)
    for c in range(chunk_size[1]):
        overlap_chunks[:, c, :, :] = hs[:, c * overlap: c * overlap + (2 * overlap), :]
    return overlap_chunks

q = torch.randn(1, 8, 768)
k = torch.randn(1, 8, 768)

q_new = sw(q, 2)
k_new = sw(k, 2)

print(q_new.shape)

attn = torch.einsum("bsxd, bsyd -> bsxy", q_new, k_new)
print(attn.shape)



