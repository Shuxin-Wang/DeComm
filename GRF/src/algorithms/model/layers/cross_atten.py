import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads, head_dim):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads
        self.head_dim = head_dim
        self.embed_dim = heads * head_dim

        self.query_proj = nn.Linear(query_dim, self.embed_dim)
        self.key_proj = nn.Linear(key_dim, self.embed_dim)
        self.value_proj = nn.Linear(value_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_weights = None

    def forward(self, query, key, value, mask=None):
        B, L_q, _ = query.shape
        _, L_kv, _ = key.shape

        q = self.query_proj(query)  # [B, L_q, E]
        k = self.key_proj(key)    # [B, L_kv, E]
        v = self.value_proj(value)  # [B, L_kv, E]

        q = q.view(B, L_q, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale # [B, H, L_q, L_kv]

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)    # [B, 1, N, N]
            elif mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)   # [1, 1, N, N]

            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights) 
        self.attn_weights = attn_weights

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        out = self.out_proj(out)
        return out