import torch
import torch.nn as nn
import torch.nn.functional as F


class _HGCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, hypergraph):
        x_t = F.relu(self.feature(x))
        group_ctx = torch.bmm(hypergraph.transpose(1, 2), x_t)
        x_attn, _ = self.attn(x_t, group_ctx, group_ctx)
        return F.relu(self.out(x_attn))


class HYGMACommNet(nn.Module):
    def __init__(self, args, num_inputs):
        super().__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.naction_heads = args.naction_heads
        self.num_groups = int(getattr(args, "hygma_num_groups", 2))
        self.num_layers = int(getattr(args, "hygma_num_layers", 2))

        self.encoder = nn.Linear(num_inputs, self.hid_size)
        self.group_assign = nn.Linear(self.hid_size, self.num_groups)
        self.layers = nn.ModuleList([_HGCNLayer(self.hid_size) for _ in range(self.num_layers)])
        self.mix = nn.Linear(self.hid_size * 2, self.hid_size)
        self.heads = nn.ModuleList([nn.Linear(self.hid_size, o) for o in self.naction_heads])
        self.value_head = nn.Linear(self.hid_size, 1)

    def _mask_from_info(self, x, info):
        comm_action = info.get("comm_action")
        if comm_action is None:
            return x
        mask = torch.as_tensor(comm_action, device=x.device, dtype=x.dtype).view(1, self.nagents, 1)
        return x * mask

    def forward(self, x, info={}):
        x = self._mask_from_info(x, info)
        h = torch.tanh(self.encoder(x))
        assign_logits = self.group_assign(h)
        hypergraph = torch.softmax(assign_logits, dim=-1)
        h_g = h
        for layer in self.layers:
            h_g = layer(h_g, hypergraph)
        global_ctx = h_g.mean(dim=1, keepdim=True).expand_as(h_g)
        fused = torch.tanh(self.mix(torch.cat([h_g, global_ctx], dim=-1)))
        action = [F.log_softmax(head(fused), dim=-1) for head in self.heads]
        value = self.value_head(fused)
        return action, value
