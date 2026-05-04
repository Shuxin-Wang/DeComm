import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedACMLNet(nn.Module):
    def __init__(self, args, num_inputs):
        super(GatedACMLNet, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_mode = args.comm_mode
        self.naction_heads = args.naction_heads
        self.num_inputs = num_inputs

        self.obs_encoder = nn.Linear(num_inputs, self.hid_size)
        self.msg_encoder = nn.Linear(self.hid_size, self.hid_size)
        self.importance = nn.Linear(self.hid_size, 1)
        self.post = nn.Linear(self.hid_size * 2, self.hid_size)
        self.heads = nn.ModuleList([nn.Linear(self.hid_size, o) for o in self.naction_heads])
        self.value_head = nn.Linear(self.hid_size, 1)

        self.prune_ratio = float(getattr(args, "prune_ratio", 0.5))
        self.min_keep = int(getattr(args, "prune_min_keep", 1))

    def _build_prune_mask(self, scores, info):
        bsz, n = scores.shape
        keep_k = max(self.min_keep, int(round(n * (1.0 - self.prune_ratio))))
        keep_k = min(n, keep_k)
        comm_action = info.get("comm_action")
        if comm_action is not None:
            idx = torch.as_tensor(comm_action, device=scores.device).view(-1)
            mask = idx.float().unsqueeze(0).expand(bsz, n)
            return mask
        top_idx = torch.topk(scores, k=keep_k, dim=1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(1, top_idx, 1.0)
        return mask

    def forward(self, x, info={}):
        h = torch.tanh(self.obs_encoder(x))
        msg = torch.tanh(self.msg_encoder(h))
        score = self.importance(msg).squeeze(-1)
        prune_mask = self._build_prune_mask(score, info)
        msg = msg * prune_mask.unsqueeze(-1)

        if self.comm_mode == "sum":
            pooled = msg.sum(dim=1, keepdim=True).expand_as(h)
        else:
            denom = prune_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (msg.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)).expand_as(h)

        fused = torch.tanh(self.post(torch.cat([h, pooled], dim=-1)))
        action = [F.log_softmax(head(fused), dim=-1) for head in self.heads]
        value = self.value_head(fused)
        return action, value
