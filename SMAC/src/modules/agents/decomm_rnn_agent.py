import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.cross_atten import CrossAttention


class FlecommRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FlecommRNNAgent, self).__init__()
        self.args = args

        self.n_agents = self.args.n_agents

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        try:
            self.att_heads = args.att_heads
        except:
            self.att_heads = 1

        self.att_embed_dim = args.rnn_hidden_dim // self.att_heads
        self.comm_att = CrossAttention(args.rnn_hidden_dim, input_shape, input_shape, self.att_heads, self.att_embed_dim)

        self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def _quantize_inputs(self, inputs, prev_rewards, comm_n=None):
        batch_size = inputs.shape[0]
        n_agents = self.n_agents

        quantify_mode = getattr(self.args, "quantify", "weighted")

        quantized_inputs = inputs.clone()

        if comm_n is None:
            comm_n = n_agents
        comm_n = max(0, min(int(comm_n), n_agents))

        if comm_n == 0:
            return quantized_inputs

        if quantify_mode == "weighted" or quantify_mode is True:
            _, indices = torch.sort(prev_rewards, dim=1, descending=True)

            n_top = comm_n // 3
            n_bottom = comm_n // 3
            n_mid = comm_n - n_top - n_bottom

            for b in range(batch_size):
                comm_indices = indices[b, :comm_n]

                mid_start = n_top
                mid_end = n_top + n_mid
                if mid_end > mid_start:
                    idx_mid = comm_indices[mid_start:mid_end]
                    quantized_inputs[b, idx_mid] = quantized_inputs[b, idx_mid].half().float()

                if n_bottom > 0:
                    idx_bottom = comm_indices[-n_bottom:]
                    if hasattr(torch, "float8_e4m3fn"):
                        quantized_inputs[b, idx_bottom] = quantized_inputs[b, idx_bottom].to(torch.float8_e4m3fn).float()
                    else:
                        quantized_inputs[b, idx_bottom] = quantized_inputs[b, idx_bottom].half().float()

        elif quantify_mode == "fp32":
            pass

        elif quantify_mode == "fp16":
            _, indices = torch.sort(prev_rewards, dim=1, descending=True)
            for b in range(batch_size):
                comm_indices = indices[b, :comm_n]
                quantized_inputs[b, comm_indices] = quantized_inputs[b, comm_indices].half().float()

        elif quantify_mode == "fp8":
            _, indices = torch.sort(prev_rewards, dim=1, descending=True)
            for b in range(batch_size):
                comm_indices = indices[b, :comm_n]
                if hasattr(torch, "float8_e4m3fn"):
                    quantized_inputs[b, comm_indices] = quantized_inputs[b, comm_indices].to(torch.float8_e4m3fn).float()
                else:
                    quantized_inputs[b, comm_indices] = quantized_inputs[b, comm_indices].half().float()

        return quantized_inputs

    def forward(self, inputs, hidden_state, prev_rewards=None):
        batch_size = inputs.shape[0] // self.n_agents

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        inputs_global = inputs.view(batch_size, self.n_agents, -1)
        h_global = h.view(batch_size, self.n_agents, -1)

        if not self.args.evaluate or self.args.communication:
            mask = torch.eye(self.n_agents, device=inputs.device).bool()

            if self.args.evaluate and self.args.communication:
                mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)

                comm_n = self.args.comm_agents
                other_n = self.n_agents - 1

                if prev_rewards is None:
                    prev_rewards = torch.zeros((batch_size, self.n_agents), device=inputs.device, dtype=inputs.dtype)
                elif prev_rewards.dim() == 1:
                    prev_rewards = prev_rewards.view(batch_size, 1).expand(batch_size, self.n_agents)
                elif prev_rewards.dim() == 2 and prev_rewards.size(1) == 1:
                    prev_rewards = prev_rewards.expand(batch_size, self.n_agents)
                elif prev_rewards.dim() == 3 and prev_rewards.size(-1) == 1:
                    prev_rewards = prev_rewards.squeeze(-1)

                if comm_n < other_n:
                    for b in range(batch_size):
                        for i in range(self.n_agents):
                            other_rewards = prev_rewards[b].clone()
                            other_rewards[i] = -1e9
                            topk_indices = other_rewards.topk(comm_n).indices
                            mask[b, i, :] = True
                            mask[b, i, topk_indices] = False
                            mask[b, i, i] = True

            kv_inputs = inputs_global

            quantify_arg = getattr(self.args, "quantify", False)
            if quantify_arg:
                comm_n = self.args.comm_agents if (self.args.evaluate and self.args.communication) else self.n_agents

                if quantify_arg == "weighted" and prev_rewards is not None:
                    if prev_rewards.dim() == 2:
                        prev_rewards = prev_rewards.expand(-1, self.n_agents)
                    elif prev_rewards.dim() == 3:
                        prev_rewards = prev_rewards.squeeze(-1)

                    kv_inputs = self._quantize_inputs(inputs_global, prev_rewards, comm_n=comm_n)
                elif isinstance(quantify_arg, str) and quantify_arg != "weighted":
                    kv_inputs = self._quantize_inputs(inputs_global, prev_rewards, comm_n=comm_n)
                elif quantify_arg is True:
                    if prev_rewards is not None:
                        if prev_rewards.dim() == 2:
                            prev_rewards = prev_rewards.expand(-1, self.n_agents)
                        elif prev_rewards.dim() == 3:
                            prev_rewards = prev_rewards.squeeze(-1)

                        kv_inputs = self._quantize_inputs(inputs_global, prev_rewards, comm_n=comm_n)

            m = self.comm_att(h_global, kv_inputs, kv_inputs, mask=mask)
            m = m.view(batch_size * self.n_agents, -1)

            residual = h + m
            normalized = self.layer_norm(residual)
            q = self.fc2(normalized)
        else:
            normalized = self.layer_norm(h)
            q = self.fc2(normalized)

        return q, h
