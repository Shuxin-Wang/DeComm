import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.cross_atten import CrossAttention

class DeCommNetMLP(nn.Module):
    """
    DeComm implementation for Communication codebase.
    Modeled after CommNetMLP and FlecommRNNAgent.
    """
    def __init__(self, args, num_inputs):
        super(DeCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.recurrent = args.recurrent

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        self.fc1 = nn.Linear(num_inputs, args.hid_size)
        
        if args.recurrent:
            self.rnn = nn.GRUCell(args.hid_size, args.hid_size)
        else:
            self.rnn = nn.Linear(args.hid_size, args.hid_size)

        try:
            self.att_heads = args.att_heads
        except:
            self.att_heads = 1
        self.att_embed_dim = args.hid_size // self.att_heads
        self.comm_att = CrossAttention(args.hid_size, num_inputs, num_inputs, self.att_heads, self.att_embed_dim)
        
        self.layer_norm = nn.LayerNorm(args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.hid_size)
        
        self.value_head = nn.Linear(self.hid_size, 1)

    def _quantize_inputs(self, inputs, prev_scores, comm_n=None):
        batch_size = inputs.shape[0]
        n_agents = self.nagents

        quantify_mode = getattr(self.args, "quantify", "weighted")

        quantized_inputs = inputs.clone()

        if comm_n is None:
            comm_n = n_agents
        comm_n = max(0, min(int(comm_n), n_agents))

        if comm_n == 0:
            return quantized_inputs

        if prev_scores is None:
            prev_scores = torch.zeros((batch_size, n_agents), device=inputs.device, dtype=inputs.dtype)
        elif not torch.is_tensor(prev_scores):
            prev_scores = torch.as_tensor(prev_scores, device=inputs.device)
        else:
            prev_scores = prev_scores.to(device=inputs.device)

        if prev_scores.dim() == 1:
            prev_scores = prev_scores.view(1, -1).expand(batch_size, n_agents)
        elif prev_scores.dim() == 2 and prev_scores.size(1) == 1:
            prev_scores = prev_scores.expand(batch_size, n_agents)
        elif prev_scores.dim() == 3 and prev_scores.size(-1) == 1:
            prev_scores = prev_scores.squeeze(-1)

        prev_scores = prev_scores.to(dtype=inputs.dtype)

        if quantify_mode == "weighted" or quantify_mode is True:
            _, indices = torch.sort(prev_scores, dim=1, descending=True)

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
            _, indices = torch.sort(prev_scores, dim=1, descending=True)
            for b in range(batch_size):
                comm_indices = indices[b, :comm_n]
                quantized_inputs[b, comm_indices] = quantized_inputs[b, comm_indices].half().float()
        elif quantify_mode == "fp8":
            _, indices = torch.sort(prev_scores, dim=1, descending=True)
            for b in range(batch_size):
                comm_indices = indices[b, :comm_n]
                if hasattr(torch, "float8_e4m3fn"):
                    quantized_inputs[b, comm_indices] = quantized_inputs[b, comm_indices].to(torch.float8_e4m3fn).float()
                else:
                    quantized_inputs[b, comm_indices] = quantized_inputs[b, comm_indices].half().float()

        return quantized_inputs

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask']).to(self.args.device)
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n, device=self.args.device)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1).clone()

        return num_agents_alive, agent_mask

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.nagents, self.hid_size, requires_grad=True, device=self.args.device)

    def forward(self, x, info={}):
        """
        Forward function for DeCommNetMLP.
        x: (Batch * N, num_inputs) or (Batch, N, num_inputs)
        """
        
        hidden_state = None
        if self.args.recurrent:
            x, extras = x
            hidden_state = extras[0] if isinstance(extras, tuple) else extras
            
        raw_inputs = x 
        x_encoded = F.relu(self.fc1(raw_inputs))
        
        if self.args.recurrent:
            if x_encoded.dim() == 3:
                x_encoded_flat = x_encoded.reshape(-1, x_encoded.shape[-1])
            else:
                x_encoded_flat = x_encoded
            
            if hidden_state.dim() == 3:
                hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])

            h = self.rnn(x_encoded_flat, hidden_state)
        else:
            h = F.relu(self.rnn(x_encoded))

        if h.dim() == 3:
            batch_size = h.shape[0]
            h_global = h
        else:
            total_batch_size = h.shape[0]
            batch_size = total_batch_size // self.nagents
            h_global = h.view(batch_size, self.nagents, -1)

        if raw_inputs.dim() == 3:
            inputs_global = raw_inputs
        else:
            inputs_global = raw_inputs.view(batch_size, self.nagents, -1) # [B, N, obs_dim]
        
        if not getattr(self.args, 'evaluate', False) or getattr(self.args, 'communication', True):
            mask = torch.eye(self.nagents, device=h.device).bool() 
            mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
            _, agent_mask = self.get_agent_mask(batch_size, info)
            alive_mask = agent_mask.to(device=h.device).squeeze(-1) > 0 # [B, N, N]
            mask = mask | (~alive_mask)
            alive_vec = alive_mask[:, 0, :] # [B, N]
            mask = mask | (~alive_vec).unsqueeze(-1).expand(-1, self.nagents, self.nagents)

            if 'comm_action' in info:
                comm_action = torch.as_tensor(info['comm_action'], device=h.device).view(-1).float()
                if comm_action.numel() == self.nagents:
                    comm_action = comm_action.view(1, self.nagents).expand(batch_size, self.nagents)
                    mask = mask | (comm_action <= 0).unsqueeze(1).expand(-1, self.nagents, self.nagents)
            
            kv_inputs = inputs_global
            quantify_arg = getattr(self.args, "quantify", False)
            if quantify_arg:
                comm_n = self.args.comm_agents if (getattr(self.args, "evaluate", False) and getattr(self.args, "communication", True)) else self.nagents
                prev_scores = None
                if isinstance(info, dict):
                    prev_scores = info.get("prev_values", None)
                kv_inputs = self._quantize_inputs(inputs_global, prev_scores, comm_n=comm_n)

            # Cross Attention: Query=h, Key=inputs, Value=inputs
            c = self.comm_att(h_global, kv_inputs, kv_inputs, mask=mask)
            
            # Residual + LayerNorm
            residual = h_global + c
            normalized = self.layer_norm(residual)
            
            out_features = normalized
        else:
            # No communication
            out_features = h_global

        value_head = self.value_head(out_features.reshape(-1, out_features.shape[-1]))
        
        if self.continuous:
            action_mean = self.action_mean(out_features)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            action = (action_mean, action_log_std, action_std)
        else:
            action = [F.log_softmax(head(out_features), dim=-1) for head in self.heads]
        
        if self.args.recurrent:
            return action, value_head, h_global
        else:
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)
