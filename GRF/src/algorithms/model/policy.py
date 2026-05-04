import numpy as np
import torch
from types import SimpleNamespace
from torch.distributions import Categorical

from src.utils.util import get_shape_from_obs_space, get_shape_from_act_space, update_linear_schedule
from src.algorithms.utils.util import check
from .comm import CommNetMLP
from .hygma import HYGMACommNet
from .gated_acml import GatedACMLNet
from .decomm import DeCommNetMLP
from .tar_comm import TarCommNetMLP


class _ModelAdapter:
    def __init__(self, model, num_agents, device):
        self.model = model
        self.edges = torch.zeros((num_agents, num_agents), device=device)

    def model_parameters(self):
        return self.model.parameters()

    def edge_parameters(self):
        return self.model.parameters()

    def edge_return(self, exact=True):
        return self.edges

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self):
        self.comm_args.evaluate = False
        self.model.train()

    def eval(self):
        self.comm_args.evaluate = True
        self.model.eval()


class CommunicationPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        self.act_dim = act_space.n if act_space.__class__.__name__ == "Discrete" else get_shape_from_act_space(act_space)
        self.act_num = 1
        self.hidden_size = args.hidden_size

        cargs = self._build_comm_args(args, num_agents, self.obs_dim, self.act_dim, device)
        model_cls = {
            "commnet": CommNetMLP,
            "decomm": DeCommNetMLP,
            "tarcomm": TarCommNetMLP,
            "gated_acml": GatedACMLNet,
            "hygma": HYGMACommNet,
        }[self.algorithm_name]
        self.model = model_cls(cargs, self.obs_dim).to(device)
        self.transformer = _ModelAdapter(self.model, num_agents, device)
        self.optimizer = torch.optim.Adam(self.transformer.model_parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.edge_optimizer = self.optimizer
        self.comm_args = cargs

    def _build_comm_args(self, args, num_agents, obs_dim, act_dim, device):
        return SimpleNamespace(
            nagents=num_agents,
            hid_size=args.hidden_size,
            comm_passes=getattr(args, "comm_passes", 1),
            recurrent=False,
            rnn_type="LSTM",
            continuous=False,
            dim_actions=1,
            naction_heads=[act_dim],
            init_std=0.2,
            comm_init_std=0.2,
            comm_mask_zero=getattr(args, "comm_mask_zero", False),
            comm_init=getattr(args, "comm_init", "uniform"),
            share_weights=getattr(args, "share_weights", True),
            hard_attn=getattr(args, "hard_attn", False),
            comm_mode=getattr(args, "comm_mode", "avg"),
            comm_action_one=getattr(args, "comm_action_one", True),
            batch_size=getattr(args, "n_rollout_threads", 1),
            obs_size=obs_dim,
            att_heads=getattr(args, "att_heads", 1),
            quantify=getattr(args, "quantify", False),
            communication=True,
            comm_agents=getattr(args, "comm_agents", 0),
            prune_ratio=getattr(args, "prune_ratio", 0.5),
            prune_min_keep=getattr(args, "prune_min_keep", 1),
            hygma_num_groups=getattr(args, "hygma_num_groups", 2),
            hygma_num_layers=getattr(args, "hygma_num_layers", 2),
            evaluate=False,
            device=device,
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def _apply_available(self, log_probs, available_actions):
        if available_actions is None:
            return log_probs
        mask = torch.as_tensor(available_actions, device=log_probs.device, dtype=log_probs.dtype)
        return log_probs.masked_fill(mask <= 0, -1e10)

    def _forward_model(self, obs, info, rnn_states_actor):
        obs_t = check(obs).to(**self.tpdv).reshape(-1, self.num_agents, self.obs_dim)
        if info is None:
            info = {}
        out = self.model(obs_t, info)
        if isinstance(out, tuple):
            return out[0], out[1], out[2] if len(out) > 2 else None
        return out, None, None

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False, info=None):
        action_out, values, next_hid = self._forward_model(obs, info, rnn_states_actor)
        log_probs = action_out[0]
        log_probs = self._apply_available(log_probs, available_actions.reshape(-1, self.num_agents, self.act_dim) if available_actions is not None else None)
        probs = torch.softmax(log_probs, dim=-1)
        dist = Categorical(probs=probs)
        actions = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
        action_log_probs = dist.log_prob(actions)
        values = values.reshape(-1, self.num_agents, 1)

        actions = actions.reshape(-1, 1)
        action_log_probs = action_log_probs.reshape(-1, 1)
        values = values.reshape(-1, 1)

        rnn_states_actor_t = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic_t = check(rnn_states_critic).to(**self.tpdv)
        return values, actions, action_log_probs, rnn_states_actor_t, rnn_states_critic_t

    def get_values(self, cent_obs, obs, rnn_states_critic, masks, available_actions=None, info=None):
        _, values, _ = self._forward_model(obs, info, rnn_states_critic)
        return values.reshape(-1, 1)

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks, available_actions=None, active_masks=None, steps=0, total_step=0, info=None):
        action_out, values, _ = self._forward_model(obs, info, rnn_states_actor)
        log_probs = action_out[0]
        log_probs = self._apply_available(log_probs, available_actions.reshape(-1, self.num_agents, self.act_dim) if available_actions is not None else None)
        probs = torch.softmax(log_probs, dim=-1)
        dist = Categorical(probs=probs)
        actions_t = check(actions).to(**self.tpdv).long().reshape(-1, self.num_agents)
        action_log_probs = dist.log_prob(actions_t).reshape(-1, 1)
        entropy = dist.entropy().reshape(-1, 1)
        values = values.reshape(-1, 1)
        if self._use_policy_active_masks and active_masks is not None:
            active_masks_t = check(active_masks).to(**self.tpdv)
            entropy = (entropy * active_masks_t).sum() / active_masks_t.sum()
        else:
            entropy = entropy.mean()
        return values, action_log_probs, entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True, info=None):
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, rnn_states_actor, _ = self.get_actions(
            cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions, deterministic, info=info
        )
        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.model.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        state_dict = torch.load(model_dir, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
