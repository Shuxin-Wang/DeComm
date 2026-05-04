import copy

import torch as th
from torch.optim import Adam, RMSprop
from torch.distributions import kl_divergence

from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets


class PTDELearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.last_target_update_episode = 0
        self.params = list(mac.parameters())

        if args.mixer == "qmix":
            self.mixer = QMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        else:
            raise ValueError("Mixer {} not recognised.".format(args.mixer))

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        if getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        z_dists = []
        z_dot_dists = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outs = self.mac.forward(batch, t=t, test_mode=False)
            if isinstance(outs, tuple):
                agent_outs, z_dist, _, z_dot_dist = outs
                z_dists.append(z_dist)
                z_dot_dists.append(z_dot_dist)
            else:
                agent_outs = outs
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t, test_mode=True)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out, dim=1)

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])
            targets = build_td_lambda_targets(
                rewards,
                terminated,
                mask,
                target_max_qvals,
                self.args.n_agents,
                self.args.gamma,
                self.args.td_lambda,
            )

        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = chosen_action_qvals - targets.detach()
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        td_loss = masked_td_error.sum() / mask.sum()
        loss = td_loss

        kl_loss = None
        lambda_kl = getattr(self.args, "lambda_kl", 0.0)
        if lambda_kl and len(z_dists) == len(z_dot_dists) and len(z_dists) > 0:
            kl_terms = [kl_divergence(p, q).sum(dim=-1).mean() for p, q in zip(z_dists, z_dot_dists)]
            kl_loss = sum(kl_terms) / len(kl_terms)
            loss = loss + float(lambda_kl) * kl_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            if kl_loss is not None:
                self.logger.log_stat("loss_kl", kl_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

