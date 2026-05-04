import numpy as np
import torch
from torch import optim
import torch.nn as nn
from collections import namedtuple
from multiprocessing import Pipe, Process
from functools import partial
from inspect import getfullargspec
from utils.utils import *
from utils.action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

class CloudpickleWrapper():
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def env_worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            next_state, reward, done, info = env.step(data)
            remote.send((next_state, reward, done, info))
        elif cmd == "reset":
            state = env.reset(epoch=data) if data is not None else env.reset()
            remote.send(state)
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]


    def get_episode(self, epoch):
        episode = []
        reset_args = getfullargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        state = state.to(self.args.device).float()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size).to(self.args.device)

        def _sample_comm_action_override():
            if not getattr(self.args, "evaluate", False):
                return None
            k = int(getattr(self.args, "comm_agents", 0) or 0)
            if k <= 0 or k >= int(self.args.nagents):
                return None
            idx = np.random.choice(int(self.args.nagents), size=k, replace=False)
            comm_action = np.zeros(int(self.args.nagents), dtype=int)
            comm_action[idx] = 1
            return comm_action

        comm_action_override = _sample_comm_action_override()
        if comm_action_override is not None:
            info['comm_action'] = comm_action_override
        elif self.args.hard_attn and self.args.commnet:
            info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

        for t in range(self.args.max_steps):
            misc = dict()
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                
                if self.args.gacomm:
                    action_out, value, prev_hid, comm_density = self.policy_net(x, info)
                else:
                    action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if self.args.gacomm:
                    action_out, value, comm_density = self.policy_net(x, info)
                else:
                    action_out, value = self.policy_net(x, info)             

            if self.args.gacomm:
                stat['density1'] = stat.get('density1', 0) + comm_density[0]
                stat['density2'] = stat.get('density2', 0) + comm_density[1]               
            
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)
            next_state = next_state.to(self.args.device).float()

            info['prev_rewards'] = reward
            if torch.is_tensor(value):
                v = value.detach()
                if v.dim() == 2 and v.size(-1) == 1:
                    v = v.view(-1)
                b = int(state.size(0)) if state.dim() == 3 else 1
                if v.numel() == b * int(self.args.nagents):
                    info["prev_values"] = v.view(b, int(self.args.nagents))
                elif v.numel() == int(self.args.nagents):
                    info["prev_values"] = v.view(1, int(self.args.nagents)).expand(b, int(self.args.nagents))
                else:
                    info["prev_values"] = v

            # store comm_action in info for next step
            if comm_action_override is not None:
                info['comm_action'] = comm_action_override
            elif self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(self.args.device)
        episode_masks = torch.tensor(np.array(batch.episode_mask), dtype=torch.float32).to(self.args.device)
        episode_mini_masks = torch.tensor(np.array(batch.episode_mini_mask), dtype=torch.float32).to(self.args.device)
        
        if isinstance(batch.action[0], torch.Tensor):
            actions = torch.stack(batch.action)
            if actions.dim() == 4:
                actions = actions.squeeze(-1)
        else:
            actions = torch.tensor(np.array(batch.action), dtype=torch.float32)
            
        actions = actions.to(self.args.device)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(self.args.device)

        coop_returns = torch.Tensor(batch_size, n).to(self.args.device)
        ncoop_returns = torch.Tensor(batch_size, n).to(self.args.device)
        returns = torch.Tensor(batch_size, n).to(self.args.device)
        deltas = torch.Tensor(batch_size, n).to(self.args.device)
        advantages = torch.Tensor(batch_size, n).to(self.args.device)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

class ParallelTrainer(Trainer):
    def __init__(self, args, policy_net, env_fn):
        super().__init__(args, policy_net, None)
        self.batch_size_run = args.nprocesses
        
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size_run)])
        self.ps = [Process(target=env_worker, 
                           args=(worker_conn, CloudpickleWrapper(partial(env_fn))))
                   for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        
        # Initial reset
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", epoch))
        
        # Initialize states
        states_np = [parent_conn.recv() for parent_conn in self.parent_conns]
        current_states = torch.tensor(np.stack(states_np), dtype=torch.float32, device=self.args.device)
        
        if self.args.recurrent and self.args.rnn_type == 'LSTM':
            prev_hids = [(torch.zeros(self.args.nagents, self.args.hid_size).to(self.args.device),
                          torch.zeros(self.args.nagents, self.args.hid_size).to(self.args.device))
                         for _ in range(self.batch_size_run)]
        else:
            prev_hids = [torch.zeros(self.args.nagents, self.args.hid_size).to(self.args.device) 
                         for _ in range(self.batch_size_run)]
        
        # Episode buffers
        episode_data = [[] for _ in range(self.batch_size_run)]
        episode_steps = [0] * self.batch_size_run
        
        while len(batch) < self.args.batch_size:
            # Batch prediction
            input_states = current_states # Already a tensor
            info = {}
            
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM':
                    h = torch.cat([hid[0] for hid in prev_hids], dim=0)
                    c = torch.cat([hid[1] for hid in prev_hids], dim=0)
                    cur_hid = (h, c)
                else:
                    cur_hid = torch.cat(prev_hids, dim=0)
                
                x = [input_states, cur_hid]
                if self.args.gacomm:
                    action_outs, values, next_hids, comm_densities = self.policy_net(x, info)
                else:
                    action_outs, values, next_hids = self.policy_net(x, info)
                
                # Flatten outputs
                if isinstance(action_outs, list):
                    action_outs = [a.view(-1, a.size(-1)) for a in action_outs]
                values = values.view(-1, 1)

                # Update hidden states
                for b in range(self.batch_size_run):
                    start_idx = b * self.args.nagents
                    end_idx = (b + 1) * self.args.nagents
                    if self.args.rnn_type == 'LSTM':
                        prev_hids[b] = (next_hids[0][start_idx:end_idx].detach(), next_hids[1][start_idx:end_idx].detach())
                    else:
                        prev_hids[b] = next_hids[start_idx:end_idx].detach()
            else:
                if self.args.gacomm:
                    action_outs, values, comm_densities = self.policy_net(input_states, info)
                else:
                    action_outs, values = self.policy_net(input_states, info)
                
                if isinstance(action_outs, list):
                    action_outs = [a.view(-1, a.size(-1)) for a in action_outs]
                values = values.view(-1, 1)

            # Select actions
            actions = select_action(self.args, action_outs)
            
            # Translate all actions at once (Optimization)
            _, actual_all = translate_action(self.args, None, actions)
            
            # Send actions to all workers
            for b in range(self.batch_size_run):
                start_idx = b * self.args.nagents
                end_idx = (b + 1) * self.args.nagents
                
                # Slice each head's array
                step_action = [head_act[start_idx:end_idx] for head_act in actual_all]
                
                self.parent_conns[b].send(("step", step_action))

            # Receive results (Optimization: separate recv and process)
            results = [parent_conn.recv() for parent_conn in self.parent_conns]
            
            # Batch convert next states
            next_states_np = [res[0] for res in results]
            next_states = torch.tensor(np.stack(next_states_np), dtype=torch.float32, device=self.args.device)

            for b in range(self.batch_size_run):
                _, reward, done, info_env = results[b]
                next_state = next_states[b]
                
                episode_steps[b] += 1
                done = done or episode_steps[b] >= self.args.max_steps
                
                episode_mask = np.zeros(reward.shape) if done else np.ones(reward.shape)
                
                start_idx = b * self.args.nagents
                end_idx = (b + 1) * self.args.nagents
                
                b_action_out = [a[start_idx:end_idx] for a in action_outs] if isinstance(action_outs, list) else action_outs[start_idx:end_idx]
                b_value = values[start_idx:end_idx]
                b_action = actions[:, start_idx:end_idx, :].cpu()

                # Use clone() because current_states is modified in place
                trans = Transition(current_states[b].clone(), b_action, b_action_out, b_value, 
                                  episode_mask, np.ones(reward.shape), next_state.clone(), reward, {'alive_mask': np.ones_like(reward)})
                
                episode_data[b].append(trans)
                # current_states[b] = next_state # Removed inplace update
                
                if done:
                    # Save episode to batch
                    batch += episode_data[b]
                    self.stats['num_episodes'] += 1
                    current_reward = sum([tr.reward for tr in episode_data[b]])
                    self.stats['reward'] = self.stats.get('reward', 0) + current_reward
                    
                    # Reset buffer and worker
                    episode_data[b] = []
                    episode_steps[b] = 0
                    
                    # Auto-reset worker
                    self.parent_conns[b].send(("reset", epoch))
                    new_state_np = self.parent_conns[b].recv()
                    # Update next_states for next iteration (this is the reset state)
                    next_states[b] = torch.tensor(new_state_np, dtype=torch.float32, device=self.args.device)
                    
                    # Reset hidden state for this worker
                    if self.args.recurrent:
                        if self.args.rnn_type == 'LSTM':
                            prev_hids[b] = (torch.zeros(self.args.nagents, self.args.hid_size).to(self.args.device),
                                          torch.zeros(self.args.nagents, self.args.hid_size).to(self.args.device))
                        else:
                            prev_hids[b] = torch.zeros(self.args.nagents, self.args.hid_size).to(self.args.device)

            current_states = next_states

        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def quit(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))
        for p in self.ps:
            p.join()
