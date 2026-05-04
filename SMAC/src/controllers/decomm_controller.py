from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC

class DeCommMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(DeCommMAC, self).__init__(scheme, groups, args)
        self.prev_agent_scores = None

    def init_hidden(self, batch_size):
        super().init_hidden(batch_size)
        self.prev_agent_scores = th.zeros((batch_size, self.n_agents), device=self.args.device)

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        prev_rewards = None
        if self.args.agent == "decomm_rnn" and (getattr(self.args, "quantify", False) or getattr(self.args, "communication", False)):
            if t > 0 and self.prev_agent_scores is not None:
                prev_rewards = self.prev_agent_scores
            else:
                prev_rewards = th.zeros((ep_batch.batch_size, self.n_agents), device=self.args.device)
        
        if self.args.agent == "decomm_rnn":
             agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, prev_rewards=prev_rewards)
        else:
             agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.args.agent == "decomm_rnn" and (getattr(self.args, "quantify", False) or getattr(self.args, "communication", False)):
            if self.agent_output_type == "q":
                q = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
                if avail_actions is not None and avail_actions.dim() == 3:
                    q = q.masked_fill(avail_actions == 0, -1e9)
                self.prev_agent_scores = q.max(dim=-1)[0].detach()

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

