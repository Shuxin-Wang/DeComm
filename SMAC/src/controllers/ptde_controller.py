from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from modules.agents.ptde_rnn_agent import DecCoachNet, PolicyAppModule, SharedRNN


class NptdeMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args

        input_shape = self._get_input_shape(scheme)
        self.args.obs_input_dims = self._get_inputs_dims(scheme)
        self.args.state_dims = self._get_state_shape(scheme)

        self.shared_rnn = SharedRNN(input_shape, self.args)
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.coach_net = DecCoachNet(self.args)
        self.policy_app = PolicyAppModule(self.args)

        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if test_mode:
            qvals = self.forward(ep_batch, t_ep, test_mode=True)
        else:
            qvals, _, _, _ = self.forward(ep_batch, t_ep, test_mode=False)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if test_mode:
            self.shared_rnn.eval()
            self.agent.eval()
            self.coach_net.eval()
            self.policy_app.eval()

        self.hidden_states = self.shared_rnn(agent_inputs, self.hidden_states)

        if test_mode:
            z_dot, _ = self.policy_app(agent_inputs, test_mode=True)
            agent_outs = self.agent(self.hidden_states, z_dot)
        else:
            b, a, _ = agent_inputs.size()
            z_dist, z_wog_dist = self.coach_net(ep_batch, agent_inputs, t=t, return_one=False)
            z_dot, z_dot_dist = self.policy_app(agent_inputs, test_mode=False)
            agent_outs = self.agent(self.hidden_states, z_dot.view(b, a, -1))

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        if test_mode:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), z_dist, z_wog_dist, z_dot_dist

    def init_hidden(self, batch_size):
        self.hidden_states = self.shared_rnn.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return (
            list(self.shared_rnn.parameters())
            + list(self.agent.parameters())
            + list(self.coach_net.parameters())
            + list(self.policy_app.parameters())
        )

    def load_state(self, other_mac):
        self.shared_rnn.load_state_dict(other_mac.shared_rnn.state_dict())
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.coach_net.load_state_dict(other_mac.coach_net.state_dict())
        self.policy_app.load_state_dict(other_mac.policy_app.state_dict())

    def cuda(self):
        self.shared_rnn.cuda()
        self.agent.cuda()
        self.coach_net.cuda()
        self.policy_app.cuda()

    def save_models(self, path):
        th.save(self.shared_rnn.state_dict(), "{}/shared_rnn.th".format(path))
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.coach_net.state_dict(), "{}/coach_net.th".format(path))
        th.save(self.policy_app.state_dict(), "{}/policy_app.th".format(path))

    def load_models(self, path):
        self.shared_rnn.load_state_dict(th.load("{}/shared_rnn.th".format(path), map_location=lambda storage, loc: storage))
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.coach_net.load_state_dict(th.load("{}/coach_net.th".format(path), map_location=lambda storage, loc: storage))
        self.policy_app.load_state_dict(th.load("{}/policy_app.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        return th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def _get_inputs_dims(self, scheme):
        obs_dims, action_dims = scheme["obs"]["vshape"], scheme["avail_actions"]["vshape"][0]
        obs_input_dims = obs_dims
        if self.args.obs_last_action:
            obs_input_dims += action_dims
        if self.args.obs_agent_id:
            obs_input_dims += self.n_agents
        return obs_input_dims

    def _get_state_shape(self, scheme):
        return scheme["state"]["vshape"]

