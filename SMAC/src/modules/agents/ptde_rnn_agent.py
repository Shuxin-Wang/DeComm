import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class PTDERNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PTDERNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim + args.z_dims, args.n_actions)

    def forward(self, input_hidden_out, z):
        b, a, e = input_hidden_out.size()
        h = F.relu(self.fc1(input_hidden_out.view(-1, e)), inplace=True)
        q = self.fc2(torch.concat([h, z.view(b * a, -1)], dim=-1))
        return q.view(b, a, -1)


class SharedRNN(nn.Module):
    def __init__(self, obs_input_dims, args):
        super(SharedRNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(obs_input_dims, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h = self.rnn(x, hidden_state.view(b * a, -1))
        return h.view(b, a, -1)


class PolicyAppModule(nn.Module):
    def __init__(self, args):
        super(PolicyAppModule, self).__init__()
        self.args = args

        self.poli_app1 = nn.Linear(args.obs_input_dims, args.rnn_hidden_dim)
        self.poli_app2 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.z_dims),
            nn.LeakyReLU(),
            nn.Linear(args.z_dims, args.z_dims),
        )
        self.poli_app3 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.z_dims),
            nn.LeakyReLU(),
            nn.Linear(args.z_dims, args.z_dims),
        )

    def forward(self, inputs, test_mode=False):
        b, a, e = inputs.size()
        z_hidden = F.relu(self.poli_app1(inputs.view(-1, e)), inplace=True)
        mu = self.poli_app2(z_hidden)
        sigma = self.poli_app3(z_hidden)
        if test_mode:
            sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor, max=self.args.var_floor)
        else:
            sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)
        dist = D.Normal(mu, sigma ** (1 / 2))
        z = dist.rsample()
        return z.view(b, a, -1), dist


class DecCoachNet(nn.Module):
    def __init__(self, args):
        super(DecCoachNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        if args.two_hyper_layers:
            self.w1 = nn.Sequential(
                nn.Linear(args.obs_input_dims, args.high_hyper_hidden_dims),
                nn.ReLU(),
                nn.Linear(args.high_hyper_hidden_dims, args.state_dims * args.z_dims),
            )
        else:
            self.w1 = nn.Sequential(nn.Linear(args.obs_input_dims, args.state_dims * args.z_dims))

        self.b1 = nn.Sequential(nn.Linear(args.obs_input_dims, args.z_dims))

        self.mu = nn.Sequential(
            nn.Linear(args.z_dims, args.z_dims),
            nn.LeakyReLU(),
            nn.Linear(args.z_dims, args.z_dims),
        )
        self.sigma = nn.Sequential(
            nn.Linear(args.z_dims, args.z_dims),
            nn.LeakyReLU(),
            nn.Linear(args.z_dims, args.z_dims),
        )

    def forward(self, ep_batch, obs, t, return_one=False):
        state = ep_batch["state"][:, t, :]
        b, a, e = obs.size()
        obs = obs.view(b * a, e)
        state = state.unsqueeze(1).repeat(1, a, 1).view(b * a, 1, -1)

        w1 = self.w1(obs).view(-1, self.args.state_dims, self.args.z_dims)
        b1 = self.b1(obs).view(-1, 1, self.args.z_dims)
        z_hidden = F.elu(torch.matmul(state, w1) + b1).squeeze()

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.args.var_floor)
        dist = D.Normal(mu, sigma ** (1 / 2))

        if return_one:
            return dist
        with torch.no_grad():
            dist_wog = D.Normal(mu.clone(), sigma.clone() ** (1 / 2))
        return dist, dist_wog

