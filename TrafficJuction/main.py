import sys
import time
import signal
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visdom

import data
import gym

# Import models from algs
from algs.decomm import DeCommNetMLP
from algs.magic import MAGIC
from algs.comm import CommNetMLP
from algs.ga_comm import GACommNetMLP
from algs.tar_comm import TarCommNetMLP
from models import MLP, Random, RNN

# Import Trainer
from trainer import Trainer, ParallelTrainer
from functools import partial

from utils.utils import *
from utils.action_utils import parse_action_args

gym.logger.set_level(40)

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

parser = argparse.ArgumentParser(description='Multi-Agent Graph Attention Communication and Baselines')

# training
parser.add_argument('--num_epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10, help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500, help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=1, help='How many processes to run')

# model common
parser.add_argument('--hid_size', default=64, type=int, help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False, help='make the model recurrent in time')
parser.add_argument('--rnn_type', default='MLP', type=str, help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int, help='detach hidden state and cell state for rnns at this interval')

# optimization
parser.add_argument('--gamma', type=float, default=1.0, help='discount factor')
parser.add_argument('--tau', type=float, default=1.0, help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1, help='random seed') 
parser.add_argument('--normalize_rewards', action='store_true', default=False, help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--entr', type=float, default=0, help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01, help='coefficient for value loss term')

# environment
parser.add_argument('--env_name', default="predator_prey", help='name of the environment to run: predator_prey / traffic_junction')
parser.add_argument('--max_steps', default=20, type=int, help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str, help='the number of agent actions')
parser.add_argument('--action_scale', default=1.0, type=float, help='scale action output from model')

# other
parser.add_argument('--plot', action='store_true', default=False, help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str, help='plot env name')
parser.add_argument('--plot_port', default='8097', type=str, help='plot port')
parser.add_argument('--save', action="store_true", default=False, help='save the model after training')
parser.add_argument('--save_every', default=0, type=int, help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str, help='load the model')
parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate latest saved model for this env/alg (unless --load is set)')
parser.add_argument('--eval_episodes', default=100, type=int, help='number of episodes to run for evaluation')
parser.add_argument('--display', action="store_true", default=False, help='display environment state')
parser.add_argument('--random', action='store_true', default=False, help="enable random model")

# Algorithm selection
parser.add_argument('--decomm', action='store_true', default=False, help="enable decomm model")
parser.add_argument('--magic', action='store_true', default=False, help="enable MAGIC model")
parser.add_argument('--commnet', action='store_true', default=False, help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False, help="enable ic3net model")
parser.add_argument('--tarcomm', action='store_true', default=False, help="enable tarmac model (with commnet or ic3net)")
parser.add_argument('--gacomm', action='store_true', default=False, help="enable gacomm model")

# MAGIC specific args
parser.add_argument('--directed', action='store_true', default=False, help='whether the communication graph is directed')
parser.add_argument('--self_loop_type1', default=2, type=int, help='self loop type in the first gat layer')
parser.add_argument('--self_loop_type2', default=2, type=int, help='self loop type in the second gat layer')
parser.add_argument('--gat_num_heads', default=1, type=int, help='number of heads in gat layers except the last one')
parser.add_argument('--gat_num_heads_out', default=1, type=int, help='number of heads in output gat layer')
parser.add_argument('--gat_hid_size', default=64, type=int, help='hidden size of one head in gat')
parser.add_argument('--ge_num_heads', default=4, type=int, help='number of heads in the gat encoder')
parser.add_argument('--first_gat_normalize', action='store_true', default=False, help='whether normalize the coefficients in the first gat layer')
parser.add_argument('--second_gat_normalize', action='store_true', default=False, help='whether normilize the coefficients in the second gat layer')
parser.add_argument('--gat_encoder_normalize', action='store_true', default=False, help='whether normilize the coefficients in the gat encoder')
parser.add_argument('--use_gat_encoder', action='store_true', default=False, help='whether use the gat encoder before learning the first graph')
parser.add_argument('--gat_encoder_out_size', default=64, type=int, help='hidden size of output of the gat encoder')
parser.add_argument('--first_graph_complete', action='store_true', default=False, help='whether the first communication graph is set to a complete graph')
parser.add_argument('--second_graph_complete', action='store_true', default=False, help='whether the second communication graph is set to a complete graph')
parser.add_argument('--learn_second_graph', action='store_true', default=False, help='whether learn a new communication graph at the second round of communication')
parser.add_argument('--message_encoder', action='store_true', default=False, help='whether use the message encoder')
parser.add_argument('--message_decoder', action='store_true', default=False, help='whether use the message decoder')

# CommNet/Baselines specific args
parser.add_argument('--nagents', type=int, default=1, help="Number of agents")
parser.add_argument('--comm_mode', type=str, default='avg', help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1, help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False, help="Whether communication should be there")
parser.add_argument('--comm_agents', default=0, type=int, help='during evaluation: randomly keep K agents for communication (0 disables)')
parser.add_argument('--quantify', default=False, action='store_true', help='whether to quantify the messages')
parser.add_argument('--mean_ratio', default=1.0, type=float, help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--comm_init', default='uniform', type=str, help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true', help='Whether to use hard attention')
parser.add_argument('--comm_action_one', default=False, action='store_true', help='Whether to always talk')
parser.add_argument('--advantages_per_action', default=False, action='store_true', help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true', help='Share weights for hops')
parser.add_argument('--qk_hid_size', default=16, type=int, help='key and query size for soft attention')
parser.add_argument('--value_hid_size', default=32, type=int, help='value size for soft attention')

parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

init_args_for_env(parser)
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    args.device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')

if args.evaluate:
    args.save = False
    args.plot = False
    args.save_every = 0

if args.magic:
    args.recurrent = True
    args.rnn_type = 'LSTM'

# Setup logic from run_baselines.py
if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0
    if args.env_name == "traffic_junction":
        args.comm_action_one = True
    
if args.gacomm:
    args.commnet = 1
    args.mean_ratio = 0
    if args.env_name == "traffic_junction":
        args.comm_action_one = True

# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

args.obs_size = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): 
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = args.obs_size # align naming

# Hard attention
if args.hard_attn and args.commnet:
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence logic
if args.magic:
    args.recurrent = True
    args.rnn_type = 'LSTM'
    
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)

print(args)

# Model selection
if args.magic:
    policy_net = MAGIC(args)
elif args.gacomm:
    policy_net = GACommNetMLP(args, args.num_inputs)
elif args.commnet:
    if args.tarcomm:
        policy_net = TarCommNetMLP(args, args.num_inputs)
    else:
        policy_net = CommNetMLP(args, args.num_inputs)
elif args.random:
    policy_net = Random(args, args.num_inputs)
elif args.decomm:
    policy_net = DeCommNetMLP(args, args.num_inputs)
elif args.recurrent:
    policy_net = RNN(args, args.num_inputs)
else:
    policy_net = MLP(args, args.num_inputs)

policy_net.to(args.device)

if not args.display:
    display_models([policy_net])

for p in policy_net.parameters():
    p.data.share_memory_()

disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()    
    
log = dict()
log['epoch'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
log['density1'] = LogField(list(), True, 'epoch', 'num_steps')
log['density2'] = LogField(list(), True, 'epoch', 'num_steps')

if args.plot:
    vis = visdom.Visdom(env=args.plot_env, port=args.plot_port)

# Model dir selection
if args.magic:
     model_dir = Path('./saved') / args.env_name / 'magic'
elif args.gacomm:
    model_dir = Path('./saved') / args.env_name / 'gacomm'
elif args.tarcomm:
    if args.ic3net:
        model_dir = Path('./saved') / args.env_name / 'tar_ic3net'
    elif args.commnet:
        model_dir = Path('./saved') / args.env_name / 'tar_commnet'
    else:
        model_dir = Path('./saved') / args.env_name / 'other'
elif args.ic3net:
    model_dir = Path('./saved') / args.env_name / 'ic3net'
elif args.commnet:
    model_dir = Path('./saved') / args.env_name / 'commnet'
elif args.decomm:
    model_dir = Path('./saved') / args.env_name / 'decomm'

def _latest_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {base_dir}")
    run_dirs = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("run"):
            continue
        suffix = p.name[3:]
        if not suffix.isdigit():
            continue
        run_dirs.append((int(suffix), p))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {base_dir}")
    run_dirs.sort(key=lambda x: x[0])
    return run_dirs[-1][1]

if args.evaluate:
    run_dir = _latest_run_dir(model_dir)
    if args.load == "":
        args.load = str(run_dir / "model.pt")
else:
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run 

def run(num_epochs): 
    num_episodes = 0
    if args.save:
        os.makedirs(run_dir)
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        for n in range(args.epoch_size):
            if n == args.epoch_size - 1 and args.display:
                trainer.display = True
            s = trainer.train_batch(ep)
            merge_stat(s, stat)
            trainer.display = False

        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        num_episodes += stat['num_episodes']
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)
        
        print('Epoch {}'.format(epoch))
        print('Episode: {}'.format(num_episodes))
        print('Reward: {}'.format(stat['reward']))
        print('Time: {:.2f}s'.format(epoch_time))
        
        if 'enemy_reward' in stat.keys():
            print('Enemy-Reward: {}'.format(stat['enemy_reward']))
        if 'add_rate' in stat.keys():
            print('Add-Rate: {:.2f}'.format(stat['add_rate']))

        # Log results to file
        alg_name = 'mlp'
        if args.magic: alg_name = 'magic'
        elif args.decomm: alg_name = 'decomm'
        elif args.gacomm: alg_name = 'gacomm'
        elif args.tarcomm and args.ic3net: alg_name = 'tar_ic3'
        elif args.tarcomm and args.commnet: alg_name = 'tar_comm'
        elif args.ic3net: alg_name = 'ic3net'
        elif args.commnet: alg_name = 'commnet'
        elif args.random: alg_name = 'random'
        elif args.recurrent: alg_name = 'rnn'
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        result_file_name = os.path.join(results_dir, alg_name + '_' + args.env_name + '_train.csv')
        file_exists = os.path.isfile(result_file_name)
        
        with open(result_file_name, 'a') as f:
            headers = ['epoch', 'episode', 'reward', 'time', 'success', 'steps_taken', 'comm_action', 'enemy_comm', 'density1', 'density2', 'enemy_reward', 'add_rate']
            if not file_exists:
                f.write(','.join(headers) + '\n')
            
            row = [
                str(epoch),
                str(num_episodes),
                '"{}"'.format(str(stat['reward']).replace('\n', ' ')),
                '{:.2f}'.format(epoch_time),
                '{:.4f}'.format(stat['success']) if 'success' in stat else '',
                '{:.2f}'.format(stat['steps_taken']) if 'steps_taken' in stat else '',
                '"{}"'.format(str(stat.get('comm_action', '')).replace('\n', ' ')) if 'comm_action' in stat else '',
                '"{}"'.format(str(stat.get('enemy_comm', '')).replace('\n', ' ')) if 'enemy_comm' in stat else '',
                '{:.4f}'.format(stat['density1']) if 'density1' in stat else '',
                '{:.4f}'.format(stat['density2']) if 'density2' in stat else '',
                '"{}"'.format(str(stat.get('enemy_reward', '')).replace('\n', ' ')) if 'enemy_reward' in stat else '',
                '{:.2f}'.format(stat['add_rate']) if 'add_rate' in stat else ''
            ]
            f.write(','.join(row) + '\n')

        if 'success' in stat.keys():
            print('Success: {:.4f}'.format(stat['success']))
        if 'steps_taken' in stat.keys():
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']))
        if 'comm_action' in stat.keys():
            print('Comm-Action: {}'.format(stat['comm_action']))
        if 'enemy_comm' in stat.keys():
            print('Enemy-Comm: {}'.format(stat['enemy_comm']))
        if 'density1' in stat.keys():
            print('density1: {:.4f}'.format(stat['density1']))
        if 'density2' in stat.keys():
            print('density2: {:.4f}'.format(stat['density2']))


        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))
    
        if args.save_every and ep and args.save and (ep+1) % args.save_every == 0:
            save(final=False, epoch=ep+1)

        if args.save:
            save(final=True)

def evaluate(num_episodes: int, epoch: int = 0):
    policy_net.eval()
    stat = dict()
    total_episodes = 0
    start_t = time.time()

    with torch.no_grad():
        for _ in range(num_episodes):
            _, episode_stat = trainer.get_episode(epoch)
            merge_stat(episode_stat, stat)
            total_episodes += 1

    elapsed = time.time() - start_t
    num_steps = stat.get("num_steps", 0)

    for k, v in log.items():
        if k == "epoch":
            continue
        if k not in stat:
            continue
        if v.divide_by == "num_episodes" and total_episodes > 0:
            stat[k] = stat[k] / total_episodes
        elif v.divide_by == "num_steps" and num_steps > 0:
            stat[k] = stat[k] / num_steps

    alg_name = 'mlp'
    if args.magic: alg_name = 'magic'
    elif args.decomm: alg_name = 'decomm'
    elif args.gacomm: alg_name = 'gacomm'
    elif args.tarcomm and args.ic3net: alg_name = 'tar_ic3'
    elif args.tarcomm and args.commnet: alg_name = 'tar_comm'
    elif args.ic3net: alg_name = 'ic3net'
    elif args.commnet: alg_name = 'commnet'
    elif args.random: alg_name = 'random'
    elif args.recurrent: alg_name = 'rnn'

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    result_file_name = os.path.join(results_dir, alg_name + '_' + args.env_name + '_evaluate.csv')
    file_exists = os.path.isfile(result_file_name)

    with open(result_file_name, 'a') as f:
        headers = ['epoch', 'episode', 'reward', 'time', 'success', 'steps_taken', 'comm_action', 'enemy_comm', 'density1', 'density2', 'enemy_reward', 'add_rate']
        if not file_exists:
            f.write(','.join(headers) + '\n')

        row = [
            'eval',
            str(total_episodes),
            '"{}"'.format(str(stat.get('reward', '')).replace('\n', ' ')),
            '{:.2f}'.format(elapsed),
            '{:.4f}'.format(stat['success']) if 'success' in stat else '',
            '{:.2f}'.format(stat['steps_taken']) if 'steps_taken' in stat else '',
            '"{}"'.format(str(stat.get('comm_action', '')).replace('\n', ' ')) if 'comm_action' in stat else '',
            '"{}"'.format(str(stat.get('enemy_comm', '')).replace('\n', ' ')) if 'enemy_comm' in stat else '',
            '{:.4f}'.format(stat['density1']) if 'density1' in stat else '',
            '{:.4f}'.format(stat['density2']) if 'density2' in stat else '',
            '"{}"'.format(str(stat.get('enemy_reward', '')).replace('\n', ' ')) if 'enemy_reward' in stat else '',
            '{:.2f}'.format(stat['add_rate']) if 'add_rate' in stat else ''
        ]
        f.write(','.join(row) + '\n')

    np.set_printoptions(precision=2)
    print(f"Evaluate: {args.env_name} | {model_dir.name} | {run_dir.name}")
    print(f"Checkpoint: {args.load}")
    print(f"Episodes: {total_episodes}")
    print(f"Time: {elapsed:.2f}s")
    if "reward" in stat:
        print(f"Reward: {stat['reward']}")
    if "enemy_reward" in stat:
        print(f"Enemy-Reward: {stat['enemy_reward']}")
    if "add_rate" in stat:
        print(f"Add-Rate: {stat['add_rate']:.2f}")
    if "success" in stat:
        print(f"Success: {stat['success']:.4f}")
    if "steps_taken" in stat:
        print(f"Steps-Taken: {stat['steps_taken']:.2f}")
    if "comm_action" in stat:
        print(f"Comm-Action: {stat['comm_action']}")
    if "enemy_comm" in stat:
        print(f"Enemy-Comm: {stat['enemy_comm']}")
    if "density1" in stat:
        print(f"density1: {stat['density1']:.4f}")
    if "density2" in stat:
        print(f"density2: {stat['density2']:.4f}")

def save(final, epoch=0): 
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    if final:
        torch.save(d, run_dir / 'model.pt')
    else:
        torch.save(d, run_dir / ('model_ep%i.pt' %(epoch)))

def load(path):
    try:
        d = torch.load(path, map_location=args.device, weights_only=False)
    except TypeError:
        d = torch.load(path, map_location=args.device)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    if args.nprocesses > 1:
        trainer = ParallelTrainer(args, policy_net, partial(data.init, args.env_name, args))
    else:
        trainer = Trainer(args, policy_net, data.init(args.env_name, args))

    if args.load != '':
        load(args.load)

    if args.evaluate:
        evaluate(args.eval_episodes)
    else:
        run(args.num_epochs)
    if args.display:
        env.end_display()

    if args.save:
        save(final=True)

    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        import os
        os._exit(0)
