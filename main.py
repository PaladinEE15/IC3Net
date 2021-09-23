import sys
import time
import signal
import argparse

import numpy as np
import torch
import visdom
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiEnvTrainer
import os
from inspect import getargspec
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')


# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nprocesses', type=int, default=16,
                    help='How many processes to run')
# model
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
# optimization
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed') # TODO: works in thread?
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.0003,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
# environment
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')


parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

# CommNet specific args
parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
parser.add_argument('--nagents', type=int, default=1,
                    help="Number of agents (used in multiagent)")
parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')

# Comm message design details
parser.add_argument("--comm_detail", type=str, default="raw", choices=["raw", "mlp", "triangle", "cos", "binary", "mim"], 
                    help="How to process broadcasting messages")
parser.add_argument('--test_times', default=0, type=int, 
                    help='test times')
parser.add_argument('--quant', default=False, action='store_true', 
                    help='Whether do quantification')
parser.add_argument('--msg_size', default=128, type=int,
                    help='message size')
parser.add_argument('--msg_hid_layer', default=[128,128], type=list,
                    help='message layer size')
parser.add_argument('--quant_levels', default=17, type=int,
                    help='quantification levels')                 

# set GPU
parser.add_argument('--GPU', default=0, type=int, 
                    help='run on which GPU')
# Configs for entropy loss
parser.add_argument('--mim_gauss_var', default=1/9, type=float,
                    help='the variance of ref gaussian in mutual information minimization')
parser.add_argument('--calcu_entropy', default=False, action='store_true', 
                    help='whether calculate entropy. ')
parser.add_argument('--loss_start', default=0, type=int, 
                    help='which epoch starts comm_entro loss calculate')
parser.add_argument('--loss_alpha', default=0, type=float,
                    help='the weight of entropy loss')
# Configs for gumbel-softmax
parser.add_argument('--gumbel_gamma', default=1, type=float,
                    help='gamma of gumbel-softmax')
                    
init_args_for_env(parser)
args = parser.parse_args()

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        trainer.quit()
        import os
        os._exit(0)
        #sys.exit(0)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0

        # For TJ set comm action to 1 as specified in paper to showcase
        # importance of individual rewards even in cooperative games
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
    args.action_space = env.action_space
    reset_args = getargspec(env.reset).args
    if 'epoch' in reset_args:
        args.reset_withepoch = True
    else:
        args.reset_withepoch = False
    if hasattr(env, 'reward_terminal'):
        args.reward_terminal = True
    else:
        args.reward_terminal = False

    num_inputs = env.observation_dim
    args.num_actions = env.num_actions

    # Multi-action
    if not isinstance(args.num_actions, (list, tuple)): # single action case
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = num_inputs

    # Hard attention
    if args.hard_attn and args.commnet:
        # add comm_action as last dim in actions
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1

    # Recurrence
    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'
    parse_action_args(args)

    if args.seed == -1:
        args.seed = np.random.randint(0,10000)
    torch.manual_seed(args.seed)

    print(args)

    if args.commnet:
        policy_net = CommNetMLP(args, num_inputs).to(torch.device("cuda"))
    elif args.random:
        policy_net = Random(args, num_inputs).to(torch.device("cuda"))
    elif args.recurrent:
        policy_net = RNN(args, num_inputs).to(torch.device("cuda"))
    else:
        policy_net = MLP(args, num_inputs).to(torch.device("cuda"))

    if not args.display:
        display_models([policy_net])

    # share parameters among threads, but not gradients
    #policy_net.share_memory()
    '''
    for p in policy_net.parameters():
        p.data.share_memory_()    
    '''
    trainer = MultiEnvTrainer(args, policy_net)

    train_log = trainer.train(args.num_epochs)


    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        import os
        os._exit(0)
