import sys
import time
import signal
import argparse

import numpy as np
import torch
import visdom
import data
from models import *
from tarcomm import TARMACMLP
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_processing import MultiProcessTrainer
import os

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch RL trainer')


# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--optim_type', type=str, default='adam',
                    help="Type of mode for communication tensor calculation [adam|rmsprop]")

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
parser.add_argument('--lrate', type=float, default=0.001,
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
parser.add_argument('--tarmac', action='store_true', default=False,
                    help="enable tarmac model")
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
                    help='type of rnn to use. [LSTM|MLP|GRU]')
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
parser.add_argument("--comm_detail", type=str, default="mlp", choices=["raw", "mlp", "triangle", "cos", "widecos", "bell", "mim"], 
                    help="How to process broadcasting messages")

parser.add_argument('--quant', default=False, action='store_true', 
                    help='Whether test and do quantification')
parser.add_argument('--v_size', default=32, type=int,
                    help='message size')
parser.add_argument('--qk_size', default=16, type=int,
                    help='message size')
parser.add_argument('--msg_size', default=128, type=int,
                    help='message size')
parser.add_argument('--msg_hid_layer_size', default=128, type=int,
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
parser.add_argument('--no_mask', default=False, action='store_true', 
                    help='when calculating entropy, whether consider mask')
parser.add_argument('--loss_start', default=0, type=int, 
                    help='which epoch starts comm_entro loss calculate')
parser.add_argument('--quant_start', default=0, type=int, 
                    help='which epoch starts quant')
parser.add_argument('--loss_alpha', default=0, type=float,
                    help='the weight of entropy loss')
parser.add_argument('--no_input_grad', default=False, action='store_true', 
                    help='whether treat encoding input as no-grad. True: no grad. False: grad')
# Configs for gumbel-softmax
parser.add_argument('--gumbel_gamma', default=1, type=float,
                    help='gamma of gumbel-softmax')

# Add a no-comm hyper-parameter to control
parser.add_argument('--no_comm', default=False, action='store_true', 
                    help='whether block comm. ')
                    
parser.add_argument("--map_name", type=str, default="3s_vs_4z", choices=["3s_vs_4z", "5m_vs_6m"], help='map name for starcraft')
parser.add_argument("--redirect_path", type=str, default= None, help='log of sc env')

#test_parameters
parser.add_argument('--test_times', default=0, type=int, 
                    help='test times')
parser.add_argument('--test_type', default=0, type=int, 
                    help='type0: normal. type1: pp-easy-ORI. type2: pp-easy-PEM')
parser.add_argument('--error_rate', default=0, type=float, 
                    help='bit error rate')                    
parser.add_argument('--test_models', default='', nargs='+', type=str,
                    help='name of tested models')



init_args_for_env(parser)
args = parser.parse_args()




def save(path):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    torch.save(d, path)

def load(path):
    d = torch.load(path)
    # log.clear()
    policy_net.load_state_dict(d['policy_net'])
    trainer.load_state_dict(d['trainer'])

def signal_handler(signal, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            env.end_display()
        trainer.quit()
        import os
        os._exit(0)
        #sys.exit(0)

#def disp():
#    x = disp_trainer.get_episode()
'''
def test(num_epochs):
    #running episodes equals to num_epochs*nprocess*batchsize
    stat = dict()
    success_set = []
    steps_set = []
    entropy_set = []

    for n in range(num_epochs):
        stat = trainer.test_batch(100)
        if 'comm_entropy' in stat.keys():
            entropy_set.append(stat['comm_entropy']/(args.batch_size*args.nprocesses))
        if 'success' in stat.keys():
            success_set.append(stat['success']/(args.batch_size*args.nprocesses))
        if 'steps_taken' in stat.keys():
            steps_set.append(stat['steps_taken']/(args.batch_size*args.nprocesses))
    if 'comm_entropy' in stat.keys():
        print('comm_entropy_mean: ', np.mean(entropy_set),' std: ', np.std(entropy_set))
    if 'success' in stat.keys():
        print('success_mean: ', np.mean(success_set),' std: ', np.std(success_set))
    if 'steps_taken' in stat.keys():
        print('steps_taken_mean: ', np.mean(steps_set),' std: ', np.std(steps_set))
    return

'''


def run(num_epochs):
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
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            elif k == 'comm_entropy' and k in stat.keys():
                stat[k] = stat[k] / (args.epoch_size*args.nprocesses)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] = stat[k] / stat[v.divide_by]
                v.data.append(stat.get(k, 0))

        np.set_printoptions(precision=2)

        if args.redirect_path is None:
            print('Epoch {}\tReward {}\tTime {:.2f}s'.format(
                    epoch, stat['reward'], epoch_time
            ))
            if 'comm_entropy' in stat.keys():
                print('comm_entropy: {}'.format(stat['comm_entropy']))
            if 'enemy_reward' in stat.keys():
                print('Enemy-Reward: {}'.format(stat['enemy_reward']))
            if 'add_rate' in stat.keys():
                print('Add-Rate: {:.3f}'.format(stat['add_rate']))
            if 'success' in stat.keys():
                print('Success: {:.3f}'.format(stat['success']))
            if 'steps_taken' in stat.keys():
                print('Steps-taken: {:.2f}'.format(stat['steps_taken']))
            if 'comm_action' in stat.keys():
                print('Comm-Action: {}'.format(stat['comm_action']))
            if 'enemy_comm' in stat.keys():
                print('Enemy-Comm: {}'.format(stat['enemy_comm']))
            if 'comm_entro_loss' in stat.keys():
                print('comm_entro_loss: {}'.format(stat['comm_entro_loss']))
            if 'other_loss' in stat.keys():
                print('other_loss: {}'.format(stat['other_loss']))
        else:
            with open(args.redirect_path, mode='a', encoding="utf-8") as sc_log:
                sc_log.write('Epoch:'+str(epoch)+' Time:'+str(epoch_time)+"\n")
                sc_log.write('Success:'+str(stat['success'])+"\n")
                if 'comm_action' in stat.keys():
                    sc_log.write('Comm-Action:'+str(stat['comm_action'])+"\n")
                if 'comm_entropy' in stat.keys():
                    sc_log.write('Comm-Entropy:'+str(stat['comm_entropy'])+"\n")           

        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

        if args.save_every and ep and args.save != '' and ep % args.save_every == 0:
            # fname, ext = args.save.split('.')
            # save(fname + '_' + str(ep) + '.' + ext)
            save(args.save + '_' + str(ep))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0

        # For TJ set comm action to 1 as specified in paper to showcase
        # importance of individual rewards even in cooperative games
        if args.env_name == "traffic_junction": # or "coop_search"
            args.comm_action_one = True
    if args.tarmac:
        args.mean_ratio = 0
        args.msg_size = args.v_size + args.qk_size
    # Enemy comm

    args.msg_hid_layer = [args.msg_hid_layer_size, args.msg_hid_layer_size]
    args.nfriendly = args.nagents
    if hasattr(args, 'enemy_comm') and args.enemy_comm:
        if hasattr(args, 'nenemies'):
            args.nagents += args.nenemies
        else:
            raise RuntimeError("Env. needs to pass argument 'nenemy'.")

    env = data.init(args.env_name, args, False)

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
    if args.commnet and args.recurrent and args.rnn_type == 'MLP':
        args.rnn_type = 'GRU'


    parse_action_args(args)

    if args.seed == -1:
        args.seed = np.random.randint(0,10000)
    torch.manual_seed(args.seed)
    
    print(args)


    if args.commnet:
        policy_net = CommNetMLP(args, num_inputs).to(torch.device("cuda"))
    elif args.tarmac:
        policy_net = TARMACMLP(args, num_inputs).to(torch.device("cuda"))
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

    for p in policy_net.parameters():
        p.data.share_memory_()

    if args.nprocesses > 1:
        trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
    else:
        trainer = Trainer(args, policy_net, data.init(args.env_name, args))

    #disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
    #disp_trainer.display = True
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
    log['comm_entro_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['other_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
    log['comm_entropy'] = LogField(list(), True, 'epoch', None)
    if args.test_times>0:
        entropy_set = []
        msg_len_set = []
        success_set = []
        steps_set = []
        distribution_set = []
        for model_path in args.test_models:
            load(model_path)
            if args.test_type == 0:
                entropy, success, steps, distributions = trainer.test_batch(args.test_times)
                entropy_set.append(np.mean(entropy))
                success_set.append(np.mean(success))
                steps_set.append(np.mean(steps))
                distribution_set.append(distributions)
            else:
                channel_msg_len, success, steps = trainer.test_batch(args.test_times)
                msg_len_set.append(channel_msg_len)
                success_set.append(np.mean(success))
                steps_set.append(np.mean(steps))
        if args.test_type > 0:
            success_set = np.array(success_set)
            steps_set = np.array(steps_set)
            msg_len_set = np.array(msg_len_set)   
            print('success: ', np.mean(success_set),' std: ', np.std(success_set) )
            print('steps: ', np.mean(steps_set),' std: ', np.std(steps_set))     
            print('channel msg avg len:', np.mean(msg_len_set))         
        else:
            entropy_set = np.array(entropy_set)
            success_set = np.array(success_set)
            steps_set = np.array(steps_set)
            distribution_set = list(np.mean(np.vstack(distribution_set),axis=0))
            print('entropy: ', np.mean(entropy_set),' std: ', np.std(entropy_set))
            print('success: ', np.mean(success_set),' std: ', np.std(success_set) )
            print('steps: ', np.mean(steps_set),' std: ', np.std(steps_set))     
            print('distributions: ')     
            print('[', end='')
            for items in distribution_set:
                print(items, end=',')
            print(']')


    else:
        if args.plot:
            vis = visdom.Visdom(env=args.plot_env)
        signal.signal(signal.SIGINT, signal_handler)

        if args.load != '':
            load(args.load)
            print('loading successful!')

        run(args.num_epochs)
        if args.display:
            env.end_display()

        if args.save != '':
            save(args.save)

    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        import os
        os._exit(0)
