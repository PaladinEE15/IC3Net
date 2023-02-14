import sys
import gym
import ic3net_envs
from env_wrappers import *

def init(env_name, args, final_init=True):
    if env_name == 'levers':
        env = gym.make('Levers-v0')
        env.multi_agent_init(args.total_agents, args.nagents)
        env = GymWrapper(env)
    elif env_name == 'number_pairs':
        env = gym.make('NumberPairs-v0')
        m = args.max_message
        env.multi_agent_init(args.nagents, m)
        env = GymWrapper(env)
    elif env_name == 'predator_prey':
        env = gym.make('PredatorPrey-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'traffic_junction':
        env = gym.make('TrafficJunction-v0')
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'coop_search':
        env = gym.make('CooperativeSearch-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'treasure_hunt':
        env = gym.make('TreasureHunt-v0')
        env.multi_agent_init(args)
        env = GymWrapper(env)
    elif env_name == 'joint_monitoring':
        env = gym.make('JointMonitoring-v0')
        env = GymWrapper(env)
    else:
        raise RuntimeError("wrong env name")

    return env
