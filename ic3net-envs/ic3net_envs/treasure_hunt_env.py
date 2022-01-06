#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a cooperative search environment.

Design Decisions:
Vision, movement speed and map size can be used to define different difficulties. Since map size is fixed: 1x1, change movement speed and vision is more useful 

"""

# core modules
import random
import math
import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class TreasureHuntEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.TIMESTEP_PENALTY = -0.05
        self.TREASURE_OCCUPY_REWARD = 0
        self.TREASURE_DIGGING_REWARD = 0.5
        self.ASSIST_REWARD = 0.2

        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')
        env.add_argument("--dim", type=int, default=5)
    

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        self.dim = args.dim
        self.speed = 0.75/args.dim
        self.reach_distance = 0.5/args.dim
        self.agents = args.nagents
        self.treatures = args.nagents

        self.ref_act = self.speed*np.array([[-0.71,0.71],[0,1],[0.71,0.71],[-1,0],[0,0],[1,0],[-0.71,-0.71],[0,-1],[0.71,-0.71]])
        self.naction = 9

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design


        self.obs_dim = 2*self.agents + 4
        # Observation for each agent will be 7n-1 ndarray
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,self.obs_dim), dtype=int)
        return

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.trasures_hunted = np.zeros(self.treatures)

        coord = np.arange(self.dim)
        xv, yv = np.meshgrid(coord,coord)
        self.ref_loc = np.array(list(zip(xv.flat, yv.flat)))
        #Spawn agents and targets
        spawn_locs = np.random.choice(np.arange(self.dim*self.dim),size=self.agents+self.treatures,replace=False)
        agent_loc_raw = spawn_locs[0:self.agents]
        treasure_loc_raw = spawn_locs[self.agents:]
        self.agent_loc = self.ref_loc[agent_loc_raw]*(1.0/self.dim) + 0.1
        self.treature_loc = self.ref_loc[treasure_loc_raw]*(1.0/self.dim) + 0.1
        self.stat = dict()

        # Observation will be nagent * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def check_arrival(self):
        digging_rewards = np.zeros(self.agents)
        for t_idx in range(self.treatures):
            if self.trasures_hunted[t_idx] == 0:
                digging_agents = []
                for a_idx in range(self.agents):
                    if np.linalg.norm(self.agent_loc[a_idx,:] - self.treature_loc[t_idx,:]) <= self.reach_distance and not (a_idx == t_idx):
                        digging_agents.append(a_idx)  
                    if len(digging_agents) >= 1:
                        self.trasures_hunted[t_idx] = 1
                        for a_idx in digging_agents:
                            digging_rewards[a_idx] += self.TREASURE_DIGGING_REWARD
                        digging_rewards[t_idx] += self.ASSIST_REWARD                   
                                  
        return digging_rewards

    def _get_obs(self):
        '''
        there are n agents
        0:2: self location
        2:4: self treasure location
        4:2n+4: all agents location
        '''
        new_obs_set = []
        for idx in range(self.agents):
            obs = np.zeros((self.agents+2,2))
            obs[0,:] = self.agent_loc[idx,:]
            if self.trasures_hunted[idx] == 0:
                obs[1,:] = self.treature_loc[idx,:]
            else:
                obs[1,:] = np.array([-1,-1])
            obs[2:,:] = self.agent_loc
            new_obs_set.append(obs.flatten())
        
        new_obs = np.vstack(new_obs_set)
        return new_obs.copy()


    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        action = np.array(action).squeeze()
        #action = np.atleast_1d(action)
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        trans_action = [self.ref_act[idx,:] for idx in action]
        self.agent_loc = self.agent_loc + trans_action

        #renew observations
        self.obs = self._get_obs()
        #check dones
        digging_rewards = self.check_arrival()

        if np.sum(self.trasures_hunted) == self.treatures:
            self.episode_over = True
            self.stat['success'] = 1
        else:
            self.episode_over = False 
            self.stat['success'] = 0
        reward = np.full(self.agents, self.TIMESTEP_PENALTY)
        reward += digging_rewards

        debug = {'agent_locs':self.agent_loc,'treature_locs':self.treature_loc}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return

    def reward_terminal(self):
        return np.zeros(self.agents)
