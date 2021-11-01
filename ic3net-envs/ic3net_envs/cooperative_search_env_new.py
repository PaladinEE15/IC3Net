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

def one_hot_encoding(dim, coord):
    #coord should be a 2x1 vector
    mat = np.zeros((2,dim),dtype=int)
    mat[0,coord[0]] = 1
    mat[1,coord[1]] = 1
    return mat

class CooperativeSearchEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.TIMESTEP_PENALTY = -0.05
        self.TARGET_REWARD = 0.5
        self.ASSIST_REWARD = 0.2 #reward agents for guiding teammate
        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')

        env.add_argument("--targets", type=int, default=1, 
                    help="How many targets one agent must reach")
        env.add_argument("--dim", type=int, default=5, 
                    help="How big the playground is")
                    

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG

        self.ref_act = np.array([[-1,0],[0,1],[1,0],[0,-1],[0,0]])
        self.ntargets = args.targets
        self.dim = args.dim
        self.naction = 5

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design
        coord = np.arange(self.dim)
        xv, yv = np.meshgrid(coord,coord)
        self.ref_loc = np.array(list(zip(xv.flat, yv.flat)))
        self.half_obs_dim = (2+self.ntargets)*self.dim
        # Observation for each agent will be (4n+2)d ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,self.half_obs_dim), dtype=int)
        return

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        #self.reached_target = np.zeros(2,self.ntargets) #mark which targets are reached
        self.alpha_reach = 0
        self.beta_reach = 0 #mark how many targets are reached for each agent
        #Spawn agents and targets
        #use a grid-like generation
        n = self.dim
        locations = np.random.choice(n*n,size=2+2*self.ntargets,replace=False)
        self.init_loc_raw = self.ref_loc[locations] 
        self.init_loc = [one_hot_encoding(self.dim,coord) for coord in self.init_loc_raw]
        self.alpha_agent = self.init_loc_raw[0]
        self.beta_agent = self.init_loc_raw[1]
        self.alpha_targets_onehot = self.init_loc[2:2+self.ntargets]
        self.beta_targets_onehot = self.init_loc[2+self.ntargets+2+self.ntargets*2]

        self.stat = dict()

        # Observation will be nagent * vision * vision ndarray
        self.obs, _ = self._get_obs()
        return self.obs

    def check_arrival(self):
        arrivals = np.zeros(2)
        if self.alpha_reach < self.ntargets:
            for target_locs in self.alpha_targets_onehot:
                if np.all(self.alpha_agent_onehot == target_locs):
                    self.alpha_reach += 1
                    arrivals[0] = 1
                    target_locs[:,:] = 0
                    break
        if self.beta_reach < self.ntargets:
            for target_locs in self.beta_targets_onehot:
                if np.all(self.beta_agent_onehot == target_locs):
                    self.beta_reach += 1
                    arrivals[1] = 1
                    target_locs[:,:] = 0
                    break

        #check dones
        if self.alpha_reach + self.beta_reach == 2*self.ntargets:
            dones = True
        else:
            dones = False

        return arrivals, dones

    def _get_obs(self):
        #observation space design
        '''
        there are d dims and n targets
        0-d :self location
        d-2d: teammate location
        2d-(2+n)d: teammate targets coord
        '''
        d = self.dim
        self.alpha_agent_onehot = one_hot_encoding(self.dim,self.alpha_agent)
        self.beta_agent_onehot = one_hot_encoding(self.dim,self.beta_agent)
        new_obs = np.zeros((4,self.half_obs_dim))
        new_obs[0:2,0:2*d] = self.alpha_agent_onehot
        new_obs[2:4,0:2*d] = self.beta_agent_onehot
        new_obs[0:2,2*d:4*d] = self.beta_agent_onehot
        new_obs[0:2,2*d:4*d] = self.alpha_agent_onehot

        new_obs[0:2,4*d:] = np.vstack(self.beta_targets_onehot)
        new_obs[2:4,4*d:] = np.vstack(self.alpha_targets_onehot)
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

        alpha_action = self.ref_act[action[0]]
        beta_action = self.ref_act[action[1]]

        self.alpha_agent = np.clip(self.alpha_agent + alpha_action, 0, self.dim-1)
        self.beta_agent = np.clip(self.beta_agent + beta_action, 0, self.dim-1)
        #renew observations
        self.obs = self._get_obs()
        #check dones

        arrivals, dones = self.check_arrival()

        if dones:
            self.episode_over = True
            self.stat['success'] = 1
        else:
            self.episode_over = False 
            self.stat['success'] = 0

        #get reward
        reward = np.full(2, self.TIMESTEP_PENALTY)
        reward += self.TARGET_REWARD*arrivals
        reward += self.ASSIST_REWARD*arrivals[::-1]

        debug = {'alpha_agent':self.alpha_agent,'beta_agent':self.beta_agent}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return

    def reward_terminal(self):
        return np.zeros_like(self.reached_target)
