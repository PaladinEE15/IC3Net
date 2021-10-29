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


class CooperativeSearchEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.TIMESTEP_PENALTY = -0.05
        self.TARGET_REWARD = 0.1
        self.VISION_REWARD = 0 #reward agents for keeping targets in view
        #self.POS_TARGET_REWARD = 0.05
        #self.COLLISION_PENALTY = -0.03
        
        self.episode_over = False
        self.ref_act = np.array([[-0.71,0.71],[0,1],[0.71,0.71],[-1,0],[0,0],[1,0],[-0.71,-0.71],[0,-1],[0.71,-0.71]])

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')

        env.add_argument("--targets", type=int, default=1, 
                    help="How many targets one agent must reach")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        self.vision = 0.25
        self.speed = 0.15 #max speed
        self.REACH_DISTANCE = 0.1
        self.spawndim = math.floor(0.5/self.REACH_DISTANCE)
        coord = (np.arange(self.spawndim)+0.5)/self.spawndim
        xv, yv = np.meshgrid(coord,coord)
        self.init_loc = np.array(list(zip(xv.flat, yv.flat)))

        self.ntargets = args.targets
        
        self.ref_act = self.speed*self.ref_act
        self.naction = 9

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design

        self.obs_dim = 6*self.ntargets + 2
        # Observation for each agent will be 7n-1 ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,self.obs_dim), dtype=int)
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
        self.reach_targets = np.zeros(2) #mark how many targets are reached for each agent
        #Spawn agents and targets
        #use a grid-like generation
        n = self.spawndim
        locations = np.random.choice(n*n,size=2+2*self.ntargets,replace=False)
        self.agent_loc = self.init_loc[locations[0:2]]
        self.target_loc = self.init_loc[locations[2:]].reshape(2,-1)
        self.other_target_loc = np.zeros_like(self.target_loc)
        self.other_target_loc[0,:] = self.target_loc[1,:]
        self.other_target_loc[1,:] = self.target_loc[0,:]
        self.total_target_loc = np.concatenate((self.target_loc, self.other_target_loc),axis=1)
        #Check if agents are spawned near its target
        self.stat = dict()

        # Observation will be nagent * vision * vision ndarray
        self.obs, _ = self._get_obs()
        return self.obs

    def _get_obs(self):
        #observation space design
        '''
        there are 2 agents and n targets
        0-1: self-location
        2-1+n: self targets visibility
        n+2-2n+1: other targets visibility
        2n+2-4n+1: self targets ref location
        4n+2-6n+1: other targets ref location  
        '''
        
        n = self.ntargets
        self_loc_mat = np.repeat(self.agent_loc,2*n,axis=0)
        tar_loc_mat = self.total_target_loc.reshape(-1,2)
        relative_loc = tar_loc_mat - self_loc_mat
        dist = np.linalg.norm(relative_loc, axis = 1)
        in_vision = np.ones_like(dist)
        in_vision[dist>self.vision] = 0
        relative_loc[dist>self.vision,:] = 0

        new_obs = np.zeros((2,self.obs_dim))
        new_obs[:,0:2] = self.agent_loc
        new_obs[:,2:2*n+2] = in_vision.reshape(2,-1)
        new_obs[:,2*n+2:6*n+2] = relative_loc.reshape(2,-1)

        target_dist = dist.reshape(2,-1)
        target_dist = target_dist[:,0:n]

        return new_obs.copy(), target_dist


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

        trans_action = [self.ref_act[idx] for idx in action]

        self.agent_loc = self.agent_loc + trans_action
        self.agent_loc = np.clip(self.agent_loc, 0, 1)
        #renew observations
        self.obs, target_dist = self._get_obs()
        #check dones
        if np.any(target_dist):
            dist_idx = target_dist.flatten()
            temp_target_loc = self.target_loc.reshape(-1,2)
            temp_target_loc[dist_idx] = -1
            self.target_loc = temp_target_loc.reshape(2,-1)
            self.other_target_loc[0,:] = self.target_loc[1,:]
            self.other_target_loc[1,:] = self.target_loc[0,:]
            self.total_target_loc = np.concatenate((self.target_loc, self.other_target_loc),axis=1) 
            #give rewards
            if            


        if reach_sum == 2:
            self.episode_over = True
            self.stat['success'] = 1
        else:
            self.episode_over = False 
            self.stat['success'] = 0
        n = 2
        #get reward
        reward = np.full(2, self.TIMESTEP_PENALTY)
        reward[target_mat] = self.TARGET_REWARD
        invision_idxes = (self.distances < self.vision) & (self.distances > self.REACH_DISTANCE)
        invision_reward = self.VISION_REWARD*(self.vision - self.distances[invision_idxes])/(self.vision - self.REACH_DISTANCE)
        reward[invision_idxes] += invision_reward

        debug = {'agent_locs':self.agent_loc,'target_locs':self.target_loc}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return

    def reward_terminal(self):
        return np.zeros_like(self.reached_target)
