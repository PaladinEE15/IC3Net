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
        self.TARGET_REWARD = 0.05
        self.VISION_REWARD = 0 #reward agents for keeping targets in view
        #self.POS_TARGET_REWARD = 0.05
        #self.COLLISION_PENALTY = -0.03
        
        self.episode_over = False
        self.ref_act = np.array([[-0.71,0.71],[0,1],[0.71,0.71],[-1,0],[0,0],[1,0],[-0.71,-0.71],[0,-1],[0.71,-0.71]])

    def init_args(self, parser):
        env = parser.add_argument_group('Treasure Hunt Group')

        env.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium"], 
                    help="The difficulty of the environment")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        if args.difficulty == "easy":
            self.vision = 0.3
            self.speed = 0.2 #max speed
            self.REACH_DISTANCE = 0.15
        elif args.difficulty == "medium":
            self.vision = 0.2
            self.speed = 0.15
            self.REACH_DISTANCE = 0.1

        self.ref_act = self.speed*self.ref_act
        self.naction = 9

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design
        '''
        there are 2 agents
        0-1: self-location
        2-3: two targets visibility
        4-5: self target ref location
        6-7: friend's target ref location
        '''

        self.obs_dim = 8
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
        self.reached_target = np.zeros(2)
        
        #Spawn agents and targets
        self.target_loc = np.random.rand(2,2)
        self.agent_loc = np.random.rand(2,2)
        self.other_target_loc = np.zeros_like(self.target_loc)
        self.other_target_loc[0,:] = self.target_loc[1,:]
        self.other_target_loc[1,:] = self.target_loc[0,:]
        #Check if agents are spawned near its target
        while True:
            reachs, idxs = self.check_arrival()
            if reachs>0:
                self.agent_loc[idxs,:] = np.random.rand(reachs,2)
            else:
                break
            # stat - like success ratio
        self.stat = dict()

        # Observation will be nagent * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def check_arrival(self):
        at_distances = self.target_loc - self.agent_loc
        self.distances = np.linalg.norm(at_distances,axis=1)
        target_mat = self.distances<=self.REACH_DISTANCE
        reach_sum = np.sum(target_mat)
        return reach_sum, target_mat

    def _get_obs(self):
        #observation space design
        '''
        there are 2 agents
        0-1: self-location
        2-3: two targets visibility
        4-5: self target ref location
        6-7: friend's target ref location
        '''
        new_obs = np.zeros((2,self.obs_dim))

        #get self.location
        new_obs[:,0:2] = self.agent_loc
        #get self target location
        agent_target_mat_1 = self.target_loc - self.agent_loc
        agent_target_mat_2 = self.other_target_loc - self.agent_loc
        dist_1 = np.linalg.norm(agent_target_mat_1, axis = 1)
        dist_2 = np.linalg.norm(agent_target_mat_2, axis = 1)
        visibility_1 = np.ones_like(dist_1)
        visibility_2 = np.ones_like(dist_2)
        visibility_1[dist_1>self.vision] = 0
        visibility_2[dist_2>self.vision] = 0
        agent_target_mat_1[dist_1>self.vision,:] = 0
        agent_target_mat_2[dist_2>self.vision,:] = 0
        new_obs[:,2] = visibility_1
        new_obs[:,3] = visibility_2
        new_obs[:,4:6] = agent_target_mat_1
        new_obs[:,6:8] = agent_target_mat_2
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

        #trans_action = [self.ref_act[idx] for idx in action]
        trans_action = np.zeros_like(self.agent_loc)
        for idx in range(2):
            if self.reached_target[idx]:
                continue
            else:
                trans_action[idx,:] = self.ref_act[action[idx],:]

        #lock successful agents. Is it necessary?
        self.agent_loc = self.agent_loc + trans_action
        self.agent_loc = np.clip(self.agent_loc, 0, 1)
        #renew observations
        self.obs = self._get_obs()
        #check dones
        reach_sum, target_mat = self.check_arrival()
        self.reached_target = target_mat

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
