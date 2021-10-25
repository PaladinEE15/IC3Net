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
        env = parser.add_argument_group('Cooperative Search task')

        env.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium"], 
                    help="The difficulty of the environment")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        if args.difficulty == "easy":
            self.vision = 0.3
            self.speed = 0.15 #max speed
        elif args.difficulty == "medium":
            self.vision = 0.2
            self.speed = 0.1
        self.REACH_DISTANCE = self.speed

        self.ref_act = self.speed*self.ref_act
        self.ntarget = args.nfriendly
        self.nagent = args.nfriendly
        self.naction = 9

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design
        '''
        assume there are n agents
        0~n-1: identification
        n~n+1: self-location
        n+2~3n+1: other agents' location
        3n+2~5n+1: other targets' locations
        5n+2~6n+1: other agents' visibilities
        6n+2-7n+1: other targets' visibilities
        '''

        self.obs_dim = 7*self.ntarget+2
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
        self.reached_target = np.zeros(self.nagent)
        
        #Spawn agents and targets
        self.target_loc = np.random.rand(self.ntarget,2)
        self.agent_loc = np.random.rand(self.nagent,2)
        #Check if agents are spawned near its target
        reachs, idxs = self.check_arrival()
        if reachs>0:
            self.agent_loc[idxs,:] = np.random.rand(reachs,2)
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
        n = self.nagent
        new_obs = np.zeros((n,self.obs_dim))
        #get identification first
        for i in range(n):
            new_obs[i,i] = 1
        #get self location
        new_obs[:,n:n+2] = self.agent_loc
        #calculate relative location and record
        self_loc_mat = np.repeat(self.agent_loc,n,0)
        agent_loc_mat = np.tile(self.agent_loc,(n,1))
        target_loc_mat = np.tile(self.target_loc,(n,1))
        agent_dis_mat = agent_loc_mat - self_loc_mat
        target_dis_mat = target_loc_mat - agent_loc_mat
        agent_dist = np.linalg.norm(agent_dis_mat, axis=1)
        target_dist = np.linalg.norm(target_dis_mat, axis=1)
        agent_dis_mat[agent_dist>self.vision,:] = 0
        target_dis_mat[target_dist>self.vision,:] = 0
        agent_visibility = np.ones_like(agent_dist)
        target_visibility = np.ones_like(target_dist)
        agent_visibility[agent_dist>self.vision] = 0
        target_visibility[target_dist>self.vision] = 0
        #put values in obs mat
        new_obs[:,n+2:3*n+2] = agent_dis_mat.reshape(n,-1)
        new_obs[:,3*n+2:5*n+2] = target_dis_mat.reshape(n,-1)
        new_obs[:,5*n+2:6*n+2] = target_visibility.reshape(n,-1)
        new_obs[:,6*n+2:7*n+2] = agent_visibility.reshape(n,-1)
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

        trans_action = [self.ref_act[idx] for idx in action]

        #lock successful agents. Is it necessary?
        #trans_action[self.reached_target] = 0

        self.agent_loc = self.agent_loc + trans_action
        self.agent_loc = np.clip(self.agent_loc, 0, 1)
        #renew observations
        self.obs = self._get_obs()
        #check dones
        reach_sum, target_mat = self.check_arrival()
        self.reached_target = target_mat

        if reach_sum == self.nagent:
            self.episode_over = True
            self.stat['success'] = 1
        else:
            self.episode_over = False 
            self.stat['success'] = 0
        n = self.nagent
        #get reward
        reward = np.full(self.nagent, self.TIMESTEP_PENALTY)
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
