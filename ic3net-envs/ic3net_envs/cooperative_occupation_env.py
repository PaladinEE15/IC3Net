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


class CooperativeOccupationEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.COLLECT_REWARD = 1
        self.TIME_PENALTY = -0.1
        self.POISON_PENALTY = -5
        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Occupation task')
        #env.add_argument("--reward_type", type=int, default=0, 
                    #help="0-full cooperative with occupy;1-close reward")   
        #env.add_argument("--setting", type=int, default=0, 
                    #help="0-easy;1-hard")       
        #env.add_argument('--small_obs', action='store_true', default=False,
                    #help='small obstacle')
        #there's another kind of observation: rangefinder. However the detection is complex......
        #firstly, calculate the sector according to angle
        #secondly, update corresponding sensor data (note that covering......)
        return
    

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        #assume the map is 1 x 1
        self.speed = 0.15
        self.nagent = args.nagents
        self.naction = 9
        self.ref_act = self.speed*np.array([[-0.71,0.71],[0,1],[0.71,0.71],[-1,0],[0,0],[1,0],[-0.71,-0.71],[0,-1],[0.71,-0.71]])
        self.action_space = spaces.MultiDiscrete([self.naction])
        if self.nagent == 2:
            self.info_mask = np.zeros((2,8))
            self.info_mask[0,4:] = 1
            self.info_mask[1,:4] = 1
            self.obs_mask = np.zeros((2,16))
            self.obs_mask[0,:8] = 1
            self.obs_mask[1,8:] = 1
            self.world_allocate = np.array([0,0,0,0,1,1,1,1])
        elif self.nagent == 4:
            self.info_mask = np.zeros((4,8))
            self.info_mask[0,0:2] = 1
            self.info_mask[1,2:4] = 1
            self.info_mask[2,4:6] = 1
            self.info_mask[3,6:8] = 1


        self.obs_dim = 26

        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,self.obs_dim), dtype=int)
        return

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        '''
        use grid generation to avoid ridiculous things
        1. generate the obstacle. it must be located in the center of the map. 0.3-0.7
        2. generate the targets. they cannot be located near the obstacle or near each other.
        use pre-set locations. 0.1-0.3-0.5-0.7-0.9? there are 25 possible locations......if one is close to the obstacle, just choose the next one.
        3. generate agents locations. if one agent is close to the obstacle or other agents, relocate one. do this in loop
        '''
        self.wrong_collection = 0
        self.episode_over = False
        self.stat = {}
        self.stat['success'] = 0
        if self.nagent == 2:
            #generate valid mat
            self.valid_mat = np.ones(8)
            inval_1 =  np.random.choice(4,2,replace=False)
            inval_2 =  4 + np.random.choice(4,2,replace=False)
            self.valid_mat[inval_1] = -1
            self.valid_mat[inval_2] = -1
            #spawn agents and targets
            self.agent_locs = np.random.rand(2,2)
            self.target_locs = np.random.rand(8,2)
            
        elif self.nagent == 4:
            #generate valid mat
            self.valid_mat = np.ones(8)
            inval_1 =  np.random.choice(2,1)
            inval_2 =  2 + np.random.choice(2,1)
            inval_3 =  4 + np.random.choice(2,1)
            inval_4 =  6 + np.random.choice(2,1)
            self.valid_mat[inval_1] = -1
            self.valid_mat[inval_2] = -1
            self.valid_mat[inval_3] = -1
            self.valid_mat[inval_4] = -1
            #spawn agents and targets
            self.agent_locs = np.random.rand(4,2)
            self.target_locs = np.random.rand(8,2)
            #allocate world to target
            self.world_allocate = np.random.permutation(np.array([0,0,1,1,2,2,3,3]))
            #generate obs_mask
            self.obs_mask = np.zeros((4,16))
            allocate_repeat = np.repeat(self.world_allocate,2)
            for idx in range(4):
                self.obs_mask[idx,allocate_repeat==idx] = 1        
        self.collection = np.zeros(8)
        self.info_mat = self.info_mask*np.tile(self.valid_mat.reshape((1,-1)),(self.nagent,1))


        self.obs = self._get_obs()
        return self.obs

    def _get_obs(self):

        #observation
        #infomat:8
        #self location:2
        #target location: 2*8
        
        self.target_obs = self.obs_mask*np.tile(self.target_locs.reshape((1,-1)),(self.nagent,1))
        self.target_obs[self.target_obs == 0] = -1
        new_obs = np.concatenate((self.info_mat,self.agent_locs,self.target_obs),axis=1)
        
        return new_obs


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

        #move agents according to actions
        action = np.array(action).squeeze()
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        
        trans_action = [self.ref_act[idx,:] for idx in action]
        self.agent_locs += trans_action
        self.agent_locs[self.agent_locs < 0] = -self.agent_locs[self.agent_locs < 0]
        self.agent_locs[self.agent_locs > 1] = 2 - self.agent_locs[self.agent_locs > 1]

        reward = self.TIME_PENALTY*np.ones(self.nagent)
        #check collection
        full_agentloc = np.tile(self.agent_locs, (1,8))
        relative_loc = (self.target_obs - full_agentloc).reshape((-1,2))
        collection_mat = (np.linalg.norm(relative_loc,axis=1)<=0.1).reshape((self.nagent,8))
        #get reward 
        valid_set = np.tile(self.valid_mat.reshape((1,-1)),(self.nagent,1))
        valid_set[valid_set<0] = -5
        reward_set = valid_set*collection_mat
        self.wrong_collection += np.sum(reward_set<0)
        collect_reward = np.sum(reward_set,axis=1)
        reward += collect_reward
        #record collection and change target locs
        collect_target_idx = np.sum(collection_mat,axis=0)>0
        self.collection[collect_target_idx] = 1
        self.target_locs[collect_target_idx,:] = -1
        if np.sum(self.collection[self.valid_mat>0]) == 4:
            self.episode_over = True
            if self.wrong_collection == 0:
                self.stat['success'] = 1

        self.obs = self._get_obs()

        debug = {}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return
