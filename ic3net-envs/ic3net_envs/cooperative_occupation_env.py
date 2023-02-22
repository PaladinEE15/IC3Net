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
        self.FULL_OCCUPATION_REWARD = 1
        self.TIME_PENALTY = -0.1
        self.SINGLE_OCCUPATION_REWARD = 0.1
        self.COLLISION_PENALTY = -1
        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Occupation task')
        env.add_argument("--reward_type", type=int, default=0, 
                    help="0-full cooperative with occupy;1-close reward")        

        #there's another kind of observation: rangefinder. However the detection is complex......
        #firstly, calculate the sector according to angle
        #secondly, update corresponding sensor data (note that covering......)
        return
    

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        #assume the map is 1 x 1
        self.nagent = args.nagents
        self.reward_type = args.reward_type
        if self.nagent == 4:
            self.speed = 0.1 
            self.detection_range = 0.2
            self.occupation_range = 0.1
            self.inner_col_range = 0.15
            self.outer_col_range = 0.2
            self.border_col_range = 0.075
            
        elif self.nagent == 8:
            self.speed = 0.05
            self.border_col_range = 0.0375
            self.detection_range = 0.1
            self.occupation_range = 0.05
            self.inner_col_range = 0.075
            self.outer_col_range = 0.2

        self.nonobserve_mat = -1*np.ones((self.nagent,2*self.nagent))
        self.naction = 9
        self.ref_act = self.speed*np.array([[-0.71,0.71],[0,1],[0.71,0.71],[-1,0],[0,0],[1,0],[-0.71,-0.71],[0,-1],[0.71,-0.71]])
        self.action_space = spaces.MultiDiscrete([self.naction])
        single_coords = np.array([0.1,0.3,0.5,0.7,0.9])
        temp_coords = np.zeros((5,5,2))
        temp_coords[:,:,0] = single_coords[:,np.newaxis]
        temp_coords[:,:,1] = single_coords
        self.potential_targets = temp_coords.reshape((-1,2))
        self.raw_index = np.arange(25)
        self.obs_dim = 4*self.nagent+2

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
        self.episode_over = False
        self.stat = {}
        self.stat['success'] = 0
        self.stat['collisions'] = 0
        #generate obstacles
        self.obs_locs = 0.3+0.4*np.random.rand(2).reshape((1,2))
        #generate targets
        temp_target_idx = np.random.shuffle(self.raw_index)
        total_targets = self.potential_targets[temp_target_idx,:].squeeze()
        self.target_locs = total_targets[0:self.nagent,:]
        new_idx = self.nagent
        for check_idx in range(self.nagent):
            sign = 0
            while sign == 0:
                dist = np.linalg.norm(self.target_locs[check_idx,:]-self.obs_locs[0,:])
                if dist <= 0.3:
                    self.target_locs[check_idx,:] = total_targets[new_idx,:]
                    new_idx += 1
                else:
                    sign = 1
        #generate agents
        self.agent_locs = np.random.rand(self.nagent,2)
        for check_idx in range(self.nagent):
            sign = 0
            while sign == 0:
                dist = np.linalg.norm(self.agent_locs[check_idx,:]-self.obs_locs[0,:])
                if dist <= self.outer_col_range:
                    self.agent_locs[check_idx,:] = np.random.rand(2)
                else:
                    sign = 1

        self.obs, _ = self._get_obs()
        return self.obs

    def _get_obs(self):

        #observation space design
        #self loc:2
        #others loc(absolute):2*(self.nagent-1)
        #all target loc:2*self.nagent
        #obstacle location

        #check inter-agent observation status
        
        agent_fullloc_a = np.tile(self.agent_locs,(1,self.nagent))
        agent_fullloc_b = np.tile(self.agent_locs.reshape((1,-1)),(self.nagent,1))
        inagent_distance = np.linalg.norm((agent_fullloc_b-agent_fullloc_a).reshape((-1,2)),axis=1)
        visibility_mat = np.tile((inagent_distance<=self.detection_range).reshape((-1,1)),(1,2)).reshape((self.nagent,2*self.nagent))
        final_observe = visibility_mat*agent_fullloc_b + (1-visibility_mat)*self.nonobserve_mat

        other_locs = np.zeros((self.nagent,2*self.nagent-2))
        for idx in range(self.nagent):
            other_locs[idx,:2*idx] = final_observe[idx,:2*idx]
            other_locs[idx,2*idx:] = final_observe[idx,2*idx+2:]
        
        target_locs = np.tile(self.target_locs.reshape((1,-1)),(self.nagent,1))
        obs_locs = np.tile(self.obs_locs.reshape((1,-1)),(self.nagent,1))
        
        new_obs = np.concatenate((self.agent_locs,other_locs,target_locs,obs_locs),axis=1)
        
        return new_obs, inagent_distance


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
        #check collisions with obstacle
        for idx in range(self.nagent):
            relative_loc =  self.agent_locs[idx,:] - self.obs_locs[0,:]
            distance = np.linalg.norm(relative_loc)
            if distance < self.outer_col_range:
                reward[idx] += self.COLLISION_PENALTY
                self.stat['collisions'] += 1
                true_dis = 2*self.outer_col_range - distance
                true_relative_loc = relative_loc * true_dis / distance
                self.agent_locs[idx,:] = self.obs_locs[0,:] + true_relative_loc



        self.obs, inagent_distance = self._get_obs()

        #check inagent collision
        collision_mat = ((inagent_distance>0)*(inagent_distance<self.inner_col_range)).reshape((self.nagent,self.nagent))
        collision_peragent = np.sum(collision_mat,axis=1)
        reward[collision_peragent>0] += self.COLLISION_PENALTY
        self.stat['collisions'] += np.sum(collision_peragent>0)


        #check arrival
        agent_loc_full = np.tile(self.agent_locs,(1,self.nagent))
        target_loc_full = np.tile(self.target_locs.reshape((1,-1)),(self.nagent,1))
        at_distances = np.linalg.norm((target_loc_full - agent_loc_full).reshape((-1,2)),axis=1).reshape((self.nagent,self.nagent))
        occupy_mat = at_distances<self.occupation_range
        if self.reward_type == 0:
            #occupy reward
            occupy_peragent = np.sum(occupy_mat,axis=1)
            reward[occupy_peragent>0] += self.SINGLE_OCCUPATION_REWARD
        else:
            closest_dis = np.min(at_distances, axis=1)
            reward += 2*(0.5-closest_dis)*self.SINGLE_OCCUPATION_REWARD
        #check all collection
        occupy_pertarget =  np.sum(occupy_mat,axis=0)
        if np.min(occupy_pertarget) > 0: #full occupation
            reward += self.FULL_OCCUPATION_REWARD
            if self.stat['collisions'] == 0:
                self.stat['success'] = 1
            self.episode_over = True

        debug = {}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return
