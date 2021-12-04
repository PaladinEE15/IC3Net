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
    mat = np.zeros(dim*2,dtype=int)
    mat[coord[0]] = 1
    mat[coord[1]+dim] = 1
    return mat

class CooperativeSearchEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.TIMESTEP_PENALTY = -0.05
        self.COLLECT_REWARD = 0.5
        self.COOP_COLLECT_REWARD = 0.2
        self.SUCCESS_REWARD = 0.5
        self.episode_over = False

    def init_args(self, parser):
        return
                    
    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        #init mask_mat: used to mask unseen objects
        self.mask_mat = np.ones((7,7,3,3,3))
        for x in range(7):
            for y in range(7):
                mini_mat = np.ones((3,3,3))
                for minix in range(3):
                    for miniy in range(3):
                        if (x+minix-1>6)|(x+minix-1<0)|(y+miniy-1>6)|(x+miniy-1<0):
                            mini_mat[minix,miniy,:] = 0
                if (x==1)|(x==4):
                    if y == 2:
                        mini_mat[2,1:,:] = 0
                    elif y == 3:
                        mini_mat[2,:,:] = 0
                    elif y == 4:
                        mini_mat[2,0:2,:] = 0
                elif (x==2)|(x==5):
                    if y == 2:
                        mini_mat[0,1:,:] = 0
                    elif y == 3:
                        mini_mat[0,:,:] = 0
                    elif y == 4:
                        mini_mat[0,0:2,:] = 0   
                self.mask_mat[x,y,:,:,:] = np.copy(mini_mat)
            
        self.ref_act = np.array([[-1,0],[0,1],[1,0],[0,-1],[0,0]])
        #0:left. 1:down. 2: right. 3:up. 4:stop
        self.nagents = args.nagents 
        self.ntargets = args.ntargets
        self.naction = 5
        self.agent_spawn_area = np.array([16,17,18,23,24,25,30,31,32])
        self.target_spawn_area = np.setdiff1d(np.arange(49),self.agent_spawn_area,assume_unique = True)
        coord = np.arange(7)
        xv, yv = np.meshgrid(coord,coord)
        self.ref_loc = np.array(list(zip(xv.flat, yv.flat)))
        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space: 41=3*3*3+7+7
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.nagents,41), dtype=int)
        return

    def reset(self):
        #init episode record msgs
        self.episode_over = False
        #Spawn agents and targets
        #use a grid-like generation
        self.agent_loc_raw = np.random.choice(self.agent_spawn_area,size=self.nagents,replace=False)
        self.target_loc_raw = np.random.choice(self.target_spawn_area,size=self.ntargets,replace=False)
        self.target_remain = self.ntargets
        self.agent_loc = self.ref_loc[self.agent_loc_raw] #a list of length2 array
        self.target_loc = self.ref_loc[self.target_loc_raw] 
        self.raw_env_info_mat = np.zeros((9,9,2))#0: targets; 1: agents. use padding
        #init infomat with target_loc
        for idx in range(self.ntargets):
            self.raw_env_info_mat[self.target_loc[idx][0]+1,self.target_loc[idx][1]+1,0] = 1
        self.stat = dict()
        # Observation will be nagent * vision * vision ndarray
        env_info_mat, collision_loc = self.update_env_info()
        self.obs = self._get_obs(env_info_mat, collision_loc)
        return self.obs

    def update_env_info(self):
        env_info_mat = np.copy(self.raw_env_info_mat)
        collision_loc = []
        for idx in range(self.nagents):
            x, y = self.agent_loc[idx]
            #check collision
            if env_info_mat[x+1,y+1,1] == 0:
                env_info_mat[x+1,y+1,1] = 1
            else:
                collision_loc.append([x,y])
        return env_info_mat, collision_loc

    def check_collection(self, env_info_mat, collision_loc):
        #check whether collection
        #if so, set rewards and remove targets
        collect_rewards = np.zeros(self.nagents)
        for idx, agent_coord in self.agent_loc:
            x, y = agent_coord
            if env_info_mat[x+1,y+1,0] == 1: #agent collects targets
                if agent_coord in collision_loc: #multiple agents collecting
                    collect_rewards[idx] = self.COOP_COLLECT_REWARD
                else:
                    collect_rewards[idx] = self.COLLECT_REWARD
                env_info_mat[x+1,y+1,0] = 0 #remove targets
                self.raw_env_info_mat[x+1,y+1,0] = 0 #remove targets
                self.target_remain -= 1
        return collect_rewards, env_info_mat, collision_loc

    def _get_obs(self, env_info_mat, collision_loc):
        #generate selfloc observation
        agent_loc_onehot = [one_hot_encoding(7,agent_locs) for agent_locs in self.agent_loc]
        agent_obs_selfloc = np.vstack(agent_loc_onehot)
        #generate env observation
        agents_obs_set = []
        #3*3*3, 0-can observe; 1-targets; 2-agents
        for idx in range(self.nagents):
            x, y = self.agent_loc[idx]
            mask = self.mask_mat[x,y,:,:,:].squeeze()
            agent_invision_mat = np.ones((3,3,1))
            agent_obs_mat_raw = np.copy(env_info_mat[x:x+2,y:y+2,:])
            agent_obs_mat_mid = np.concatenate((agent_invision_mat,agent_obs_mat_raw),axis=2)
            agent_obs_mat_final = agent_obs_mat_mid*mask
            assert agent_obs_mat_final[1,1,2] == 1, "Error in obs generation"
            #check self observation
            if self.agent_loc[idx] in collision_loc:
                agent_obs_mat_final[1,1,2] = 1
            else:
                agent_obs_mat_final[1,1,2] = 0
            agents_obs_set.append(agent_obs_mat_final.flatten())
        agent_obs_others = np.vstack(agents_obs_set)
        agent_obs_final = np.concatenate((agent_obs_selfloc,agent_obs_others),dim=1)
        return agent_obs_final.copy()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        for idx, act in enumerate(action):
            x,y = self.agent_loc[idx]
            agent_act = np.copy(self.ref_act[act])
            if y <= 4 & y >= 2:
                if x == 1 | x == 4:
                    if act == 2: #cannot go right
                        agent_act = np.copy(self.ref_act[4])
                elif x == 2 | x == 5:
                    if act == 0: #cannot go left
                        agent_act = np.copy(self.ref_act[4]) 
            
            self.agent_loc[idx] = self.agent_loc[idx] + agent_act
        self.agent_loc = np.clip(self.agent_loc, 0, 6)
                           
        env_info_mat, collision_loc = self.update_env_info()
        collect_rewards, env_info_mat, collision_loc = self.check_collection(env_info_mat, collision_loc)
        self.obs = self._get_obs(env_info_mat, collision_loc)
        #check dones
        if self.target_remain == 0:
            self.episode_over = True
            self.stat['success'] = 1
        else:
            self.episode_over = False 
            self.stat['success'] = 0

        #get reward
        reward = np.full(self.nagents, self.TIMESTEP_PENALTY)
        reward += collect_rewards
        debug = {'target_remain':self.target_remain}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return

    def reward_terminal(self):
        return self.SUCCESS_REWARD*np.ones(self.nagents)
