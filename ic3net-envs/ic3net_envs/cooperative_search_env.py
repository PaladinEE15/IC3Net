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
        env = parser.add_argument_group('Cooperative search task')
        env.add_argument('--ntargets', type=int, default=4,
                         help="targets need to be collect ")
        return
                    
    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        #init mask_mat: used to mask unseen objects
        self.mask_mat = np.ones((7,7,3,3,5))
        for x in range(7):
            for y in range(7):
                mini_mat = np.ones((3,3,5))
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
        self.halfagents = int(self.nagents/2)
        self.ntargets = args.ntargets
        self.halftargets = int(self.ntargets/2)
        self.naction = 5
        #self.agent_spawn_area = np.array([16,17,18,23,24,25,30,31,32])
        #self.target_spawn_area = np.setdiff1d(np.arange(49),self.agent_spawn_area,assume_unique = True)
        coord = np.arange(7)
        xv, yv = np.meshgrid(coord,coord)
        self.ref_loc = np.array(list(zip(xv.flat, yv.flat)))
        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space: 43=2+3*3*5+7+7
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,61), dtype=int)
        return

    def reset(self):
        #init episode record msgs
        self.episode_over = False
        #Spawn agents and targets
        #use a grid-like generation
        #self.agent_loc_raw = np.random.choice(self.agent_spawn_area,size=self.nagents,replace=False)
        #self.target_loc_raw = np.random.choice(self.target_spawn_area,size=self.ntargets,replace=False)

        spawn_locs = np.random.choice(np.arange(49),size=self.nagents+self.ntargets,replace=False)
        self.agent_loc_raw_a = spawn_locs[0:self.halfagents]
        self.agent_loc_raw_b = spawn_locs[self.halfagents:self.nagents]
        self.target_loc_raw_a = spawn_locs[self.ntargets:self.nagents+self.halftargets]
        self.target_loc_raw_b = spawn_locs[self.ntargets+self.halftargets:]
        self.target_remain = self.ntargets
        self.agent_loc_a = self.ref_loc[self.agent_loc_raw_a] #a list of length2 array
        self.agent_loc_b = self.ref_loc[self.agent_loc_raw_b]
        self.target_loc_a = self.ref_loc[self.target_loc_raw_a] 
        self.target_loc_b = self.ref_loc[self.target_loc_raw_b]
        self.raw_env_info_mat = np.zeros((9,9,4))#0: targets_a; 1: targets_b; 2: agents_a; 3: agents_b. use padding
        #init infomat with target_loc
        for idx in range(self.halftargets):
            self.raw_env_info_mat[self.target_loc_a[idx][0]+1,self.target_loc_a[idx][1]+1,0] = 1
            self.raw_env_info_mat[self.target_loc_b[idx][0]+1,self.target_loc_b[idx][1]+1,1] = 1
        self.stat = dict()
        # Observation will be nagent * vision * vision ndarray
        self.update_env_info()
        self.obs = self._get_obs()
        return self.obs

    def update_env_info(self):
        self.env_info_mat = np.copy(self.raw_env_info_mat)
        self.collision_loc_a = []
        self.collision_loc_b = []
        for idx in range(self.halfagents):
            x_a, y_a = self.agent_loc_a[idx]
            #check collision
            if self.env_info_mat[x_a+1,y_a+1,2] == 0:
                self.env_info_mat[x_a+1,y_a+1,2] = 1
            else:
                self.collision_loc_a.append([x_a,y_a])
            x_b, y_b = self.agent_loc_b[idx]
            #check collision
            if self.env_info_mat[x_b+1,y_b+1,3] == 0:
                self.env_info_mat[x_b+1,y_b+1,3] = 1
            else:
                self.collision_loc_b.append([x_b,y_b])
        return 

    def check_collection(self):
        #check whether collection
        #if so, set rewards and remove targets
        collect_rewards = np.zeros(self.nagents)
        for idx in range(self.halfagents):
            x_a, y_a = self.agent_loc_a[idx]
            if self.env_info_mat[x_a+1,y_a+1,0] == 1: #agent collects targets
                if [x_a,y_a] in self.collision_loc_a: #multiple agents collecting
                    collect_rewards[idx] = self.COOP_COLLECT_REWARD
                else:
                    collect_rewards[idx] = self.COLLECT_REWARD
                self.env_info_mat[x_a+1,x_a+1,0] = 0 #remove targets
                self.raw_env_info_mat[x_a+1,x_a+1,0] = 0 #remove targets
                self.target_remain -= 1
            x_b, y_b = self.agent_loc_b[idx]
            if self.env_info_mat[x_b+1,y_b+1,1] == 1: #agent collects targets
                if [x_b,y_b] in self.collision_loc_b: #multiple agents collecting
                    collect_rewards[idx+self.halfagents] = self.COOP_COLLECT_REWARD
                else:
                    collect_rewards[idx+self.halfagents] = self.COLLECT_REWARD
                self.env_info_mat[x_b+1,x_b+1,1] = 0 #remove targets
                self.raw_env_info_mat[x_b+1,x_b+1,1] = 0 #remove targets
                self.target_remain -= 1            
        return collect_rewards

    def _get_obs(self):
        #self identification
        agent_self_id = np.zeros((self.nagents,2))
        agent_self_id[0:self.halfagents,0] = 1
        agent_self_id[self.halfagents:,1] = 1
        #generate selfloc observation
        self.agent_loc = np.concatenate((self.agent_loc_a,self.agent_loc_b),axis=0)
        agent_loc_onehot = [one_hot_encoding(7,agent_locs) for agent_locs in self.agent_loc]
        agent_obs_selfloc = np.vstack(agent_loc_onehot)
        #generate env observation
        agents_obs_set_a = []
        agents_obs_set_b = []
        #3*3*5, 0-can observe; 1-targets_a; 2-targets_b; 3-agents_a; 4-agents_b
        for idx in range(self.halfagents):
            x_a, y_a = self.agent_loc_a[idx]
            mask_a = self.mask_mat[x_a,y_a,:,:,:].squeeze()
            agent_invision_mat_a = np.ones((3,3,1))
            agent_obs_mat_raw_a = np.copy(self.env_info_mat[x_a:x_a+3,y_a:y_a+3,:])
            agent_obs_mat_mid_a = np.concatenate((agent_invision_mat_a,agent_obs_mat_raw_a),axis=2)
            agent_obs_mat_final_a = agent_obs_mat_mid_a*mask_a
            assert agent_obs_mat_final_a[1,1,3] == 1, "Error in obs generation"
            #check self observation
            if [x_a,y_a] in self.collision_loc_a:
                agent_obs_mat_final_a[1,1,3] = 1
            else:
                agent_obs_mat_final_a[1,1,3] = 0
            agents_obs_set_a.append(agent_obs_mat_final_a.flatten())

            x_b, y_b = self.agent_loc_b[idx]
            mask_b = self.mask_mat[x_b,y_b,:,:,:].squeeze()
            agent_invision_mat_b = np.ones((3,3,1))
            agent_obs_mat_raw_b = np.copy(self.env_info_mat[x_b:x_b+3,y_b:y_b+3,:])
            agent_obs_mat_mid_b = np.concatenate((agent_invision_mat_b,agent_obs_mat_raw_b),axis=2)
            agent_obs_mat_final_b = agent_obs_mat_mid_b*mask_b
            assert agent_obs_mat_final_b[1,1,4] == 1, "Error in obs generation"
            #check self observation
            if [x_b,y_b] in self.collision_loc_b:
                agent_obs_mat_final_b[1,1,4] = 1
            else:
                agent_obs_mat_final_b[1,1,4] = 0
            agents_obs_set_b.append(agent_obs_mat_final_b.flatten())
        agents_obs_set = agents_obs_set_a + agents_obs_set_b
        agent_obs_others = np.vstack(agents_obs_set)
        agent_obs_final = np.concatenate((agent_self_id,agent_obs_selfloc,agent_obs_others),axis=1)
        return agent_obs_final.copy()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        self.agent_loc = np.concatenate((self.agent_loc_a,self.agent_loc_b),axis=0)
        for idx in range(self.nagents):
            act = action[idx]
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
        self.agent_loc_a = self.agent_loc[0:self.halfagents,:]
        self.agent_loc_b = self.agent_loc[self.halfagents:,:]                   
        self.update_env_info()
        collect_rewards = self.check_collection()
        self.obs = self._get_obs()
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
