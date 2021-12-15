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
        self.NOCOOP_COLLECT_REWARD = 0.3
        self.COOP_COLLECT_REWARD = 0.5
        self.COOP_STAY_REWARD = 0.05
        self.NOCOOP_MULTICOLLECT_REWARD = 0.1
        self.SUCCESS_REWARD = 0.5
        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative search task')
        env.add_argument('--coop_targets', type=int, default=2,
                         help="targets need to be collect cooperatively")
        env.add_argument('--nocoop_targets', type=int, default=4,
                         help="targets need to be collect separately")
        env.add_argument('--lock_agent', default=False, action='store_true', 
                    help='lock agent after reaching a coop_target') 
        env.add_argument('--spawn_type',type=str, default="A", choices=["A", "B","C"], help="how to spawn agents and targets; A: both random; B: agents spawn together; C:agents spawn in corners")
        #env.add_argument('--areas',type=int, default=4, choices=[4, 8, 16], help='how many areas in this env') #use 16 areas
        #env.add_argument("--map_type", type=str, default="empty", choices=["cross", "parallel"], help="map type") #abandoned for now
        return
                    
    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        #init mask_mat: used to mask unseen objects
        #self.mask_mat = np.ones((8,8,3,3,4))
        self.ref_act = np.array([[-1,0],[0,1],[1,0],[0,-1],[0,0]])
        #0:left. 1:up. 2: right. 3:down. 4:stop
        self.nagents = args.nagents 
        self.coop_targets = args.coop_targets
        self.noncoop_targets = args.nocoop_targets
        self.spawn_type = args.spawn_type
        self.naction = 5
        #self.agent_spawn_area = np.array([16,18,18,23,24,25,30,31,32])
        #self.target_spawn_area = np.setdiff1d(np.arange(49),self.agent_spawn_area,assume_unique = True)
        coord = np.arange(8)
        xv, yv = np.meshgrid(coord,coord)
        self.ref_loc = np.array(list(zip(xv.flat, yv.flat)))
        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space: 43=3*3*3+8+8
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,43), dtype=int)
        return

    def reset(self):
        #init episode record msgs
        self.episode_over = False
        #Spawn agents and targets
        coord = np.arange(4)
        xv, yv = np.meshgrid(coord,coord)
        areas_ref = list(zip(xv.flat, yv.flat))
        if self.spawn_type == "A":
            spawn_locs = np.random.choice(np.arange(64),size=self.nagents+self.coop_targets+self.noncoop_targets,replace=False)
            self.agent_loc_raw = spawn_locs[0:self.nagents]
            self.coop_target_loc_raw = spawn_locs[self.nagents:self.nagents+self.coop_targets]
            self.noncoop_target_loc_raw = spawn_locs[self.nagents+self.coop_targets:self.nagents+self.coop_targets+self.noncoop_targets]
        elif self.spawn_type == "B":
            area_idx = np.random.choice(16, size=1, replace=True)
            area_x, area_y = areas_ref[area_idx]
            agent_spawn_area = np.array([area_x*2+area_y*8,area_x*2+1+area_y*8,area_x*2+(area_y+1)*8,area_x*2+1+(area_y+1)*8])
            target_spawn_hyper_area = np.setdiff1d(np.arange(16), area_idx,assume_unique = True)
            self.agent_loc_raw = np.random.choice(agent_spawn_area,size=self.nagents, replace = True)
            target_spawn_area = np.random.choice(target_spawn_hyper_area, size=self.coop_targets+self.noncoop_targets, replace=False)
            target_loc_set = []
            for locs in target_spawn_area:
                area_x, area_y = areas_ref[locs]
                spawn_area = np.array([area_x*2+area_y*8,area_x*2+1+area_y*8,area_x*2+(area_y+1)*8,area_x*2+1+(area_y+1)*8])
                loc_idx = np.random.choice(spawn_area, size=1)
                target_loc_set.append(loc_idx)
            target_loc_array = np.array(target_loc_set)
            self.coop_target_loc_raw = target_loc_array[0:self.coop_targets]
            self.noncoop_target_loc_raw = target_loc_array[self.coop_targets:self.coop_targets+self.noncoop_targets]
        elif self.spawn_type == "C":
            agent_spawn_area = []
            for area_idx in [0,3,12,15]:
                area_x, area_y = areas_ref[area_idx]
                mini_agent_spawn_area = [area_x*2+area_y*8,area_x*2+1+area_y*8,area_x*2+(area_y+1)*8,area_x*2+1+(area_y+1)*8]
                agent_spawn_area += mini_agent_spawn_area
            
            self.agent_loc_raw = np.random.choice(np.array(agent_spawn_area),size=self.nagents, replace = True)
            target_spawn_hyper_area = np.setdiff1d(np.arange(16), np.array([0,3,12,15]), assume_unique = True)
            target_spawn_area = np.random.choice(target_spawn_hyper_area, size=self.coop_targets+self.noncoop_targets, replace=False)
            target_loc_set = []
            for locs in target_spawn_area:
                area_x, area_y = areas_ref[locs]
                spawn_area = np.array([area_x*2+area_y*8,area_x*2+1+area_y*8,area_x*2+(area_y+1)*8,area_x*2+1+(area_y+1)*8])
                loc_idx = np.random.choice(spawn_area, size=1)
                target_loc_set.append(loc_idx)
            target_loc_array = np.array(target_loc_set)
            self.coop_target_loc_raw = target_loc_array[0:self.coop_targets]
            self.noncoop_target_loc_raw = target_loc_array[self.coop_targets:self.coop_targets+self.noncoop_targets]

        self.target_remain = self.coop_targets + self.noncoop_targets
        self.agent_loc = self.ref_loc[self.agent_loc_raw].squeeze() #a list of length2 array
        self.coop_target_loc = self.ref_loc[self.coop_target_loc_raw].squeeze() 
        self.noncoop_target_loc = self.ref_loc[self.noncoop_target_loc_raw].squeeze() 

        self.raw_env_info_mat = np.zeros((10,10,3))#0: coop_targets; 1: noncoop_targets; 2: agents. use padding
        #init infomat with target_loc
        for idx in range(self.coop_targets):
            self.raw_env_info_mat[self.coop_target_loc[idx][0]+1,self.coop_target_loc[idx][1]+1,0] = 1
        for idx in range(self.noncoop_targets):
            self.raw_env_info_mat[self.noncoop_target_loc[idx][0]+1,self.noncoop_target_loc[idx][1]+1,1] = 1
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
            if env_info_mat[x+1,y+1,2] == 0:
                env_info_mat[x+1,y+1,2] = 1
            else:
                collision_loc.append([x,y])
        return env_info_mat, collision_loc

    def check_collection(self, env_info_mat, collision_loc):
        #check whether collection
        #if so, set rewards and remove targets
        collect_rewards = np.zeros(self.nagents)
        collected_noncoop_target = []
        collected_coop_target = []
        for idx in range(self.nagents):
            x, y = self.agent_loc[idx]
            if env_info_mat[x+1,y+1,1] == 1: #agent collects noncoop targets
                if [x,y] in collision_loc: #multiple agents collecting
                    collect_rewards[idx] = self.NOCOOP_MULTICOLLECT_REWARD
                else:
                    collect_rewards[idx] = self.NOCOOP_COLLECT_REWARD
                if [x,y] not in collected_noncoop_target:
                    collected_noncoop_target.append([x,y])

            if env_info_mat[x+1,y+1,0] == 1: #agent collects coop targets
                if [x,y] in collision_loc: #multiple agents collect successfully
                    collect_rewards[idx] = self.COOP_COLLECT_REWARD
                    if [x,y] not in collected_coop_target:
                        collected_coop_target.append([x,y]) 
                else:
                    collect_rewards[idx] = self.COOP_STAY_REWARD
            for [x,y] in collected_noncoop_target:
                env_info_mat[x+1,y+1,1] = 0 #remove targets
                self.raw_env_info_mat[x+1,y+1,1] = 0 #remove targets
                self.target_remain -= 1     
            for [x,y] in collected_coop_target:            
                env_info_mat[x+1,y+1,0] = 0 #remove targets
                self.raw_env_info_mat[x+1,y+1,0] = 0 #remove targets
                self.target_remain -= 1                 
         
        return collect_rewards, env_info_mat, collision_loc

    def _get_obs(self, env_info_mat, collision_loc):
        #generate selfloc observation
        agent_loc_onehot = [one_hot_encoding(8,agent_locs) for agent_locs in self.agent_loc]
        agent_obs_selfloc = np.vstack(agent_loc_onehot)
        #generate env observation
        agents_obs_set = []
        #3*3*3, 0-coop-targets; 1-noncoop-targets; 2-agents
        for idx in range(self.nagents):
            x, y = self.agent_loc[idx]
            agent_obs_mat = np.copy(env_info_mat[x:x+3,y:y+3,:])
            assert agent_obs_mat[1,1,2] == 1, "Error in obs generation"
            #check self observation
            if [x,y] in collision_loc:
                agent_obs_mat[1,1,2] = 1
            else:
                agent_obs_mat[1,1,2] = 0
            agents_obs_set.append(agent_obs_mat.flatten())
        agent_obs_others = np.vstack(agents_obs_set)
        agent_obs_final = np.concatenate((agent_obs_selfloc,agent_obs_others),axis=1)
        return agent_obs_final.copy()

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        for idx in range(self.nagents):
            act = action[idx]
            agent_act = np.copy(self.ref_act[act])
            self.agent_loc[idx] = self.agent_loc[idx] + agent_act
        self.agent_loc = np.clip(self.agent_loc, 0, 7)
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
