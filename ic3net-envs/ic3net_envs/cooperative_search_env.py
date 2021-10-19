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
        self.TARGET_REWARD = 0
        self.POS_TARGET_REWARD = 0.05
        self.COLLISION_PENALTY = -0.03
        self.REACH_DISTANCE = 0.05
        self.episode_over = False
        


    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')

        env.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium"], 
                    help="The difficulty of the environment")

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        if args.difficulty == "easy":
            self.vision = 0.15
            self.speed = 0.075 #max speed
        elif args.difficulty == "medium":
            self.vision = 0.1
            self.speed = 0.05

        self.ntarget = args.nfriendly
        self.nagent = args.nfriendly
        self.naction = 9

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design
        '''
        assume there are n agents
        0~n-1: identification
        n~n+1: self-location
        n+2~3n-1: other agents' location
        3n~5n-1: other targets' locations
        5n~6n-1: other targets' visibilities
        6n-7n-2: other agents' visibilities
        '''

        self.obs_dim = 7*self.ntarget-1
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
        self.target_loc = np.random.rand(2,self.ntarget)
        self.agent_loc = np.random.rand(2,self.nagent)
        #Check if agents are spawned near its target
        reachs, idxs = self.check_arrival()
        if reachs>0:
            self.agent_loc[:, idxs] = np.random.rand(2,reachs)
        # stat - like success ratio
        self.stat = dict()

        # Observation will be nagent * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def check_arrival():
        at_distances = self.target_loc - self.agent_loc
        self.distances = np.linalg.norm(at_distances,axis=0)
        target_mat = self.distances<self.REACH_DISTANCE
        reach_sum = np.sum(target_mat)
        return reach_sum, target_mat

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
        action = np.atleast_1d(action)

        for i, a in enumerate(action):
            self._take_action(i, a)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."


        self.episode_over = False
        self.obs = self._get_obs()

        debug = {'agent_locs':self.agent_loc,'target_locs':self.target_loc}
        return self.obs, self._get_reward(), self.episode_over, debug



    def seed(self):
        return

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.agent_loc[:,0], self.agent_loc[:,1]] = self.agent_ids
        # self.grid[self.target_loc[:,0], self.target_loc[:,1]] = self.target_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _get_obs(self):

        return

    def _take_action(self, idx, act):
        # target action
        if idx >= self.nagent:
            # fixed target
            if not self.moving_target:
                return
            else:
                raise NotImplementedError

        if self.reached_target[idx] == 1:
            return

        # STAY action
        if act==5:
            return

        # UP
        if act==0 and self.grid[max(0,
                                self.agent_loc[idx][0] + self.vision - 1),
                                self.agent_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.agent_loc[idx][0] = max(0, self.agent_loc[idx][0]-1)

        # RIGHT
        elif act==1 and self.grid[self.agent_loc[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.agent_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.agent_loc[idx][1] = min(self.dims[1]-1,
                                            self.agent_loc[idx][1]+1)

        # DOWN
        elif act==2 and self.grid[min(self.dims[0]-1,
                                    self.agent_loc[idx][0] + self.vision + 1),
                                    self.agent_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.agent_loc[idx][0] = min(self.dims[0]-1,
                                            self.agent_loc[idx][0]+1)

        # LEFT
        elif act==3 and self.grid[self.agent_loc[idx][0] + self.vision,
                                    max(0,
                                    self.agent_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.agent_loc[idx][1] = max(0, self.agent_loc[idx][1]-1)

    def _get_reward(self):
        n = self.nagent 
        reward = np.full(n, self.TIMESTEP_PENALTY)

        #occuping reward
        for agent_idx in range(n):
            if self.reached_target[agent_idx] == 1:
                reward[agent_idx] = self.TARGET_REWARD

        for target_idx in range(n):
            if self.target_occupied[target_idx] == 1:
                continue
            else:
                for agent_idx in range(n):
                    if np.all(self.agent_loc[agent_idx] == self.target_loc[target_idx]):
                        reward[agent_idx] = self.TARGET_REWARD
                        if self.target_occupied[target_idx] == 0:
                            #the agent successfully occupy the target
                            self.reached_target[agent_idx] = 1
                            self.target_occupied[target_idx] = 1
                        #we encourage agents to occupy targets at the same time. However, after that,
                        
        
        #collision penalty:
        for agent_x in range(n):
            for agent_y in range(agent_x+1, n):
                if np.all(self.agent_loc[agent_y] == self.agent_loc[agent_x]):
                    #collision!
                    reward[agent_x] -= self.COLLISION_PENALTY
                    reward[agent_y] -= self.COLLISION_PENALTY

        if np.all(self.reached_target == 1):
            self.episode_over = True

        # Success ratio
        if np.all(self.reached_target == 1):
            self.stat['success'] = 1
        else:
            self.stat['success'] = 0

        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())


    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.agent_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for p in self.target_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()
