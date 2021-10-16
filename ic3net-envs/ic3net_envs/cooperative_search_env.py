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
        self.OUTSIDE_CLASS = 1
        self.TARGET_CLASS = 2
        self.PREDATOR_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.target_REWARD = 0
        self.POS_target_REWARD = 0.05
        self.episode_over = False

    def init_curses(self):#useless function- we do not display for now
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)


    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')

        env.add_argument('--vision', type=int, default=1,
                         help="Vision of predator")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        params = ['dim', 'vision']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.ntarget = args.nfriendly
        self.npredator = args.nfriendly
        self.dims = dims = (self.dim, self.dim)
        self.stay = True

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.TARGET_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE

        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + target + grid + outside

        # Observation for each agent will be vision * vision ndarray
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1), dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return

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

        debug = {'predator_locs':self.predator_loc,'target_locs':self.target_loc}
        return self.obs, self._get_reward(), self.episode_over, debug

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.reached_target = np.zeros(self.npredator)

        self.target_occupied = np.zeros(self.ntarget)
        # Locations
        locs = self._get_cordinates()
        self.predator_loc, self.target_loc = locs[:self.npredator], locs[self.npredator:]

        self._set_grid()

        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def seed(self):
        return

    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims),(self.npredator + self.ntarget), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.target_loc[:,0], self.target_loc[:,1]] = self.target_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _get_obs(self):
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1

        for i, p in enumerate(self.target_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.TARGET_CLASS] += 1

        obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])

        obs = np.stack(obs)
        return obs

    def _take_action(self, idx, act):
        # target action
        if idx >= self.npredator:
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
                                self.predator_loc[idx][0] + self.vision - 1),
                                self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0]-1)

        # RIGHT
        elif act==1 and self.grid[self.predator_loc[idx][0] + self.vision,
                                min(self.dims[1] -1,
                                    self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1]-1,
                                            self.predator_loc[idx][1]+1)

        # DOWN
        elif act==2 and self.grid[min(self.dims[0]-1,
                                    self.predator_loc[idx][0] + self.vision + 1),
                                    self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0]-1,
                                            self.predator_loc[idx][0]+1)

        # LEFT
        elif act==3 and self.grid[self.predator_loc[idx][0] + self.vision,
                                    max(0,
                                    self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1]-1)

    def _get_reward(self):
        n = self.npredator 
        reward = np.full(n, self.TIMESTEP_PENALTY)

        for target_idx in range(n):
            if self.target_occupied[target_idx] == 1:
                continue
            else:
                for predator_idx in range(n):
                    if np.all(self.predator_loc[predator_idx] == self.target_loc[target_idx]):
                        self.reached_target[predator_idx] = 1
                        self.target_occupied[target_idx] = 1
                        reward[predator_idx] = self.target_REWARD
                    break


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

        for p in self.predator_loc:
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
