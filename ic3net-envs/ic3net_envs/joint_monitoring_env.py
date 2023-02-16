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


class JointMonitoringEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.FULL_MONITORING_REWARD = 1
        self.IN_MONITORING_REWARD = 0.2
        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')
        env.add_argument("--evader_speed", type=float, default=0.2, 
                    help="How many targets one agent must reach")
        env.add_argument("--monitor_angle", type=float, default=0.25, 
                    help="Monitor observation angle")
        env.add_argument("--observation_type", type=int, default=0, 
                    help="0-self abs coords + partial target observation;1-all others relative coords + partial target observation;2-self abs ccords+ full...")
        #there's another kind of observation: rangefinder. However the detection is complex......
        #firstly, calculate the sector according to angle
        #secondly, update corresponding sensor data (note that covering......)
        return
    

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        self.evader_speed = args.evader_speed
        self.monitor_angle = args.monitor_angle
        if args.nagents == 4:
            self.xlen = 2.414
            self.ylen = 2.414
            self.monitors = 4
            self.evaders = 5
            self.monitor_locs = np.array([[0.707,0.707],[0.707,1.707],[1.707,0.707],[1.707,1.707]])
        elif args.nagents == 6:
            self.xlen = 3.414
            self.ylen = 2.414
            self.monitors = 6
            self.evaders = 8        
            self.monitor_locs = np.array([[0.707,0.707],[0.707,1.707],[1.707,0.707],[1.707,1.707],[2.707,1.707],[2.707,0.707]])    
        else:
            return

        self.observation_type = args.observation_type
        self.ref_act = np.array([0.5*math.pi,-0.5*math.pi,1/6*math.pi,-1/6*math.pi])
        self.naction = 4

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design
        #self angle:1
        #self abs coords:2
        #others relative coords: 2*(n-1)
        #target observation: 2*evaders
        if (self.observation_type == 0) or (self.observation_type == 2):
            self.obs_dim = 3 + 2*self.evaders
        else:
            self.obs_dim = 1 + 2*self.monitors + 2*self.evaders
        
        #calculate monitors locs and relative locs first
        if (self.observation_type == 1) or (self.observation_type == 3):
            monitor_locs_d1expand = np.tile(self.monitor_locs,(1,self.monitors))
            monitor_locs_d0expand = np.tile(self.monitor_locs.reshape((1,-1)),(self.monitors,1))
            monitor_locs_xy = monitor_locs_d0expand - monitor_locs_d1expand
            monitor_locs_mid = monitor_locs_xy.reshape((-1,2))
            monitor_locs_d = np.linalg.norm(monitor_locs_mid,axis=1,keepdims=True).reshape((-1,1))
            monitor_locs_theta = np.arctan((monitor_locs_mid[:,1]/monitor_locs_mid[:,0])).reshape((-1,1))
            self.monitor_relative_locs = np.concatenate((monitor_locs_d,monitor_locs_theta),axis=1).reshape((self.monitors,-1))

        self.observation_space = spaces.Box(low=-4, high=4, shape=(1,self.obs_dim), dtype=int)
        return

    def calcu_evader2monitor(self):
        #calculate the relative locations of evaders to monitors
        fullmonitoring = False
        evader_locs_full = np.tile(self.evader_locs.reshape((1,-1)),(self.monitors,1))
        monitors_locs_full = np.tile(self.monitor_locs,(1,self.evaders))
        monitors_angles_full = np.tile(self.monitor_angles,(1,self.evaders))
        relative_locs_xy = evader_locs_full - monitors_locs_full
        relative_locs_mid = relative_locs_xy.reshape((-1,2))
        self.relative_locs_d = np.linalg.norm(relative_locs_mid,axis=1,keepdims=True).reshape((self.monitors,self.evaders))
        self.relative_locs_theta = np.arctan((relative_locs_mid[:,1]/relative_locs_mid[:,0])).reshape((self.monitors, self.evaders))
        inrange = self.relative_locs_d < 1
        inangle = (self.relative_locs_theta>monitors_angles_full)*(self.relative_locs_theta<(monitors_angles_full+np.pi*self.monitor_angle))+((2*np.pi+self.relative_locs_theta)>monitors_angles_full)*((2*np.pi+self.relative_locs_theta)<(monitors_angles_full+np.pi*self.monitor_angle)) 


        self.monitoring_mat = inrange*inangle
        #shape: (monitors,evaders)
        monitoring_perevader = np.sum(self.monitoring_mat,axis=0)
        if np.sum(monitoring_perevader==0)==0:
            fullmonitoring = True
        else:
            fullmonitoring = False
        return fullmonitoring

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        #spawn evaders randomly 
        self.evader_locs = np.random.rand(self.evaders,2)
        self.evader_locs[:,0] = self.evader_locs[:,0]*self.xlen
        self.evader_locs[:,1] = self.evader_locs[:,1]*self.ylen
        self.monitor_angles = 2*math.pi*np.random.rand(self.monitors,1) - math.pi

        self.episode_over = False
        self.evaders_displacement = np.zeros((self.evaders,2))
        self.stat = dict()
        self.stat['full_monitoring'] = 0

        # Observation will be nagent * vision * vision ndarray
        self.obs = self._get_obs()
        return self.obs

    def _get_obs(self):
        '''
        there are n agents
        0:1, self angle
        1:2*evaders+1, evaders loc
        2*evaders+1:2*evaders+3, if abs coords
        2*evaders+1:2*evaders+1+2*monitors, if relative locs
        '''
        if self.observation_type <= 1:
            temp_relative_locs_d = self.relative_locs_d 
            temp_relative_locs_theta = self.relative_locs_theta
            temp_relative_locs_d[self.monitoring_mat==False] = -1
            temp_relative_locs_theta[self.monitoring_mat==False] = 0
            evader_locs = np.concatenate((temp_relative_locs_d.reshape((-1,1)),temp_relative_locs_theta.reshape((-1,1))),axis=1).reshape((self.monitors,-1))
        else:
            evader_locs = np.concatenate((self.relative_locs_d.reshape((-1,1)),self.relative_locs_theta.reshape((-1,1))),axis=1).reshape((self.monitors,-1))
        
        if (self.observation_type == 0) or (self.observation_type == 2):
            monitor_locs = self.monitor_locs
        else:
            monitor_locs = self.monitor_relative_locs

        new_obs = np.concatenate((self.monitor_angles, evader_locs, monitor_locs),axis=1)
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

        #adjust monitors according to actions
        action = np.array(action).squeeze()
        #action = np.atleast_1d(action)
        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."
        trans_action = [self.ref_act[idx] for idx in action]
        self.monitor_angles = self.monitor_angles + trans_action
        self.monitor_angles[self.monitor_angles>=math.pi] = self.monitor_angles[self.monitor_angles>=math.pi] - 2*math.pi
        self.monitor_angles[self.monitor_angles<-math.pi] = self.monitor_angles[self.monitor_angles<-math.pi] + 2*math.pi

        #let evaders run randomly
        evaders_angle = 2*math.pi*np.random.rand(self.evaders)
        self.evaders_displacement[:,0] = np.cos(evaders_angle)
        self.evaders_displacement[:,1] = np.sin(evaders_angle)
        self.evader_locs += self.evader_speed*self.evaders_displacement
        x = self.evader_locs[:,0]
        y = self.evader_locs[:,1]
        x[x<0] = -x[x<0]
        y[y<0] = -y[y<0]
        x[x>self.xlen] = 2*self.xlen - x[x>self.xlen]
        y[y>self.ylen] = 2*self.ylen - y[y>self.ylen]
        self.evader_locs[:,0] = x
        self.evader_locs[:,1] = y
        #renew observations
        self.obs = self._get_obs()
        full_monitoring = self.calcu_evader2monitor()

        if full_monitoring == True:
            reward = self.FULL_MONITORING_REWARD*np.ones(self.monitors)
            self.stat['full_monitoring'] += 1

        monitoring_permonitor = np.sum(self.monitoring_mat,axis=1)
        reward[monitoring_permonitor>0] += self.IN_MONITORING_REWARD

        debug = {'monitor_angles':self.monitor_angles,'evader_locs':self.evader_locs}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return
