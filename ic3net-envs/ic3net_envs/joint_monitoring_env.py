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
        self.TIME_PENALTY = -0.02
        self.IN_MONITORING_REWARD = 0.01
        self.episode_over = False

    def init_args(self, parser):
        env = parser.add_argument_group('Cooperative Search task')
        env.add_argument("--evader_speed", type=float, default=0, 
                    help="speed of evaders")
        env.add_argument("--observation_type", type=int, default=0, 
                    help="0-self abs coords + relative target observation;1-self abs coords + target obsolute observation;2- one-hot id + obs")
        env.add_argument("--reward_type", type=int, default=0, 
                    help="0-full_monitor;1-monitor as much as possible;2-sparse reward;3-solo observe reward")
        env.add_argument("--monitor_type", type=int, default=0, 
                    help="0-not in corners, 1-in corners")
        env.add_argument("--add_evaders", type=int, default=0, 
                    help="number of additional evaders")   
        env.add_argument('--catch', default=False, action='store_true', 
                    help='catch or monitor')     
        env.add_argument('--random_monitor', default=False, action='store_true', 
                    help='spawn monitor randomly')          
        #there's another kind of observation: rangefinder. However the detection is complex......
        #firstly, calculate the sector according to angle
        #secondly, update corresponding sensor data (note that covering......)
        return
    

    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        self.evader_speed = args.evader_speed
        self.reward_type = args.reward_type
        self.monitor_type = args.monitor_type
        self.catch = args.catch
        self.monitors = args.nagents
        self.evaders = args.nagents + args.add_evaders
        self.random_monitor = args.random_monitor
        if self.monitor_type == 0: #square in 
            self.single_monitor_angle = 0.5
            self.ref_act = np.array([0.5*math.pi,-0.5*math.pi,0.25*math.pi,-0.25*math.pi,0])
            if args.nagents == 4:
                self.xlen = 2.414
                self.ylen = 2.414              
                self.monitor_locs = np.array([[0.707,0.707],[0.707,1.707],[1.707,0.707],[1.707,1.707]])
            elif args.nagents == 6: 
                self.xlen = 3.414
                self.ylen = 2.414        
                self.monitor_locs = np.array([[0.707,0.707],[0.707,1.707],[1.707,0.707],[1.707,1.707],[2.707,1.707],[2.707,0.707]])    
        elif self.monitor_type == 1: #square border
            self.ref_act = np.array([1/6*math.pi,-1/6*math.pi,1/12*math.pi,-1/12*math.pi,0])
            self.single_monitor_angle = 1/6
            if args.nagents == 3:
                self.xlen = 1
                self.ylen = 1
                self.single_monitor_angle = 1/4
                self.ref_act = np.array([1/4*math.pi,-1/4*math.pi,1/8*math.pi,-1/8*math.pi,0])
                self.monitor_locs = np.array([[0,0],[1,0],[0.5,1]])
            elif args.nagents == 4:
                self.xlen = 1
                self.ylen = 1
                self.monitor_locs = np.array([[0,0],[0,1],[1,0],[1,1]])
            elif args.nagents == 5:
                self.xlen = math.sqrt(2)
                self.ylen = math.sqrt(2)        
                self.monitor_locs = np.array([[0,0],[0,math.sqrt(2)],[math.sqrt(2),0],[math.sqrt(2),math.sqrt(2)],[math.sqrt(2)/2,math.sqrt(2)/2]])    
            elif args.nagents == 6:
                self.xlen = math.sqrt(2)
                self.ylen = math.sqrt(2)        
                self.monitor_locs = np.array([[0,0],[0,math.sqrt(2)],[math.sqrt(2),0],[math.sqrt(2),math.sqrt(2)],[math.sqrt(2)/2+0.1,math.sqrt(2)/2],[math.sqrt(2)/2-0.1,math.sqrt(2)/2]])  

        elif self.monitor_type == 2: #round
            if args.nagents == 6:
                self.single_monitor_angle = 1/6
                self.ref_act = np.array([1/6*math.pi,-1/6*math.pi,1/12*math.pi,-1/12*math.pi,0])      
                self.monitor_locs = 0.6*np.array([[0,1],[0,-1],[np.sqrt(3)/2,0.5],[np.sqrt(3)/2,-0.5],[-np.sqrt(3)/2,0.5],[-np.sqrt(3)/2,-0.5]]) 
        elif self.monitor_type == 3:
            if args.nagents == 4:
                self.xlen = math.sqrt(2)
                self.ylen = math.sqrt(2)
                self.single_monitor_angle = 1/4+(1/12)*args.add_evaders
                self.ref_act = np.array([1/4*math.pi,-1/4*math.pi,1/8*math.pi,-1/8*math.pi,0])      
                self.monitor_locs = math.sqrt(2)/2 * np.ones((4,2))
            elif args.nagents == 3:
                self.xlen = math.sqrt(2)
                self.ylen = math.sqrt(2)
                self.single_monitor_angle = 1/3+(1/12)*args.add_evaders
                self.ref_act = np.array([1/4*math.pi,-1/4*math.pi,1/8*math.pi,-1/8*math.pi,0])      
                self.monitor_locs = math.sqrt(2)/2 * np.ones((3,2))
        else:
            return

        self.observation_type = args.observation_type
        
        self.naction = 5

        self.action_space = spaces.MultiDiscrete([self.naction])
        #observation space design
        #self angle:1
        #self abs coords:2
        #evaders coords + whether observes: 3*evaders
        self.obs_dim = 3 + 3*self.evaders
        if self.observation_type == 2:
            self.monitor_ids = np.identity(self.monitors)
            self.obs_dim = 1+self.monitors+self.evaders

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
        self.relative_locs_theta = np.arctan((relative_locs_mid[:,1]/(relative_locs_mid[:,0]+1e-4))).reshape((self.monitors, self.evaders))
        inrange = self.relative_locs_d < 1

        inangle = (self.relative_locs_theta>monitors_angles_full)*(self.relative_locs_theta<(monitors_angles_full+np.pi*self.single_monitor_angle))+((2*np.pi+self.relative_locs_theta)>monitors_angles_full)*((2*np.pi+self.relative_locs_theta)<(monitors_angles_full+np.pi*self.single_monitor_angle)) 


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
        if self.random_monitor:
            self.monitor_locs = np.random.rand(self.monitors,2)
            self.monitor_locs[:,0] = self.monitor_locs[:,0]*self.xlen
            self.monitor_locs[:,1] = self.monitor_locs[:,1]*self.ylen            
        if self.monitor_type != 2:
            self.evader_locs = np.random.rand(self.evaders,2)

            self.evader_locs[:,0] = self.evader_locs[:,0]*self.xlen
            self.evader_locs[:,1] = self.evader_locs[:,1]*self.ylen
        else:
            evader_locs_raw = np.random.rand(self.evaders,2)
            self.evader_locs = np.zeros((self.evaders,2))
            evader_locs_tho = np.sqrt(evader_locs_raw[:,0])
            evader_locs_theta = 2*math.pi*evader_locs_raw[:,1]
            self.evader_locs[:,0] = evader_locs_tho*np.cos(evader_locs_theta)
            self.evader_locs[:,1] = evader_locs_tho*np.sin(evader_locs_theta)
            self.evader_locs = 0.6*self.evader_locs
        self.monitor_angles = 2*math.pi*np.random.rand(self.monitors,1) - math.pi

        self.episode_over = False
        self.evaders_displacement = np.zeros((self.evaders,2))
        self.stat = dict()
        if self.catch:
            self.stat['success'] = 0
        else:
            self.stat['full_monitoring'] = 0

        # Observation will be nagent * vision * vision ndarray
        _ = self.calcu_evader2monitor()
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
        if self.observation_type == 0:
            temp_relative_locs_d = self.relative_locs_d 
            temp_relative_locs_theta = self.relative_locs_theta
            temp_relative_locs_d[self.monitoring_mat==False] = -1
            temp_relative_locs_theta[self.monitoring_mat==False] = 0
            evader_locs = np.concatenate((temp_relative_locs_d.reshape((-1,1)),temp_relative_locs_theta.reshape((-1,1))),axis=1).reshape((self.monitors,-1))
        elif self.observation_type == 1:
            evader_locs = np.tile(self.evader_locs.reshape((1,-1)),(self.monitors,1))
            indicate_mat = np.repeat(self.monitoring_mat,2,axis=1)
            evader_locs[indicate_mat == False] = 0
        elif self.observation_type == 2:
            new_obs = np.concatenate((self.monitor_angles, self.monitor_ids, self.monitoring_mat.astype(np.float32)),axis=1)
            return new_obs

        
        monitor_locs = self.monitor_locs
        
        '''
        type4:
        0:1, self angle
        1:3, self coordinates
        3:11, eight sensor range
        11:11+nevaders, one hot, denote whether evader is observed
        self.relative_locs_d
        self.relative_locs_theta
        self.monitoring_mat
        monitors_angles_full = np.tile(self.monitor_angles,(1,self.evaders))
        '''
        if self.observation_type == 4:
            monitor_sensors_status = np.ones((self.monitors,9))
            monitors_angles_full = np.tile(self.monitor_angles,(1,self.evaders))
            sec_angle = self.single_monitor_angle*np.pi/8
            add_idx = (self.relative_locs_theta<monitors_angles_full)*self.monitoring_mat
            self.relative_locs_theta[add_idx] += 2*np.pi
            sectors = (np.ceil((self.relative_locs_theta - monitors_angles_full)/sec_angle)*self.monitoring_mat).astype(int)
            x_idx = np.arange(self.monitors).reshape(-1,1)
            monitor_sensors_status[x_idx, sectors] = self.relative_locs_d
            new_obs = np.concatenate((self.monitor_angles, self.monitor_locs, monitor_sensors_status[:,1:], self.monitoring_mat.astype(np.float32)),axis=1)
        else:
            new_obs = np.concatenate((self.monitor_angles, evader_locs, monitor_locs,self.monitoring_mat.astype(np.float32)),axis=1)
        
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
        self.monitor_angles = self.monitor_angles + np.array(trans_action).reshape((-1,1))
        self.monitor_angles[self.monitor_angles>=math.pi] = self.monitor_angles[self.monitor_angles>=math.pi] - 2*math.pi
        self.monitor_angles[self.monitor_angles<-math.pi] = self.monitor_angles[self.monitor_angles<-math.pi] + 2*math.pi

        #let evaders run randomly
        if self.evader_speed >0:
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
        full_monitoring = self.calcu_evader2monitor()
        self.obs = self._get_obs()
        
        if self.reward_type == 0:
            if full_monitoring == True:
                reward = self.FULL_MONITORING_REWARD*np.ones(self.monitors)
                if self.catch:
                    self.stat['success'] = 1
                    self.episode_over = True
                else:
                    self.stat['full_monitoring'] += 1
            else: 
                reward = self.TIME_PENALTY*np.ones(self.monitors)
                monitoring_permonitor = np.sum(self.monitoring_mat,axis=1)
                if self.reward_type == 0:          
                    reward[monitoring_permonitor>0] += self.IN_MONITORING_REWARD
        elif self.reward_type == 1:
            monitoring_perevader = np.sum(self.monitoring_mat,axis=0)
            monitor_numbers = np.sum(monitoring_perevader>0)
            self.stat['full_monitoring'] += monitor_numbers
            reward = monitor_numbers*self.IN_MONITORING_REWARD*np.ones(self.monitors)
        elif self.reward_type == 2:
            if full_monitoring == True:
                reward = self.FULL_MONITORING_REWARD*np.ones(self.monitors)
                self.stat['full_monitoring'] += 1
            else:
                reward = self.TIME_PENALTY*np.ones(self.monitors)
        elif self.reward_type == 3:
            if full_monitoring == True:
                reward = self.FULL_MONITORING_REWARD*np.ones(self.monitors)
                self.stat['full_monitoring'] += 1
            else:
                reward = self.TIME_PENALTY*np.ones(self.monitors)
                monitoring_perevader = np.sum(self.monitoring_mat,axis=0,keepdims=True)
                check_idx = np.tile((monitoring_perevader==1).astype(int),(self.monitors,1))
                solo_monitoring = np.sum(check_idx*self.monitoring_mat,axis=1)
                reward[solo_monitoring>0] += self.IN_MONITORING_REWARD

        debug = {'monitor_angles':self.monitor_angles,'evader_locs':self.evader_locs}
        return self.obs, reward, self.episode_over, debug

    def seed(self):
        return
