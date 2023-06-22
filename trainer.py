from collections import namedtuple
from inspect import getfullargspec
import numpy as np
import math
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

def myquant(x):
    for k in range(9):
        x[(x>k*0.25-1.125)&(x<k*0.25-0.875)] = k*0.25 - 1

class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        if self.args.optim_type == 'adam':
            self.optimizer = optim.Adam(policy_net.parameters(),lr = args.lrate)
        elif self.args.optim_type == 'rmsprop':
            self.optimizer = optim.RMSprop(policy_net.parameters(), lr = args.lrate, alpha=0.97, eps=1e-6)
        
        self.params = [p for p in self.policy_net.parameters()]
        self.mark_reftensor = False #mark whether the reftensor is created
    
    def get_distribution_simple(self,input_comm):
        input_comm = (input_comm+1)*0.5
        input_comm = input_comm*(self.args.quant_levels-1)  
        calcu_comm = np.rint(input_comm)      
        counts = np.array(list(map(lambda x: np.sum(calcu_comm==x,axis=1),range(self.args.quant_levels))))
        probs = counts/input_comm.shape[1]
        return probs

    def get_episode(self, epoch, details = False):
        episode = []
        reset_args = getfullargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        state = torch.from_numpy(state).double().to(torch.device("cuda"))
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1
        if self.args.quant and epoch >= self.args.quant_start:
            quant = True
        else:
            quant = False

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size).to(torch.device("cuda"))

        stat['quant_time'] = 0
        stat['generate_time'] = 0
        stat['process_time'] = 0
        stat['interact_time'] = 0
        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                comm, action_out, value, prev_hid, time_dict = self.policy_net(x, info, quant)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:#GRU
                        prev_hid = prev_hid.detach()
            else:
                x = state
                comm, action_out, value = self.policy_net(x, info, quant)
            
            
            if self.args.calcu_entropy:
                if t == 0:
                    #init comm
                    comm_stat = comm.view(self.args.nagents,-1)
                else:
                    comm_stat = torch.cat((comm_stat, comm.view(self.args.nagents,-1)),dim = 0)
            else:
                comm_stat = torch.zeros(1)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            interact_start_t = time.time()
            next_state, reward, done, info = self.env.step(actual)

            stat['quant_time'] += time_dict['quant']
            stat['generate_time'] += time_dict['generate']
            stat['process_time'] += time_dict['process']
            stat['interact_time'] += time.time() - interact_start_t
            next_state = torch.from_numpy(next_state).double().to(torch.device("cuda"))
            if details:
                ready_comm = comm.view(self.args.nagents,-1).detach().cpu().numpy()
                myquant(ready_comm)
                print('timestep:',t)
                for idx in range(self.args.nagents):
                    arr_str = ','.join(str(x) for x in ready_comm[idx])
                    print(arr_str)
                print('new env info - predator locs:',info['predator_locs'])
                print('new env info - prey locs:',info['prey_locs'])
                if done:
                    print('done!!!')

            #here, begin env data display
            if self.args.detailed_info:
                print('info begin!timestep:',t)
                print('predator locs:', info['predator_locs'])
                print('prey locs:', info['prey_locs'])
                print('comm info:',self.get_distribution_simple(comm.view(self.args.nagents,-1).detach().cpu().numpy()))

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat, comm_stat)




    def compute_grad(self, comm_info, batch, loss_alpha):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward).to(torch.device("cuda"))
        episode_masks = torch.Tensor(batch.episode_mask).to(torch.device("cuda"))
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask).to(torch.device("cuda"))
        actions = torch.Tensor(batch.action).to(torch.device("cuda"))
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(torch.device("cuda"))

        coop_returns = torch.Tensor(batch_size, n).to(torch.device("cuda"))
        ncoop_returns = torch.Tensor(batch_size, n).to(torch.device("cuda"))
        returns = torch.Tensor(batch_size, n).to(torch.device("cuda"))
        deltas = torch.Tensor(batch_size, n).to(torch.device("cuda"))
        advantages = torch.Tensor(batch_size, n).to(torch.device("cuda"))
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0
        
        if loss_alpha > 0:
            if self.args.comm_detail == 'triangle':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = torch.min(ref_info-target+1, -ref_info+target+1)
                    mid_mat = torch.clamp(mid_mat, min=0)
                    square_mat = (ref_info>target-0.5)*(ref_info<target+0.5)*torch.ones_like(ref_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat)+1e-4
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += freq
            elif self.args.comm_detail == 'cos':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = 0.5*(ref_info>target-1)*(ref_info<target+1)*(torch.cos(math.pi*(ref_info-target))+1)
                    square_mat = (ref_info>target-0.5)*(ref_info<target+0.5)*torch.ones_like(ref_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat)+1e-4
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += freq
            elif self.args.comm_detail == 'bar':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = 0.5*(ref_info>target-1)*(ref_info<target+1)*(torch.cos(math.pi*(ref_info-target))+1)
                    square_mat = (ref_info>target-0.5)*(ref_info<target+0.5)*torch.ones_like(ref_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat)+self.args.epsilon
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += freq
                comm_entro_loss = F.relu(comm_entro_loss - self.args.entropy_limit) 
            elif self.args.comm_detail == 'widecos':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = (ref_info>target-1)*(ref_info<target+1)*torch.cos(0.5*math.pi*(ref_info-target))
                    square_mat = (ref_info>target-0.5)*(ref_info<target+0.5)*torch.ones_like(ref_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat)+1e-4
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += freq
            elif self.args.comm_detail == 'bell':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = (ref_info>target-1)*(ref_info<target+1)*torch.exp(-4*(ref_info-target)**2)
                    square_mat = (ref_info>target-0.5)*(ref_info<target+0.5)*torch.ones_like(ref_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat)+1e-4
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += freq
            elif self.args.comm_detail == 'mim':
                split_size = int(comm_info.size()[1]/3)
                _, mu, lnsigma = torch.split(comm_info,split_size,1) 
                loss_mat = 0.5*(mu**2 + (torch.exp(lnsigma))**2)/self.args.mim_gauss_var - lnsigma    
                comm_entro_loss = torch.mean(loss_mat)    
            elif self.args.comm_detail == 'ndq':
                split_size = int(comm_info.size()[1]/2)
                _, mu = torch.split(comm_info,split_size,1) 
                loss_mat = 0.5*(mu**2)/self.args.mim_gauss_var   
                comm_entro_loss = torch.mean(loss_mat)     
        else:
            comm_entro_loss = torch.Tensor([0]).cuda()

        
        for i in reversed(range (rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        comm_entro_loss = comm_entro_loss*batch_size #to keep in pace with other loss

        if not self.args.continuous:
            # entropy regularization term
            if self.args.entr > 0:
                entropy = 0
                for i in range(len(log_p_a)):
                    entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
                stat['entropy'] = entropy.item()
                loss -= self.args.entr * entropy
        stat['other_loss'] = loss.item()
        stat['comm_entro_loss'] = comm_entro_loss.item()*loss_alpha
        loss = loss + comm_entro_loss*loss_alpha #we want to maximize comm_entro

        loss.backward()

        return stat

    def test(self, run_times):
        steps_taken = []
        success_times = []
        for idx in range(run_times):
            episode, episode_stat, comm_stat = self.get_episode(2000)
            if idx == 0:
                comm_stat_acc = comm_stat.detach().cpu().numpy()
            else:
                comm_stat_acc = np.concatenate((comm_stat_acc,comm_stat.detach().cpu().numpy()))
            if 'success' in episode_stat.keys():
                success_times.append(episode_stat['success'])
            else:
                success_times.append(episode_stat['full_monitoring'])
            steps_taken.append(episode_stat['steps_taken'])   

        return comm_stat_acc, np.array(steps_taken), np.array(success_times)

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat, comm_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            if len(batch) == 0:
                comm_stat_acc = comm_stat
            else:
                comm_stat_acc = torch.cat((comm_stat_acc,comm_stat),dim=0)
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats, comm_stat_acc


    # only used when nprocesses=1
    def test_batch(self,times):
        # run its own trainer
        comm_stat_acc, steps_taken_acc, success_times_acc = self.trainer.test(times)
        
        print('success: ', np.mean(success_times_acc),' std: ', np.std(success_times_acc) )
        print('steps: ', np.mean(steps_taken_acc),' std: ', np.std(steps_taken_acc))

        if self.args.calcu_entropy:
            #calculate entropy here
            comm_np_list = np.hsplit(comm_stat_acc,self.args.msg_size) #split matrix for parallelization
            entropy_set = map(self.calcu_entropy, comm_np_list)
            final_entropy = sum(entropy_set)
            print('entropy: ', final_entropy)
        return
    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat, comm_info_acc = self.run_batch(epoch)
        self.optimizer.zero_grad()

        comm_np = comm_info_acc.detach().numpy()    
        comm_np_list = np.hsplit(comm_np,self.args.msg_size) #split matrix for parallelization
        entropy_set = map(self.calcu_entropy, comm_np_list)
        final_entropy = sum(entropy_set)
        entro_stat = {'comm_entropy':final_entropy}
        merge_stat(entro_stat, stat)

        s = self.compute_grad(comm_info_acc, batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
