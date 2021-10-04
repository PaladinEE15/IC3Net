import time
from utils import *
import torch
import torch.multiprocessing as mp
from collections import Counter
import data
from inspect import getargspec
from utils import *
from action_utils import *
import time

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

ctx = mp.get_context("spawn")

def calcu_entropy_binary(comm):
    comm = np.rint(comm)
    freq = np.mean(comm, axis=0)
    entropy_set = -(freq+1e-20)*np.log(freq+1e-20) -(1-freq+1e-20)*np.log(1-freq+1e-20)
    entropy = np.sum(entropy_set)
    return entropy

def multi_env_process(id, comm, main_args):
    env = data.init(main_args.env_name, main_args, False)
    torch.manual_seed(id + 1)
    np.random.seed(id + 1)
    while True:
        task = comm.recv()
        the_data = None
        if type(task) == list:
            task, the_data = task  
        if task == 'reset':
            if the_data == None:
                comm.send(env.reset())
            else:
                comm.send(env.reset(the_data))  
        elif task == 'step':
            comm.send(env.step(the_data))  
        elif task == 'reward_terminal':
            comm.send(env.reward_terminal())   
        elif task == 'get_stat':
            comm.send(env.get_stat())   
        elif task == 'quit':
            return 

class MultiEnvTrainer(object):
    def __init__(self, args, policy_net):
        self.policy_net = policy_net
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=args.lrate)
        self.args = args
        self.parent_pipes, self.child_pipes = zip(*[mp.Pipe() for _ in range(self.args.nprocesses)])
        self.workers = []

        for idx, child_pipe in enumerate(self.child_pipes):
            process = ctx.Process(target=multi_env_process, args=(idx, child_pipe, self.args), daemon=True)
            self.workers.append(process)
            process.start()
    
    def quit(self):
        for parent_pipe in self.parent_pipes:
            parent_pipe.send('quit')
    def save(self, train_log, path):
        d = dict()
        d['policy_net'] = self.policy_net.state_dict()
        d['log'] = train_log
        d['trainer'] = self.state_dict()
        torch.save(d, path)        

    def load(self, path):
        d = torch.load(path)
        self.policy_net.load_state_dict(d['policy_net'])
        self.load_state_dict(d['trainer'])

    def train(self, total_epoch):
        n_envs = self.args.nprocesses
        train_log = dict()
        train_log['success'] = []
        train_log['steps_mean'] = []
        train_log['steps_std'] = []
        train_log['main_loss'] = []
        train_log['main_loss_std'] = []


        for epoch in range(total_epoch):
            success_set = []
            steps_set = []
            main_loss_set = []
            entropy_loss_set = []
            epoch_begin_time = time.time()
            whether_comm_record = None
        
            for mini_epoch in range(self.args.epoch_size):
                misc = dict()
                misc['alive_mask'] = np.ones(self.args.nagents) #ignore starcraft scenerio
                episode_set = [[] for i in range(n_envs)]
                #prepare for collecting episodes
                state_set = None
                if self.args.reset_withepoch:
                    for parent_pipe in self.parent_pipes:
                        parent_pipe.send(['reset',epoch])
                else:
                    for parent_pipe in self.parent_pipes:
                        parent_pipe.send('reset')  
                info = dict()

                for parent_pipe in self.parent_pipes:
                    state = parent_pipe.recv()
                    if state_set == None:
                        state_set = state
                    else:
                        state_set = torch.cat((state_set, state), 0)
                comm_acc_set = [None for i in range(n_envs)]
                total_comm_set = None
                t_set = np.zeros(n_envs) #record t of each environment. remember to make zero after reset
                continue_training = np.ones(n_envs) #this is used to indicate whether the training should be stopped
                total_steps = 0
                if self.args.hard_attn and self.args.commnet:
                    info['comm_action'] = np.zeros((n_envs,self.args.nagents), dtype=int)
                if self.args.recurrent:
                    prev_hid_set = self.policy_net.init_hidden(batch_size=n_envs)
                else:
                    prev_hid_set = torch.zeros(n_envs, self.args.nagents, self.args.hid_size).to(torch.device("cuda"))
                n_agents = self.args.nagents
                success_record = []
                steps_record = []
                entropy_record = []
                while True:
                    #the envs will run asynchronously. if one done, just reset related info and continue, unless steps reach max batch size
                    if self.args.recurrent:
                        prev_hid_set = prev_hid_set.view(n_envs*n_agents,self.args.hid_size)
                        x = [state_set, prev_hid_set]
                        comm_set, action_out_set, value_set, prev_hid_set = self.policy_net(x, info)
                        prev_hid_set = prev_hid_set.view(n_envs,n_agents,self.args.hid_size)
                        for i in range(n_envs):
                            if t_set[i]+1 % self.args.detach_gap == 0:
                                prev_hid_set[i,:] = prev_hid_set[i,:].detach()
                    else:
                        x = state_set
                        comm_set, action_out_set, value_set = self.policy_net(x, info)
                
                    if self.args.calcu_entropy:
                        total_comm = torch.flatten(comm_set,0,-2)
                        if total_comm_set == None:
                            total_comm_set = total_comm
                        else:
                            total_comm_set = torch.cat((total_comm_set, total_comm), 0)
                        for i in range(n_envs):
                            if comm_acc_set[i] == None:
                                comm_acc_set[i] = comm_set[i,:]
                            else:
                                comm_acc_set[i] = torch.cat((comm_acc_set[i],comm_set[i,:]),0)
                    action_set = select_action(self.args, action_out_set)
                    action_set, actual = translate_action(self.args, action_set)
                    #need to split it 
                    #action_set and actual are both [batchsize x n, batchsize x n]     
                    actual = list(zip(actual[0],actual[1]))
                    for idx, parent_pipe in enumerate(self.parent_pipes):
                        parent_pipe.send(['step', actual[idx]])
                    # store comm_action in info for next step
                    if self.args.hard_attn and self.args.commnet:
                        info['comm_action'] = action_set[1] if not self.args.comm_action_one else np.ones((n_envs,self.args.nagents), dtype=int)

                    if whether_comm_record is None:
                        whether_comm_record = action_set[1]
                    else:
                        whether_comm_record = np.concatenate((whether_comm_record, action_set[1]),0)
                    action_set = list(zip(action_set[0],action_set[1]))
                    action_out_set = list(zip(action_out_set[0],action_out_set[1]))
                    #we do not need to record comm_action        
                    #the following are used to avoid inplace
                    new_state_set = torch.zeros_like(state_set)
                    for idx, parent_pipe in enumerate(self.parent_pipes):

                        next_state, reward, done, env_info = parent_pipe.recv()
                        real_done = done or t_set[idx] == self.args.max_steps - 1

                        if real_done:
                            episode_mask = np.zeros(reward.shape)
                            episode_mini_mask = np.ones(reward.shape)
                        else:
                            episode_mask = np.ones(reward.shape)
                            if 'is_completed' in env_info:
                                episode_mini_mask = 1 - env_info['is_completed'].reshape(-1)
                            else:
                                episode_mini_mask = np.ones(reward.shape)
                        state = state_set[idx,:].unsqueeze(0)
                        
                        action = action_set[idx]
                        action_out = action_out_set[idx]
                        temp1 = action_out[0].unsqueeze(0)
                        temp2 = action_out[1].unsqueeze(0)
                        action_out = [temp1, temp2]
                        value = value_set[idx,:]
                        if real_done:
                            if self.args.reward_terminal:
                                parent_pipe.send('reward_terminal')
                                add_reward = parent_pipe.recv()
                                reward += add_reward

                            parent_pipe.send('get_stat')
                            prev_hid_set[idx,:] = self.policy_net.init_hidden(batch_size=1)
                            info['comm_action'][idx,:] = np.zeros(self.args.nagents, dtype=int) 
                            steps_record.append(t_set[idx])
                            success_record.append(parent_pipe.recv()['success'])
                            t_set[idx] = 0
                            if continue_training[idx] == 1:
                                trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
                                episode_set[idx].append(trans)
                            if total_steps >= self.args.batch_size:
                                continue_training[idx] = 0
                            if self.args.reset_withepoch:
                                parent_pipe.send(['reset',epoch])
                            else:
                                parent_pipe.send('reset')      
                            new_state_set[idx,:] = parent_pipe.recv() #inplace action
                        else:
                            t_set[idx] += 1   
                            new_state_set[idx,:] = next_state #inplace action       
                            if continue_training[idx] == 1: #the training is not complete, continue adding trans. else, trans should not be added to buffer                                           
                                trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
                                episode_set[idx].append(trans)
                    total_steps += n_envs
                    state_set = new_state_set
                    if np.sum(continue_training) == 0:
                        break
                #begin training
                #collect batch
                loss = 0
                if self.args.loss_alpha>0 and epoch>self.args.loss_start:
                    for idx, episodes in enumerate(episode_set):
                        batch = Transition(*zip(*episodes))
                        batch_loss = self.compute_loss(batch,comm_acc_set[idx])
                        loss += batch_loss
                else:
                    for idx, episodes in enumerate(episode_set):
                        batch = Transition(*zip(*episodes))
                        batch_loss = self.compute_loss(batch)
                        loss += batch_loss                    
                loss = loss/n_envs
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #calculate entropy if needed

                if self.args.calcu_entropy:
                    entropy = self.calcu_entropy(total_comm_set.detach().cpu().numpy())
                    entropy_record.append(entropy)
                #store training info and prepare for output
                success_set += success_record
                steps_set += steps_record
                #main_loss_set.append(train_info['main_loss'])
                #entropy_loss_set.append(train_info['comm_entro_loss'])
            
            #a batch is completed, print stat and record  
            epoch_end_time = time.time()
            epoch_run_time = epoch_end_time - epoch_begin_time
            mean_success = np.mean(np.array(success_set))
            mean_steps, std_steps = np.mean(np.array(steps_set)), np.std(np.array(steps_set))
            #mean_main_loss, std_main_loss = np.mean(np.array(main_loss_set)), np.std(np.array(main_loss_set))
            mean_ifcomm = np.mean(whether_comm_record,0)
            print('epoch: ', epoch, ' time: ', epoch_run_time, 's')
            print('success: ', mean_success)
            print('steps: ', mean_steps, ' std: ', std_steps)
            print('comm action:', mean_ifcomm)
            #print('main loss: ', mean_main_loss, ' std: ', std_main_loss)
            
            if self.args.calcu_entropy:
                mean_entropy, std_entropy = np.mean(np.array(entropy_record)), np.std(np.array(entropy_record))
            #    mean_entropy_loss, std_entropy_loss = np.mean(np.array(entropy_loss_set)), np.std(np.array(entropy_loss_set))
            #    print('entropy loss: ', mean_entropy_loss, ' std: ', std_entropy_loss)
                print('entropy: ', mean_entropy, ' std: ', std_entropy)
            train_log['success'].append(mean_success)
            train_log['steps_mean'].append(mean_steps)
            train_log['steps_std'].append(std_steps)
            #train_log['main_loss'].append(mean_main_loss)
            #train_log['main_loss_std'].append(std_main_loss)

            if self.args.save_every and epoch and self.args.save != '' and epoch % self.args.save_every == 0:
                self.save(train_log, self.args.save + '_' + str(epoch))
        self.save(train_log, self.args.save)
        return train_log


    def compute_loss(self, batch, comm_info=None):
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward).to(torch.device("cuda"))
        episode_masks = torch.Tensor(batch.episode_mask).to(torch.device("cuda"))
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask).to(torch.device("cuda"))
        actions = torch.Tensor(batch.action).to(torch.device("cuda"))
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

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
        
        if comm_info != None:
            if self.args.comm_detail == 'triangle':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = torch.min(1.25*(ref_info-target+0.8), -1.25*(ref_info-target-0.8))
                    mid_mat = torch.clamp(mid_mat,min=0,max=1)
                    square_mat = (comm_info>target-0.5)*(comm_info<target+0.5)*torch.ones_like(comm_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat,dim=0)+1e-20
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += torch.mean(freq)
            elif self.args.comm_detail == 'cos':
                ref_info = (comm_info+1)*0.5
                ref_info = ref_info*(self.args.quant_levels-1) 
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = 0.5*(ref_info>target-1)*(ref_info<target+1)*(torch.cos(math.pi*(ref_info-target))+1)
                    square_mat = (ref_info>target-0.5)*(ref_info<target+0.5)*torch.ones_like(ref_info).to(torch.device("cuda"))
                    final_mat = (square_mat-mid_mat).detach()+mid_mat
                    freq = torch.mean(final_mat,dim=0)+1e-20
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += torch.mean(freq)
            elif self.args.comm_detail == 'binary':
                freq = torch.mean(comm_info, dim=0)
                entropy_set = -(freq+1e-20)*torch.log(freq+1e-20) -(1-freq+1e-20)*torch.log(1-freq+1e-20)
                comm_entro_loss = torch.mean(entropy_set)
            elif self.args.comm_detail == 'mim':
                _, mu, lnsigma = torch.split(comm_info,3,1) 
                loss_mat = 0.5*(mu**2 + (torch.exp(lnsigma))**2)/self.args.mim_gauss_var - lnsigma    
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

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            if self.args.entr > 0:
                entropy = 0
                for i in range(len(log_p_a)):
                    entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
                loss -= self.args.entr * entropy
        loss = loss/batch_size #get mean
        loss = loss + comm_entro_loss*self.args.loss_alpha #we want to maximize comm_entro
      
        return loss
    

    def calcu_entropy(self, comm):
        if self.args.comm_detail == 'mim':
            comm = np.split(comm,3,1)[0]
        elif self.args.comm_detail == 'binary':
            comm = np.rint(comm)
            freq = np.mean(comm, axis=0)
            entropy_set = -(freq+1e-20)*np.log(freq+1e-20) -(1-freq+1e-20)*np.log(1-freq+1e-20)
            entropy = np.sum(entropy_set)
            return entropy
        comm = (comm+1)*0.5
        comm = comm*(self.args.quant_levels-1)  
        calcu_comm = np.rint(comm)      
        counts = np.array(list(map(lambda x: np.sum(calcu_comm==x,axis=0),range(self.args.quant_levels))))
        probs = counts/comm.shape[0]
        probs[probs==0] = 1 #avoid ln0
        entropy = -np.sum(probs*np.log(probs))
        return entropy                           

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)