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
                    mid_mat = torch.min(nn.functional.relu(ref_info-target+1), nn.functional.relu(-ref_info+target+1),dim=0)
                    freq = torch.mean(mid_mat,dim=0)+1e-20
                    freq = -freq*torch.log(freq)
                    comm_entro_loss += torch.mean(freq)
            elif self.args.comm_detail == 'cos':
                comm_entro_loss = 0
                for target in range(self.args.quant_levels):
                    mid_mat = 0.5*(comm_info>target-1)*(comm_info<target+1)*(torch.cos(math.pi*(comm_info-target))+1)
                    freq = torch.mean(mid_mat,dim=0)+1e-20
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
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

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
