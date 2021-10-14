import torch
import torch.nn.functional as F
from torch import nn
import sys
from models import MLP
from action_utils import select_action, translate_action

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info
    between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions)).to(torch.device("cuda"))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents).to(torch.device("cuda"))
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents).to(torch.device("cuda")) - torch.eye(self.nagents, self.nagents).to(torch.device("cuda"))

        if self.args.comm_detail == 'mim':
            self.msg_encoder = nn.Sequential()
            msg_layer_num = len(self.args.msg_hid_layer)
            for i in range(msg_layer_num):
                if i == 0:
                    self.msg_encoder.add_module('fc1',nn.Linear(args.hid_size, self.args.msg_hid_layer[0]))
                    self.msg_encoder.add_module('activate1',nn.ReLU())
                else:
                    self.msg_encoder.add_module('fc2',nn.Linear(self.args.msg_hid_layer[i-1], self.args.msg_hid_layer[i]))
                    self.msg_encoder.add_module('activate2',nn.ReLU())
            self.mu_layer = nn.Sequential()    
            self.mu_layer.add_module('mu_out',nn.Linear(self.args.msg_hid_layer[i], self.args.msg_size))
            self.mu_layer.add_module('activate3',nn.Tanh())    
            self.lnsigma_layer = nn.Linear(self.args.msg_hid_layer[i], self.args.msg_size)
        elif self.args.comm_detail != 'raw':
            self.msg_encoder = nn.Sequential()
            msg_layer_num = len(self.args.msg_hid_layer)
            for i in range(msg_layer_num):
                if i == 0:
                    self.msg_encoder.add_module('fc1',nn.Linear(args.hid_size, self.args.msg_hid_layer[0]))
                    self.msg_encoder.add_module('activate1',nn.ReLU())
                else:
                    self.msg_encoder.add_module('fc2',nn.Linear(self.args.msg_hid_layer[i-1], self.args.msg_hid_layer[i]))
                    self.msg_encoder.add_module('activate2',nn.ReLU())
            self.msg_encoder.add_module('fc3',nn.Linear(self.args.msg_hid_layer[i], self.args.msg_size))
            if self.args.comm_detail =='binary':
                self.msg_encoder.add_module('activate3',nn.Sigmoid())
            else:
                self.msg_encoder.add_module('activate3',nn.Tanh())

        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoder = nn.Linear(num_inputs, args.hid_size)

        # if self.args.env_name == 'starcraft':
        #     self.state_encoder = nn.Linear(num_inputs, num_inputs)
        #     self.encoder = nn.Linear(num_inputs * 2, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:
            self.init_hidden(args.batch_size)
            if args.rnn_type == 'LSTM':
                self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)
            else:
                self.f_module = nn.GRUCell(args.hid_size, args.hid_size)
        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        # self.f = nn.Linear(args.hid_size, args.hid_size)
        if args.share_weights:
            self.C_module = nn.Linear(args.msg_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.msg_size, args.hid_size)
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.msg_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # print(self.C)
        # self.C.weight.data.zero_()
        # Init weights for linear layers
        # self.apply(self.init_weights)

        self.value_head = nn.Linear(self.hid_size, 1)


    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask']).to(torch.device("cuda"))
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n).to(torch.device("cuda"))
            num_agents_alive = n    
        
        record_mask = agent_mask.view(n,1)
        record_mask = record_mask.expand(n,self.args.msg_size)

        agent_mask = agent_mask.view(1, 1, n)
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)

        return num_agents_alive, agent_mask, record_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
            # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state

    def generate_comm(self,raw_comm):
        if self.args.comm_detail == 'raw':
            comm = raw_comm
        else :
            comm = self.msg_encoder(raw_comm) 
        comm_inuse = comm
        comm_info = comm     
        if self.args.comm_detail == 'binary':
            U = torch.rand(2, self.args.msg_size).cuda()
            noise_0 = -torch.log(-torch.log(U[0,:]))
            noise_1 = -torch.log(-torch.log(U[1,:]))
            comm_0 = torch.exp((torch.log(comm)+noise_0)/self.args.gumbel_gamma)
            comm_1 = torch.exp((torch.log(comm)+noise_1)/self.args.gumbel_gamma)
            comm = comm_1/(comm_0+comm_1)
            comm_info = comm

            if self.args.quant:
                qt_comm = torch.round(comm)
                comm_inuse = (qt_comm-comm).detach() + comm
            else:
                comm_inuse = comm
            return comm_inuse, comm_info
        elif self.args.comm_detail == 'mim':
            mu = self.mu_layer(comm)
            lnsigma = self.lnsigma_layer(comm)
            comm = mu + torch.exp(lnsigma)*(torch.randn_like(lnsigma).cuda())
            comm = torch.clamp(comm,min=-1,max=1)
            comm_info = torch.cat((comm,mu,lnsigma),-1)
            comm_inuse = comm
        #the message range is (-1, 1)
        if self.args.quant:
            qt_comm = (comm+1)*0.5
            qt_comm = qt_comm*(self.args.quant_levels-1)
            qt_comm = torch.round(qt_comm)
            qt_comm = qt_comm/(self.args.quant_levels-1)
            qt_comm = qt_comm*2-1
            comm_inuse = (qt_comm-comm).detach()+comm
        return comm_inuse, comm_info
        

    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (1 x N x hid_size)
            comm_in {tensor} -- Communication tensor for the network. (1 x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        # if self.args.env_name == 'starcraft':
        #     maxi = x.max(dim=-2)[0]
        #     x = self.state_encoder(x)
        #     x = x.sum(dim=-2)
        #     x = torch.cat([x, maxi], dim=-1)
        #     x = self.tanh(x)

        x, hidden_state, cell_state = self.forward_state_encoder(x)

        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask, record_mask = self.get_agent_mask(batch_size, info)

        #get mask for record


        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action']).to(torch.device("cuda"))
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask = agent_mask*comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes): #decide how many times to communicate, default 1
            # Choose current or prev depending on recurrent
            raw_comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state
            comm, broad_comm = self.generate_comm(raw_comm)
            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.args.msg_size)

            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            mask = mask.expand_as(comm)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)


            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)

                if self.args.rnn_type == 'LSTM':
                    output = self.f_module(inp, (hidden_state, cell_state))
                    hidden_state = output[0]
                    cell_state = output[1]
                else: #GRU
                    hidden_state = self.f_module(inp, hidden_state)

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # v = torch.stack([self.value_head(hidden_state[:, i, :]) for i in range(n)])
        # v = v.view(hidden_state.size(0), n, -1)
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent :
            if self.args.rnn_type == 'LSTM':
                return broad_comm, action, value_head, (hidden_state.clone(), cell_state.clone())
            else:
                return broad_comm, action, value_head, hidden_state.clone()
        else:
            return broad_comm, action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        if self.args.rnn_type == 'LSTM':
            return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True).to(torch.device("cuda")),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True).to(torch.device("cuda"))))
        else:
            return torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True).to(torch.device("cuda"))

