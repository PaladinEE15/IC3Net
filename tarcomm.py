import torch
import torch.nn.functional as F
from torch import nn
import sys
from models import MLP
from action_utils import select_action, translate_action

class TARMACMLP(nn.Module):
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

        super(TARMACMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        self.continuous = False
        self.action_generator = nn.Linear(args.hid_size, args.naction_heads[0])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents).to(torch.device("cuda"))
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents).to(torch.device("cuda")) - torch.eye(self.nagents, self.nagents).to(torch.device("cuda"))

        if self.args.comm_detail == 'mim':
            self.msg_initializer = nn.Sequential()
            msg_layer_num = len(self.args.msg_hid_layer)
            for i in range(msg_layer_num):
                if i == 0:
                    self.msg_initializer.add_module('fc1',nn.Linear(args.hid_size, self.args.msg_hid_layer[0]))
                    self.msg_initializer.add_module('activate1',nn.ReLU())
                else:
                    self.msg_initializer.add_module('fc2',nn.Linear(self.args.msg_hid_layer[i-1], self.args.msg_hid_layer[i]))
                    self.msg_initializer.add_module('activate2',nn.ReLU())
            self.mu_layer = nn.Sequential()    
            self.mu_layer.add_module('mu_out',nn.Linear(self.args.msg_hid_layer[i], self.args.msg_size))
            self.mu_layer.add_module('activate3',nn.Tanh())    
            self.lnsigma_layer = nn.Linear(self.args.msg_hid_layer[i], self.args.msg_size)
            self.k_generator = nn.Sequential()   
            self.k_generator.add_module('k_out',nn.Linear(self.args.msg_hid_layer[i], self.args.qk_size))
            self.k_generator.add_module('activate4',nn.Tanh())
        elif self.args.comm_detail != 'raw':
            self.msg_initializer = nn.Sequential()
            msg_layer_num = len(self.args.msg_hid_layer)
            for i in range(msg_layer_num):
                if i == 0:
                    self.msg_initializer.add_module('fc1',nn.Linear(args.hid_size, self.args.msg_hid_layer[0]))
                    self.msg_initializer.add_module('activate1',nn.ReLU())
                else:
                    self.msg_initializer.add_module('fc2',nn.Linear(self.args.msg_hid_layer[i-1], self.args.msg_hid_layer[i]))
                    self.msg_initializer.add_module('activate2',nn.ReLU())

            self.m_generator = nn.Sequential()   
            self.m_generator.add_module('msg_out',nn.Linear(self.args.msg_hid_layer[i], self.args.msg_size))
            self.m_generator.add_module('activate3',nn.Tanh())
            self.k_generator = nn.Sequential()   
            self.k_generator.add_module('k_out',nn.Linear(self.args.msg_hid_layer[i], self.args.qk_size))
            self.k_generator.add_module('activate4',nn.Tanh())

        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        self.encoding_size = int(args.hid_size/2)
        self.encoder = nn.Linear(num_inputs, self.encoding_size)

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
            self.C_module = nn.Linear(args.msg_size, int(args.hid_size/2))
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.v_size, int(args.hid_size/2))
                                            for _ in range(self.comm_passes)])
        # self.C = nn.Linear(args.msg_size, args.hid_size)

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()

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

        x, extras = x
        x = self.encoder(x)

        if self.args.rnn_type == 'LSTM':
            hidden_state, cell_state = extras
        else:
            hidden_state = extras
        # hidden_state = self.tanh( self.hidd_encoder(prev_hidden_state) + x)

        return x, hidden_state, cell_state

    def generate_comm(self,raw_comm):
        if self.args.no_comm:
            return torch.zeros_like(raw_comm), torch.zeros_like(raw_comm)
        
        mid_comm = self.msg_initializer(raw_comm) 
        self_key = self.k_generator(mid_comm)

        if self.args.comm_detail == 'mim':
            mu = self.mu_layer(mid_comm)
            lnsigma = self.lnsigma_layer(mid_comm)
            comm = mu + torch.exp(lnsigma)*(torch.randn_like(lnsigma).cuda())
            comm = torch.clamp(comm,min=-1,max=1)
            comm_info = torch.cat((comm,mu,lnsigma),-1)
            comm_inuse = comm
        else: #assume mlp
            comm = self.m_generator(mid_comm)
            comm_info = comm
            comm_inuse = comm

        #the message range is (-1, 1)
        if self.quant:
            qt_comm = (comm+1)*0.5
            qt_comm = qt_comm*(self.args.quant_levels-1)
            qt_comm = torch.round(qt_comm)
            qt_comm = qt_comm/(self.args.quant_levels-1)
            qt_comm = qt_comm*2-1
            comm_inuse = (qt_comm-comm).detach()+comm
        return comm_inuse, comm_info, self_key

    def forward(self, x, info={}, quant=False):
        # TODO: Update dimensions

        x, hidden_state, cell_state = self.forward_state_encoder(x)
        self.quant = quant
        batch_size = x.size()[0]
        n = self.nagents

        num_agents_alive, agent_mask, record_mask = self.get_agent_mask(batch_size, info)

        #get mask for record
        agent_mask_transpose = agent_mask.transpose(1, 2)

        for i in range(self.comm_passes): #decide how many times to communicate, default 1
            # Choose current or prev depending on recurrent
            raw_comm = hidden_state.view(n, self.hid_size) if self.args.recurrent else hidden_state
            comm, broad_comm, key = self.generate_comm(raw_comm)
            #comm shape: n x msg_size
            #key shape: n x qk_size
            # Get the next communication vector based on next hidden state
            value, query = torch.split(comm, [self.args.v_size,self.args.qk_size], dim=1)
            attention = torch.mm(key, query.t())/4 - 100*torch.eye(n).to(torch.device("cuda"))
            msg_weight = F.softmax(attention,dim=1)
            msg_weight_mat = msg_weight.view(1,n,n,1).expand(1,n,n,self.args.v_size)
            value_mat = value.view(1,n,1,self.args.v_size).expand(1,n,n,self.args.v_size)
            comm = msg_weight_mat*value_mat
            # Create mask for masking self communication
            mask = self.comm_mask.view(1, n, n)
            mask = mask.expand(comm.shape[0], n, n)
            mask = mask.unsqueeze(-1)

            mask = mask.expand_as(comm)
            comm = comm * mask

            # Mask comm_in
            # Mask communcation from dead agents
            comm = comm * agent_mask
            # Mask communication to dead agents
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)

            x = x.view(batch_size * n, self.encoding_size)
            c = c.view(batch_size * n, self.encoding_size)

            inp = torch.cat([x,c], dim=1)
            if self.args.rnn_type == 'LSTM':
                output = self.f_module(inp, (hidden_state, cell_state))
                hidden_state = output[0]
                cell_state = output[1]
            else: #GRU
                hidden_state = self.f_module(inp, hidden_state)

        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)

        action = [F.log_softmax(self.action_generator(h), dim=-1)]

        if self.args.rnn_type == 'LSTM':
            return broad_comm, action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return broad_comm, action, value_head, hidden_state.clone()


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

