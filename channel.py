
import numpy as np
import sk_dsp_comm.fec_conv as fec
import torch
from bitarray import bitarray
class Node(object):
    def __init__(self,name=None,value=None):
        self._name=name
        self._value=value
        self._left=None
        self._right=None

class HuffmanTree(object):

    def __init__(self,char_weights):
        self.Leav=[Node(part[0],part[1]) for part in char_weights]  
        while len(self.Leav)!=1:    
            self.Leav.sort(key=lambda node:node._value,reverse=True)
            c=Node(value=(self.Leav[-1]._value+self.Leav[-2]._value))
            c._left=self.Leav.pop(-1)
            c._right=self.Leav.pop(-1)
            self.Leav.append(c)
        self.root=self.Leav[0]
        self.Buffer=list(range(10)) 
    def pre(self,tree,length):
        node=tree
        if (not node):
            return
        elif node._name:
            print (node._name + '    encoding:',end=''),
            for i in range(length):
                print (self.Buffer[i],end='')
            print ('\n')
            return
        self.Buffer[length]=0
        self.pre(node._left,length+1)
        self.Buffer[length]=1
        self.pre(node._right,length+1)
    def get_code(self):
        self.pre(self.root,0)

huffman_dict_2 = { #type2-PEM
    0: bitarray('1'), 0.25: bitarray('01'), 0.5: bitarray('0001'), 0.75: bitarray('000001'), 1: bitarray('00000000'), -1: bitarray('00000001'), -0.75: bitarray('0000001'), -0.5:bitarray('00001'), -0.25:bitarray('001')}

huffman_dict_1 = { #type1-ORI
    0: bitarray('0'), 0.25: bitarray('10'), 0.5: bitarray('1100'), 0.75: bitarray('110100'), 1: bitarray('11010100'), -1: bitarray('11010101'), -0.75: bitarray('1101011'), -0.5:bitarray('11011'), -0.25:bitarray('111')}

huffman_set = [0,huffman_dict_1,huffman_dict_2]
prefix_set = [0,[0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
cc1 = fec.FECConv(('101','111'),10)
cc2 = fec.FECConv(('111','111','101'),10)

cc_set = [0,cc1,cc2]

def modify_message(msg, exp_type, ber):
    #preprocess msg
    raw_msg_set = msg.detach().cpu().numpy()
    total_y,raw_len = raw_msg_set.shape[0],raw_msg_set.shape[1]
    output_set = []
    length_set = []
    for yidx in range(total_y):
        raw_msg = raw_msg_set[yidx,:]

        #huffman encoding
        s_e_seq = bitarray()
        s_e_seq.encode(huffman_set[exp_type], raw_msg)
        s_e_seq = list(s_e_seq)
        #convolutional encoding
        s_e_seq.extend([0,0,0,0,0,0,0,0,0]) #the constrain is 10
        state = '000'
        c_e_seq, state = cc_set[exp_type].conv_encoder(s_e_seq, state)
        #add error
        #c_e_seq is a float64 ndarray
        noise = np.random.binomial(1,ber,c_e_seq.shape[0])
        length_set.append(c_e_seq.shape[0])
        #convolutional decoding
        recv_seq = np.abs(c_e_seq-noise)
        recv_seq = recv_seq.astype(int)
        c_d_seq = cc_set[exp_type].viterbi_decoder(recv_seq,'hard')
        c_d_seq = list(c_d_seq.astype(int))
        #huffman decoding
        s_d_seq = bitarray()
        c_d_seq += prefix_set[exp_type]
        s_d_seq.extend(c_d_seq)
        s_d_seq = s_d_seq.decode(huffman_set[exp_type])
        s_d_seq = np.append(np.array(s_d_seq),np.zeros(48))
        s_d_seq = s_d_seq[0:raw_len] 
        output_set.append(s_d_seq)
        #add error
    output = np.vstack(output_set)
    output = torch.from_numpy(output).to(torch.device("cuda"))
    return output, torch.Tensor(length_set)