import numpy as np
import torch
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
model = ['models/th/sem/n3_100_s1','models/th/sem/n5_100_s1','models/pp/sem/n5_100_s2','models/pp/sem/n3_100_s1','models/pp/sem/n3_10_s1','models/th/sem/n3_10_s1','models/th/sem/n5_25_s2','models/pp/sem/n5_10_s10_q_01']

for item in model:
    print('check:',item)
    d = torch.load(item)
    the_log = (d['log'])['comm_entropy'].data
    print(the_log)


#thalog

model_raw = 'models/th/sem/n3_100_s1'
model_sem = 'models/th/sem/n3_10_s1'

d = torch.load(model_raw)
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem)
sem_comm = (d['log'])['comm_entropy'].data
sem_steps = (d['log'])['steps_taken'].data

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

ax1.plot(x,raw_steps,color='#7cd6cf',linestyle='-',label='raw steps')
ax1.plot(x,sem_steps,color='#63b2ee',linestyle='-',label='ours steps')

ax2.plot(x,raw_comm,color='#76da91',linestyle='--',label='raw ent')
ax2.plot(x,sem_comm,color='#007f54',linestyle='--',label='ours ent')

ax1.set_xlabel('training epochs')
ax1.set_ylabel('timesteps')
ax2.set_ylabel('entropy')

ax1.legend(loc='lower left')
ax2.legend(loc='lower right')

#ax1.set_ylim(5,20)
#ax2.set_ylim(0,2)