import numpy as np
import torch
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
ref_model = ['models/th/sem/n3_100_s1_new','models/th/sem/n3_10_s1_new','models/pp/sem/n3_100_s1_new', 'models/pp/sem/n3_10_s1_new', 'models/pp/sem/n5_100_s2_new','models/th/sem/n5_25_s2_new','models/th/sem/n5_100_s1_new','models/pp/sem/n5_10_s10_q_01', 'models/tj/sem/a_s1', 'models/tj/sem/a_10_s1_cur1000', 'models/tj/sem/b_s10_base_cur1000','models/tj/sem/b_s1_10_cur1000']




#thalog
model_raw = 'models/th/sem/n3_100_s1_new'
model_sem = 'models/th/sem/n3_10_s1_new'

d = torch.load(model_raw)
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem)
sem_comm = (d['log'])['comm_entropy'].data
sem_steps = (d['log'])['steps_taken'].data

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#c82d31',linestyle='-',label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#194f97',linestyle='-',label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#c82d31',linestyle='--',label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#194f97',linestyle='--',label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.legend(handles, labels, fontsize=14, loc='center right')

plt.title('Treasure Hunt A',fontsize=20)
plt.savefig('figs/tha_curve.pdf',bbox_inches='tight')

#thblog
model_raw = 'models/th/sem/n5_100_s1_new'
model_sem = 'models/th/sem/n5_25_s2_new'

d = torch.load(model_raw)
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem)
sem_comm = (d['log'])['comm_entropy'].data
sem_steps = (d['log'])['steps_taken'].data

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#c82d31',linestyle='-',label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#194f97',linestyle='-',label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#c82d31',linestyle='--',label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#194f97',linestyle='--',label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.legend(handles, labels, fontsize=14, loc='center right')

plt.title('Treasure Hunt B',fontsize=20)
plt.savefig('figs/thb_curve.pdf',bbox_inches='tight')

#ppalog
model_raw = 'models/pp/sem/n3_100_s1_new'
model_sem = 'models/th/sem/n3_10_s1_new'

d = torch.load(model_raw)
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem)
sem_comm = (d['log'])['comm_entropy'].data
sem_steps = (d['log'])['steps_taken'].data

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#c82d31',linestyle='-',label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#194f97',linestyle='-',label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#c82d31',linestyle='--',label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#194f97',linestyle='--',label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.legend(handles, labels, fontsize=14, loc='center right')

plt.title('Predator Prey A',fontsize=20)
plt.savefig('figs/ppa_curve.pdf',bbox_inches='tight')

#ppblog
model_raw = 'models/th/sem/n5_100_s1_new'
model_sem = 'models/pp/sem/n5_10_s10_q_01'

d = torch.load(model_raw)
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem)
sem_comm = (d['log'])['comm_entropy'].data
sem_steps = (d['log'])['steps_taken'].data

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#c82d31',linestyle='-',label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#194f97',linestyle='-',label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#c82d31',linestyle='--',label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#194f97',linestyle='--',label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.legend(handles, labels, fontsize=14, loc='center right')

plt.title('Predator Prey B',fontsize=20)
plt.savefig('figs/ppb_curve.pdf',bbox_inches='tight')

#tjalog
model_raw = 'models/tj/sem/a_s1'
model_sem = 'models/tj/sem/a_10_s1_cur1000'

d = torch.load(model_raw)
raw_comm = ((d['log'])['comm_entropy'].data)[:1200]
raw_steps = ((d['log'])['success'].data)[:1200]

d = torch.load(model_sem)
sem_comm = ((d['log'])['comm_entropy'].data)[:1200]
sem_steps = ((d['log'])['success'].data)[:1200]

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#c82d31',linestyle='-',label='baseline, success rates')
p2 = ax1.plot(x,sem_steps,color='#194f97',linestyle='-',label='ours, success rates')

p3 = ax2.plot(x,raw_comm,color='#c82d31',linestyle='--',label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#194f97',linestyle='--',label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('success rates↑',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.legend(handles, labels, fontsize=14, loc='center right')

plt.title('Traffic Junction A',fontsize=20)
plt.savefig('figs/tja_curve.pdf',bbox_inches='tight')

#tjblog
model_raw = 'models/tj/sem/b_s10_base_cur1000'
model_sem = 'models/tj/sem/b_s1_10_cur1000'

d = torch.load(model_raw)
raw_comm = ((d['log'])['comm_entropy'].data)[:1200]
raw_steps = ((d['log'])['success'].data)[:1200]

d = torch.load(model_sem)
sem_comm = ((d['log'])['comm_entropy'].data)[:1200]
sem_steps = ((d['log'])['success'].data)[:1200]

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#c82d31',linestyle='-',label='baseline, success rates')
p2 = ax1.plot(x,sem_steps,color='#194f97',linestyle='-',label='ours, success rates')

p3 = ax2.plot(x,raw_comm,color='#c82d31',linestyle='--',label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#194f97',linestyle='--',label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('success rates↑',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.legend(handles, labels, fontsize=14, loc='center right')

plt.title('Traffic Junction B',fontsize=20)
plt.savefig('figs/tjb_curve.pdf',bbox_inches='tight')


#ax1.set_ylim(5,20)
#ax2.set_ylim(0,2)