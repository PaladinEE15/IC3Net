import numpy as np
import torch
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#ref_model = ['models/th/sem/n3_100_s1_new','models/th/sem/n3_10_s1_new','models/pp/sem/n3_100_s1_new', 'models/pp/sem/n3_10_s1_new', 'models/pp/sem/n5_100_s2_new','models/th/sem/n5_25_s2_new','models/th/sem/n5_100_s1_new','models/pp/sem/n5_10_s10_q_01', 'models/tj/sem/a_s1', 'models/tj/sem/a_10_s1_cur1000', 'models/tj/sem/b_s10_base_cur1000','models/tj/sem/b_s1_10_cur1000']

colors = ['#194f97','#555555','#bd6b08','#00686b','#c82d31']

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

p1 = ax1.plot(x,raw_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.set_ylim(0,20)
ax2.set_ylim(0,2)

plt.axhline(y=0.17, color='#555555',lw=2.5,alpha=0.8,linestyle='-.')
plt.text(150,0.2, 'entropy limit', color='#555555', fontsize=12)


plt.legend(handles, labels, fontsize=12, loc='upper right', framealpha=0.5)

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

p1 = ax1.plot(x,raw_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.set_ylim(0,40)
ax2.set_ylim(0,2)

plt.axhline(y=0.36, color='#555555', lw=2.5,alpha=0.8 ,linestyle='-.')
plt.text(150,0.39, 'entropy limit', color='#555555', fontsize=12)

plt.legend(handles, labels, fontsize=12, loc='upper right', framealpha=0.5)

plt.title('Treasure Hunt B',fontsize=20)
plt.savefig('figs/thb_curve.pdf',bbox_inches='tight')

#ppalog
model_raw = 'models/pp/sem/n3_100_s1_new'
model_sem = 'models/pp/sem/n3_10_s1_new2'

d = torch.load(model_raw)
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem)
sem_comm = (d['log'])['comm_entropy'].data
sem_steps = (d['log'])['steps_taken'].data

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.set_ylim(0,20)
ax2.set_ylim(0,2)
plt.axhline(y=0.14, color='#555555', lw=2.5,alpha=0.8 ,linestyle='-.')
plt.text(150,0.17, 'entropy limit', color='#555555', fontsize=12)
plt.legend(handles, labels, fontsize=12, loc='upper right', framealpha=0.5)

plt.title('Predator Prey A',fontsize=20)
plt.savefig('figs/ppa_curve.pdf',bbox_inches='tight')

#ppblog
model_raw = 'models/pp/sem/n5_100_s2_new'
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

p1 = ax1.plot(x,raw_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('timesteps↓',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)

ax1.set_ylim(0,40)
ax2.set_ylim(0,2)
plt.axhline(y=0.15, color='#555555', lw=2.5,alpha=0.8 ,linestyle='-.')
plt.text(150,0.18, 'entropy limit', color='#555555', fontsize=12)
plt.legend(handles, labels, fontsize=12, loc='upper right', framealpha=0.5)

plt.title('Predator Prey B',fontsize=20)
plt.savefig('figs/ppb_curve.pdf',bbox_inches='tight')

#tjalog
plt.figure()
model_raw_set = ['models/tj/sem/a_s1', 'models/tj/sem/a_base_50_cur1000', 'models/tj/sem/a_base_90_batch2000']
model_sem_set = ['models/tj/sem/a_10_s1_cur1000','models/tj/sem/a_10_40_cur1000','models/tj/sem/a_10_80_cur1000','models/tj/sem/a_10_90_batch2000','models/tj/sem/a_10_100_batch2000']

raw_comm_set = []
raw_steps_set = []
for model_raw in model_raw_set:
    d = torch.load(model_raw)
    raw_comm_set.append(((d['log'])['comm_entropy'].data)[:1200])
    raw_steps_set.append(((d['log'])['success'].data)[:1200])
raw_comm = np.mean(np.stack(raw_comm_set),axis=0)
raw_steps = np.mean(np.stack(raw_steps_set),axis=0)

sem_comm_set = []
sem_steps_set = []
for model_sem in model_sem_set:
    d = torch.load(model_sem)
    sem_comm_set.append(((d['log'])['comm_entropy'].data)[:1200])
    sem_steps_set.append(((d['log'])['success'].data)[:1200])
sem_comm = np.mean(np.stack(sem_comm_set),axis=0)
sem_steps = np.mean(np.stack(sem_steps_set),axis=0)

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('success rates↑',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)
ax1.set_ylim(0,1)
ax2.set_ylim(0,2)

plt.axhline(y=0.15, color='#555555', lw=2.5,alpha=0.8 , linestyle='-.')
plt.text(900,0.18, 'entropy limit', color='#555555', fontsize=12)
plt.legend(handles, labels, fontsize=12, loc='lower left', framealpha=0.5)
plt.title('Traffic Junction A',fontsize=20)
plt.savefig('figs/tja_curve.pdf',bbox_inches='tight')

#tjblog
plt.figure()
model_raw_set = ['models/tj/sem/b_s10_base_cur1000','models/tj/sem/b_base_20_cur1000','models/tj/sem/b_base_30_cur1000','models/tj/sem/b_base_50_cur1000']
model_sem_set = ['models/tj/sem/b_s1_10_cur1000','models/tj/sem/b_10_30_cur1000','models/tj/sem/b_10_50_cur1000','models/tj/sem/b_10_70_cur1000']

raw_comm_set = []
raw_steps_set = []
for model_raw in model_raw_set:
    d = torch.load(model_raw)
    raw_comm_set.append(((d['log'])['comm_entropy'].data)[:1200])
    raw_steps_set.append(((d['log'])['success'].data)[:1200])
raw_comm = np.mean(np.stack(raw_comm_set),axis=0)
raw_steps = np.mean(np.stack(raw_steps_set),axis=0)

sem_comm_set = []
sem_steps_set = []
for model_sem in model_sem_set:
    d = torch.load(model_sem)
    sem_comm_set.append(((d['log'])['comm_entropy'].data)[:1200])
    sem_steps_set.append(((d['log'])['success'].data)[:1200])
sem_comm = np.mean(np.stack(sem_comm_set),axis=0)
sem_steps = np.mean(np.stack(sem_steps_set),axis=0)

fig, ax1 = plt.subplots() 
ax2 = ax1.twinx()

x = np.arange(len(sem_comm))

p1 = ax1.plot(x,raw_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='baseline, timesteps')
p2 = ax1.plot(x,sem_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='ours, timesteps')

p3 = ax2.plot(x,raw_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline, entropy')
p4 = ax2.plot(x,sem_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='ours, entropy')

handles, labels = [], []
for p in [p1[0], p2[0], p3[0], p4[0]]:
    handles.append(p)
    labels.append(p.get_label())

ax1.set_xlabel('training epochs',fontsize=17)
ax1.set_ylabel('success rates↑',fontsize=17)
ax2.set_ylabel('entropy↓',fontsize=17)
ax1.set_ylim(0,1)
ax2.set_ylim(0,2)

plt.axhline(y=0.12, color='#555555', lw=2.5,alpha=0.8 ,linestyle='-.')
plt.text(900,0.15, 'entropy limit', color='#555555', fontsize=12)
plt.legend(handles, labels, fontsize=12, loc='lower left', framealpha=0.5)
plt.title('Traffic Junction B',fontsize=20)
plt.savefig('figs/tjb_curve.pdf',bbox_inches='tight')

#ax1.set_ylim(5,20)
#ax2.set_ylim(0,2)


#below are ablation study log
model_comp = 'models/pp/sem/n3_10_s1_new2'
model_sem_1 = 'models/pp/sem/n3_all01'
model_sem_2 = 'models/pp/sem/n3_all0001'
model_sem_3 = 'revise_models/pp3_increase01'


d = torch.load('models/pp/sem/n3_100_s1_new')
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data

d = torch.load(model_comp)
comp_comm = (d['log'])['comm_entropy'].data
comp_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem_1)
sem1_comm = (d['log'])['comm_entropy'].data
sem1_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem_2)
sem2_comm = (d['log'])['comm_entropy'].data
sem2_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem_3)
sem3_comm = (d['log'])['comm_entropy'].data
sem3_steps = (d['log'])['steps_taken'].data

x = np.arange(len(comp_comm))

colors = ['#194f97','#555555','#bd6b08','#00686b','#c82d31']
plt.figure()
plt.plot(x,raw_steps,color='#555555',linestyle='-',alpha=0.8,lw=3,label='baseline')
plt.plot(x,comp_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='SEM+SBM')
plt.plot(x,sem1_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='SEM-1')
plt.plot(x,sem2_steps,color='#00686b',linestyle='-',alpha=0.8,lw=3,label='SEM-2')
plt.plot(x,sem3_steps,color='#c82d31',linestyle='-',alpha=0.8,lw=3,label='SEM-3')

plt.xlabel('training epochs',fontsize=18)
plt.ylabel('timesteps↓',fontsize=18)
plt.ylim(0,20)
plt.legend(fontsize=14,loc="lower left")

plt.title('PP-A Timesteps',fontsize=22)
plt.savefig('figs/ppa_ab1.pdf',bbox_inches='tight')

x = np.arange(len(comp_comm))

plt.figure()
plt.plot(x,raw_comm,color='#555555',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline')
plt.plot(x,comp_comm,color='#194f97',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM+SBM')
plt.plot(x,sem1_comm,color='#bd6b08',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM-1')
plt.plot(x,sem2_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM-2')
plt.plot(x,sem3_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM-3')

plt.xlabel('training epochs',fontsize=18)
plt.ylabel('entropy↓',fontsize=18)
plt.ylim(-0.25,2)
plt.legend(fontsize=14,loc="lower left")

plt.title('PP-A Entropy',fontsize=22)
plt.savefig('figs/ppa_ab2.pdf',bbox_inches='tight')

model_comp = 'models/pp/sem/n5_10_s10_q_01'
model_sem_1 = 'models/pp/sem/n5_all01'
model_sem_2 = 'models/pp/sem/n5_all0001'
model_sem_3 = 'revise_models/pp5_increase01'

d = torch.load('models/pp/sem/n5_100_s2_new')
raw_comm = (d['log'])['comm_entropy'].data
raw_steps = (d['log'])['steps_taken'].data


d = torch.load(model_comp)
comp_comm = (d['log'])['comm_entropy'].data
comp_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem_1)
sem1_comm = (d['log'])['comm_entropy'].data
sem1_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem_2)
sem2_comm = (d['log'])['comm_entropy'].data
sem2_steps = (d['log'])['steps_taken'].data

d = torch.load(model_sem_3)
sem3_comm = (d['log'])['comm_entropy'].data
sem3_steps = (d['log'])['steps_taken'].data

x = np.arange(len(comp_comm))

plt.figure()
plt.plot(x,raw_steps,color='#555555',linestyle='-',alpha=0.8,lw=3,label='baseline')
plt.plot(x,comp_steps,color='#194f97',linestyle='-',alpha=0.8,lw=3,label='SEM+SBM')
plt.plot(x,sem1_steps,color='#bd6b08',linestyle='-',alpha=0.8,lw=3,label='SEM-1')
plt.plot(x,sem2_steps,color='#00686b',linestyle='-',alpha=0.8,lw=3,label='SEM-2')
plt.plot(x,sem3_steps,color='#c82d31',linestyle='-',alpha=0.8,lw=3,label='SEM-3')

plt.xlabel('training epochs',fontsize=18)
plt.ylabel('timesteps↓',fontsize=18)
plt.ylim(0,40)
plt.legend(fontsize=14,loc="lower left")

plt.title('PP-B Timesteps',fontsize=22)
plt.savefig('figs/ppb_ab1.pdf',bbox_inches='tight')

x = np.arange(len(comp_comm))

plt.figure()
plt.plot(x,raw_comm,color='#555555',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='baseline')
plt.plot(x,comp_comm,color='#194f97',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM+SBM')
plt.plot(x,sem1_comm,color='#bd6b08',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM-1')
plt.plot(x,sem2_comm,color='#00686b',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM-2')
plt.plot(x,sem3_comm,color='#c82d31',linestyle=(0, (5, 1)),alpha=0.8,lw=3,label='SEM-3')

plt.xlabel('training epochs',fontsize=18)
plt.ylabel('entropy↓',fontsize=18)
plt.ylim(-0.25,2)
plt.legend(fontsize=14,loc="lower left")

plt.title('PP-B Entropy',fontsize=22)
plt.savefig('figs/ppb_ab2.pdf',bbox_inches='tight')

