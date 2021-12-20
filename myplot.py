import numpy as np
import matplotlib.pyplot as plt
import gym


ColorPool = ['cornflowerblue', 'lightgreen', 'lightsteelblue', 'moccasin']


#the following is used for assist experiment 1
#datainput area
steps_1 = []
entropy_1 = []
steps_2 = []
entropy_2 = []
steps_3 = []
entropy_3 = []
steps_4 = []
entropy_4 = []
steps_5 = []
entropy_5 = []
steps_6 = []
entropy_6 = []
step_set = [steps_1,steps_2,steps_3,steps_4,steps_5,steps_6]
entropy_set = [entropy_1,entropy_2,entropy_3,entropy_4,entropy_5,entropy_6]
name_set = ['Predator-Prey A', 'Predator-Prey B','Cooperative Search A', 'Cooperative Search B', 'Traffic Junction A', 'Traffic Junction B']
xticks = ['None','17','9','5','3']
x = range(5)
plt.figure()
for idx in range(1,7):  
    ax1 = plt.add_subplot([2,3,idx])
    ax1.plot(x, step_set[idx],linewidth=1,label='steps',color='deepskyblue')
    ax2 = ax1.twinx()
    ax2.plot(x, entropy_set[idx],linewidth=1,label='entropy',color='orangered')

    ax1.set_ylabel('steps',fontsize=14)
    ax2.set_ylabel('entropy (nat)',fontsize=14)
    ax1.set_xlabel('quantization levels',fontsize=14)
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')
    ax1.title(name_set[idx])

plt.tight_layout()
plt.savefig('ABC.pdf', bbox_inches='tight')

#the following is used for assist experiment 3
raw_entropy_set = []
raw_steps_set = []
pem_entropy_set = []
pem_steps_set = []
plt.figure()
axe = plt.subplot(111)
ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')

txt = ['msg_size 48', 'msg_size 24', 'msg_size 12', 'msg_size 6', 'msg_size 3']
for idx in range(len(raw_entropy_set)):
    axe.annotate(txt[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytest = (raw_entropy_set[idx]+0.1, raw_steps_set[idx]+0.1))
    axe.annotate(txt[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytest = (pem_entropy_set[idx]+0.1, pem_steps_set[idx]+0.1))

axe.legend((ax1,ax2),('RAW','PEM'))
plt.savefig('DEF.pdf', bbox_inches='tight')

