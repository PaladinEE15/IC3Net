import numpy as np
import matplotlib.pyplot as plt
import gym
from textwrap import fill
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


#the following is used for reducing message size
raw_entropy_set = np.array([20,11,5.99,3.26])
raw_steps_set = np.array([10.35,10.54,10.10,10.26])
pem_entropy_set = np.array([2.81,1.24,0.53])
pem_steps_set = np.array([10.52,10.70,11.04])
plt.figure()
plt.subplot(111)
#ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
#ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')
plt.scatter(raw_entropy_set, raw_steps_set, c='#9192ab',label='Original',linewidths=5)
plt.scatter(pem_entropy_set, pem_steps_set, c='#7cd6cf',label='DisEM',linewidths=5)

txt_ori = ['16', '8', '4', '2']
txt_pem = ['16', '8', '4']
for idx in range(len(raw_entropy_set)):
    plt.annotate(txt_ori[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytext = (raw_entropy_set[idx]+0.2, raw_steps_set[idx]-0.15),fontsize=18)
    if idx <=2:
        plt.annotate(txt_pem[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytext = (pem_entropy_set[idx]+0.2, pem_steps_set[idx]-0.15),fontsize=18)
plt.xlim(0,25)
plt.ylim(7,12)
plt.tick_params(labelsize=16)
plt.ylabel('timesteps↓',fontsize=20)
plt.xlabel('entropy',fontsize=20)
plt.legend(loc = 'lower right',fontsize=18)
plt.title('Predator Prey A',fontsize=22)
#plt.show()
plt.savefig('D:\Git\IC3Net\Figures\lengthreduceA.pdf', bbox_inches='tight')

#the following is used for reducing message size
raw_entropy_set = np.array([22,12.4,6.7,4.3])
raw_steps_set = np.array([15.9,15.1,16.2,19.8])
pem_entropy_set = np.array([4.25,2.88,1.24])
pem_steps_set = np.array([15.05,15.4,16.4])
plt.figure()
plt.subplot(111)
#ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
#ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')
plt.scatter(raw_entropy_set, raw_steps_set, c='#9192ab',label='Original',linewidths=5)
plt.scatter(pem_entropy_set, pem_steps_set, c='#7cd6cf',label='DisEM',linewidths=5)

txt_ori = ['16', '8', '4', '2']
txt_pem = ['16', '8', '4']
for idx in range(len(raw_entropy_set)):
    plt.annotate(txt_ori[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytext = (raw_entropy_set[idx]+0.2, raw_steps_set[idx]-0.4),fontsize=18)
    if idx <=2:
        plt.annotate(txt_pem[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytext = (pem_entropy_set[idx]+0.2, pem_steps_set[idx]-0.4),fontsize=18)
plt.xlim(0,30)
plt.ylim(14,20)
plt.tick_params(labelsize=16)
plt.ylabel('timesteps↓',fontsize=20)
plt.xlabel('entropy',fontsize=20)
plt.legend(loc = 'lower right',fontsize=18)
plt.title('Predator Prey B',fontsize=22)
#plt.show()
plt.savefig('D:\Git\IC3Net\Figures\lengthreduceB.pdf', bbox_inches='tight')



'''
#the following is used for assist experiment 1
#datainput area
steps_1 = np.array([8.28,8.30,8.14,9.54,18.44])
entropy_1 = np.array([1.8,1.16,0.59,0.11])
steps_2 = np.array([23.81,24.34,23.95,59.52,59.99])
entropy_2 = np.array([1.76,1.09,0.32,0.09])
steps_3 = np.array([10.11,10.28,10.48,13.05,14.94])
entropy_3 = np.array([1.56,0.92,0.19,0.04])
steps_4 = np.array([15.65,15.11,16.33,23.1,23.39])
entropy_4 = np.array([1.55,0.89,0.21,0.12])
steps_5 = np.array([0.958,0.966,0.972,0.948,0.956])
entropy_5 = np.array([2.35,1.73,1.15,0.75])
steps_6 = np.array([0.90,0.90,0.90,0.90,0.87])
entropy_6 = np.array([1.51,0.91,0.30,0.07])
step_set = [steps_1,steps_2,steps_3,steps_4,steps_5,steps_6]
entropy_set = [entropy_1,entropy_2,entropy_3,entropy_4,entropy_5,entropy_6]
name_set = ['Treasure Hunt A', 'Treasure Hunt B', 'Predator-Prey A', 'Predator-Prey B', 'Traffic Junction A', 'Traffic Junction B']
xticks = [0,0.125,0.25,0.5,1]
x1 = np.arange(5)
x2 = np.arange(1,5)

#subplotset = [231,232,233,234,235,236]
ylegend = ['timesteps↓','timesteps↓','timesteps↓','timesteps↓','success rates↑','success rates↑']
ylabel = ['timesteps↓','timesteps↓','timesteps↓','timesteps↓','success rates↑','success rates↑']
bar_width = 0.3
for idx in range(6):  
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    bar1 = ax1.bar(x1-0.5*bar_width, step_set[idx],bar_width,bottom = 0, label=ylegend[idx], ec='black',color='#63b2ee')
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x2+0.5*bar_width, entropy_set[idx],bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
    ax1.set_ylabel(ylabel[idx],fontsize=18)
    ax2.set_ylabel('entropy (nat)',fontsize=18)
    ax1.set_xlabel('value of $\Delta$',fontsize=18)
    ax1.set_xticks(ticks=x1)
    ax1.set_xticklabels(xticks)
    lns = bar1 + bar2
    labs = [l.get_label() for l in lns]
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)
    #ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 1))
    #ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.92))
    ax1.set_title(name_set[idx],fontsize=18)
    #plt.tight_layout()
    plt.savefig('D:\Git\IC3Net\TNNLS_Figures\quant'+str(idx)+'.pdf', bbox_inches='tight')
    #plt.show()




plt.figure()
plt.bar([0], [8.15], 0.4, bottom = 0, ec='black', label='Baseline',color='#05f8d6')
plt.bar([1,2], [9.29,11.46], 0.4, bottom = 0, ec='black', label='RMS',color='#0082fc')
plt.bar([2.4], [9.06], 0.4, bottom = 0, ec='black', label='RMS+SEM',color='#0082fc',hatch='//')
plt.ylabel('timesteps↓',fontsize=20)
plt.xlabel('bandwidth limits',fontsize=20)
plt.xticks([0,1,2.2],['100%','5%','2.5%'],fontsize=18)
plt.legend(fontsize=14,loc="lower left")
plt.title('Treasure Hunt A',fontsize=22)
#plt.show()
plt.savefig('D:\Git\IC3Net\TNNLS_Figures\RS1.pdf', bbox_inches='tight')

plt.figure()
plt.bar([0], [23.36], 0.4, bottom = 0, ec='black', label='Baseline',color='#05f8d6')
plt.bar([1,2], [27.73,39.41], 0.4, bottom = 0, ec='black', label='RMS',color='#0082fc')
plt.bar([2.4], [24.64], 0.4, bottom = 0, ec='black', label='RMS+SEM',color='#0082fc',hatch='//')
plt.ylabel('timesteps↓',fontsize=20)
plt.xlabel('bandwidth limits',fontsize=20)
plt.xticks([0,1,2.2],['100%','20%','10%'],fontsize=18)
plt.legend(fontsize=14,loc="lower left")
plt.title('Treasure Hunt B',fontsize=22)
#plt.show()
plt.savefig('D:\Git\IC3Net\TNNLS_Figures\RS2.pdf', bbox_inches='tight')




x1 = np.arange(5)
x2 = np.arange(1,5)
bar_width = 0.3
xticks = [0,0.125,0.25,0.5,1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, step_set[idx],bar_width,bottom = 0, label='success rates↑', ec='black',color='#d5d6d8')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, entropy_set[idx],bar_width,bottom = 0, label='entropy', ec='black', color='#555555')
ax1.set_ylabel('success rates↑',fontsize=18)
ax2.set_ylabel('entropy (nat)',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)
#ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 1))
#ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.92))
ax1.set_title('Traffic Junction B',fontsize=18)
#plt.tight_layout()
plt.savefig('D:\Git\IC3Net\TNNLS_Figures\quant5.pdf', bbox_inches='tight')



title_set = ['Distribution of digit 0','Distribution of digit 1','Distribution of digit 2','Distribution of digit 3']
name_set = ['ppb4ent0', 'ppb4ent1', 'ppb4ent2', 'ppb4ent3']
dataset = []
dataset.append(np.array([0.0007722212990199904, 0.005746556433392236, 0.00867390339285093, 0.01776235706021928, 0.9196218528336695, 0.03264807067492905, 0.00918113882901719, 0.005593899476901916, 0.0, ]))
dataset.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ]))
dataset.append(np.array([0.0, 0.0004154261863861756, 0.002764902709509585, 0.009939339896259593, 0.6058201009253656, 0.3630892345875737, 0.01192631444968453, 0.0052739264623493706, 0.0007707547828714094, ]))
dataset.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, ]))

xset = np.arange(9)

for idx in range(4):
    plt.figure()
    plt.bar(xset,dataset[idx],1,bottom=0,color = '#7cd6cf')
    plt.ylabel('probability',fontsize=18)
    plt.ylim(0,1)
    plt.xlabel('reconstruction points',fontsize=18)
    plt.xticks(xset,labels=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1],fontsize=14)
    plt.title(title_set[idx])
    plt.savefig('D:\Git\IC3Net\Figures\distribution_'+name_set[idx]+'.pdf', bbox_inches='tight')

x = np.arange(5)
bar_width = 0.3
ori_set = 20 - np.array([10.15, 9.62, 10.89, 15.28, 16.65])
pem_set = 20 - np.array([10.27, 10.35, 10.2, 11.40, 14.57])
#ori_set = 40 - np.array([16.25, 15.41, 17.43, 25.78, 31.13])
#pem_set = 40 - np.array([16.24, 16.6, 16.69, 17.4, 23.39])

plt.figure()
plt.bar(x-0.5*bar_width, ori_set, bar_width, bottom = 0, label='TARMAC-ORI',color='#9192ab')
plt.bar(x+0.5*bar_width, pem_set, bar_width, bottom = 0, label='TARMAC-PEM',color='#7cd6cf')
plt.ylabel('remaining timesteps',fontsize=20)
plt.xlabel('crossover probability p',fontsize=20)
plt.xticks(x,[0,0.01,0.05,0.1,0.15],fontsize=18)
plt.legend(fontsize=14,loc="lower left")
plt.title('Predator Prey A',fontsize=22)
plt.savefig('D:\Git\IC3Net\Figures\PPAcomm.pdf', bbox_inches='tight')


#


#the following is used for reducing message size
raw_entropy_set = np.array([20,11,5.99,3.26])
raw_steps_set = 20 - np.array([10.35,10.54,10.10,10.26])
pem_entropy_set = np.array([2.81,1.24,0.53])
pem_steps_set = 20 - np.array([10.52,10.70,11.04])
plt.figure()
plt.subplot(111)
#ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
#ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')
plt.scatter(raw_entropy_set, raw_steps_set, c='#9192ab',label='ORI',linewidths=5)
plt.scatter(pem_entropy_set, pem_steps_set, c='#7cd6cf',label='PEM',linewidths=5)

txt_ori = ['ORI, 16', 'ORI, 8', 'ORI, 4', 'ORI, 2']
txt_pem = ['PEM, 16', 'PEM, 8', 'PEM, 4']
for idx in range(len(raw_entropy_set)):
    plt.annotate(txt_ori[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytext = (raw_entropy_set[idx]+0.2, raw_steps_set[idx]-0.15),fontsize=18)
    if idx <=2:
        plt.annotate(txt_pem[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytext = (pem_entropy_set[idx]+0.2, pem_steps_set[idx]-0.15),fontsize=18)
plt.xlim(0,25)
plt.ylim(7.5,10.5)
plt.tick_params(labelsize=16)
plt.ylabel('remaining timesteps',fontsize=20)
plt.xlabel('entropy',fontsize=20)PEM
plt.legend(loc = 'lower right',fontsize=18)
plt.title('Predator Prey A',fontsize=22)
#plt.show()
plt.savefig('D:\Git\IC3Net\Figures\lengthreduceA.pdf', bbox_inches='tight')

#the following is used for reducing message size
raw_entropy_set = np.array([22,12.4,6.7,4.3])
raw_steps_set = 40 - np.array([15.9,15.1,16.2,19.8])
pem_entropy_set = np.array([4.25,2.88,1.24])
pem_steps_set = 40 - np.array([15.05,15.4,16.4])
plt.figure()
plt.subplot(111)
#ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
#ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')
plt.scatter(raw_entropy_set, raw_steps_set, c='#9192ab',label='ORI',linewidths=5)
plt.scatter(pem_entropy_set, pem_steps_set, c='#7cd6cf',label='PEM',linewidths=5)

txt_ori = ['ORI, 16', 'ORI, 8', 'ORI, 4', 'ORI, 2']
txt_pem = ['PEM, 16', 'PEM, 8', 'PEM, 4']
for idx in range(len(raw_entropy_set)):
    plt.annotate(txt_ori[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytext = (raw_entropy_set[idx]+0.2, raw_steps_set[idx]-0.4),fontsize=18)
    if idx <=2:
        plt.annotate(txt_pem[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytext = (pem_entropy_set[idx]+0.2, pem_steps_set[idx]-0.4),fontsize=18)
plt.xlim(0,30)
plt.ylim(19,26)
plt.tick_params(labelsize=16)
plt.ylabel('remaining timesteps',fontsize=20)
plt.xlabel('entropy',fontsize=20)
plt.legend(loc = 'lower right',fontsize=18)
plt.title('Predator Prey B',fontsize=22)
#plt.show()
plt.savefig('D:\Git\IC3Net\Figures\lengthreduceB.pdf', bbox_inches='tight')




steps_1 = np.array([8.28,8.30,8.14,9.54,18.44])
entropy_1 = np.array([1.8,1.16,0.59,0.11])
steps_2 = np.array([23.81,24.34,23.95,59.52,59.99])
entropy_2 = np.array([1.76,1.09,0.32,0.09])
steps_3 = np.array([10.11,10.28,10.48,13.05,14.94])
entropy_3 = np.array([1.56,0.92,0.19,0.04])
steps_4 = np.array([15.65,15.11,16.33,23.1,23.39])
entropy_4 = np.array([1.55,0.89,0.21,0.12])
steps_5 = np.array([0.958,0.966,0.972,0.948,0.956])
entropy_5 = np.array([2.35,1.73,1.15,0.75])
steps_6 = np.array([0.95,0.946,0.954,0.95,0.934])
entropy_6 = np.array([1.92,1.62,1.32,1.06])
step_set = [steps_1,steps_2,steps_3,steps_4,steps_5,steps_6]
entropy_set = [entropy_1,entropy_2,entropy_3,entropy_4,entropy_5,entropy_6]
name_set = ['Treasure Hunt A', 'Treasure Hunt B', 'Predator-Prey A', 'Predator-Prey B', 'Traffic Junction A', 'Traffic Junction B']
xticks = [0,0.125,0.25,0.5,1]
x1 = np.arange(5)
x2 = np.arange(1,5)

#subplotset = [231,232,233,234,235,236]
ylegend = ['timesteps↓','timesteps↓','timesteps↓','timesteps↓','success rates↑','success rates↑']
ylabel = ['timesteps↓','timesteps↓','timesteps↓','timesteps↓','success rates↑','success rates↑']
bar_width = 0.3
for idx in range(6):  
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    bar1 = ax1.bar(x1-0.5*bar_width, step_set[idx],bar_width,bottom = 0, label=ylegend[idx], ec='black',color='#09b0d3')
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x2+0.5*bar_width, entropy_set[idx],bar_width,bottom = 0, label='entropy', ec='black', color='#1d27c9')
    ax1.set_ylabel(ylabel[idx],fontsize=18)
    ax2.set_ylabel('entropy (nat)',fontsize=18)
    ax1.set_xlabel('value of $\Delta$',fontsize=18)
    ax1.set_xticks(ticks=x1)
    ax1.set_xticklabels(xticks)
    lns = bar1 + bar2
    labs = [l.get_label() for l in lns]
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)
    #ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 1))
    #ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.92))
    ax1.set_title(name_set[idx],fontsize=18)
    #plt.tight_layout()
    plt.savefig('D:\Git\IC3Net\TNNLS_Figures\quant'+str(idx)+'.pdf', bbox_inches='tight')
    #plt.show()







x = np.arange(0.5,1.5,0.02)
y1 = 0.5*(1+np.cos(2*np.pi*(x-1)))
y2 = np.ones_like(y1)
y2[0:12] = 0
y2[-12:] = 0
plt.figure()
plt.plot(x,y1,linestyle='dashed',c='#05f8d6',linewidth=2)
plt.plot(x,y2,c='#0082fc',linewidth=2)
plt.xlim(-1.5,1.5)
ax = plt.gca()
ax.set_aspect(1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_linewidth((2))
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth((2))
plt.xticks([-1,0,1],['-1','0','1'],fontsize=20)
plt.yticks([1],['1'],fontsize=20)
#plt.show()
plt.savefig('D:\Git\IC3Net\TNNLS_Figures\Rsn2.png', bbox_inches='tight')


x = np.arange(-1.5,-0.5,0.02)
y1 = 0.5*(1+np.cos(2*np.pi*(x+1)))
y2 = np.ones_like(y1)
y2[0:12] = 0
y2[-12:] = 0
plt.figure()
#plt.bar([1],[1],0.5,color='white', ec='black')
plt.plot(x,y1,linestyle='dashed',c='#05f8d6',linewidth=2)
plt.plot(x,y2,c='#0082fc',linewidth=2)
plt.xlim(-1.5,1.5)
plt.xticks([-1,0,1])
plt.yticks([1])
ax = plt.gca()
ax.set_aspect(1)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_linewidth((2))
ax.spines['left'].set_position(('data',0))
ax.spines['left'].set_linewidth((2))
plt.xticks([-1,0,1],['-1','0','1'],fontsize=20)
plt.yticks([1],['1'],fontsize=20)
#plt.show()
plt.savefig('D:\Git\IC3Net\TNNLS_Figures\Rsn1.png', bbox_inches='tight')
    '''
