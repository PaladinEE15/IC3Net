import numpy as np
import matplotlib.pyplot as plt
#import gym
#from textwrap import fill

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



#the following is used for reducing message size
raw_entropy_set = np.array([20,11,5.99,3.26])
raw_steps_set = 20 - np.array([10.35,10.54,10.23,10.26])
pem_entropy_set = np.array([2.81,1.6,0.53])
pem_steps_set = 20 - np.array([10.59,10.7,11.17])
plt.figure()
#plt.subplot(111)
#ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
#ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')
plt.scatter(raw_entropy_set, raw_steps_set, c='#9192ab',label='ORI',linewidths=3)
plt.scatter(pem_entropy_set, pem_steps_set, c='#7cd6cf',label='PEM',linewidths=3)
txt_ori = ['ORI, 16', 'ORI, 8', 'ORI, 4', 'ORI, 2']
txt_pem = ['PEM, 16', 'PEM, 8', 'PEM, 4']
for idx in range(len(raw_entropy_set)):
    plt.annotate(txt_ori[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytext = (raw_entropy_set[idx]+0.2, raw_steps_set[idx]-0.13),fontsize=12)
    if idx <=2:
        plt.annotate(txt_pem[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytext = (pem_entropy_set[idx]+0.2, pem_steps_set[idx]-0.13),fontsize=12)
plt.xlim(0,25)
plt.ylim(7.5,10.5)
plt.ylabel('remaining timesteps',fontsize=16)
plt.xlabel('entropy',fontsize=16)
plt.legend(loc = 'lower right',fontsize=12)
#plt.show()
plt.title('Predator Prey A',fontsize=18)
plt.savefig('E:\lengthreduceA.pdf', bbox_inches='tight')


title_set = ['Distribution of digit 0','Distribution of digit 1','Distribution of digit 2','Distribution of digit 3']
name_set = ['ppb4mlp0', 'ppb4mlp1', 'ppb4mlp2', 'ppb4mlp3']
dataset = []
dataset.append(np.array([0.0064990259628353725, 0.01810794178381246, 0.13040584279333292, 0.1879038130212624, 0.24274426036115604, 0.17208707681430896, 0.1971659370328116, 0.03087791764166766, 0.014208184588812633, ]))
dataset.append(np.array([0.007253720942187772, 0.016750492196120226, 0.09538639060347498, 0.26985276293472, 0.22258704435110355, 0.2153695645284115, 0.15067983188224768, 0.014148867361844277, 0.007971325199890042, ]))
dataset.append(np.array([0.011656214720292957, 0.02115859110757057, 0.09906299463340698, 0.2367368062277564, 0.22188654908932598, 0.2489464341186666, 0.14060673656716688, 0.013079248885549863, 0.006866424650263758, ]))
dataset.append(np.array([0.003891854817505162, 0.009338410745518808, 0.05583064748339237, 0.3528437014759584, 0.3154339055446483, 0.19677412458283097, 0.03344980545424783, 0.02070535792749324, 0.011732191968404932, ]))




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
'''
xset = [1,2,3,4,5]
dataset = 20-np.array([10.30,10.60,11.56,11.44,15.40])
plt.figure()
plt.bar(xset,dataset,0.5,bottom=0,color = '#9192ab')
plt.ylabel('remaining timesteps',fontsize=18)
plt.xlabel('entropy limits',fontsize=18)
plt.xticks(xset,labels=['100%','25%','10%','5%','2%'],fontsize=14)
plt.title('Predator Prey A',fontsize=22)
#plt.show()
plt.savefig('E:\entropy_limit_A.pdf', bbox_inches='tight')

xset = [1,2,3,4,5]
dataset = 40-np.array([17.41,18.66,17.20,20.59,26.13])
plt.figure()
plt.bar(xset,dataset,0.5,bottom=0,color = '#9192ab')
plt.ylabel('remaining timesteps',fontsize=18)
plt.xlabel('entropy limits',fontsize=18)
plt.xticks(xset,labels=['100%','50%','25%','10%','2%'],fontsize=14)
plt.title('Predator Prey B',fontsize=22)
#plt.show()
plt.savefig('E:\entropy_limit_B.pdf', bbox_inches='tight')
