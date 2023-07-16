import numpy as np
import matplotlib.pyplot as plt
import seaborn
import matplotlib.ticker as ticker
#main experiments




sem_thb = [21.6,23.2,25.2,29.2]
sem_ppb = [16.4,16.9,17.4,18.9]
sem_tjb = [0.69,0.69,0.69,0.69]


fn_thb = [36.9,36.8,37,37]
fn_ppb = [20.9,21.1,21.6,23]
fn_tjb = [0.72,0.72,0.72,0.72]


etc_thb = [25.6,26.8,28.4,32.5]
etc_ppb = [22.5,22.9,23.5,25.1]
etc_tjb = [0.7,0.7,0.7,0.7]

imac_thb = [33.3,33.8,34.3,35.4]
imac_ppb = [18.6,18.6,18.8,19.6]
imac_tjb = [0.73,0.73,0.73,0.73]


colors = ['#194f97','#555555','#bd6b08','#00686b']
labels = ['SEM+SBM','FN','ETCNET','IMAC']

#thb
ys = [sem_thb,fn_thb,etc_thb,imac_thb]
fig = plt.figure()
x = [1,2,3,4]
for idx in range(4):
    plt.plot(x,ys[idx],'-o',color=colors[idx],label=labels[idx],lw=2.5)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['0.1','0.2','0.3','0.5'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,4.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Treasure Hunt B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/71436/OneDrive/Paperwork/【TMC2023】/revision/thbdrop.pdf',bbox_inches='tight')

#ppb
ys = [sem_ppb,fn_ppb,etc_ppb,imac_ppb]
fig = plt.figure()
x = [1,2,3,4]
for idx in range(4):
    plt.plot(x,ys[idx],'-o',color=colors[idx],label=labels[idx],lw=2.5)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['0.1','0.2','0.3','0.5'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,4.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Predator Prey B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/71436/OneDrive/Paperwork/【TMC2023】/revision/ppbdrop.pdf',bbox_inches='tight')

#tjb
ys = [sem_tjb,fn_tjb,etc_tjb,imac_tjb]
fig = plt.figure()
x = [1,2,3,4]
for idx in range(4):
    plt.plot(x,ys[idx],'-o',color=colors[idx],label=labels[idx],lw=2.5)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['0.1','0.2','0.3','0.5'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,4.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Traffic Junction B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/71436/OneDrive/Paperwork/【TMC2023】/revision/tjbdrop.pdf',bbox_inches='tight')


