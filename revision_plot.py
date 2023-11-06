import numpy as np
import matplotlib.pyplot as plt
import seaborn
import matplotlib.ticker as ticker
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
#main experiments




sem_thb = [20.3,21.6,23.2,25.2,29.2]
sem_ppb = [17.2,17.3,17.3,17.4,18.9]
sem_tjb = [0.89,0.72,0.72,0.72,0.72]


fn_thb = [24.8,36.9,36.8,37,37]
fn_ppb = [19.3,20.9,21.1,21.6,23]
fn_tjb = [0.718,0.718,0.718,0.718,0.718]

etc_thb = [25.6,25.6,26.8,28.4,32.5]
etc_ppb = [22.3,22.5,22.9,23.5,25.1]
etc_tjb = [0.71,0.71,0.71,0.71,0.71]

imac_thb = [36.6,36.7,36.8,36.8,36.8]
imac_ppb = [18.6,18.6,18.6,18.8,19.6]
imac_tjb = [0.89,0.712,0.712,0.712,0.712]

do_thb = [25.28,25.30,26.10,27.50,30.02]
do_ppb = [17.65,17.66,17.69,17.70,17.75]
do_tjb = [0.708,0.708,0.708,0.70,0.70]
markers = ['-s','-v','-*','-P','-X']
colors = ['#194f97','#555555','#bd6b08','#00686b','#c82d31']
labels = ['SEM+SBM','FN','ETCNET','IMAC','DO']

#thb
ys = [sem_thb,fn_thb,etc_thb,imac_thb,do_thb]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('timesteps↓',fontsize=17)
plt.ylim(20,40)
plt.xticks(x,['0','0.1','0.2','0.3','0.5'])
plt.xlabel('message loss probability',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Treasure Hunt B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/isaac/OneDrive/Paperwork/【TMC2023】/minor2/thbdrop.pdf',bbox_inches='tight')

#ppb
ys = [sem_ppb,fn_ppb,etc_ppb,imac_ppb,do_ppb]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('timesteps↓',fontsize=17)
plt.ylim(16,28)
plt.xticks(x,['0','0.1','0.2','0.3','0.5'])
plt.xlabel('message loss probability',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Predator Prey B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/isaac/OneDrive/Paperwork/【TMC2023】/minor2/ppbdrop.pdf',bbox_inches='tight')

#tjb
ys = [sem_tjb,fn_tjb,etc_tjb,imac_tjb,do_tjb]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('success rates↑',fontsize=17)
plt.ylim(0.68,0.9)
plt.xticks(x,['0','0.1','0.2','0.3','0.5'])
plt.xlabel('message loss probability',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Traffic Junction B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.savefig('C:/Users/isaac/OneDrive/Paperwork/【TMC2023】/minor2/tjbdrop.pdf',bbox_inches='tight')


