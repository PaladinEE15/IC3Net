import numpy as np
import matplotlib.pyplot as plt
import seaborn
'''
main experiments
'''
'''
sem_tha = [8.16,8.49,8.48,8.42,9.13]
sem_thb = [19.56,20.16,20.31,20.75,35.95]
sem_ppa = [9.94,10.46,10.20,10.44,10.03]
sem_ppb = [15.30,17.98,17.22,17.55,18.23]
sem_tja = [0.94,0.94,0.94,0.94,0.94]
sem_tjb = [0.89,0.89,0.89,0.89,0.89]

fn_tha = [8.16,8.82,9.57,14.66,19.26]
fn_thb = [19.56,22.33,24.79,33.59,38.05]
fn_ppa = [9.94,10.86,11.36,12.61,14.29]
fn_ppb = [15.30,17.37,19.27,21.40,22.90]
fn_tja = [0.95,0.95,0.3,0.3,0.3]
fn_tjb = [0.89,0.89,0.72,0.72,0.72]

etc_tha = [8.29,10.07,10.67,11.71,14.77]
etc_thb = [20.48,25.98,24.31,39.82,39.64]
etc_ppa = [9.93,12.38,12.24,13.18,15.56]
etc_ppb = [15.25,22.70,22.39,21.68,27.81]
etc_tja = [0.95,0.94,0.94,0.94,0.94]
etc_tjb = [0.844,0.71,0.71,0.71,0.71]

imac_tha = [8.78,8.98,9.00,9.20,12.00]
imac_thb = [36.58,33,36.95,36.81,37.54]
imac_ppa = [10.87,10.60,10.51,12.24,16.56]
imac_ppb = [17.75,15.63,18.62,24.74,28.24]
imac_tja = [0.85,0.85,0.85,0.85,0.85]
imac_tjb = [0.90,]

do_tha = [8.16,10.03,10.89,12.37,15.50]
do_thb = [19.56,23.11,25.28,30.02,33.20]
do_ppa = [9.94,10.85,11.59,12.96,15.19]
do_ppb = [15.30,16.38,17.65,17.75,23.24]
do_tja = [0.95,0.923,0.752,0.503,0.357]
do_tjb = [0.89,0.743,0.711,0.696,0.703]

colors = ['#63b2ee','#76da91','#f8cb7f','#f89588','#9192ab']
labels = ['SEM','FN','ETC','IMAC','DO']
#tha
ys = [sem_tha,fn_tha,etc_tha,imac_tha,do_tha]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],'-o',color=colors[idx],label=labels[idx],lw=2)

plt.ylabel('timesteps↓',fontsize=18)
plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=18)
plt.xlim(0.5,5.5)
plt.legend(fontsize=14,loc="upper left")
plt.title('Treasure Hunt A',fontsize=22)
plt.show()
'''

'''
quantization experiments
tha_ent = []
tha_steps = []
thb_ent = []
thb_steps = []
ppa_ent = []
ppa_steps = []
ppb_ent = []
ppb_steps = []
tja_ent = []
tja_steps = []
tjb_ent = []
tjb_steps = []


x1 = np.arange(5)
x2 = np.arange(1,5)
bar_width = 0.3
xticks = [0,0.125,0.25,0.5,1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, tha_steps[idx],bar_width,bottom = 0, label='timesteps', ec='black',color='#d5d6d8')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, tha_ent[idx],bar_width,bottom = 0, label='entropy', ec='black', color='#555555')
ax1.set_ylabel('timesteps↓',fontsize=18)
ax2.set_ylabel('entropy↓(nat)',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Trasure Hunt A',fontsize=18)
plt.show()

'''
'''
hot map plot
'''

'''
sem
predator locs: [[3 4]
 [1 1]
 [0 2]]
prey locs: [[2 2]]
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,


predator locs: [[2 2]
 [4 0]
 [2 0]]
prey locs: [[2 2]]
timestep: 5
0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.25,0.0,0.0,0.0,-0.25,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,


new env info - predator locs: [[4 2]
 [0 2]
 [2 3]]
new env info - prey locs: [[2 1]]
timestep: 2
0.0,0.25,-0.25,0.25,0.0,0.5,0.0,-0.25,0.25,0.5,0.0,0.0,0.0,0.5,-0.5,0.0,0.0,-0.25,0.25,0.0,0.0,-0.25,0.0,0.0,0.25,0.25,0.0,0.0,-0.25,-0.25,0.0,0.0
-0.25,0.0,0.0,-0.25,0.0,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,-0.25,0.0,0.25,-0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,0.0,-0.25
0.0,-0.25,0.0,-0.25,0.0,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,-0.25,0.0,0.25,-0.25,0.0,0.0,0.25,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,0.0,-0.25

new env info - predator locs: [[3 1]
 [1 1]
 [2 1]]
new env info - prey locs: [[2 1]]
timestep: 4
0.0,0.0,-0.25,-0.25,-0.25,0.0,0.25,0.25,0.0,0.0,-0.25,0.0,0.25,0.0,0.0,0.0,0.0,0.25,-0.25,0.25,0.25,0.0,-0.25,0.0,0.0,0.25,0.0,0.0,0.0,0.0,0.25,-0.25
0.0,0.0,0.0,0.0,0.0,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,0.0,0.0,0.25,-0.25,0.25,0.0,0.25,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,0.0,-0.25
0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,0.0,0.0,0.25,-0.25,0.25,0.0,0.25,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.0,0.0,-0.25


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
plt.subplots_adjust(wspace=-0.8, hspace=0.2)
data1 = np.array([0.0,0.0,-0.25,-0.25,-0.25,0.0,0.25,0.25,0.0,0.0,-0.25,0.0,0.25,0.0,0.0,0.0,0.0,0.25,-0.25,0.25,0.25,0.0,-0.25,0.0,0.0,0.25,0.0,0.0,0.0,0.0,0.25,-0.25]).reshape((16,2))
data2 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,0.0,0.0,0.25,-0.25,0.25,0.0,0.25,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,0.0,-0.25]).reshape((16,2))
data3 = np.array([0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,0.0,0.0,0.25,-0.25,0.25,0.0,0.25,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.0,0.0,-0.25]).reshape((16,2))

seaborn.heatmap(data1, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax1, square=True, cbar=False)
seaborn.heatmap(data2, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax2, square=True, cbar=False)
seaborn.heatmap(data3, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax3, square=True, cbar=False)
#seaborn.heatmap(data,center=0,cmap='bwr',vmin=-1,vmax=1,square=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('agent 1')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('agent 2')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('agent 3')
cbar_ax = fig.add_axes([0.70, 0.12, 0.02, 0.75])
_ = plt.colorbar(ax3.collections[0], cax=cbar_ax)
#plt.show()
plt.savefig('D:/Git/IC3Net/raw_2.pdf',bbox_inches='tight')


fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
plt.subplots_adjust(wspace=-0.8, hspace=0.2)
data1 = np.array([0.0,0.25,-0.25,0.25,0.0,0.5,0.0,-0.25,0.25,0.5,0.0,0.0,0.0,0.5,-0.5,0.0,0.0,-0.25,0.25,0.0,0.0,-0.25,0.0,0.0,0.25,0.25,0.0,0.0,-0.25,-0.25,0.0,0.0]).reshape((16,2))
data2 = np.array([-0.25,0.0,0.0,-0.25,0.0,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,-0.25,0.0,0.25,-0.25,0.25,0.25,0.0,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,0.0,-0.25]).reshape((16,2))
data3 = np.array([0.0,-0.25,0.0,-0.25,0.0,0.0,0.25,0.25,-0.25,0.0,-0.25,0.0,0.25,-0.25,0.25,-0.25,0.0,0.25,-0.25,0.0,0.0,0.25,0.0,0.0,0.0,0.0,-0.25,0.0,0.25,0.25,0.0,-0.25]).reshape((16,2))

seaborn.heatmap(data1, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax1, square=True, cbar=False)
seaborn.heatmap(data2, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax2, square=True, cbar=False)
seaborn.heatmap(data3, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax3, square=True, cbar=False)
#seaborn.heatmap(data,center=0,cmap='bwr',vmin=-1,vmax=1,square=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('agent 1')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('agent 2')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('agent 3')
cbar_ax = fig.add_axes([0.70, 0.12, 0.02, 0.75])
_ = plt.colorbar(ax3.collections[0], cax=cbar_ax)
#plt.show()
plt.savefig('D:/Git/IC3Net/raw_1.pdf',bbox_inches='tight')

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
plt.subplots_adjust(wspace=-0.8, hspace=0.2)
data1 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((16,2))
data2 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((16,2))
data3 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((16,2))

seaborn.heatmap(data1, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax1, square=True, cbar=False)
seaborn.heatmap(data2, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax2, square=True, cbar=False)
seaborn.heatmap(data3, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax3, square=True, cbar=False)
#seaborn.heatmap(data,center=0,cmap='bwr',vmin=-1,vmax=1,square=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('agent 1')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('agent 2')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('agent 3')
cbar_ax = fig.add_axes([0.70, 0.12, 0.02, 0.75])
_ = plt.colorbar(ax3.collections[0], cax=cbar_ax)
#plt.show()
plt.savefig('D:/Git/IC3Net/sem_1.pdf',bbox_inches='tight')

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
plt.subplots_adjust(wspace=-0.8, hspace=0.2)
data3 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.25,0.0,0.0,0.0,-0.25,-0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((16,2))
data2 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((16,2))
data1 = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).reshape((16,2))

seaborn.heatmap(data1, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax1, square=True, cbar=False)
seaborn.heatmap(data2, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax2, square=True, cbar=False)
seaborn.heatmap(data3, center=0, cmap='Blues', vmin=-1, vmax=1,  ax=ax3, square=True, cbar=False)
#seaborn.heatmap(data,center=0,cmap='bwr',vmin=-1,vmax=1,square=True)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('agent 1')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('agent 2')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel('agent 3')
cbar_ax = fig.add_axes([0.70, 0.12, 0.02, 0.75])
_ = plt.colorbar(ax3.collections[0], cax=cbar_ax)
#plt.show()
plt.savefig('D:/Git/IC3Net/sem_2.pdf',bbox_inches='tight')
'''



