import numpy as np
import matplotlib.pyplot as plt
import seaborn
import matplotlib.ticker as ticker
import matplotlib
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
#main experiments



sem_tha = [8.16,8.49,8.48,8.42,9.13]
sem_thb = [19.56,20.16,20.31,20.75,35.95]
sem_ppa = [9.94,10.16,10.20,10.44,10.53]
sem_ppb = [15.30,17.02,17.22,17.55,18.23]
sem_tja = [0.94,0.94,0.94,0.94,0.94]
sem_tjb = [0.89,0.89,0.89,0.89,0.89]

fn_tha = [8.16,8.82,9.57,14.66,19.26]
fn_thb = [19.56,22.33,24.79,33.59,38.05]
fn_ppa = [9.94,10.86,11.36,12.61,14.29]
fn_ppb = [15.30,17.37,19.27,21.40,22.90]
fn_tja = [0.95,0.95,0.3,0.3,0.3]
fn_tjb = [0.89,0.89,0.72,0.72,0.72]

etc_tha = [8.29,10.07,10.67,11.71,14.77]
etc_thb = [20.48,24.31,25.98,39.82,39.64]
etc_ppa = [9.93,12.24,12.38,13.18,15.56]
etc_ppb = [15.25,21.68,22.39,23.22,27.81]
etc_tja = [0.95,0.94,0.94,0.94,0.94]
etc_tjb = [0.844,0.71,0.71,0.71,0.71]

imac_tha = [8.78,8.98,9.00,9.20,12.00]
imac_thb = [33,36.58,36.95,36.81,37.54]
imac_ppa = [10.51,10.60,10.87,12.24,16.56]
imac_ppb = [17.75,17.83,18.62,24.74,28.24]
imac_tja = [0.85,0.85,0.85,0.85,0.85]
imac_tjb = [0.90,0.89,0.89,0.88,0.83]

do_tha = [8.16,10.03,10.89,12.37,15.50]
do_thb = [19.56,23.11,25.28,30.02,33.20]
do_ppa = [9.94,10.85,11.59,12.96,15.19]
do_ppb = [15.30,16.38,17.65,17.75,23.24]
do_tja = [0.95,0.923,0.752,0.503,0.357]
do_tjb = [0.89,0.743,0.711,0.703,0.696]

colors = ['#194f97','#555555','#bd6b08','#00686b','#c82d31']
labels = ['SEM+SBM','FN','ETCNET','IMAC','DO']
markers = ['-s','-v','-*','-P','-X']
#tha
ys = [sem_tha,fn_tha,etc_tha,imac_tha,do_tha]
fig, ax = plt.subplots()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Treasure Hunt A',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/thamain.pdf',bbox_inches='tight')

#thb
ys = [sem_thb,fn_thb,etc_thb,imac_thb,do_thb]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Treasure Hunt B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/thbmain.pdf',bbox_inches='tight')


#ppa
ys = [sem_ppa,fn_ppa,etc_ppa,imac_ppa,do_ppa]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Predator Prey A',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/ppamain.pdf',bbox_inches='tight')

#ppb
ys = [sem_ppb,fn_ppb,etc_ppb,imac_ppb,do_ppb]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('timesteps↓',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="upper left")
plt.title('Predator Prey B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/ppbmain.pdf',bbox_inches='tight')

#tja
ys = [sem_tja,fn_tja,etc_tja,imac_tja,do_tja]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('success rates↑',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="lower left")
plt.title('Traffic Junction A',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/tjamain.pdf',bbox_inches='tight')

#tjb
ys = [sem_tjb,fn_tjb,etc_tjb,imac_tjb,do_tjb]
fig = plt.figure()
x = [1,2,3,4,5]
for idx in range(5):
    plt.plot(x,ys[idx],markers[idx],color=colors[idx],label=labels[idx],lw=3,alpha=0.8,markersize=10)

plt.ylabel('success rates↑',fontsize=17)
#plt.ylim(5,20)
plt.xticks(x,['100%','75%','50%','25%','10%'])
plt.xlabel('bandwidth limits',fontsize=17)
plt.xlim(0.5,5.5)
plt.legend(fontsize=12,loc="lower left")
plt.title('Traffic Junction B',fontsize=20)
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/tjbmain.pdf',bbox_inches='tight')


'''



#quantization experiments
tha_ent = [2.39,1.72,1.09,0.56]
tha_steps = [8.225,8.228,8.355,8.38,12.91]
thb_ent = [2.03,1.38,0.76,0.31]
thb_steps = [19.44,19.51,20.25,24.46,36.15]
ppa_ent = [2.02,1.38,0.69,0.07]
ppa_steps = [9.89,9.99,9.99,10.96,14.51]
ppb_ent = [2.10,1.47,0.47,0.19]
ppb_steps = [15.285,15.11,15.19,21.415,22.075]
tja_ent = [1.82,1.46,1.13,1.02]
tja_steps = [0.94,0.94,0.96,0.94,0.94]
tjb_ent = [1.37,1.16,1.04,0.93]
tjb_steps = [0.89,0.89,0.89,0.89,0.89]


x1 = np.arange(5)
x2 = np.arange(1,5)
bar_width = 0.3
xticks = [0,0.125,0.25,0.5,1]


fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, tha_steps,bar_width,bottom = 0, label='timesteps', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, tha_ent,bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
ax1.set_ylabel('timesteps↓',fontsize=18)
ax2.set_ylabel('entropy↓',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Treasure Hunt A',fontsize=18)
#plt.show()
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/tha_quant.pdf',bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, thb_steps,bar_width,bottom = 0, label='timesteps', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, thb_ent,bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
ax1.set_ylabel('timesteps↓',fontsize=18)
ax2.set_ylabel('entropy↓',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Treasure Hunt B',fontsize=18)
#plt.show()
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/thb_quant.pdf',bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, ppa_steps,bar_width,bottom = 0, label='timesteps', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, ppa_ent,bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
ax1.set_ylabel('timesteps↓',fontsize=18)
ax2.set_ylabel('entropy↓',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Predator Prey A',fontsize=18)
#plt.show()
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/ppa_quant.pdf',bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, ppb_steps,bar_width,bottom = 0, label='timesteps', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, ppb_ent,bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
ax1.set_ylabel('timesteps↓',fontsize=18)
ax2.set_ylabel('entropy↓',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Predator Prey B',fontsize=18)
#plt.show()
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/ppb_quant.pdf',bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, tja_steps,bar_width,bottom = 0, label='success rates', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, tja_ent,bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
ax1.set_ylabel('success rates↑',fontsize=18)
ax2.set_ylabel('entropy↓',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Traffic Junction A',fontsize=18)
#plt.show()
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/tja_quant.pdf',bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, tjb_steps,bar_width,bottom = 0, label='success rates', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, tjb_ent,bar_width,bottom = 0, label='entropy', ec='black', color='#9192ab')
ax1.set_ylabel('success rates↑',fontsize=18)
ax2.set_ylabel('entropy↓',fontsize=18)
ax1.set_xlabel('value of $\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('Traffic Junction B',fontsize=18)
#plt.show()
plt.savefig('C:/Users/1/OneDrive/Paperwork/【TMC2023】/minor2/tjb_quant.pdf',bbox_inches='tight')


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



