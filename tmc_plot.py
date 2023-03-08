import numpy as np
import matplotlib.pyplot as plt

#main experiments
sem_tha = [8.52,8.49,8.48,8.42,8.50]
sem_thb = [20.64,20.16,20.31,21.04,35.95]
sem_ppa = [10.06,10.46,10.20,10.44,11.31]
sem_ppb = [15.15,17.98,17.22,17.55,18.23]
sem_tja = []
sem_tjb = []

fn_tha = [8.52,8.82,9.57,14.66,19.26]
fn_thb = [20.64,22.33,24.79,33.59,38.05]
fn_ppa = [10.06,10.86,11.36,12.61,14.29]
fn_ppb = [15.15,17.37,19.27,21.40,22.90]
fn_tja = [0.95,0.95,0.3,0.3,0.3]
fn_tjb = [0.89,0.89,0.72,0.72,0.72]

etc_tha = [8.29,10.07,10.67,11.71,14.77]
etc_thb = [20.48,25.98,24.31,39.82,39.64]
etc_ppa = [9.93,12.38,12.24,13.18,15.56]
etc_ppb = [15.25,22.70,22.39,21.68,27.81]
etc_tja = []
etc_tjb = []

imac_tha = [8.78,8.98,9.00,9.20,12.00]
imac_thb = [36.58,33,36.95,36.81,37.54]
imac_ppa = [10.87,10.60,10.51,12.24,16.56]
imac_ppb = [17.75,15.63,18.62,24.74,28.24]
imac_tja = []
imac_tjb = []

do_tha = [8.52,10.03,10.89,12.37,15.50]
do_thb = [20.64,23.11,25.28,30.02,33.20]
do_ppa = [10.06,10.85,11.59,12.96,15.19]
do_ppb = [15.15,16.38,17.65,17.75,23.24]
do_tja = []
do_tjb = []

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