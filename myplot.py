import numpy as np
import matplotlib.pyplot as plt
import gym
from textwrap import fill

ColorPool = ['cornflowerblue', 'lightgreen', 'lightsteelblue', 'moccasin']

'''
#the following is used for assist experiment 1
#datainput area
steps_1 = 20 - np.array([8.3,8.2,8.3,9.5,18.3])
entropy_1 = np.array([200,125,60,9.8])
steps_2 = 60 - np.array([23.9,23.7,25,59,60])
entropy_2 = np.array([194,121,33,9])
steps_3 = 20 - np.array([10.2,10.4,10.3,13.2,15.2])
entropy_3 = np.array([178,103,21.3,4.9])
steps_4 = 40 - np.array([15.7,15.2,16.2,22.2,23.4])
entropy_4 = np.array([175,99,25,12])
steps_5 = np.array([0.97,0.96,0.96,0.96,0.97])
entropy_5 = np.array([171,128,88,58])
steps_6 = np.array([0.95,0.94,0.94,0.94,0.93])
entropy_6 = np.array([140,118,96,78])
step_set = [steps_1,steps_2,steps_3,steps_4,steps_5,steps_6]
entropy_set = [entropy_1,entropy_2,entropy_3,entropy_4,entropy_5,entropy_6]
name_set = ['Treasure Hunt A', 'Treasure Hunt B', 'Predator-Prey A', 'Predator-Prey B', 'Traffic Junction A', 'Traffic Junction B']
xticks = [0,0.125,0.25,0.5,1]
x1 = np.arange(5)
x2 = np.arange(1,5)

#subplotset = [231,232,233,234,235,236]
ylegend = ['remaining\ntimesteps','remaining\ntimesteps','remaining\ntimesteps','remaining\ntimesteps','success\nrates','success\nrates']
ylabel = ['remaining timesteps','remaining timesteps','remaining timesteps','remaining timesteps','success rates','success rates']
bar_width = 0.3
for idx in range(6):  
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    bar1 = ax1.bar(x1-0.5*bar_width, step_set[idx],bar_width,bottom = 0, label=ylegend[idx],color='#7cd6cf')
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x2+0.5*bar_width, entropy_set[idx],bar_width,bottom = 0, label='entropy',color='#9192ab')
    ax1.set_ylabel(ylabel[idx],fontsize=16)

    ax2.set_ylabel('entropy (nat)',fontsize=16)
    ax1.set_xlabel('value of $\Delta$',fontsize=16)
    ax1.set_xticks(ticks=x1)
    ax1.set_xticklabels(xticks)
    lns = bar1 + bar2
    labs = [l.get_label() for l in lns]
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=14)
    #ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 1))
    #ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.92))
    ax1.set_title(name_set[idx],fontsize=16)
    #plt.tight_layout()
    plt.savefig('D:\Git\IC3Net\Figures\quant'+str(idx)+'.pdf', bbox_inches='tight')
    #plt.show()


#the following is used for reducing message size
raw_entropy_set = np.array([22,12.4,6.7,4.3])
raw_steps_set = 40 - np.array([15.9,15.1,16.2,19.8])
pem_entropy_set = np.array([4.25,2.88,1.24])
pem_steps_set = 40 - np.array([15.05,15.3,16.4])
plt.figure()
plt.subplot(111)
#ax1 = axe.scatter(raw_entropy_set, raw_steps_set, c='deepskyblue')
#ax2 = axe.scatter(pem_entropy_set, pem_steps_set, c='orangered')
plt.scatter(raw_entropy_set, raw_steps_set, c='#7898e1',label='ORI',linewidths=3)
plt.scatter(pem_entropy_set, pem_steps_set, c='#efa666',label='PEM',linewidths=3)

txt_ori = ['ORI, 16', 'ORI, 8', 'ORI, 4', 'ORI, 2']
txt_pem = ['PEM, 16', 'PEM, 8', 'PEM, 4']
for idx in range(len(raw_entropy_set)):
    plt.annotate(txt_ori[idx], xy = (raw_entropy_set[idx], raw_steps_set[idx]), xytext = (raw_entropy_set[idx]+0.2, raw_steps_set[idx]-0.3),fontsize=12)
    if idx <=2:
        plt.annotate(txt_pem[idx], xy = (pem_entropy_set[idx], pem_steps_set[idx]), xytext = (pem_entropy_set[idx]+0.2, pem_steps_set[idx]-0.3),fontsize=12)
plt.xlim(0,30)
plt.ylim(19,26)
plt.ylabel('remaining timesteps',fontsize=16)
plt.xlabel('entropy',fontsize=16)
plt.legend(loc = 'lower right',fontsize=12)
#plt.show()
plt.savefig('D:\Git\IC3Net\Figures\lengthreduce.pdf', bbox_inches='tight')


title_set = ['IC3NET-PEM in PP-A', 'IC3NET-PEM in PP-B', 'IC3NET-PEM in TJ-A', 'IC3NET-PEM in TJ-B', 'IC3NET-PEM in TH-A', 'IC3NET-PEM in TH-B', 'TARMAC-PEM in PP-A', 'TARMAC-PEM in PP-B', 'TARMAC-PEM in TJ-A', 'TARMAC-PEM in TJ-B', 'TARMAC-PEM in TH-A', 'TARMAC-PEM in TH-B']
name_set = ['ic3pemppa','ic3pemppb','ic3pemtja','ic3pemtjb','ic3pemtha','ic3pemthb','tarpemppa','tarpemppb','tarpemtja','tarpemtjb','tarpemtha','tarpemthb']
dataset = []
dataset.append(np.array([0.00000000e+00, 1.53146826e-05, 2.80155902e-04, 1.39689924e-02, 9.72521403e-01, 1.28779419e-02, 3.03393175e-04, 3.27990674e-05, 0.00000000e+00]))
dataset.append(np.array([2.15706476e-05, 4.31056093e-04, 1.55028799e-03, 2.64743746e-02, 9.43811659e-01, 2.56459561e-02, 1.67846248e-03, 3.74627222e-04, 1.20061874e-05]))
dataset.append(np.array([1.59495742e-01, 1.60526953e-02, 8.93828125e-03, 4.42834766e-02, 5.44791055e-01, 5.09656641e-02, 2.57187500e-04, 5.69609375e-03, 1.69519805e-01]))
dataset.append(np.array([0.20190191, 0.00191798, 0.00149872, 0.07313392, 0.48993311, 0.05811698, 0.00436896, 0.00139788, 0.16773055]))
dataset.append(np.array([0.00000000e+00, 7.84236544e-05, 2.42309741e-03, 7.99871834e-02, 8.27814553e-01, 8.49995140e-02, 4.48545942e-03, 2.11768909e-04, 0.00000000e+00]))
dataset.append(np.array([1.66999419e-05, 2.19477606e-03, 1.30176213e-02, 9.09409345e-02, 7.61276802e-01, 1.13649717e-01, 1.55072523e-02, 3.35378671e-03, 4.24099929e-05]))
dataset.append(np.array([0.00000000e+00, 5.43763419e-05, 9.65831329e-04, 9.30569456e-02, 7.91187467e-01, 1.06511305e-01, 8.11308259e-03, 1.10992854e-04, 0.00000000e+00]))
dataset.append(np.array([9.25732913e-06, 3.04067892e-04, 2.58500577e-03, 1.17929473e-01, 6.57462992e-01, 2.19320497e-01, 2.11730248e-03, 2.59446439e-04, 1.19580516e-05]))
dataset.append(np.array([4.70003438e-01, 4.06145833e-04, 1.76614583e-03, 9.27208333e-03, 2.30692708e-02, 1.11094792e-02, 4.39229167e-03, 5.14270833e-04, 4.79466875e-01]))
dataset.append(np.array([0.27683781, 0.07811721, 0.08018112, 0.04814846, 0.02906424, 0.05922534, 0.06316805, 0.11542781, 0.24982995]))
dataset.append(np.array([0.00487007, 0.02007273, 0.05984069, 0.17501647, 0.42603862, 0.20906011, 0.03807667, 0.04477016, 0.02225448]))
dataset.append(np.array([3.65660962e-04, 1.00656170e-02, 2.58982936e-02, 1.29526830e-01, 6.47330787e-01, 1.47149555e-01, 2.62890058e-02, 1.24676503e-02, 9.06599593e-04]))


xset = np.arange(9)

for idx in range(12):
    plt.figure()
    plt.bar(xset,dataset[idx],1,bottom=0,color = '#63b2ee')
    plt.ylabel('probability',fontsize=16)
    plt.ylim(0,1)
    plt.xlabel('reconstruction points',fontsize=16)
    plt.xticks(xset,labels=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
    plt.title(title_set[idx])
    plt.savefig('D:\Git\IC3Net\Figures\distribution_'+name_set[idx]+'.pdf', bbox_inches='tight')
    #plt.show()

'''


title_set = ['IC3NET-ORI in PP-A', 'IC3NET-ORI in PP-B', 'IC3NET-ORI in TJ-A', 'IC3NET-ORI in TJ-B', 'IC3NET-ORI in TH-A', 'IC3NET-ORI in TH-B', 'TARMAC-ORI in PP-A', 'TARMAC-ORI in PP-B', 'TARMAC-ORI in TJ-A', 'TARMAC-ORI in TJ-B', 'TARMAC-ORI in TH-A', 'TARMAC-ORI in TH-B']
name_set = ['ic3orippa','ic3orippb','ic3oritja','ic3oritjb','ic3oritha','ic3orithb','tarorippa','tarorippb','taroritja','taroritjb','taroritha','tarorithb']
dataset = []
dataset.append(np.array([8.52507160e-05, 2.93874160e-03, 1.07700406e-02, 1.24122191e-01 ,7.17929564e-01, 1.31785541e-01, 9.86431505e-03, 2.46964860e-03, 3.47076144e-05]))
dataset.append(np.array([0.0023842,  0.00807539, 0.01439627, 0.13091714, 0.70600621, 0.11452867, 0.01457515, 0.00723294, 0.00188405]))
dataset.append(np.array([0.16991109, 0.02037086, 0.01942613, 0.09055953, 0.41227332, 0.09678141, 0.01691961, 0.02003527, 0.15372277]))
dataset.append(np.array([0.19982733, 0.00544474, 0.00732121, 0.08343657, 0.42475255, 0.08895726, 0.00991636, 0.00696706, 0.17337692]))
dataset.append(np.array([1.15794890e-05, 3.23210151e-03, 2.88837587e-02, 1.78454280e-01, 5.37193740e-01, 2.08469714e-01, 3.77100529e-02, 5.87154470e-03, 1.73228555e-04]))
dataset.append(np.array([1.61661866e-04, 6.65649250e-03, 2.31230653e-02, 1.54474156e-01, 6.09662762e-01, 1.68736022e-01, 2.78488083e-02, 9.00274387e-03, 3.34288033e-04]))
dataset.append(np.array([0.00509865, 0.01480316, 0.0517002,  0.22061981, 0.38277391, 0.24670127, 0.05803671, 0.01559188, 0.00467442]))
dataset.append(np.array([0.00378952, 0.01308105, 0.03958712, 0.20198536, 0.40561395, 0.24477246, 0.05750206, 0.02417157, 0.0094969 ]))
dataset.append(np.array([0.48524573, 0.00415177, 0.00464354, 0.01018427, 0.01641396, 0.01418719, 0.00622687, 0.00527073, 0.45367594]))
dataset.append(np.array([0.18572672, 0.10282349, 0.08096247, 0.06999437, 0.08730044, 0.11237034, 0.07701253, 0.08544432, 0.19836531]))
dataset.append(np.array([0.00386729, 0.026317,   0.06435445, 0.18602838, 0.32363837, 0.24006365, 0.08675517, 0.04301758, 0.02595811]))
dataset.append(np.array([0.01120228, 0.04745614, 0.07616783, 0.19099994, 0.32701317, 0.19480843, 0.08327309, 0.05393314, 0.01514598]))


xset = np.arange(9)

for idx in range(12):
    plt.figure()
    plt.bar(xset,dataset[idx],1,bottom=0,color = '#f8cb7f')
    plt.ylabel('probability',fontsize=16)
    plt.ylim(0,1)
    plt.xlabel('reconstruction points',fontsize=16)
    plt.xticks(xset,labels=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
    plt.title(title_set[idx])
    plt.savefig('D:\Git\IC3Net\Figures\distribution_'+name_set[idx]+'.pdf', bbox_inches='tight')
