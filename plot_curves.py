import numpy as np
import torch
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#PPA plot
name_set = ['SEM-25%','SEM-10%','MIM-25%','MIM-10%']
color_set = ['#05f8d6','#0082fc','#eab026','#ac2026']
steps_mean_set = []
steps_up_set = []
steps_down_set = []

models_a_1 = ['models_final/pp_easy/mlpD/model_150','tnnls_models/PPAbar/model_00202']
models_a_2 = ['models_final/pp_easy/mlpE/model_150','tnnls_models/PPAbar/model_00202_1']
models_a_3 = ['models_final/pp_easy/mlpF/model_150','tnnls_models/PPAbar/model_00202_2']
models_a_4 = ['models_final/pp_easy/mlpA/model_150','tnnls_models/PPAbar/model_00202_3']
models_a_5 = ['models/pp_easy/mlpC/model','models_final/pp_easy/mlpC/model_50','tnnls_models/PPAbar/model_00202_4']

models_b_1 = ['models_final/pp_easy/mlpD/model_150','tnnls_models/PPAbar/model_005008']
models_b_2 = ['models_final/pp_easy/mlpE/model_150','tnnls_models/PPAbar/model_005008_1']
models_b_3 = ['models_final/pp_easy/mlpF/model_150','tnnls_models/PPAbar/model_005008_2']
models_b_4 = ['models_final/pp_easy/mlpA/model_150','tnnls_models/PPAbar/model_005008_3']
models_b_5 = ['models/pp_easy/mlpC/model','models_final/pp_easy/mlpC/model_50','tnnls_models/PPAbar/model_01008_4']

models_c_1 = ['models_final/pp_easy/mimD/model_100','models_ultimate/pp_easy_ic3net/mimC/model_50','tnnls_models/PPAmim/model_003']
models_c_2 = ['models_final/pp_easy/mimE/model_100','models_ultimate/pp_easy_ic3net/mimA/model_50','tnnls_models/PPAmim/model_004_1']
models_c_3 = ['models_final/pp_easy/mimF/model_100','models_ultimate/pp_easy_ic3net/mimB/model_50','tnnls_models/PPAmim/model_001_2']
models_c_4 = ['models_final/pp_easy/mimA/model_100','models_ultimate/pp_easy_ic3net/mimD/model_50','tnnls_models/PPAmim/model_003_3']
models_c_5 = ['tnnls_models/PPAmim/base_model','tnnls_models/PPAmim/model_002_6']

models_d_1 = ['models_final/pp_easy/mimD/model_100','models_ultimate/pp_easy_ic3net/mimC/model_50','tnnls_models/PPAmim/model_004']
models_d_2 = ['models_final/pp_easy/mimE/model_100','models_ultimate/pp_easy_ic3net/mimA/model_50','tnnls_models/PPAmim/model_002_1']
models_d_3 = ['models_final/pp_easy/mimF/model_100','models_ultimate/pp_easy_ic3net/mimB/model_50','tnnls_models/PPAmim/model_002_2']
models_d_4 = ['models_final/pp_easy/mimA/model_100','models_ultimate/pp_easy_ic3net/mimD/model_50','tnnls_models/PPAmim/model_004_3']
models_d_5 = ['tnnls_models/PPAmim/base_model','tnnls_models/PPAmim/model_002_4']


models_a = [models_a_1,models_a_2,models_a_3,models_a_4,models_a_5]
models_b = [models_b_1,models_b_2,models_b_3,models_b_4,models_b_5]
models_c = [models_c_1,models_c_2,models_c_3,models_c_4,models_c_5]
models_d = [models_d_1,models_d_2,models_d_3,models_d_4,models_d_5]

meta_model_set = [models_a,models_b,models_c,models_d]

for model_set in meta_model_set:
    steps_set = None
    for mini_model_set in model_set:
        mini_step_set = None
        for model in mini_model_set:
            d = torch.load(model)
            steps = np.array((d['log'])['steps_taken'].data).reshape(1,-1)
            if mini_step_set is None:
                mini_step_set = steps
            else: 
                mini_step_set = np.concatenate((mini_step_set,steps),axis=1)
            mini_step_set = mini_step_set[:,0:200]
        if steps_set is None:
            steps_set = mini_step_set
        else:
            steps_set = np.concatenate((steps_set,mini_step_set),axis=0)
    steps_mean = np.mean(steps_set,axis=0)
    steps_std = np.std(steps_set,axis=0)
    steps_up = steps_mean + steps_std
    steps_down = steps_mean - steps_std

    steps_mean_set.append(steps_mean)
    steps_up_set.append(steps_up)
    steps_down_set.append(steps_down)

plt.figure()
data_num = steps_set.shape[1]
x = np.arange(data_num)
for idx in range(4):
    datamean = steps_mean_set[idx]
    dataup = steps_up_set[idx]
    datadown = steps_down_set[idx]
    plt.plot(x,datamean,label=name_set[idx],c=color_set[idx])
    plt.fill_between(x, dataup, datadown, facecolor=color_set[idx], alpha=0.3)
plt.ylabel('timesteps',fontsize=18)
plt.xlabel('training epochs',fontsize=18)
plt.xticks([0,50,100,150,200],['0','500','1000','1500','2000'],fontsize=18)
plt.legend(fontsize=16,loc="lower left")
plt.title('PPA-timesteps',fontsize=22)
plt.savefig('/home/nfs_data/yulebin/IC3NET/figs/PPA_time.pdf', bbox_inches='tight')



#PPB plot
name_set = ['SEM-50%','SEM-25%','MIM-50%','MIM-25%']
color_set = ['#05f8d6','#0082fc','#eab026','#ac2026']
steps_mean_set = []
steps_up_set = []
steps_down_set = []

models_a_1 = ['models_final/pp_hard/mlpD/model_150','tnnls_models/PPBbar/model_00505']
models_a_2 = ['models_final/pp_hard/mlpE/model_150','tnnls_models/PPBbar/model_00505_1']
models_a_3 = ['models_final/pp_hard/mlpF/model_150','tnnls_models/PPBbar/model_00505_1']
models_a_4 = ['models_final/pp_hard/mlpA/model_150','tnnls_models/PPBbar/model_00505_3']
models_a_5 = ['models_old/pp_medium/mlpE/model_100','models/pp_medium/mlpF/model_50','tnnls_models/PPBbar/model_00505_4']

models_b_1 = ['models_final/pp_hard/mlpD/model_150','tnnls_models/PPBbar/model_02025']
models_b_2 = ['models_final/pp_hard/mlpE/model_150','tnnls_models/PPBbar/model_02025_1']
models_b_3 = ['models_final/pp_hard/mlpF/model_150','tnnls_models/PPBbar/model_02025_2']
models_b_4 = ['models_final/pp_hard/mlpA/model_150','tnnls_models/PPBbar/model_02025_3']
models_b_5 = ['models_old/pp_medium/mlpE/model_100','models/pp_medium/mlpF/model_50','tnnls_models/PPBbar/model_02025_4']

models_c_1 = ['models_app/PPBmim_1','tnnls_models/PPBmim/model_002']
models_c_2 = ['models_app/PPBmim_2','tnnls_models/PPBmim/model_002_1']
models_c_3 = ['models_app/PPBmim_3','tnnls_models/PPBmim/model_002_2']
models_c_4 = ['models_app/PPBmim_4','tnnls_models/PPBmim/model_002_3']
models_c_5 = ['models_app/PPBmim_5','tnnls_models/PPBmim/model_002_4']

models_d_1 = ['models_app/PPBmim_1','tnnls_models/PPBmim/model_004']
models_d_2 = ['models_app/PPBmim_2','tnnls_models/PPBmim/model_004_1']
models_d_3 = ['models_app/PPBmim_3','tnnls_models/PPBmim/model_004_2']
models_d_4 = ['models_app/PPBmim_4','tnnls_models/PPBmim/model_003_3']
models_d_5 = ['models_app/PPBmim_5','tnnls_models/PPBmim/model_003_4']


models_a = [models_a_1,models_a_2,models_a_3,models_a_4,models_a_5]
models_b = [models_b_1,models_b_2,models_b_3,models_b_4,models_b_5]
models_c = [models_c_1,models_c_2,models_c_3,models_c_4,models_c_5]
models_d = [models_d_1,models_d_2,models_d_3,models_d_4,models_d_5]

meta_model_set = [models_a,models_b,models_c,models_d]

for model_set in meta_model_set:
    steps_set = None
    for mini_model_set in model_set:
        mini_step_set = None
        for model in mini_model_set:
            d = torch.load(model)
            steps = np.array((d['log'])['steps_taken'].data).reshape(1,-1)
            if mini_step_set is None:
                mini_step_set = steps
            else: 
                mini_step_set = np.concatenate((mini_step_set,steps),axis=1)
            mini_step_set = mini_step_set[:,0:200]
        if steps_set is None:
            steps_set = mini_step_set
        else:
            steps_set = np.concatenate((steps_set,mini_step_set),axis=0)
    steps_mean = np.mean(steps_set,axis=0)
    steps_std = np.std(steps_set,axis=0)
    steps_up = steps_mean + steps_std
    steps_down = steps_mean - steps_std

    steps_mean_set.append(steps_mean)
    steps_up_set.append(steps_up)
    steps_down_set.append(steps_down)

plt.figure()
data_num = steps_set.shape[1]
x = np.arange(data_num)
for idx in range(4):
    datamean = steps_mean_set[idx]
    dataup = steps_up_set[idx]
    datadown = steps_down_set[idx]
    plt.plot(x,datamean,label=name_set[idx],c=color_set[idx])
    plt.fill_between(x, dataup, datadown, facecolor=color_set[idx], alpha=0.3)
plt.ylabel('timesteps',fontsize=18)
plt.xlabel('training epochs',fontsize=18)
plt.xticks([0,50,100,150,200],['0','500','1000','1500','2000'],fontsize=18)
plt.legend(fontsize=16,loc="lower left")
plt.title('PPB-timesteps',fontsize=22)
plt.savefig('/home/nfs_data/yulebin/IC3NET/figs/PPB_time.pdf', bbox_inches='tight')


#THA plot
name_set = ['SEM-25%','SEM-10%','MIM-25%','MIM-10%']
color_set = ['#05f8d6','#0082fc','#eab026','#ac2026']
steps_mean_set = []
steps_up_set = []
steps_down_set = []

models_a_1 = ['models_final/th_easy/mlpA/model_100','tnnls_models/THAbar/model_0203']
models_a_2 = ['models_final/th_easy/mlpC/model_100','tnnls_models/THAbar/model_0203_1']
models_a_3 = ['models_final/th_easy/mlpD/model_100','tnnls_models/THAbar/model_0203_2']
models_a_4 = ['models_final/th_easy/mlpE/model_100','tnnls_models/THAbar/model_0203_3']
models_a_5 = ['models_final/th_easy/mlpF/model_100','tnnls_models/THAbar/model_0203_4']

models_b_1 = ['models_final/th_easy/mlpA/model_100','tnnls_models/THAbar/model_02013']
models_b_2 = ['models_final/th_easy/mlpC/model_100','tnnls_models/THAbar/model_05013_1']
models_b_3 = ['models_final/th_easy/mlpE/model_100','tnnls_models/THAbar/model_05013_2']
models_b_4 = ['models_final/th_easy/mlpE/model_100','tnnls_models/THAbar/model_05013_3']
models_b_5 = ['models_final/th_easy/mlpF/model_100','tnnls_models/THAbar/model_05013_4']

models_c_1 = ['models_final/th_easy/mimB/model_100','tnnls_models/THAmim/model_008']
models_c_2 = ['models_final/th_easy/mimC/model_100','tnnls_models/THAmim/model_006_1']
models_c_3 = ['models_final/th_easy/mimD/model_100','tnnls_models/THAmim/model_01_2']
models_c_4 = ['models_final/th_easy/mimE/model_100','tnnls_models/THAmim/model_008_3']


models_d_1 = ['models_final/th_easy/mimB/model_100','tnnls_models/THAmim/model_01_1']
models_d_2 = ['models_final/th_easy/mimC/model_100','tnnls_models/THAmim/model_012_2']
models_d_3 = ['models_final/th_easy/mimD/model_100','tnnls_models/THAmim/model_01_3']
models_d_4 = ['models_final/th_easy/mimE/model_100','tnnls_models/THAmim/model_01_4']


models_a = [models_a_1,models_a_2,models_a_3,models_a_4,models_a_5]
models_b = [models_b_1,models_b_2,models_b_3,models_b_4,models_b_5]
models_c = [models_c_1,models_c_2,models_c_3,models_c_4]
models_d = [models_d_1,models_d_2,models_d_3,models_d_4]

meta_model_set = [models_a,models_b,models_c,models_d]

for model_set in meta_model_set:
    steps_set = None
    for mini_model_set in model_set:
        mini_step_set = None
        for model in mini_model_set:
            d = torch.load(model)
            steps = np.array((d['log'])['steps_taken'].data).reshape(1,-1)
            if mini_step_set is None:
                mini_step_set = steps
            else: 
                mini_step_set = np.concatenate((mini_step_set,steps),axis=1)
            mini_step_set = mini_step_set[:,0:200]
        if steps_set is None:
            steps_set = mini_step_set
        else:
            steps_set = np.concatenate((steps_set,mini_step_set),axis=0)
    steps_mean = np.mean(steps_set,axis=0)
    steps_std = np.std(steps_set,axis=0)
    steps_up = steps_mean + steps_std
    steps_down = steps_mean - steps_std

    steps_mean_set.append(steps_mean)
    steps_up_set.append(steps_up)
    steps_down_set.append(steps_down)

plt.figure()
data_num = steps_set.shape[1]
x = np.arange(data_num)
for idx in range(4):
    datamean = steps_mean_set[idx]
    dataup = steps_up_set[idx]
    datadown = steps_down_set[idx]
    plt.plot(x,datamean,label=name_set[idx],c=color_set[idx])
    plt.fill_between(x, dataup, datadown, facecolor=color_set[idx], alpha=0.3)
plt.ylabel('timesteps',fontsize=18)
plt.xlabel('training epochs',fontsize=18)
plt.xticks([0,50,100,150,200],['0','500','1000','1500','2000'],fontsize=18)
plt.legend(fontsize=16,loc="lower left")
plt.title('THA-timesteps',fontsize=22)
plt.savefig('/home/nfs_data/yulebin/IC3NET/figs/THA_time.pdf', bbox_inches='tight')


#THB plot
name_set = ['SEM-50%','SEM-25%','MIM-50%','MIM-25%']
color_set = ['#05f8d6','#0082fc','#eab026','#ac2026']
steps_mean_set = []
steps_up_set = []
steps_down_set = []

models_a_1 = ['models_final/th_hard/mlpA/model_100','tnnls_models/THBbar/model_0205']
models_a_2 = ['models_final/th_hard/mlpC/model_100','tnnls_models/THBbar/model_0205_1']
models_a_3 = ['models_final/th_hard/mlpD/model_100','tnnls_models/THBbar/model_0205_2']
models_a_4 = ['models_final/th_hard/mlpE/model_100','tnnls_models/THBbar/model_0205_3']
models_a_5 = ['models_final/th_hard/mlpF/model_100','tnnls_models/THBbar/model_0205_4']

models_b_1 = ['models_final/th_hard/mlpA/model_100','tnnls_models/THBbar/model_05025']
models_b_2 = ['models_final/th_hard/mlpC/model_100','tnnls_models/THBbar/model_05025_1']
models_b_3 = ['models_final/th_hard/mlpD/model_100','tnnls_models/THBbar/model_05025_2']
models_b_4 = ['models_final/th_hard/mlpE/model_100','tnnls_models/THBbar/model_05025_3']
models_b_5 = ['models_final/th_hard/mlpF/model_100','tnnls_models/THBbar/model_05025_4']

models_c_1 = ['models_final/th_hard/mimA/model_100','tnnls_models/THBmim/model_002']
models_c_2 = ['models_final/th_hard/mimB/model_100','tnnls_models/THBmim/model_002_1']
models_c_3 = ['models_final/th_hard/mimD/model_100','tnnls_models/THBmim/model_001_3']
models_c_4 = ['models_final/th_hard/mimE/model_100','tnnls_models/THBmim/model_001_4']
models_c_5 = ['tnnls_models/THBmim/base_model_100','tnnls_models/THBmim/model_002']

models_d_1 = ['models_final/th_hard/mimA/model_100','tnnls_models/THBmim/model_004']
models_d_2 = ['models_final/th_hard/mimB/model_100','tnnls_models/THBmim/model_004_1']
models_d_3 = ['models_final/th_hard/mimD/model_100','tnnls_models/THBmim/model_002_3']
models_d_4 = ['models_final/th_hard/mimE/model_100','tnnls_models/THAmim/model_008_4']
models_d_5 = ['tnnls_models/THBmim/base_model_100','tnnls_models/THBmim/model_001_6']


models_a = [models_a_1,models_a_2,models_a_3,models_a_4,models_a_5]
models_b = [models_b_1,models_b_2,models_b_3,models_b_4,models_b_5]
models_c = [models_c_1,models_c_2,models_c_3,models_c_4,models_c_5]
models_d = [models_d_1,models_d_2,models_d_3,models_d_4,models_d_5]

meta_model_set = [models_a,models_b,models_c,models_d]

for model_set in meta_model_set:
    steps_set = None
    for mini_model_set in model_set:
        mini_step_set = None
        for model in mini_model_set:
            d = torch.load(model)
            steps = np.array((d['log'])['steps_taken'].data).reshape(1,-1)
            if mini_step_set is None:
                mini_step_set = steps
            else: 
                mini_step_set = np.concatenate((mini_step_set,steps),axis=1)
            mini_step_set = mini_step_set[:,0:200]
        if steps_set is None:
            steps_set = mini_step_set
        else:
            steps_set = np.concatenate((steps_set,mini_step_set),axis=0)
    steps_mean = np.mean(steps_set,axis=0)
    steps_std = np.std(steps_set,axis=0)
    steps_up = steps_mean + steps_std
    steps_down = steps_mean - steps_std

    steps_mean_set.append(steps_mean)
    steps_up_set.append(steps_up)
    steps_down_set.append(steps_down)

plt.figure()
data_num = steps_set.shape[1]
x = np.arange(data_num)
for idx in range(4):
    datamean = steps_mean_set[idx]
    dataup = steps_up_set[idx]
    datadown = steps_down_set[idx]
    plt.plot(x,datamean,label=name_set[idx],c=color_set[idx])
    plt.fill_between(x, dataup, datadown, facecolor=color_set[idx], alpha=0.3)
plt.ylabel('timesteps',fontsize=18)
plt.xlabel('training epochs',fontsize=18)
plt.xticks([0,50,100,150,200],['0','500','1000','1500','2000'],fontsize=18)
plt.legend(fontsize=16,loc="lower left")
plt.title('THB-timesteps',fontsize=22)
plt.savefig('/home/nfs_data/yulebin/IC3NET/figs/THB_time.pdf', bbox_inches='tight')

#TJB plot
name_set = ['SEM-50%','SEM-25%','MIM-50%','MIM-25%']
color_set = ['#05f8d6','#0082fc','#eab026','#ac2026']
steps_mean_set = []
steps_up_set = []
steps_down_set = []



models_a_1 = ['tnnls_models/TJBbar/model_base_a','tnnls_models/TJBbar/model_00505_5']
models_a_2 = ['models_final/tj_hard/mlpD/model_1000', 'tnnls_models/TJBbar/model_00505_2']
models_a_3 = ['models_final/tj_hard/mlpA/model_1000','tnnls_models/TJBbar/model_00505_3']

models_b_1 = ['tnnls_models/TJBbar/model_base_a','tnnls_models/TJBbar/model_005025_5']
models_b_2 = ['models_final/tj_hard/mlpD/model_1000','tnnls_models/TJBbar/model_00505_2']
models_b_3 = ['models_final/tj_hard/mlpA/model_1000','tnnls_models/TJBbar/model_005025_3']

models_c_1 = ['models_final/tj_hard/mimA/model_500','tnnls_models/TJBmim/model_test_0001_1']
models_c_2 = ['models_final/tj_hard/mimD/model_500','tnnls_models/TJBmim/model_test_0001_2']
models_c_3 = ['models/tj_medium/mimA/model_500','tnnls_models/TJBmim/model_test_0001_3']

models_d_1 = ['models_final/tj_hard/mimA/model_500','tnnls_models/TJBmim/model_test_0005_1']
models_d_2 = ['models_final/tj_hard/mimD/model_500','tnnls_models/TJBmim/model_test_0005_2']
models_d_3 = ['models/tj_medium/mimA/model_500','tnnls_models/TJBmim/model_test_0005_3']




models_a = [models_a_1,models_a_2,models_a_3]
models_b = [models_b_1,models_b_2,models_b_3]
models_c = [models_c_1,models_c_2,models_c_3]
models_d = [models_d_1,models_d_2,models_d_3]

meta_model_set = [models_a,models_b,models_c,models_d]

for model_set in meta_model_set:
    steps_set = None
    for mini_model_set in model_set:
        mini_step_set = None
        for model in mini_model_set:
            d = torch.load(model)
            steps = np.array((d['log'])['success'].data).reshape(1,-1)
            if mini_step_set is None:
                mini_step_set = steps
            else: 
                mini_step_set = np.concatenate((mini_step_set,steps),axis=1)
            mini_step_set = mini_step_set[:,0:1250]
        if steps_set is None:
            steps_set = mini_step_set
        else:
            steps_set = np.concatenate((steps_set,mini_step_set),axis=0)
    steps_mean = np.mean(steps_set,axis=0)
    steps_std = np.std(steps_set,axis=0)
    steps_up = steps_mean + steps_std
    steps_down = steps_mean - steps_std

    steps_mean_set.append(steps_mean)
    steps_up_set.append(steps_up)
    steps_down_set.append(steps_down)

plt.figure()
data_num = steps_set.shape[1]
x = np.arange(data_num)
for idx in range(4):
    datamean = steps_mean_set[idx]
    dataup = steps_up_set[idx]
    datadown = steps_down_set[idx]
    plt.plot(x,datamean,label=name_set[idx],c=color_set[idx])
    plt.fill_between(x, dataup, datadown, facecolor=color_set[idx], alpha=0.3)
plt.ylabel('success rates',fontsize=18)
plt.xlabel('training epochs',fontsize=18)
plt.xticks([0,250,500,750,1000,1250],['0','2500','5000','7500','10000','12500'],fontsize=18)
plt.legend(fontsize=16,loc="lower left")
plt.title('TJB-success rates',fontsize=22)
plt.savefig('/home/nfs_data/yulebin/IC3NET/figs/TJB_suc.pdf', bbox_inches='tight')

'''
#TJA plot
name_set = ['SEM-50%','SEM-25%','MIM-50%','MIM-25%']
color_set = ['#05f8d6','#0082fc','#eab026','#ac2026']
steps_mean_set = []
steps_up_set = []
steps_down_set = []

models_a_1 = ['tnnls_models/TJAbar/model_0108_4']
models_a_2 = ['tnnls_models/TJAbar/model_0108_conti_a_1000', 'tnnls_models/TJAbar/model_0108_final']
models_a_3 = ['tnnls_models/TJAbar/model_0108_conti_c']
models_a_4 = ['tnnls_models/TJAbar/model_0108_5']

models_b_1 = ['tnnls_models/TJAbar/model_0104_1']
models_b_2 = ['tnnls_models/TJAbar/model_0104_2']
models_b_3 = ['tnnls_models/TJAbar/model_0104_3']
models_b_4 = ['tnnls_models/TJAbar/model_0104_4']
models_b_5 = ['tnnls_models/TJAbar/model_0104_5']

models_c_1 = ['models_final/tj_easy/mimC/model_500','tnnls_models/TJAmim/model_test_00005_1']
models_c_2 = ['models_final/tj_easy/mimA/model_500','tnnls_models/TJAmim/model_test_0001_2']
models_c_3 = ['models_final/tj_easy/mimB/model_500','tnnls_models/TJAmim/model_test_0002_3']
models_c_4 = ['models/tj_easy/mimA/model_500','tnnls_models/TJAmim/model_test_00005_5']
models_c_5 = ['models_final/tj_easy/mimD/model_500', 'tnnls_models/TJAmim/model_test_00005_4']

models_d_1 = ['models_final/tj_easy/mimC/model_500','tnnls_models/TJAmim/model_test_0005_1']
models_d_2 = ['models_final/tj_easy/mimA/model_500','tnnls_models/TJAmim/model_test_0005_2']
models_d_3 = ['models_final/tj_easy/mimB/model_500','tnnls_models/TJAmim/model_test_0005_3']
models_d_4 = ['models/tj_easy/mimA/model_500','tnnls_models/TJAmim/model_test_0002_4']
models_d_5 = ['models_final/tj_easy/mimD/model_500','tnnls_models/TJAmim/model_test_0005_5']


models_a = [models_a_1,models_a_2,models_a_3,models_a_4]
models_b = [models_b_1,models_b_2,models_b_3,models_b_4,models_b_5]
models_c = [models_c_1,models_c_2,models_c_3,models_c_4,models_c_5]
models_d = [models_d_1,models_d_2,models_d_3,models_d_4,models_d_5]

meta_model_set = [models_a,models_b,models_c,models_d]

for model_set in meta_model_set:
    steps_set = None
    for mini_model_set in model_set:
        mini_step_set = None
        for model in mini_model_set:
            d = torch.load(model)
            steps = np.array((d['log'])['success'].data).reshape(1,-1)
            if mini_step_set is None:
                mini_step_set = steps
            else: 
                mini_step_set = np.concatenate((mini_step_set,steps),axis=1)
            mini_step_set = mini_step_set[:,0:1250]
        if steps_set is None:
            steps_set = mini_step_set
        else:
            steps_set = np.concatenate((steps_set,mini_step_set),axis=0)
    steps_mean = np.mean(steps_set,axis=0)
    steps_std = np.std(steps_set,axis=0)
    steps_up = steps_mean + steps_std
    steps_down = steps_mean - steps_std

    steps_mean_set.append(steps_mean)
    steps_up_set.append(steps_up)
    steps_down_set.append(steps_down)

plt.figure()
data_num = steps_set.shape[1]
x = np.arange(data_num)
for idx in range(4):
    datamean = steps_mean_set[idx]
    dataup = steps_up_set[idx]
    datadown = steps_down_set[idx]
    plt.plot(x,datamean,label=name_set[idx],c=color_set[idx])
    plt.fill_between(x, dataup, datadown, facecolor=color_set[idx], alpha=0.3)
plt.ylabel('success rates',fontsize=18)
plt.xlabel('training epochs',fontsize=18)
plt.xticks([0,250,500,750,1000,1250],['0','2500','5000','7500','10000','12500'],fontsize=18)
plt.legend(fontsize=16,loc="lower left")
plt.title('TJA-success rates',fontsize=22)
plt.savefig('/home/nfs_data/yulebin/IC3NET/figs/TJA_suc.pdf', bbox_inches='tight')
'''