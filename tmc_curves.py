import numpy as np
import torch
import os
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
model = ['models/th/sem/n3_100_s1','models/th/sem/n5_100_s1','models/pp/sem/n5_100_s2','models/pp/sem/n3_100_s1','models/pp/sem/n3_10_s1','models/th/sem/n3_10_s1','models/th/sem/n5_25_s2','models/pp/sem/n5_10_s10_q_01']

for item in model:
    print('check:',item)
    d = torch.load(item)
    the_log = (d['log'])['comm_entropy'].data
    print(the_log)
