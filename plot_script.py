import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch 
import numpy as np
import sys
import glob


'''
suggestions for colors:
raw: yellow, gold, orange, goldenrod
reduce_layer: green, palegreen, darkgreen
my_method: darkturquoise, skyblue, deepskyblue, blue, royalblue, c, dodgerblue, darkcyan
'''

#parameter set part

train_file_path = []
legends = []
xlabel = 'epochs'
ylabel = 'success rate'
color_set = []
content_name = 'comm_entropy'
fig_path = None
title = 'aa'


#main part

plot_data = []
for i, file in enumerate(train_file_path):
    d = torch.load(file)
    plot_data = np.asarray(d['log'][content_name].data)
    plt.plot(np.arange(len(plot_data)),plot_data,c=color_set[i],label=legends[i])
plt.title(title,fontsize=18)
plt.ylabel(ylabel,fontsize=13)
plt.xlabel(xlabel,fontsize=13)
plt.legend(fontsize=13,loc="lower left")
if fig_path is not None:
    plt.savefig(fig_path, bbox_inches='tight')
plt.show()

