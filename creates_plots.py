'''
Create plots for speed optimization result
'''

import matplotlib.pyplot as plt
import numpy as np

x = ['output1', 'output2']
base_acc = [0.8845, 0.9588]
base_time = 998.1831
base_gpu1 = [4872746752, 7204740608]
base_gpu2 = [592120320, 5063051776]

gpu1_acc = [0.8617, 0.9494]
gpu1_time = 1249.436
gpu1_mem = [4873275136, 7090025728]

noXLA_acc = [0.8466, 0.9433]
noXLA_time = 1285.3337
noXLA_gpu1 = [4869491456, 7621772800]
noXLA_gpu2 = [599273984, 5484239360]

f32_acc = [0.8466, 0.9433]
f32_time = 1322.106
f32_gpu1 = [4869113600, 7609946112]
f32_gpu2 = [589446656, 5490722816]

ind = np.arange(2)
ind2 = [0.3, 1.0]
time = [base_time, f32_time]
mem1 = [base_gpu1[0], f32_gpu1[0]]
mem2 = [base_gpu2[0], f32_gpu2[0]]
l = ['float16', 'float32']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.bar(ind, base_acc, width=0.4, color='darkorange')
ax1.bar(ind+0.4, f32_acc, width=0.4, color='gold')
ax1.set_xticks(ind+0.2) 
ax1.set_xticklabels(x)
ax1.legend(l,loc='lower right')
ax1.set_title('Accuracy')
ax2.bar(ind2, time, width=0.5, color='orange', align='center')
ax2.set_xticks(ind2) 
ax2.set_xticklabels(l)
ax2.set_title('Time')
ax3.bar(ind2, mem1, width=0.5, color='gold', align='center')
ax3.bar(ind2, mem2, bottom=mem1, width=0.5, color='darkorange', align='center')
ax3.set_xticks(ind2) 
ax3.set_xticklabels(l)
ax3.legend(['GPU1', 'GPU2'],loc='lower right')
ax3.set_title('Memory usage')
 
plt.tight_layout()
plt.show()