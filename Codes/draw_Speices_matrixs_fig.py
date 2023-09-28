# author:Lenovo
# datetime:2023/8/21 20:15
# software: PyCharm
# project:PlantNh-Kcr

import numpy as np
import matplotlib.pyplot as plt


#categories
categories=['Wheat','Tabacum','Rice','Peanut','Papaya','Total']

#labels
x_labels = ['Sn','Sp','ACC', 'MCC', 'AUC']

#colors
colors=['pink','orange','cyan','lime','deepskyblue','b']
# colors=['r','deepskyblue','lime','cyan','orange','y']

values = [

# SN、SP、ACC、MCC、AUC
# Wheat	0.6406	0.8619	0.8140	0.4800	0.8568
# Tabacum	0.7102	0.8885	0.8486	0.5800	0.8960
# Rice	0.4890	0.9393	0.8353	0.4929	0.8857
# Peanut	0.6856	0.9398	0.8883	0.6448	0.9320
# Papaya	0.7091	0.9037	0.8664	0.5883	0.9197
# Total	0.6597	0.9074	0.8560	0.5646	0.9002

    [0.6406,0.8619,0.8140,0.4800,0.8568],#wheat
    [0.7102,0.8885,0.8486,0.5800,0.8960], #Tabacum
    [0.4890,0.9393,0.8353,0.4929,0.8857], #Rice
    [0.6856,0.9398,0.8883,0.6448,0.9320], #Peanut
    [0.7091,0.9037,0.8664,0.5883,0.9197], #Papaya
    [0.6597,0.9074,0.8560,0.5646,0.9002],#Total

]

x_ticks= np.arange(len(x_labels))
x_tick_labels=x_labels
# 绘制柱状图
plt.figure(figsize=(10, 6))
for i in range(len(categories)):
    plt.bar(np.arange(len(x_labels)) + i*0.15, values[i], width=0.15,label=categories[i],color=colors[i])

ax=plt.gca()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(x_ticks + (len(categories)-1)*0.15/2, x_tick_labels)

from matplotlib.pyplot import MultipleLocator

# 设置标题和标签
plt.ylabel('Score')
y_major_locator=MultipleLocator(0.1)
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim((0,1))

# 设置图例横向显示在上方区域
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=len(categories),frameon=False)

#save figure
plt.savefig('../figures/species_ind_test.png')
# 显示柱状图
plt.show()