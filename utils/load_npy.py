# -*- coding:utf-8 -*-

# @Filename: load_npy
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-09 10:35
# @Author  : Linshan

import numpy as np
data = np.load('/media/hlf/Luffy/WLS/LoveDA/Unsupervised_Domian_Adaptation/log/try/2urban/confusion_matrix-1638800456.9436584.npy')
# print(data)
# print(data/data.sum(0))
d = data/data.sum(0)
d = np.around(d, decimals=2)
print(d)