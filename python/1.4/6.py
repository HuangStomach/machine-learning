# %matplotlib inline
import matplotlib
matplotlib.use('MacOSX')

import numpy as np

from matplotlib import pyplot as plt
# 在-10和10之间生成一个100个数的数列
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker = "x")
plt.show()
