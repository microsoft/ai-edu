from turtle import color
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plot
import matplotlib as mpl
import matplotlib.colors as mcolors

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False
colors = list(mcolors.TABLEAU_COLORS.keys())

a = np.random.beta(10, 20,size=(5))
print(a)
plt.scatter(a, np.zeros_like(a), color=mcolors.TABLEAU_COLORS[colors[1]])

a = np.random.beta(105.3,32.5,size=(5))
print(a)
plt.scatter(a, np.zeros_like(a), color=mcolors.TABLEAU_COLORS[colors[2]])


# 定义一组alpha 跟 beta值
alpha_beta_values = [[50,50], [10,20], [105.3,32.5], [1,1] ]
markers = ['.', 'o', 'v', 's', 'x', '>', 'p', '*']

lines = ["-", "--", "-.", ":"]  # 线条风格
# 定义 x 值
x = np.linspace(0, 1, 1002)[1:-1]
for i, alpha_beta_value in enumerate(alpha_beta_values):
  print(alpha_beta_value)
  dist = beta(alpha_beta_value[0], alpha_beta_value[1])
  dist_y = dist.pdf(x)
  # 添加图例
  # plot.legend('alpha=')
  # 创建 beta 曲线
  plot.plot(x, dist_y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (alpha_beta_value[0], alpha_beta_value[1]), linestyle=lines[i])

# 设置标题
plot.title(u'Beta分布')
# 设置 x,y 轴取值范围
plot.xlim(0, 1)
#plot.ylim(0, 2.5)
plot.legend()
plot.grid()
plot.show()
