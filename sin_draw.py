import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.axisartist as axisartist


x_1 = np.array([0,0.4,0.8,1.2,1.6,2.0])
x_2 = np.array([-0.4,0,0.4,0.8,1.2,1.6,2.0])
x_3 = np.array([0.4,0.8,1.2,1.6,0,2.0])
y_1 = np.sin(np.pi*x_1+np.pi*0.1)
y_2 = np.sin(np.pi*x_2+np.pi*0.5)
y_3 = np.sin(np.pi*x_3-np.pi*0.3)
x1 = np.linspace(0, 2, 200, endpoint=True)
x2 = np.linspace(-0.5, 2, 250, endpoint=True)
x3 = np.linspace(0, 2, 200, endpoint=True)
y1 = np.sin(np.pi*x1+np.pi*0.1)
y2 = np.sin(np.pi*x2+np.pi*0.5)
y3 = np.sin(np.pi*x3-np.pi*0.3)
plt.scatter(x_1,y_1,marker='^',color='r')
plt.scatter(x_2,y_2,marker='^',color='r')
plt.scatter(x_3,y_3,marker='^',color='r')
plt.plot(x1,y1,label=r'$\sin(\pi t+0.1\pi)$')
plt.plot(x2,y2,label = r'$\sin(\pi t+0.5\pi)$')
plt.plot(x3,y3,label = r'$\sin(\pi t-0.3\pi)$')
plt.axvline(0.4,linestyle = '-.', color = 'k',linewidth = 0.7)
plt.axvline(0.8,linestyle = '-.', color = 'k',linewidth = 0.7)
plt.text(0.45,1.1,r'$\Delta\theta=72°$')
plt.xlim(-0.5,2)
plt.legend(loc='upper right')


ax = plt.gca()  # get current axis 获得坐标轴对象

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')  # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴 指定左边的边为 y 轴

ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
ax.spines['left'].set_position(('data', 0))



plt.show()