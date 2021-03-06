import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


# c_out_v = [0.   ,      0.07843137 ,0.15384615 ,0.22641509, 0.2962963 , 0.36363636,
#  0.42857143, 0.49122807, 0.55172414 ,0.61016949 ,0.66666667 ,0.72131148,
#  0.77419355 ,0.82539683 ,0.875 ,     0.92307692 ,0.96969697 ,1.01492537,
#  1.05882353 ,1.10144928 ,1.14285714 ,1.18309859 ,1.22222222, 1.26027397,
#  1.2972973 , 1.33333333, 1.36842105 ,1.4025974 , 1.43589744 ,1.46835443,
#  1.5   ,     1.5308642 , 1.56097561 ,1.59036145 ,1.61904762 ,1.64705882,
#  1.6744186 , 1.70114943, 1.72727273 ,1.75280899 ,1.77777778 ,1.8021978,
#  1.82608696 ,1.84946237 ,1.87234043 ,1.89473684 ,1.91666667 ,1.93814433,
#  1.95918367, 1.97979798]

c_out_v = np.array([0.   ,      0.07843137 ,0.15384615 ,0.22641509, 0.2962963 , 0.36363636,
  0.42857143, 0.49122807, 0.55172414 ,0.61016949 ,0.66666667 ,0.72131148,
  0.77419355 ,0.82539683 ,0.875 ,     0.92307692 ,0.96969697 ,1.01492537,
  1.05882353 ,1.10144928 ,1.14285714 ,1.18309859 ,1.22222222, 1.26027397,
  1.2972973 , 1.33333333, 1.36842105 ,1.4025974 , 1.43589744 ,1.46835443,
  1.5   ,     1.5308642 , 1.56097561 ,1.59036145 ,1.61904762 ,1.64705882,
  1.6744186 , 1.70114943, 1.72727273 ,1.75280899 ,1.77777778 ,1.8021978,
  1.82608696 ,1.84946237 ,1.87234043 ,1.89473684 ,1.91666667 ,1.93814433,
  1.95918367, 1.97979798])

c_in_v = np.array([4. ,        3.92156863 ,3.84615385 ,3.77358491 ,3.7037037 , 3.63636364,
 3.57142857, 3.50877193, 3.44827586 ,3.38983051 ,3.33333333 ,3.27868852,
 3.22580645 ,3.17460317 ,3.125 ,     3.07692308 ,3.03030303 ,2.98507463,
 2.94117647 ,2.89855072, 2.85714286 ,2.81690141 ,2.77777778 ,2.73972603,
 2.7027027 , 2.66666667 ,2.63157895 ,2.5974026 , 2.56410256 ,2.53164557,
 2.5    ,    2.4691358 , 2.43902439 ,2.40963855 ,2.38095238 ,2.35294118,
 2.3255814 , 2.29885057 ,2.27272727 ,2.24719101 ,2.22222222 ,2.1978022,
 2.17391304, 2.15053763, 2.12765957 ,2.10526316 ,2.08333333, 2.06185567,
 2.04081633, 2.02020202])

# y1 =  [0.72033333, 0.72849765, 0.71281842 ,0.6859244 , 0.66548115, 0.65980316,
#  0.63958495 ,0.60668235, 0.56370041 ,0.53484767 ,0.50542416, 0.47407451,
#  0.44528754 ,0.40303242, 0.36190146, 0.32510289 ,0.27601295, 0.22418913,
#  0.16638154 ,0.06839486 ,0.   ,      0.    ,     0.   ,      0.,
#  0.   ,      0. ,        0. ,        0. ,        0.,         0.,
#  0.   ,      0. ,        0. ,        0. ,        0.,         0.,
#  0.   ,      0.  ,       0. ,        0. ,        0.,         0.,
#  0.  ,       0.,       0. ,        0. ,        0. ,        0.,
#  0.  ,       0.       ]

y1 = np.array([0.72 ,    0.7115037 , 0.69721797 ,0.68881069 ,0.66769404 ,0.6489773,
 0.62032781 ,0.59698091, 0.56040213 ,0.52372882 ,0.48827733 ,0.47085855,
 0.43473613, 0.39461073 ,0.35224935 ,0.31573431, 0.27453571 ,0.22585277,
 0.16512882 ,0.06747151 ,0.  ,       0.    ,     0.   ,      0.,
 0.  ,       0.   ,      0. ,        0.   ,      0. ,        0.,
 0.  ,       0.   ,      0.  ,       0.  ,       0. ,        0.,
 0. ,        0.   ,      0.  ,       0.  ,       0. ,        0.,
 0.  ,       0.   ,      0.  ,       0.   ,      0.    ,     0.,
 0.  ,       0.        ])

y2 = np.array([  0.825, 0.82385875 ,0.82022015 ,0.80892816 ,0.78837799 ,0.76631892,
 0.74197955, 0.72361395, 0.70739084, 0.66827325, 0.64245691 ,0.61547528,
 0.58051129, 0.55533238, 0.52100996 ,0.4834783 , 0.44964875, 0.41732014,
 0.38047714 ,0.34194213, 0.29291144, 0.24036834, 0.18207859 ,0.09232176,
 0.     ,    0. ,        0. ,        0. ,        0.,         0.,
 0.    ,     0. ,        0.  ,       0. ,        0.,         0.,
 0.    ,     0.  ,       0. ,        0. ,        0. ,        0.,
 0.    ,     0.  ,       0. ,        0.  ,       0. ,        0.,
 0.   ,      0.        ])

y3 = np.array([ 0.89, 0.88156312, 0.86768554, 0.86418865, 0.85188915 ,0.84307146,
 0.82889318, 0.80021424 ,0.77493499, 0.74995176, 0.72700059 ,0.69335877,
 0.67713674 ,0.63583551 ,0.61324716 ,0.59019854 ,0.55397461 ,0.5218381,
 0.48667859, 0.45195967 ,0.41850338 ,0.37028584 ,0.33344622 ,0.28867825,
 0.23283835 ,0.16448074, 0.04231173 ,0. ,        0.   ,      0.,
 0.     ,    0. ,        0.  ,       0. ,        0.   ,      0.,
 0.    ,     0. ,        0.  ,       0.  ,       0.  ,       0.,
 0.   ,      0. ,        0. ,        0. ,        0.  ,       0.,
 0.   ,      0.        ])

y4 = np.array([ 0.91 ,0.908524833, 0.9070192 , 0.89401295 ,0.88353168 ,0.86009931,
 0.85599321, 0.82971792 ,0.80839962, 0.77738118, 0.76216876 ,0.74206426,
 0.72183  ,  0.6921377 , 0.66022489 ,0.63531102 ,0.60600035 ,0.5722303,
 0.54709746, 0.50857894 ,0.47459322 ,0.43715214 ,0.40334431 ,0.36081798,
 0.32134674 ,0.2792415 , 0.22368199 ,0.15867584 ,0.03139512 ,0.,
 0.      ,   0.  ,       0.   ,      0. ,        0.      ,   0.,
 0.    ,     0.   ,      0.  ,       0.  ,       0.    ,     0.,
 0.    ,     0.      ,   0.  ,       0.  ,       0.   ,      0.,
 0.    ,     0.        ])

ms = 8
lw = 2
fs = 18
lg = 1.25

fig = plt.figure(figsize = (8,7))

plt.plot(c_out_v / c_in_v, y1, color='orchid', linewidth=3 * lw)
plt.plot(c_out_v / c_in_v, y1, color='indigo', linewidth=lw, label='c=2')
plt.plot(c_out_v / c_in_v, y2, color='#87CEFA', linewidth=3 * lw)
plt.plot(c_out_v / c_in_v, y2, color='b', linewidth=lw, label='c=3')
plt.plot(c_out_v / c_in_v, y3, color='#7FFFAA', linewidth=3 * lw)
plt.plot(c_out_v / c_in_v, y3, color='g', linewidth=lw, label='c=4')
plt.plot(c_out_v / c_in_v, y4, color='#FFDAB9', linewidth=3 * lw)
plt.plot(c_out_v / c_in_v, y4, color='r', linewidth=lw, label='c=5')
plt.title('Nodes predicted with different degree', fontsize=fs);
plt.xlabel(r'$c_{out}/c_{in}$', fontsize=fs + 5)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.ylim(-0.05, 1.05)
plt.tick_params(axis="x", labelsize=fs)
plt.tick_params(axis="y", labelsize=fs)
plt.ylabel('Overlap', fontsize=fs)
plt.grid(linestyle='--', color='dimgrey', alpha=1, linewidth=lg)
plt.legend(fontsize=fs)
plt.savefig('BH2.png')
plt.show()
