import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


cora_data = [70.18874019381634, 72.36286919831224, 75.13846153846154, 77.25258493353027, 78.78228782287823, 79.05781057810579, 81.12177121771218]
citeseer_data = [51.77986476333584, 53.52984113353371, 54.58187280921381, 56.030769230769226, 55.09992486851991, 56.15615615615616, 58.5045045045045]
pubmed_data = [78.17294281729428, 78.40168091580931, 79.5875243005663, 79.39953342123948, 79.54697603651579, 79.41176470588235, 80.50202839756592]

dcsbm_cora = [[77.1725888324873, 77.54176280572219, 76.28011075219197], [78.11603375527426, 78.64345991561181, 79.59282700421942], [79.1076923076923, 79.41538461538461, 80.84615384615384], [80.42836041358936, 81.75775480059085, 81.6440177252585], [81.19557195571956, 81.47232472324724, 82.39483394833948], [82.91881918819189, 82.28782287822878, 83.02583025830258], [82.65682656826569, 83.02583025830258, 83.57933579335793]]
dcsbm_citeseer = [[55.438016528925615, 55.73854244928626, 56.07663410969196], [56.16702447402319, 56.16702447402319, 57.16702447402319], [56.19028542814221, 57.34051076614922, 57.44066099148723], [58.35336538461539, 59.254807692307686, 59.036057692307686], [59.25544703230654, 60.08189331329827, 60.30728775356874], [60.46046046046046, 60.76076076076076, 61.76176176176176], [60.06006006006006, 61.810810810810814, 62.111111111111114]]
dcsbm_pubmed = [[78.6,78.2,78.1],[78.5,78.7,79.1],[78.5,78.9,79.1],[79.4,79.2,78.8],[79.5,79.2,79.1],[79.7,79.3,79.1],[79.5,79.8,80.0]]

cora_midium = []
citeseer_midium = []
pubmed_midium = []

for i in range(len(dcsbm_cora)):
    a = (dcsbm_cora[i][0]+dcsbm_cora[i][1]+dcsbm_cora[i][2])/3
    cora_midium.append(a)
    b = (dcsbm_citeseer[i][0]+dcsbm_citeseer[i][1]+dcsbm_citeseer[i][2])/3
    citeseer_midium.append(b)
    c = (dcsbm_pubmed[i][0]+dcsbm_pubmed[i][1]+dcsbm_pubmed[i][2])/3
    pubmed_midium.append(c)
print(cora_midium)
print(citeseer_midium)

x = [1, 2, 3, 4, 5, 6, 7]
xticklabes = ['20%', '30%', '40%', '50%', '60%', '70%', '80%']
plt.plot(x, cora_data, c='#87CEEB',linewidth = 4)
plt.plot(x, cora_midium, c='#87CEEB',linewidth = 4)
plt.plot(x, cora_data, c='#00008B',linewidth = 1.5,label = "cora")
plt.plot(x, cora_midium, c='#00008B',linestyle='--',linewidth = 1.5)

plt.plot(x, citeseer_data, c='#C2E27C',linewidth = 4)
plt.plot(x, citeseer_midium, c='#C2E27C',linewidth = 4)
plt.plot(x, citeseer_data, c='#006400',linewidth = 1.5,label = "citeseer")
plt.plot(x, citeseer_midium, c='#006400',linestyle='--',linewidth = 1.5)

plt.plot(x, pubmed_data, c='#FFDEAD',linewidth = 4)
plt.plot(x, pubmed_midium, c='#FFDEAD',linewidth = 4)
plt.plot(x, pubmed_data, c='#FF4500',linewidth = 1.5,label = "pubmed")
plt.plot(x, pubmed_midium, c='#FF4500',linestyle='--',linewidth = 1.5)
dt1 = pd.DataFrame({
    'a': dcsbm_cora[0],
    'b': dcsbm_cora[1],
    'c': dcsbm_cora[2],
    'd': dcsbm_cora[3],
    'e': dcsbm_cora[4],
    'f': dcsbm_cora[5],
    'g': dcsbm_cora[6]
})
dt2 = pd.DataFrame({
    'a': dcsbm_citeseer[0],
    'b': dcsbm_citeseer[1],
    'c': dcsbm_citeseer[2],
    'd': dcsbm_citeseer[3],
    'e': dcsbm_citeseer[4],
    'f': dcsbm_citeseer[5],
    'g': dcsbm_citeseer[6]
})
dt3 = pd.DataFrame({
    'a': dcsbm_pubmed[0],
    'b': dcsbm_pubmed[1],
    'c': dcsbm_pubmed[2],
    'd': dcsbm_pubmed[3],
    'e': dcsbm_pubmed[4],
    'f': dcsbm_pubmed[5],
    'g': dcsbm_pubmed[6]
})
plt.boxplot(dt1, widths=0.1, patch_artist=True, boxprops={'color': '#00008B', 'facecolor': '#00008B'},
            medianprops={'linestyle': '--', 'color': '#00008B'})
plt.boxplot(dt2, widths=0.1, patch_artist=True, boxprops={'color': '#006400', 'facecolor': '#006400'},
            medianprops={'linestyle': '--', 'color': '#006400'})
plt.boxplot(dt3, widths=0.1, patch_artist=True, boxprops={'color': 'r', 'facecolor': 'r'},
            medianprops={'linestyle': '--', 'color': 'r'})
plt.title("The training results of ProNE", fontsize='xx-large')
plt.xlabel("Training node ratio")
plt.ylabel("Accuracy")
plt.xticks(x, xticklabes)
def to_percent(temp, position):
  return '%1.0f'%(1*temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.legend()
plt.grid(linestyle='-.')
plt.savefig('box2.png')
plt.show()