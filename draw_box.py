import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


cora_data = [78.63405629903092, 80.90717299578058, 82.49630723781388, 82.65682656826569, 83.63076923076923, 86.10086100861008, 86.53136531365314]
citeseer_data = [58.414725770097675, 60.62687848862173, 61.84276414621933, 62.980769230769226, 62.05860255447032,  63.813813813813816, 65.26526526526526,]
pubmed_data = [80.17294281729428, 82.40168091580931, 85.5875243005663, 86.39953342123948, 86.54697603651579, 87.41176470588235, 87.50202839756592]

dcsbm_cora = [[78.58790955237656, 80.54937701892017, 79.18781725888326], [80.90717299578058, 80.64345991561181, 80.09071729957806], [81.41538461538461, 80.98461538461538, 80.40000000000001], [82.6440177252585, 82.57016248153619, 82.27474150664698,], [83.21033210332104, 83.11808118081181, 84.47232472324724], [85.77982779827798, 85.65682656826569, 84.41082410824109], [87.68634686346863, 85.57933579335793, 86.21033210332104]]
dcsbm_citeseer = [[58.04012021036814, 58.75281743050338, 58.79038317054845], [61.49806784027479, 60.455130957492486, 60.41219407471018], [61.79268903355033, 61.942914371557336, 61.24261392088132], [62.620192307692314, 62.379807692307686, 62.25961538461539], [61.98347107438017, 62.43425995492111, 62.73478587528174], [63.76376376376376, 62.96296296296296, 62.56256256256256], [63.213213213213216, 64.01201201201201, 64.56156156156156]]
dcsbm_pubmed = [[77.6,78.2,79.1],[84.5,81.7,81.1],[83.5,85.9,83.1],[85.4,84.2,84.8],[85.5,83.2,84.1],[85.7,86.3,85.1],[85.5,86.8,84.0]]

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
plt.title("The training results of NetMF", fontsize='xx-large')
plt.xlabel("Training node ratio")
plt.ylabel("Accuracy")
plt.xticks(x, xticklabes)
def to_percent(temp, position):
  return '%1.0f'%(1*temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.legend()
plt.grid(linestyle='-.')
plt.savefig('box1.png')
plt.show()