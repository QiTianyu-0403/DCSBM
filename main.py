from models import data_to_G, SBM, DCSBM, power_law
from get_func import p_random, p_random_simple, zipf, get_weight, statistic_nodes, statistic_edges,\
                    get_ave_degree, get_data_ave_degree
from draw import draw_degree, draw, draw_data_x, draw_degree_fit_power,\
                    draw_degree_fit_line, draw_old, draw_network
from graph import init
from genetic import genetic_func
from BH_model import BH_draw
import numpy as np
import networkx as nx

if __name__ == '__main__':
    data, degree = init()

    real_degree = get_data_ave_degree(degree)
    G_real = data_to_G(data)
    draw_old(G_real)
    #P,dict = statistic_edges(data)
    #G2 = DCSBM(sizes=[dict[0], dict[1],dict[2],dict[3],dict[4],dict[5],dict[6]], p=P, theta=get_weight(degree), sparse=True)
    #G2 = SBM(sizes=[dict[0], dict[1],dict[2],dict[3],dict[4],dict[5],dict[6]], p=P, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)
    #G=data_to_G(data)
    #draw_old(G2)
    #draw_degree_fit_line(G)



    '''
    *******   Generate DCSBM from random sequence   ************
    '''
    '''
    power=[]
    for i in range(1,10001):
        power.append(zipf(i,450,1.3))

    random.shuffle(power)
    G=DCSBM(sizes=[5000,5000], p=p_random_simple(2), theta=power, sparse=True)
    draw_old(G)
    '''



    '''
    *******   Generate DCSBM according to the real graph distribution  ************
    '''
    '''
    get_data_ave_degree(degree)
    p=p_random_simple(2)
    weight = get_weight(degree)
    G2=DCSBM(sizes=[1354, 1354], p=p,theta=weight, sparse=True)
    #e=list(G2.nodes)
    #G2.remove_nodes_from(e)
    get_ave_degree(G2)
    draw_old(G2)
    '''

    '''
    *******   GA  ************
    '''
    '''
    p = p_random_simple(2)
    size = [1354, 1354]
    genetic_func(degree, p, size, real_degree)
    '''

    '''
    *******   test the degree formula(SBM)  ************
    '''
    '''
    p = p_random_simple(2)
    G = SBM(sizes=[5000,5000], p=p, nodelist=None, seed=None, directed=False, selfloops=False, sparse=True)


    print('left = ',5000*(p[0][0]+p[0][1]))
    degree_ave = get_ave_degree(G)
    '''



    '''
    *******   test the degree formula(DCSBM)  ************
    '''
    '''
    p = p_random_simple(2)
    weight = get_weight(degree)

    sum_weight = 0
    variance = 0
    for i in range(len(weight)):
        weight[i] = weight[i]*19717/88648
    for i in range(0,10000):
        sum_weight += weight[i]
        variance = weight[i] * weight[i] + variance
    print('mean value(weight):',sum_weight/10000)
    print('variance(weight):', variance/10000)

    G2 = DCSBM(sizes=[5000, 5000], p=p, theta=weight, sparse=True)
    print('cin=', p[0][0] * 10000)
    print('cout=', p[0][1] * 10000)
    print('cin + cout = ', (p[0][0] * 10000 + p[0][1] * 10000))
    get_ave_degree(G2)
    draw_old(G2)
    '''


    p = p_random_simple(2)
    weight = get_weight(degree,1)
    weight_1 = np.array(weight)
    weight_1 = weight_1/np.mean(weight_1)
    weight = weight_1.tolist()
    cin = p[0][0]*3000
    cout = p[0][1]*3000

    #weight = np.random.uniform(3, 30, size=2708)
    #weight = weight / np.mean(weight)

    print(np.mean(weight_1))
    FAI = np.mean(weight_1**2)
    print(FAI)
    G = DCSBM(sizes=[1500, 1500], p=p_random_simple(2), theta=weight, sparse=True)
    draw_old(G)

    print('cin,cout = ',cin,cout)
    BH_draw(size=1000, weight=weight_1, c_average=3, FAI = FAI)


    print('hello')