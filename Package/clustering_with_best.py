import numpy as np
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np
import random
random.seed;
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.sparse import linalg
import time
import math
from scipy import stats
from sklearn.cluster import KMeans


''' This file is equal to clustering.py except for the functions that perform the clustering
For some of them, instead of choosing the minimal number of eigenvectors to perform the k-means algorithm on,
an arbitrary number of eigenvalues is computed and the overlap taken over the best
'''


def overlap(x,y):

    ''' This function computes the overlap between two vectors in the case of two classes. 

    Use: ov = overlap(x,y)
    Inputs: x, y the two vectors to evaluate 
    Outputs: the overlap 

    '''

    n = len(x)
    return (np.sum(x==y)/n - 1/2)*2


def adj(c_in,c_out,q, fraction):

    ''' This function creates the adjacency matrix for two classes, according to the DC-SBM model 

    Use: A = adj(c_in,c_out,q, fraction)
    Inputs: c_in (scalar), c_out (scalar), q (vector) are the parameters of the DC-SBM model. fraction is the size of the two classes. For even sizes fraction = n/2
    Outputs: The adjacency matrix A

    '''

    n = len(q)  
    c_matrix = np.ones((n,n))*c_out
    c_matrix[:fraction,:fraction] = c_in
    c_matrix[fraction:,fraction:] = c_in # we create the block matrix
    c_matrix = c_matrix - np.diag(np.diag(c_matrix)) # remove the diagonal term
    c_matrix = np.multiply(q*np.ones((n,n))*q[:, np.newaxis],c_matrix) # introduce the q_iq_j dependence


    A = np.random.binomial(1,c_matrix/n) # create the matrix  A


    i_lower = np.tril_indices(n, -1)
    A[i_lower] = A.T[i_lower]  # make the matrix symmetric

     
    # connect the unconncted components	

    d = np.sum(A,axis = 0)
    for i in range(n):
        if d[i] == 0:
            first_node = i
            if first_node > int(fraction): # if the element is in the second family
                r = np.random.uniform(0,1)
                if r < (c_in/(c_in+c_out)): # if it connects with an element of the same family
                    second_node = np.random.choice(np.arange(fraction,n), p=q[fraction:]/sum(q[fraction:])) # second node is in the second family
                else:
                    second_node = np.random.choice(np.arange(fraction), p=q[0:fraction]/sum(q[0:fraction])) # second node is in the first family
            else:
                r = np.random.uniform(0,1)
                if r < (c_out/(c_in+c_out)): # if the first element is in the first family and the second is in the same
                    second_node = np.random.choice(np.arange(fraction,n), p=q[fraction:]/sum(q[fraction:])) # second node is in the first family
                else:
                    second_node = np.random.choice(np.arange(fraction), p=q[0:fraction]/sum(q[0:fraction])) # second node is in the second family

            A[first_node][second_node] = 1 # create connections
            A[second_node][first_node] = 1
        
    return A




def BH(A,classes,r,assortativity):

    ''' This function performs spectral clustering on the second (resp. first) eigenvector of the deformed Laplacian D - rA, if one seeks assortative (resp. disassortative) blocks. 

    Use: y_kmeans,eigenvalues, X, precision = BH(A,classes,r,assortativity)
    Inputs: A a (symmetric) n x n adjacency matrix of a graph, classes a vector of size n with the underlying ground truth class assignment, r a real scalar, and assortativity is set to 1 if one seeks assortative blocks or to -1 if one seeks disassortative blocks. 
    Outputs: y_kmeans the vector of size n of detected classes, eigenvalues the values of the two smallest eigenvalues of D - rA (algebraic), X the eigenvector of size n used for the classification and precision the overlap between the detected classes y_kmeans and the ground truth classes.

    '''

    n = len(A)
    H = - r*A + np.diag(np.sum(A,axis = 0)) 
    H = H.astype(float)
    d = np.sum(A.astype(float),axis = 0) # degree vector
    eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(H, 2, which='SR') # computation of the smallest eigenvalues
    eigenvalues = eigenvalues.real # the matrix is  symmetric so the eigenvalues are real
    eigenvectors = eigenvectors.real # the matrix is  symmetric so the eigenvectors are real
    idx = eigenvalues.argsort()[::-1] # sort the eigenvalues
    idx = idx[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    if assortativity > 0:
        X = np.column_stack((np.ones(len(eigenvectors[:,0])),eigenvectors[:,1]))
    else:
        X = np.column_stack((np.ones(len(eigenvectors[:,0])),eigenvectors[:,0]))

    kmeans = KMeans(n_clusters = 2) # perform kmeans on the informative eigenvector
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    precision1 = overlap(y_kmeans,classes) # choose between the class assignment 0 -> A, 1 -> B and 0 -> B, 1 -> A, A and B being the two classes. This keeps the overlap positive
    precision2 = overlap(1-y_kmeans,classes)
    precision = max(precision1, precision2)
    if precision == precision2:
        y_kmeans = 1-y_kmeans


    return y_kmeans,eigenvalues, X, precision




def adj_cluster(A,classes, assortativity,number_eig):

    ''' This function performs spectral clustering on the best eigenvector of the adjacency matrix A

    Use: y_kmeans,eigenvalues, X, ov = adj_cluster(A,classes, assortativity,number_eig)
    Inputs: A a (symmetric) n x n adjacency matrix of a graph, classes a vector of size n with the underlying ground truth class assignment and assortativity is set to 1 if one seeks assortative blocks or to -1 if one seeks disassortative blocks, number_eig is the number of eigenvalues among which look for the best 
    Outputs: y_kmeans the vector of size n of detected classes, eigenvalues the values of the comoputed eigenvalues of A, X the eigenvector of size n used for the classification and ov the overlap between the detected classes y_kmeans and the ground truth classes.

    '''

    n = len(A)
    A = A.astype(float)
    if assortativity > 0:
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(A, number_eig, which='LR')
    else:
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(A, number_eig, which='SR')

    eigenvalues = eigenvalues.real # the matrix is  symmetric so the eigenvalues are real
    eigenvectors = eigenvectors.real # the matrix is  symmetric so the eigenvectors are real

    idx = eigenvalues.argsort()[::-1] # order the eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    current_ov = 0
    y = np.zeros(n)
    vector = np.zeros((len(eigenvectors[:,0]),2))

    # This cycle picks the best out of number_eig computed eigenvectors

    for i in range(1,number_eig):

        X = np.ones(len(eigenvectors[:,0]))
        X = np.column_stack((X,eigenvectors[:,i]))
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        precision1 = overlap(y_kmeans,classes) # choose between the class assignment 0 -> A, 1 -> B and 0 -> B, 1 -> A, A and B being the two classes. This keeps the overlap positive
        precision2 = overlap(1-y_kmeans,classes)
        ov = max(precision1, precision2)
        if ov > current_ov:
            current_ov = ov
            y = y_kmeans
            vector = np.zeros(len(eigenvectors[:,0]))
            vector = np.column_stack((vector,eigenvectors[:,i]))
            if ov == precision2:
                y = 1-y_kmeans

    return y,eigenvalues, vector, current_ov



def sym_lap(A,classes, assortativity,number_eig):

    ''' This function performs spectral clustering on the best eigenvector of the symmetric laplacian matrix D^{-1/2}AD^{-1/2}. 

    Use: y_kmeans,eigenvalues, X, ov = sym_lap(A,classes, assortativity,number_eig)
    Inputs: A a (symmetric) n x n adjacency matrix of a graph, classes a vector of size n with the underlying ground truth class assignment and assortativity is set to 1 if one seeks assortative blocks or to -1 if one seeks disassortative blocks, number_eig is the number of eigenvalues among which look for the best  
    Outputs: y_kmeans the vector of size n of detected classes, eigenvalues the values of the two smallest (largest) eigenvalues of A, X the eigenvector of size n used for the classification and ov the overlap between the detected classes y_kmeans and the ground truth classes.

    '''

    n = len(A)
    A = A.astype(float)
    d = np.sum(A,axis = 0)
    D_05 = np.diag(d**(-0.5))
    L = np.dot(D_05,np.dot(A,D_05))

    if assortativity > 0:
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(L, number_eig, which='LR')
    else:
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(L, number_eig, which='SR')
    eigenvalues = eigenvalues.real # the matrix is  symmetric so the eigenvalues are real
    eigenvectors = eigenvectors.real # the matrix is  symmetric so the eigenvectors are real
    idx = eigenvalues.argsort()[::-1] # order the eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    current_ov = 0
    y = np.zeros(n)
    vector = np.zeros((len(eigenvectors[:,0]),2))

    # This cycle picks the best out of number_eig computed eigenvectors

    for i in range(1,number_eig):

        X = np.ones(len(eigenvectors[:,0]))
        X = np.column_stack((X,eigenvectors[:,i]))
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        precision1 = overlap(y_kmeans,classes) # choose between the class assignment 0 -> A, 1 -> B and 0 -> B, 1 -> A, A and B being the two classes. This keeps the overlap positive
        precision2 = overlap(1-y_kmeans,classes)
        ov = max(precision1, precision2)
        if ov > current_ov:
            current_ov = ov
            y = y_kmeans
            vector = np.zeros(len(eigenvectors[:,0]))
            vector = np.column_stack((vector,eigenvectors[:,i]))
            if ov == precision2:
                y = 1-y_kmeans

    return y,eigenvalues, vector, current_ov



def lap(A,classes, assortativity,number_eig):

    ''' This function performs spectral clustering on the best eigenvector of the random walk laplacian matrix D^{-1}A. 

    Use: y_kmeans,eigenvalues, X, ov = lap(A,classes, assortativity,number_eig)
    Inputs: A a (symmetric) n x n adjacency matrix of a graph, classes a vector of size n with the underlying ground truth class assignment and assortativity is set to 1 if one seeks assortative blocks or to -1 if one seeks disassortative blocks, number_eig is the number of eigenvalues among which look for the best 
    Outputs: y_kmeans the vector of size n of detected classes, eigenvalues the values of the two smallest (largest) eigenvalues of A, X the eigenvector of size n used for the classification and ov the overlap between the detected classes y_kmeans and the ground truth classes.

    '''

    n = len(A)
    A = A.astype(float)
    d = np.sum(A,axis = 0)
    D_1 = np.diag(d**(-1))
    L = np.dot(D_1,A)

    if assortativity > 0:
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(L, number_eig, which='LR')
    else:
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(L, number_eig, which='SR')

    eigenvalues = eigenvalues.real # the matrix is  symmetric so the eigenvalues are real
    eigenvectors = eigenvectors.real # the matrix is  symmetric so the eigenvectors are real
    idx = eigenvalues.argsort()[::-1] # order the eigenvectors
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    current_ov = 0
    y = np.zeros(n)
    vector = np.zeros((len(eigenvectors[:,0]),2))

    # This cycle picks the best out of number_eig computed eigenvectors

    for i in range(1,number_eig):

        X = np.ones(len(eigenvectors[:,0]))
        X = np.column_stack((X,eigenvectors[:,i]))
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)
        precision1 = overlap(y_kmeans,classes) # choose between the class assignment 0 -> A, 1 -> B and 0 -> B, 1 -> A, A and B being the two classes. This keeps the overlap positive
        precision2 = overlap(1-y_kmeans,classes)
        ov = max(precision1, precision2)
        if ov > current_ov:
            current_ov = ov
            y = y_kmeans
            vector = np.zeros(len(eigenvectors[:,0]))
            vector = np.column_stack((vector,eigenvectors[:,i]))
            if ov == precision2:
                y = 1-y_kmeans

    return y,eigenvalues, vector, current_ov


def find_r(A,n_cycles,assortativity):

    ''' This function computes find the estimate of zeta as a ratio between the two largest real eigenvalues of the non-backatracking matrix B. 

    Use: r = find_r(A,n_cycles,assortativity)
    Inputs: A the adjacency matrix, n_cycles is the number of iterations to perform line-search, assortativity is sign(c_in - c_out)
    Outputs: r, the estimate of (c_in + c_out)/(c_in - c_out)

    '''

    d = np.sum(A,axis = 0)
    c = np.mean(d) # average degree
    phi = np.mean(d**2)/c**2 # estimation of the second moment of the q
    gamma1 = newton(2*c*phi/3,4*c*phi/3,A,0,n_cycles) # line search of gamma_1 around c*phi
    if assortativity > 0:
        gamma2 = newton(np.sqrt(gamma1),gamma1,A,1,n_cycles) # linesearch of gamma_2 in (\sqrt(rho),rho)
    else:
        gamma2 = newton(-np.sqrt(gamma1),-gamma1,A,0,n_cycles) # linesearch of gamma_2 in (-\sqrt(rho),-rho)

    return gamma1/gamma2

def find_r_inside(A,n_cycles,assortativity):

    ''' This function computes find the estimate of zeta as the eigenvalue inside the bulk of the non-backatracking matrix B. 

    Use: r = find_r_inside(A,n_cycles,assortativity)
    Inputs: A the adjacency matrix, n_cycles is the number of iterations to perform line-search, assortativity is sign(c_in - c_out)
    Outputs: r, the estimate of (c_in + c_out)/(c_in - c_out)

    '''

    d = np.sum(A,axis = 0)
    c = np.mean(d) # average degree
    phi = np.mean(d**2)/c**2 # estimate of the second moment of q
    if assortativity > 0:
        r = newton(np.sqrt(c*phi),1,A,1,n_cycles) # linesearch of zeta in (\sqrt(rho),1)
    else:
        r = newton(-np.sqrt(c*phi),-1,A,1,n_cycles) # linesearch of zeta in (-\sqrt(rho),-1)

    return r


def newton(a,b,A,pos,n_cycles):

    ''' This function computes performs the line search of r such that the Bethe Hessian matrix has a zero eigenvalue

    Use: x = newton(a,b,A,pos,n_cycles)
    Inputs: a,b are the exttremes of the interval, A is the  adjacency matrix, pos is the position of the zero eigenvalue (0,1), n_cycle is the number of iterations to perform
    Outputs: x, the estimate of the position of the zero eigenvalue

    '''
    
    n = len(A)
    d = np.sum(A,axis = 0)

    
    for i in range(n_cycles):
    
        r = 0.5*(a+b)    
        H = (r**2-1)*np.diag(np.ones(n)) + np.diag(d) - r*A
        eigenvalues,eigenvectors = scipy.sparse.linalg.eigs(H, 2, which='SR')
        idx = eigenvalues.argsort()[::-1] 
        idx = idx[::-1]
        if eigenvalues[pos] > 0:
            b = r
        else:
            a = r
            
    return 0.5*(a+b)


def reg_L(A,classes):
        
 
    n_clusters = 2
    d = np.sum(A, axis = 0)
    n = len(A)
    d_tau = d + np.mean(d)*np.ones(n)
    D_05 = np.diag(d_tau**(-0.5))
    L = np.dot(D_05,np.dot(A,D_05))
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, n_clusters, which='LA')
    idx = eigenvalues.argsort()[::-1] # sort the eigenvalues
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    X = eigenvectors
    for i in range(len(X)):
        X[i] = X[i]/np.sqrt(np.sum(X[i]**2))
    kmeans = KMeans(n_clusters = n_clusters) # perform kmeans on the informative eigenvector
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    precision1 = overlap(y_kmeans,classes) # choose between the class assignment 0 -> A, 1 -> B and 0 -> B, 1 -> A, A and B being the two classes. This keeps the overlap positive
    precision2 = overlap(1-y_kmeans,classes)
    ov = max(precision1, precision2)
    if ov == precision2:
        y_kmeans = 1-y_kmeans
    
    return y_kmeans,eigenvalues,X,ov


