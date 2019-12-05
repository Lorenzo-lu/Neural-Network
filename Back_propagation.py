# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:35:05 2019

@author: Lorenzo_Lu
"""

class Neural_network:
    import numpy as np;
    import matplotlib.pyplot as plt;
    
    def __init__(self,X,T,layer_nodes):
        #layer_nodes[0] = X.shape[1];
        #layer_nodes[-1] = T.shape[1];
        self.Target = T;
        # L is the number of networks layers
        self.L = len(layer_nodes) - 1;
        self.W = [0] * self.L;
        self.b = [0] * self.L;
        # --------------------------------------------------------------------
        # init the nodes in neural network
        # nodes has L + 1 layers !
        self.nodes = [0] * (self.L+1);
        self.nodes[0] = X;
        # --------------------------------------------------------------------
        for i in range(self.L):            
            self.W[i] = np.random.randn(layer_nodes[i],layer_nodes[i+1]);
            self.b[i] = np.random.randn(layer_nodes[i+1]);  
            
    def classification_rate(self,T,Y): 
        #n_correct = 0;
        #n_total = 0;
        ground_true = np.argmax(T,axis = 1);
        pred = np.argmax(Y,axis = 1);        
        #for i in range(len(pred)):
            #n_total += 1;
            #if pred[i] == ground_true[i]:
                #n_correct += 1;            
        #return n_correct / n_total;
        # !!! the easier implement
        return np.mean(ground_true == pred);
    
    # these following two functions are non-linearity
    # ------------------------------------------------------------------------
    def softmax(self,A):
        expA = np.exp(A);
        return expA / expA.sum(axis = 1 , keepdims = True);
    
    def tanh(self,A):
        return 1/(np.exp(-A) + 1);
    # ------------------------------------------------------------------------
            
    def forward(self,L,X,W,b): 
        nodes = [0] * (self.L+1);
        nodes[0] = X;
        for i in range(1,L):
            A = nodes[i-1].dot(W[i-1]) + b[i-1];
            nodes[i] = self.tanh(A);
        
        A = nodes[L-1].dot(W[L-1]) + b[L-1];
        nodes[L] = self.softmax(A);
        
        return nodes;
       
        
    def derivative(self):
        # back probagation
        delta_W = [0] * self.L;
        delta_b = [0] * self.L;
        j = self.L - 1;
        
        recursive_kernel = 0;
        while j >= 0:
            if j == self.L - 1:
                recursive_kernel = self.Target - self.nodes[j+1];            
            else:
                recursive_kernel = (recursive_kernel.dot(self.W[j+1].T)) * self.nodes[j+1] * (1 - self.nodes[j+1]);
            
            delta_W[j] = self.nodes[j].T.dot(recursive_kernel);
            delta_b[j] = recursive_kernel.sum(axis = 0);            
            j -= 1;
            
        return delta_W, delta_b;
            
            
    def cost(self,T,Y): # cross entropy 
        tot = T * np.log(Y);
        return np.mean(tot);
        #tot = self.Target * np.log(self.nodes[-1]);
        #return tot.sum();
    
    def optimal(self):
        learning_rate =  5*10e-7;
        costs = [];
        for epoch in range(100000):            
            self.nodes = self.forward(self.L,self.nodes[0],self.W,self.b);
            
            if epoch%1000 == 0:
                
                Y = self.nodes[-1];  
                c = self.cost(self.Target,Y);
                r = self.classification_rate(self.Target,Y);
                print("cost:",c,"classification rate:",r);
                costs.append(c);
                
            delta_W,delta_b = self.derivative();    
            for i in range(self.L):
                self.W[i] += learning_rate * delta_W[i];
                self.b[i] += learning_rate * delta_b[i];
            
        plt.plot(costs);
        plt.show();
        
        
import numpy as np
import matplotlib.pyplot as plt

def main():
    Nclass = 200;
    D = 2;
    M1 = 3;
    #M2 = 4;
    K = 3;
    
    X1 = np.random.randn(Nclass,2) + np.array([0,-2]);
    X2 = np.random.randn(Nclass,2) + np.array([2,2]);
    X3 = np.random.randn(Nclass,2) + np.array([-2,2]);
    X = np.vstack([X1,X2,X3]);
    Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass);
    
    N = len(Y);
    T = np.zeros((N,K));
    for i in range(N):
        T[i,Y[i]] = 1;
        
    plt.scatter(X[:,0],X[:,1],c = Y, s = 10, alpha =100);
    plt.show();
    
    NN = Neural_network(X,T,[D,M1,K]);
    NN.optimal();
    
if __name__ == '__main__':
    main();
        
