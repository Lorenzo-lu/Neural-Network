# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 00:10:02 2019

@author: yluea

"""

import numpy as np;
import matplotlib.pyplot as plt;

class Neural_network_classification:
    
    def __init__(self,X,T,layer_nodes,Activation = 'sigmoid'):
        self.Activation = Activation;
        
        self.kinds = layer_nodes[-1];
        # how many different kinds
        self.Target = self.y2indicator(T,self.kinds);        
        # (self.L + 1) total layers ||| self.L is the number of W and b
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
            #self.W[i] = np.zeros((layer_nodes[i],layer_nodes[i+1]));
            #self.b[i] = np.zeros((layer_nodes[i+1]));
            
    def y2indicator(self,y,K): 
        N = len(y);
        ind = np.zeros((N,K));
        for i in range(N):
            ind[i,y[i]] = 1;        
        return ind;
    
    def classification_rate(self,T,Y):
        
        ground_true = np.argmax(T,axis = 1);
        pred = np.argmax(Y,axis = 1);        
        
        return np.mean(ground_true == pred);
    
    # these following two functions are non-linearity
    # ------------------------------------------------------------------------
    def softmax(self,A):
        expA = np.exp(A);
        return expA / expA.sum(axis = 1 , keepdims = True);
    
    def sigmoid(self,A):
        return 1/(np.exp(-A) + 1);
    
    def relu(self,A):
        return A*(A>0);
    
    def leaky_relu(self,A):
        return A*((A > 0) * 1 + (A < 0) * 0.1);
    
    def tanh(self,A):
        return np.tanh(A);
    # ------------------------------------------------------------------------
            
    def forward(self,L,X,W,b): 
        nodes = [0] * (L);
        L = L - 1;
        nodes[0] = X;
        if self.Activation == 'relu':
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];
                nodes[i] = self.relu(A);
        elif self.Activation == 'leaky_relu':
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];
                nodes[i] = self.leaky_relu(A);
                
        elif self.Activation == 'tanh':
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];
                nodes[i] = self.tanh(A);            
        
        else: # for sigmoid
            #if self.Activation != 'sigmoid':
                #print('Wrong activation statement. Automatically using sigmoid.');
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];            
                nodes[i] = self.sigmoid(A);
            
        # the output nodes are below:
        A = nodes[L-1].dot(W[L-1]) + b[L-1];
        nodes[L] = self.softmax(A);
        
        return nodes;
        
    def derivative(self,L,Target,nodes,W):
        # back probagation
        delta_W = [0] * L;
        delta_b = [0] * L;
        
        recursive_kernel = (Target - nodes[-1]);  
        delta_W[L-1] = nodes[L-1].T.dot(recursive_kernel);
        delta_b[L-1] = recursive_kernel.sum(axis = 0); 
        j = L - 2;
        # if we have 1 hidden layer, L+1 = 3; L = 2; j = 0, we get the delta W[0],b[0]
        # if we have 0 hidden layers, L = 1, we already get the delta W[0] b[0]
                  
        if self.Activation == 'relu':
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * (nodes[j+1] > 0); # relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
                
                
        elif self.Activation == 'leaky_relu':
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * ((nodes[j+1] > 0)*1 + (nodes[j+1] < 0)*0.1); # leaky relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
                
        elif self.Activation == 'tanh':
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * (1 - nodes[j+1] * nodes[j+1]); # leaky relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
            
        else:
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (1 - nodes[j+1]); # sigmoid
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
                
#        j = L - 1;
#        recursive_kernel = 0;
#        while j >= 0:
#            if j == L-1:
#                recursive_kernel = Target - nodes[j+1];
#            else:
#                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (1 - nodes[j+1]);
#                #recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (nodes[j+1] > 0);
#                #recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * ((nodes[j+1] > 0)*1 + (nodes[j+1] <= 0)*0.1 ); # leaky relu
#            
#            delta_W[j] = nodes[j].T.dot(recursive_kernel);
#            delta_b[j] = recursive_kernel.sum(axis = 0);            
#            j -= 1;
            
        return delta_W, delta_b;
            
    def cost(self,T,Y): # cross entropy 
        tot = T * np.log(Y);
        return np.mean(tot);
        
    
    def optimal(self,learning_rate = 1e-5,converge = 1e-7,regularization = 0, plot_step = 1000):
        
        costs = [];
        c0 = 0;
        c = 1;
        
        epoch = 0;
        
        while abs(c - c0) >  converge:         
            c0 = c;
            
            self.nodes = self.forward(self.L+1,self.nodes[0],self.W,self.b);            
            Y = self.nodes[-1];  
            c = self.cost(self.Target,Y);           
            
            if epoch%(plot_step) == 0:                
                
                r = self.classification_rate(self.Target,Y);
                print("cost:",c,"classification rate:",r);
                costs.append(c);
                
            epoch += 1; 
                
            delta_W,delta_b = self.derivative(self.L,self.Target,self.nodes,self.W);    
            for i in range(self.L):
                self.W[i] += learning_rate * (delta_W[i] - regularization * np.abs(self.W[i]));
                self.b[i] += learning_rate * (delta_b[i] - regularization * np.abs(self.b[i]));
            
        plt.plot(costs);
        plt.title("Optimization");
        plt.xlabel("steps");
        plt.ylabel("cross entropy likelihood");
        plt.show();
        
    def momentum_optimal(self,learning_rate = 1e-5,converge = 1e-7 ,beta = 0.5,regularization = 0,plot_step = 1000):
        if beta < 0 or beta > 1:
            print('wrong beta input: beta should between 0 and 1');
            return;
        costs = [];
        c0 = 0;
        c = 1;        
        epoch = 0;
        
        # init the momentum!
        # --------------------------------------------------------------------
        VdW = [0] * self.L;
        Vdb = [0] * self.L;        
        for i in range(self.L):
            r,c =  self.W[i].shape;           
            VdW[i] = np.zeros((r,c));
            Vdb[i] = np.zeros(c); 
        # --------------------------------------------------------------------        
         
        while abs(c - c0) >  converge:
            
            c0 = c;            
            self.nodes = self.forward(self.L+1,self.nodes[0],self.W,self.b);            
            Y = self.nodes[-1];  
            c = self.cost(self.Target,Y);
            if epoch%(plot_step) == 0:
                r = self.classification_rate(self.Target,Y);
                print("cost:",c,"classification rate:",r);
                costs.append(c);
            epoch += 1; 
            
            #if epoch > 1/converge:
                #break;
                
            delta_W,delta_b = self.derivative(self.L,self.Target,self.nodes,self.W);
            for i in range(self.L):
                VdW[i] = beta * VdW[i] + (1-beta) * delta_W[i];
                Vdb[i] = beta * Vdb[i] + (1-beta) * delta_b[i];
                self.W[i] += learning_rate * (VdW[i] - regularization * self.W[i]);
                self.b[i] += learning_rate * (Vdb[i] - regularization * self.b[i]);
            
        plt.plot(costs);        
        plt.title("Optimization with momentum beta = %s" %(beta));
        plt.xlabel("steps");
        plt.ylabel("cross entropy likelihood");
        plt.show();
        
    def test_classification_rate(self,Xtest,Ytest):
        test_Target = self.y2indicator(Ytest,self.kinds);
        test_nodes = self.forward(self.L+1,Xtest,self.W,self.b);
        r = self.classification_rate(test_Target,test_nodes[-1]);
        
        print('The classification rate of the test set is: ',r);
        return r;
    

# -----------------------------------------------------------------------------
    
class Neural_network_regression:
    def __init__(self,X,T,layer_nodes,Activation = 'sigmoid'):
        self.Activation = Activation;
        self.Target = T.reshape(len(T),1);
        if layer_nodes[-1] != 1:
            layer_nodes[-1] = 1;
            print("Wrong input on the last layer. But has been fixed.");                  
        # self.L+1 is the number of networks layers ||| self.L is the number of W and b
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
            #self.W[i] = np.zeros((layer_nodes[i],layer_nodes[i+1]));
            #self.b[i] = np.zeros((layer_nodes[i+1]));
            
    def sigmoid(self,A):
        return 1/(np.exp(-A) + 1);
    def relu(self,A):
        return A*(A>0);
    def leaky_relu(self,A):
        return A * ((A>0) + (A<=0) * 0.1);
    def tanh(self,A):
        return np.tanh(A);    
    
    def forward(self,L,X,W,b): 
        
        nodes = [0] * (L);
        L = L - 1;
        nodes[0] = X;
        if self.Activation == 'relu':
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];
                nodes[i] = self.relu(A);
        elif self.Activation == 'leaky_relu':
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];
                nodes[i] = self.leaky_relu(A);
                
        elif self.Activation == 'tanh':
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];
                nodes[i] = self.tanh(A);    
        
        else: # for sigmoid
            #if self.Activation != 'sigmoid':
                #print('Wrong activation statement. Automatically using sigmoid.');
            for i in range(1,L):
                A = nodes[i-1].dot(W[i-1]) + b[i-1];            
                nodes[i] = self.sigmoid(A);
            
        # the output nodes are below:
        nodes[L] = nodes[L-1].dot(W[L-1]) + b[L-1]; 
        
        return nodes;
            
            
    def cost(self,T,Y): # cross entropy 
        #tot = T * np.log(Y);
        tot = ((T-Y)**2);
        #tot = np.abs(T-Y);
        return np.mean(tot);     
        #tot = np.abs((T-Y));
        #tot = np.mean(tot);
        #return tot;
            
            
    def derivative(self,L,Target,nodes,W):
        # back probagation
        
        delta_W = [0] * L;
        delta_b = [0] * L;
        
        recursive_kernel = 2 * (Target - nodes[-1]); 
        delta_W[L-1] = nodes[L-1].T.dot(recursive_kernel);
        delta_b[L-1] = recursive_kernel.sum(axis = 0); 
        j = L - 2;
                  
        if self.Activation == 'relu':
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * (nodes[j+1] > 0); # relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
                
                
        elif self.Activation == 'leaky_relu':
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * ((nodes[j+1] > 0)*1 + (nodes[j+1] <= 0)*0.1); # leaky relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
                
        elif self.Activation == 'tanh':
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * (1 - nodes[j+1] * nodes[j+1]); # leaky relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
                
        else:
            while j >= 0:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (1 - nodes[j+1]); # sigmoid
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);            
                j -= 1;
              
            
        return delta_W, delta_b;       
            
            
    def optimal(self,learning_rate = 1e-5,converge = 1e-7 ,regularization = 0, plot_step = 1000):
        costs = [];
        c0 = 0;
        c = 1;
        
        epoch = 0;
        
        while abs(c - c0) >  converge:         
            c0 = c;            
            self.nodes = self.forward(self.L+1,self.nodes[0],self.W,self.b);            
            Y = self.nodes[-1];  
            c = self.cost(self.Target,Y);
            if epoch%(plot_step) == 0:
                #r = self.classification_rate(self.Target,Y);
                #print("cost:",c,"classification rate:",r);
                costs.append(c);
                print("cost:",c);
                
            epoch += 1; 
                
            delta_W,delta_b = self.derivative(self.L,self.Target,self.nodes,self.W);    
            for i in range(self.L):
                self.W[i] += learning_rate * (delta_W[i] - regularization * np.abs(self.W[i]));
                self.b[i] += learning_rate * (delta_b[i] - regularization * np.abs(self.b[i]));
        plt.plot(costs);
        plt.title("Optimization");
        plt.xlabel("steps");
        plt.ylabel("cost");
        plt.show();   
        
    def momentum_optimal(self,learning_rate = 1e-5,converge = 1e-7,beta = 0.5,regularization = 0,plot_step = 1000):
        if beta < 0 or beta > 1:
            print('wrong beta input: beta should between 0 and 1');
            return;
        costs = [];
        
        c0 = 0;
        c = 1;        
        epoch = 0;
        
        # init the momentum!
        # --------------------------------------------------------------------
        VdW = [0] * self.L;
        Vdb = [0] * self.L;        
        for i in range(self.L):
            r,c =  self.W[i].shape;           
            VdW[i] = np.zeros((r,c));
            Vdb[i] = np.zeros(c); 
        # --------------------------------------------------------------------        
         
        while (abs(c - c0) >  converge):         
            c0 = c;            
            self.nodes = self.forward(self.L+1,self.nodes[0],self.W,self.b);            
            Y = self.nodes[-1];  
            c = self.cost(self.Target,Y);
            if epoch%(plot_step) == 0:
                #r = self.classification_rate(self.Target,Y);
                #print("cost:",c,"classification rate:",r);
                print("cost:",c);
                costs.append(c);                
                
            epoch += 1; 
            
            #if epoch > 1/converge:
                #break;
                
            delta_W,delta_b = self.derivative(self.L,self.Target,self.nodes,self.W);
            for i in range(self.L):
                VdW[i] = beta * VdW[i] + (1-beta) * delta_W[i];
                Vdb[i] = beta * Vdb[i] + (1-beta) * delta_b[i];
                self.W[i] += learning_rate * (VdW[i] - regularization * self.W[i]);
                self.b[i] += learning_rate * (Vdb[i] - regularization * self.b[i]);
            
        plt.plot(costs);        
        plt.title("Optimization with momentum beta = %s" %(beta));
        plt.xlabel("steps");
        plt.ylabel("cost");
        plt.show();