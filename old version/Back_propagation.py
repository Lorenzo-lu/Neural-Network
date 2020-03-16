# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:35:05 2019

@author: Lorenzo_Lu
"""
class Neural_network:
    import numpy as np;
    import matplotlib.pyplot as plt;
    
    def __init__(self,X,T,layer_nodes):
        self.kinds = layer_nodes[-1];
        # how many different kinds
        self.Target = self.y2indicator(T,self.kinds);
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
            
    def y2indicator(self,y,K):        
        # change y (a vector like [1]) into a matrix like [0 1 0 0]
        # and [0] into [1 0 0 0] etc
        N = len(y);
        ind = np.zeros((N,K));
        for i in range(N):
            ind[i,y[i]] = 1;        
        return ind;
    
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
        # the output nodes are below:
        A = nodes[L-1].dot(W[L-1]) + b[L-1];
        nodes[L] = self.softmax(A);
        
        return nodes;
        
    def derivative(self,L,Target,nodes,W):
        # back probagation
        delta_W = [0] * L;
        delta_b = [0] * L;
        j = L - 1;
        recursive_kernel = 0;
        while j >= 0:
            if j == L-1:
                recursive_kernel = Target - nodes[j+1];
            else:
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (1 - nodes[j+1]);
            
            delta_W[j] = nodes[j].T.dot(recursive_kernel);
            delta_b[j] = recursive_kernel.sum(axis = 0);            
            j -= 1;
            
        return delta_W, delta_b;
            
    def cost(self,T,Y): # cross entropy 
        tot = T * np.log(Y);
        return np.mean(tot);
        #tot = self.Target * np.log(self.nodes[-1]);
        #return tot.sum();
    
    def optimal(self,learning_rate,converge):
        #learning_rate =  10*10e-7;
        costs = [];
        c0 = 0;
        c = 1;
        
        epoch = 0;
        #for epoch in range(100000): 
        while abs(c - c0) >  converge:         
            c0 = c;
            
            self.nodes = self.forward(self.L,self.nodes[0],self.W,self.b);            
            Y = self.nodes[-1];  
            c = self.cost(self.Target,Y);           
            
            if epoch%1000 == 0:                
                #Y = self.nodes[-1];  
                #c = self.cost(self.Target,Y);
                r = self.classification_rate(self.Target,Y);
                print("cost:",c,"classification rate:",r);
                costs.append(c);
                
            epoch += 1;            
            #if epoch > 1/converge:
                #break;
                
            delta_W,delta_b = self.derivative(self.L,self.Target,self.nodes,self.W);    
            for i in range(self.L):
                self.W[i] += learning_rate * delta_W[i];
                self.b[i] += learning_rate * delta_b[i];
            
        plt.plot(costs);
        plt.title("Optimization");
        plt.xlabel("steps");
        plt.ylabel("cross entropy likelihood");
        plt.show();
        
    def momentum_optimal(self,learning_rate,converge,beta):
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
            self.nodes = self.forward(self.L,self.nodes[0],self.W,self.b);            
            Y = self.nodes[-1];  
            c = self.cost(self.Target,Y);
            if epoch%1000 == 0:
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
                self.W[i] += learning_rate * VdW[i];
                self.b[i] += learning_rate * Vdb[i];
            
        plt.plot(costs);        
        plt.title("Optimization with momentum");
        plt.xlabel("steps");
        plt.ylabel("cross entropy likelihood");
        plt.show();
        
    def test_classification_rate(self,Xtest,Ytest):
        test_Target = self.y2indicator(Ytest,self.kinds);
        test_nodes = self.forward(self.L,Xtest,self.W,self.b);
        r = self.classification_rate(test_Target,test_nodes[-1]);
        
        print('The classification rate of the test set is: ',r);
        return r;
        
