# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:13:28 2020

@author: yluea
"""



import numpy as np;
import matplotlib.pyplot as plt;

class YZ_ANN:
    ## statement: 
    ## currently, support four acitivation function: 1. Sigmoid 2. ReLU 3. Leaky_ReLU 4.Tanh
    ## support three outputs format : 1. Softmax (for multiple classfification) 2. Sigmoid 3.Linear(for regression)!
    
    # ========================================================================================
    ## Section 1 : init the class! include func 1 & 2 & 3
    ## func 1:
    def __init__(self, X, Target, layer_nodes, W_init = 'Random', \
                 output_format = 'Softmax'):
        
        print('support three outputs format : 1. Softmax (for multiple \
                classfification) 2. Sigmoid 3.Linear(for regression)!');
        # (self.L + 1) total layers ||| self.L is the number of W and b
        self.L = len(layer_nodes) - 1;
        self.W = [0] * self.L;
        self.b = [0] * self.L;
        # init the nodes in neural network
        # nodes has L + 1 layers !
        self.nodes = [0] * (self.L+1);
        self.nodes[0] = X;
        
        ## Define the activation function for each layer        
        self.Activation_map = {1:'Sigmoid',2:'ReLU',3:'Leaky_ReLU',4:'Tanh',\
                               5:'Softmax',6:'Linear'};
        self.Activation_sequence();           
        
        # *******************************************************
        # initialize the output form:
        if output_format == 'Softmax':
            print("You are now doing a multiple classification by 'Softmax'");
            if layer_nodes[-1] <= 1:
                print("Fatal Error: The last layer should be no fewer than 2!\
                      Or you should choose 'Sigmoid' or 'Linear'");
            else:
                self.kinds = layer_nodes[-1];
                self.Target = self.Y2indicator(Target,self.kinds);
                self.layer_activation[-1] = 5;
        else:
            if layer_nodes[-1] != 1:
                print("Warning: The last layer has wrong nodes number. \
                      Automatically convert to 1!");
                layer_nodes[-1] = 1;
            self.kinds = 1;
            self.Target = Target.reshape(-1,1);
            if output_format == 'Sigmoid':
                print("You are now doing a binary classification by 'Sigmoid'");
                self.layer_activation[-1] = 1;
            elif  output_format == 'Linear':
                print("You are now doing a regression by 'Linear'");  
                self.layer_activation[-1] = 6;
            else:
                print("Fatal Error: Wrong output format!");
        print("The final activation functions are:\n")
        self.Show_layers_act(self.L);
        
        # *******************************************************
        # initialize the W and b by either random (0-1) or zeros
        if W_init == 'Zeros':
            for i in range(self.L): 
                self.W[i] = np.zeros((layer_nodes[i],layer_nodes[i+1]));
                self.b[i] = np.zeros((layer_nodes[i+1]));
        elif W_init == 'Random':
            for i in range(self.L):            
                self.W[i] = np.random.randn(layer_nodes[i],layer_nodes[i+1]);
                self.b[i] = np.random.randn(layer_nodes[i+1]); 
        
        
        
    ## func 2 and 3:  
    def Show_layers_act(self, L): 
        # print the layers and their corresponding activation functions
        acts = [];
        for i in range(L):
            acts.append([i, self.Activation_map[self.layer_activation[i]]]);       
        print(acts);
        
        
    def Activation_sequence(self):
        def Read_str(s, TYPE):
            
            if TYPE == int:
                return int(s);
            elif TYPE == float:
                return float(s);
            elif TYPE == list:
                num_map = set(['1','2','3','4','5','6','7','8','9','0']);
                i = 1;
                ans = [];
                while i < len(s):
                    if s[i] == '[':
                        j1 = i+1;
                        while s[j1] not in num_map:
                            j1 += 1;
                        j2 = j1+1;
                        while s[j2] in num_map:
                            j2 += 1;                            
                        j3 = j2 + 1;
                        while s[j3] not in num_map:
                            j3 += 1;
                        j4 = j3 + 1;
                        while s[j4] in num_map:
                            j4 += 1;                        
                        ans.append([Read_str(s[j1:j2],int), Read_str(s[j3:j4],int)]);
                        i = j4;
                    i += 1;
                return ans; 
        
        # You can even change the activation functions after you declare this class !!!
        print("The current activations you can choose are:");
        print(self.Activation_map);
        print("Now choosing your most frequent activation function in this case:");
        choice = Read_str(input(), int);
        self.layer_activation = [choice] * self.L; ## the activation function for each layer!   
            
        Judge = 'N';
        while Judge == 'N':
            print("The current activations for each layers (without last layer) are:");
            self.Show_layers_act(self.L -1);
            ## The last activation will be output later!
            Judge = input("Good with current activation sequence?\nY/N\n");
            if Judge == 'N':
                print('Please input the index of the layer and the activation \
                      you want::\n e.g. [[1,2],[4,1]], means change the second layer\
                into ReLU and the fifth into Sigmoid');
                change = Read_str(input(), list);
                for i in change:
                    index = i[0]
                    self.layer_activation[index] = i[1];                    
    # ========================================================================================
    ## Section2: include func 4 & 5
    ## func 4
    def Y2indicator(self,y,K): 
        N = len(y);
        ind = np.zeros((N,K));
        for i in range(N):
            ind[i,y[i]] = 1;        
        return ind;
    ## func 5
    def Classification_rate(self,T,Y,last_layer_act = False):
        if not last_layer_act:
            last_layer_act = self.layer_activation[-1];
        
        if last_layer_act == 5: ## Softmax
            ground_true = np.argmax(T,axis = 1);
            pred = np.argmax(Y,axis = 1);
            return np.mean(ground_true == pred);
        elif last_layer_act == 1: ## Sigmoid
            pred = np.round(Y);
            return np.mean(T == pred);
        
        elif last_layer_act == 6: ## Linear
            return False; ## No classification rate for a regression problem
    
    # ========================================================================================
    ## Section 3: define the activating!    
    ## func 6
    def Activating(self, A, activation):
        # a is the node and the activation is the element in self.Activation_map
        if activation == 1: ## Sigmoid
            return 1/(np.exp(-A) + 1);
        elif activation == 2: ## ReLU
            return A*(A>0);
        elif activation == 3: ## Leaky_ReLU
            return A*((A > 0) * 1 + (A < 0) * 0.1);
        elif activation == 4: ## Tanh
            return np.tanh(A);
        elif activation == 5: ## Softmax
            expA = np.exp(A);
            return expA / expA.sum(axis = 1 , keepdims = True);
        elif activation == 6: ## for regression
            return A;
        else:
            print("Fatal Error! No activation function matches!");
            
    # ========================================================================================
    ## Section 4: define the forward!
    ## func 7
    def Forward(self,L,X,W,b,layer_activation = False):
        if not layer_activation:
            layer_activation = self.layer_activation;
        
        nodes = [0] * (L); ## L layers of nodes, L-1 layers of W and b
        #L = L - 1;
        nodes[0] = X;        
        for i in range(1,L):
            A = nodes[i-1].dot(W[i-1]) + b[i-1];
            nodes[i] = self.Activating(A,layer_activation[i-1]);        
        return nodes;
    # ========================================================================================
    ## Section 5: define the back_probagation!    
    ## func 8
    def Derivative(self,L,Target,nodes,W,layer_activation = False):
        if not layer_activation:
            layer_activation = self.layer_activation;
            
        L = L - 1; 
        ## initially please input the total layers of W instead of nodes! That is self.L + 1, in this step it will be self.L finally
        delta_W = [0] * L;
        delta_b = [0] * L;
        ###### for either softmax,linear or sigmoid, the recursive kernel could be the same!
        ###### if you set linear loss = 1/2(Y-T)^2;
        #recursive_kernel = (Target - nodes[-1]) * (1 - nodes[-1]) * (nodes[-1]) / len(Target);
        recursive_kernel = (Target - nodes[-1]) / len(Target);
        delta_W[L-1] = (nodes[L-1]).T.dot(recursive_kernel);
        delta_b[L-1] = recursive_kernel.sum(axis = 0); 
        ## This two delta are for the last W and b (same for softmax, sigmoid, and linear )
        ## slightly different; but the good news is we don't have to consider the activation!!!
        
        j = L - 2; ## begin from the last but one (the last one is W[L-1]), and also consider the activation_sequence!
        ## now from the last but 1 
        while j >= 0:
            if layer_activation[j] == 2: ## ReLU
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * (nodes[j+1] > 0);
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);
            
            elif layer_activation[j] == 3: ## Leaky_ReLU
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * ((nodes[j+1] > 0)*1 + (nodes[j+1] < 0)*0.1); # leaky relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);  
            
            elif layer_activation[j] == 4: ## Tanh
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * (1 - nodes[j+1] * nodes[j+1]); # leaky relu
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0); 
                
            elif layer_activation[j] == 1: ## Sigmoid
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (1 - nodes[j+1]); # sigmoid
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);
                
            elif layer_activation[j] == 5: ##Softmax
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)) * nodes[j+1] * (1 - nodes[j+1]); 
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);
                
            elif layer_activation[j] == 6: ##Linear
                recursive_kernel = (recursive_kernel.dot(W[j+1].T)); # Linear
                delta_W[j] = nodes[j].T.dot(recursive_kernel);
                delta_b[j] = recursive_kernel.sum(axis = 0);
                
            else:
                print("Fatal Error: No such Activation Function");
                return;
            
            j -= 1;            
        return delta_W, delta_b;
    # ========================================================================================
    ## Section 6: cost and optimal!    
    ## func 9
    def Cost(self,T,Y,last_layer_act = False):
        if not last_layer_act:
            last_layer_act = self.layer_activation[-1];
        if last_layer_act == 1: ## Sigmoid, y has shape [-1,1], so
            total = T * np.log(Y) + (1-T) * np.log(1-Y);
        elif last_layer_act == 5:## Softmax, y has shape [-1, self.kinds], so
            total = T * np.log(Y);
        elif last_layer_act == 6: ## linear
            total = (T-Y)**2/2;
        else:
            print("Fatal Error: No such Activation Function");
            return;
        return np.sum(total)/len(total);
    
    
    def Optimal(self,learning_rate = 1e-2, beta = 0, regularization = 0, plot_step = 1000, max_epoch = False, batch_size = 'all'):
        if not max_epoch:
            max_epoch = plot_step * 50;
            
        name_map = {1:"1.Learning rate", 2:"2.Momentum", 3:"3.Regularization",\
                    4:"4.Steps for error plotting", 5:"5.Maximum runing steps",\
                    6:"6.The size for training batch"};        
        para_map = {1:(learning_rate),
                    2:(beta), 
                    3:(regularization), 
                    4:(plot_step),
                    5:(max_epoch), 
                    6:(batch_size)};
        
        Judge = 'N';        

        while Judge != 'Y':
            para_map[1] = float(para_map[1]);
            para_map[2] = float(para_map[2]);
            para_map[3] = float(para_map[3]);
            para_map[4] = int(para_map[4]);
            para_map[5] = int(para_map[5]);  
            if (para_map[6]) != 'all':
                para_map[6] = int((para_map[6]))
                
            
            self.Gradient_Ascent(para_map[1], para_map[2], para_map[3], para_map[4], para_map[5], para_map[6]);
            print('Is the Optimal achieved?');
            Judge = input('Y/N\n');

            if Judge != 'Y':                
                Adjust = 'N';
                while Adjust != 'Y':
                    print('The current parameters are :\n');
                    for i in name_map:
                        print(name_map[i],para_map[i]);
                        
                    index = int(input("Which parameter to change? Please input its index.\n Input 0 if no change.\n"));
                    if index in para_map:
                        value = (input("Please input the desired value\n"));
                        para_map[index] = value;
                    
                    Adjust = input("ALL SET?\nY/N\n");
                
                    
                
            

    def Gradient_Ascent(self,learning_rate, beta = 0, regularization = 0, plot_step = 1000, max_epoch = False, batch_size = 'all'):
        if type(batch_size) == int:
            if batch_size == 1:
                print("Stochastic Gradient Ascent!");
            else:
                print("Mini-batch (size %s) Gradient Ascent!" %batch_size);
        else:
            print("Batch Gradient Ascent!");

        data = np.hstack((self.nodes[0], self.Target));                                

        def Make_Batch(size,perm = data):            
            if type(size) == int:
                if (size <= len(self.nodes[0])) and (size >= 1):
                    cols = len(self.nodes[0][0,:]);
                    rnd_indices = np.random.randint(0, len(perm), size);
                    selection = perm[rnd_indices];


                    x = np.reshape(selection[:,:cols],(size,-1));
                    y = np.reshape(selection[:,cols:],(size,-1));
                    

                    return x,y;
            return self.nodes[0],self.Target;
            
        
        if beta < 0 or beta > 1:
            print('wrong beta input: beta should between 0 and 1');
            return;
        costs = [];
        #c0 = 0;
        c = 1;

        #N_sample = len(self.nodes[0]); ## The number of samples
      
        
        epoch = 0;
        
        # init the momentum!
        # --------------------------------------------------------------------
        VdW = [0] * self.L;
        Vdb = [0] * self.L;        
        for i in range(self.L):            
            VdW[i] = False;
            Vdb[i] = False;
        # --------------------------------------------------------------------     
         
        #while abs(c - c0) >  converge:
        
        #if not max_epoch:
            #max_epoch = plot_step * 50;
            
        while epoch <= max_epoch:

            X, T = Make_Batch(batch_size); ## X,Y and T are selected in the batch! 

            nodes_selected = self.Forward(self.L+1,X,self.W,self.b);
            #Y = nodes_selected[-1];
            
            if epoch%(plot_step) == 0:                
                
                self.nodes = self.Forward(self.L+1,self.nodes[0],self.W,self.b);
                Y_global = self.nodes[-1];
                c = self.Cost(self.Target,Y_global);
                #c = np.log(c);
                r = self.Classification_rate(self.Target,Y_global);
                if r:
                    print("cost:",c ,"classification rate:",r);
                else:
                    print("cost:",c);
                costs.append(c);              
                
            epoch += 1;

            #if (epoch == max_epoch) and abs(costs[-1] - costs[-2]) > (converge/plot_step):
                #max_epoch += plot_step;
                
           
            delta_W,delta_b = self.Derivative(self.L+1, T, nodes_selected, self.W);
            for i in range(self.L):
                if type(VdW[i]) == bool:
                    VdW[i] = delta_W[i]; 
                if type(Vdb[i]) == bool:
                    Vdb[i] = delta_b[i];
                VdW[i] = beta * VdW[i] + (1-beta) * delta_W[i];
                Vdb[i] = beta * Vdb[i] + (1-beta) * delta_b[i];
                self.W[i] += learning_rate * (VdW[i] - regularization * self.W[i]);
                self.b[i] += learning_rate * (Vdb[i] - regularization * self.b[i]);

        plt.figure();     
        plt.plot(np.log(costs));        
        plt.title("Optimization with momentum beta = %s & regularization = %s" %(beta, regularization));
        plt.xlabel("steps /(%s)" %(plot_step));       

        plt.ylabel("Loss (in log)");
        plt.show();
        
        self.Loss = costs;
        
        
    def Test_classification_rate(self,Xtest,Ytest):
        test_Target = self.Y2indicator(Ytest,self.kinds);
        test_nodes = self.Forward(self.L+1,Xtest,self.W,self.b,self.layer_activation);
        r = self.Classification_rate(test_Target,test_nodes[-1],self.layer_activation[-1]);
        
        print('The classification rate of the test set is: ',r);
        return r;
    
