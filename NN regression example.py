# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:24:18 2020

@author: yluea
"""

import numpy as np;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import time;

#from YZL_NN import YZL_NN;
from YZ_ANN import YZ_ANN;

N = 500;
X = np.random.random((N,2)) * 4 - 2;

#Y = X[:,0]**3 + X[:,1]**2;
Y = (2**3 - X[:,0]**2 - X[:,1]**2)**0.5;
#Y = X[:,0] * X[:,1];
#Y.reshape(N,1);
#Y.reshape(400,1)


fig = plt.figure();
ax = fig.add_subplot(1,1,1,projection = '3d');
ax.scatter(X[:,0],X[:,1],Y);
plt.show();

D = 2;
M1 = 10;
M2 = 10;

K = 1;

layer = [D,20,K];
#layer = [D,20,K];

reg = YZ_ANN(X,Y,layer, W_init = 'Random', output_format = 'Linear');

#learning_rate = 2e-6;
learning_rate = 5e-2;
converge = 1e-6;

regularization = 0.;
beta = 0.5;


start = time.clock();


#reg.Optimal(learning_rate,converge, beta, regularization, plot_step = 100, \
           #max_epoch = 100000, batch_size = 'all');
            
reg.Optimal(learning_rate, beta, regularization,plot_step = 100 ,batch_size = 50);
#reg.optimal(learning_rate,converge,regularization);


end = time.clock();

print("time used = ",(end - start));

line = np.linspace(-2,2,20);
xx,yy = np.meshgrid(line,line);
Xgrid = np.vstack((xx.flatten(),yy.flatten())).T;

nodes = reg.Forward(len(layer),Xgrid,reg.W,reg.b , reg.layer_activation); # all the nodes are matrices
Y_test = nodes[-1][:,0];

fig = plt.figure();
ax = fig.add_subplot(1,1,1,projection = '3d');
ax.scatter(X[:,0],X[:,1],Y,label = 'Training',c = 'Blue',s = 10);
#ax.scatter(Xgrid[:,0],Xgrid[:,1],Y_test)
ax.plot_trisurf(Xgrid[:,0],Xgrid[:,1],Y_test,linewidth = 0.2, antialiased = True,color = 'Yellow');

plt.legend();
#plt.show();

Ygrid = (2**3 - Xgrid[:,0]**2 - Xgrid[:,1]**2)**0.5;
#Ygrid = Xgrid[:,0] * Xgrid[:,1];
print('test_cost:',reg.Cost(Ygrid,Y_test, 6));

fig2 = plt.figure();
plt.scatter(Xgrid[:,0],Xgrid[:,1], c = np.abs(Ygrid - Y_test));
plt.show();
