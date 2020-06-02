
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

from YZ_ANN import YZ_ANN;


df_train = pd.read_csv("train.csv");

dt = df_train.values;

np.random.shuffle(dt);

divide = int(len(dt) * 0.7);


target = dt[:divide,0];
X = dt[:divide,1:];
X = X/np.max(X);

target_test = dt[divide:,0];
X_test = dt[divide:,1:];
X_test = X_test/np.max(X_test);


Layer_nodes = [len(X[0,:]), 128, 10];
myNet = YZ_ANN();
myNet.Train(X,target, Layer_nodes);
myNet.Load_valid(X_test, target_test);


myNet.Optimal(learning_rate = 2e-2, beta = 0.5, 
regularization = 1e-4, plot_step =len(X[:,0]), 
max_epoch = len(X[:,0]) * 5, batch_size = 10);


Cr = myNet.Test_classification_rate(X_test, target_test);
print("Test_classification is %.2f%%" %(Cr*100));

