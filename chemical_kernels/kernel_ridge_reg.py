import numpy as np
import math
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from numpy.linalg import inv


# tunning parameters for kernel ridge regression
# 5-fold cross validation
def tune_lamda(sim, y_true):
    results = []
    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    kf.get_n_splits(y_true)
    i = 0
    for train_index, test_index in kf.split(y_true):
        gram = sim
        gram = np.delete(gram, [test_index], 1)
        gram_test = gram[test_index,:]
        gram = np.delete(gram, [test_index], 0)

        y = np.array(y_true[:])
        y_test = np.array(y_true)[test_index]
        y = np.delete(y,[test_index],None)

        lamda_range = [0.0005, 0.001, 0.002, 0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.05,1.0,1.5]
        results.append([])
        
        n = gram.shape[0]

        for lamda in lamda_range:
            y_pred = np.dot(np.dot(gram_test, inv(gram + lamda*np.eye(n))), y)
            results[i].append(r2_score(y_test, y_pred))

        i = i + 1

    results = np.array(results)
    R_square = np.mean(results, axis=0)
    lamda_opt = lamda_range[np.argmax(R_square)]
    return (lamda_opt)


def evaluate_model(sim, y_true):
    # hold out 20% of data points as test set, 10% as validation set
    np.random.seed(0)
    N = sim.shape[0]
    indices = np.random.permutation(N)
    trunc1 = int(N * 0.2)
    trunc2 = int(N * 0.3)
    test_index = np.sort(indices[:trunc1])
    val_index = np.sort(indices[trunc1+1:trunc2])
    train_index = np.sort(indices[trunc2+1:])
    gram = sim
    gram_test = gram[test_index][:,train_index]
    gram_val = gram[val_index][:,val_index]
    gram_train = gram[train_index][:,train_index]
    gram = gram[:][:,train_index]
    y = np.array(y_true)
    y_test = y[test_index]
    y_val = y[val_index]
    y_train = y[train_index]
    
    # perform model selection on training data
    lamda = tune_lamda(gram_val, y_val)
    # predict using test set 
    n = gram_train.shape[0]
    y_test_pred = np.dot(np.dot(gram_test, inv(gram_train + lamda*np.eye(n))), y_train)
    # compute R square
    R_square_test = r2_score(y_test, y_test_pred)
    # predict using the entire data set
    y_pred = np.dot(np.dot(gram, inv(gram_train + lamda*np.eye(n))), y_train)
    R_square = r2_score(y, y_pred)
    
    return (R_square_test, R_square)

