import numpy as np
import pandas as pd
import operator 
from operator import itemgetter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
        

def predict(self, X_test):
    """
    To get the predicted class, iterate from 1 to the total number of training data points
    Calculate the distance between test data and each row of training data. Euclidean distance is used as our distance metric
    Get top k rows from the sorted array
    Get the most frequent class of these rows
    Return the predicted class
    """

    predictions = [] 
    for i in range(len(X_test)):
        dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
        dist_sorted = dist.argsort()[:self.K]
        neigh_count = {}
        for idx in dist_sorted:
            if self.Y_train[idx] in neigh_count:
                neigh_count[self.Y_train[idx]] += 1
            else:
                neigh_count[self.Y_train[idx]] = 1
        sorted_neigh_count = sorted(neigh_count.items(),    
        key=operator.itemgetter(1), reverse=True)
        predictions.append(sorted_neigh_count[0][0]) 
    return predictions
    
def prepare_data():
#     mnist = load_digits()
#     print(mnist.data.shape)
#     X = mnist.data 
#     y = mnist.target
    
    
    
    se_data = pd.read_csv('data//IST_MIR.csv') 
    print(se_data.shape)
    X = se_data.iloc[:, 2:14]
    y = se_data['defect_status']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

    print(X_train.shape, y_train.shape)
#     print(X_test.shape, y_test.shape)
    print("y_train count: ", np.unique(y_train,return_counts=True))
#     print(np.unique(y_test,return_counts=True))
    
    return X_train, X_test, X_val, y_train, y_test, y_val
    
def calculate_k(X_train, X_test, y_train, y_test):
    """
    Training our model on all possible K values (odd) from 3 to 10  
    """
    kVals = np.arange(3,10,2)
    accuracies = []
    for k in kVals:
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies.append(acc)
#         print("K = "+str(k)+"; Accuracy: "+str(acc))
        
    max_index = accuracies.index(max(accuracies))
    print("selected k = " + str(2 * (max_index + 1) + 1))
    
#     plt.plot(kVals, accuracies) 
#     plt.xlabel("K Value") 
#     plt.ylabel("Accuracy")

    return (2 * (max_index + 1) + 1)
    

def calculate_metrics(k, X_train, y_train):
    """
    Checking for Precision, Recall and F-score for the most accurate K value
    """
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train) 
    pred = model.predict(X_train)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_train, pred, average = 'binary')
    fpr, tpr, thresholds = roc_curve(y_train, pred)
    auc_score = auc(fpr, tpr)
    print("----------training----------")
    print("Precision \n", precision)
    print("\nRecall \n", recall)
    print("\nF-score \n", fscore)
    print("\nAUC \n", auc_score)
    print("----------training----------")
    

def perform_inference(k, X_train, X_test, y_train, y_test):
    """
    Performing inference of the trained model on the testing set:
    """
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, pred, average = 'binary')
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    auc_score = auc(fpr, tpr)
    print("----------testing----------")
    print("Precision \n", precision)
    print("\nRecall \n", recall)
    print("\nF-score \n", fscore)
    print("\nAUC \n", auc_score)
    print("----------testing----------")
    return precision, recall, fscore, auc_score