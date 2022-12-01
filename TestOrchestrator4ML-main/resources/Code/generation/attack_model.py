import numpy as np
import pandas as pd
import operator 
from operator import itemgetter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

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
    max_index = accuracies.index(max(accuracies))
    print("selected k = " + str(2 * (max_index + 1) + 1))
    return (2 * (max_index + 1) + 1)
    

def keras_model():
    model = Sequential()
    model.add(Dense(12, input_dim=12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
    
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
    
    

def perform_inference(X_train, X_test, y_train, y_test, model_name):
    """
    Performing inference of the trained model on the testing set:
    """
    
    if (model_name == 'KNeighborsClassifier') : 
        k = calculate_k(X_train, X_test, y_train, y_test)
        model = KNeighborsClassifier(n_neighbors = k)
    elif (model_name == 'DecisionTreeClassifier') : 
        model = DecisionTreeClassifier()
    elif (model_name == 'SVC') : 
        model = SVC()
    elif (model_name == 'LinearSVC') : 
        model = LinearSVC()
    elif (model_name == 'Dense' or model_name == 'SimpleRNN' or model_name == 'Bidirectional') : 
        model = keras_model()
    else:
        print("else")
        model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.round(abs(pred))
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