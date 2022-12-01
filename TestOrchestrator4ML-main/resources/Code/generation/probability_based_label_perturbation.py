import numpy as np
import pandas as pd
import random
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def generate_malicious_instance(X_train, y_train, g):
    M = X_train.shape[1]
    generated_X = [0] * M
    no_of_elements_in_bin = int(len(y_train)/g)
    for j in range(0, M):
        weight = [0] * g
        total_weight = 0
        attribute_prob = [0] * g
        for k in range(0, g):
            start_index = no_of_elements_in_bin * k
            end_index = no_of_elements_in_bin * (k + 1)
            bin_element = y_train[start_index:end_index]
            no_of_entries_in_attacking_class = np.count_nonzero(bin_element == 1)  # consider 1 as attacking class
            no_of_entries_in_attacked_class = no_of_elements_in_bin - no_of_entries_in_attacking_class 
            
            weight[k] = no_of_entries_in_attacked_class/no_of_entries_in_attacking_class 
            total_weight += weight[k]
        for k in range(0, g):
            attribute_prob[k] = weight[k]/total_weight
        generated_X[j] = attribute_prob[random.randrange(0, g)]
     
    generated_y = 1  # consider 1 as attacking class
    
    return generated_X, generated_y
    
    
def poisonous_data_perturbation(X_train, X_test, X_val, y_train, y_test, y_val, change_unit, I, g):    
    D_1_X = X_train
    D_1_y = y_train
    N_1 = int(y_train.size * change_unit)
    
    acc = [0] * I
    
    for k in range(0, N_1):
        temp_D_1_X = D_1_X
        temp_D_1_y = D_1_y
        
        for i in range(0, I):
            generated_X, generated_y = generate_malicious_instance(X_train, y_train, g)
            generated_X = pd.DataFrame([generated_X], columns= X_train.columns)
#             temp_D_1_X = np.append(temp_D_1_X, generated_X)
            temp_D_1_X = temp_D_1_X.append(generated_X)
            temp_D_1_y = np.append(temp_D_1_y, generated_y)
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
            clf.fit(temp_D_1_X, temp_D_1_y)
            pred = clf.predict(X_val)
            acc[i] = accuracy_score(y_val, pred)
        
        min_value = min(acc)
        i_1 = acc.index(min_value)
        
#         np.append(D_1_X, temp_D_1_X[len(D_1_X) + i_1])
        D_1_X = D_1_X.append(temp_D_1_X.iloc[len(D_1_X) + i_1])
        D_1_y = np.append(D_1_y, temp_D_1_y[len(D_1_y) + i_1])
    
    print("after perturbation y_train count: ", np.unique(D_1_y,return_counts=True))
    
    return D_1_X, D_1_y