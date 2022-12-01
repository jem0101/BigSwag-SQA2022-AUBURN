import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def label_flip_perturbation(X_train, X_test, X_val, y_train, y_test, y_val, change_unit):    
    S_p = y_train
    p = int(y_train.size * change_unit)
    I = list(range(0, y_train.size)) 
    e = [0] * y_train.size
    
    for k in range(0, p):
        for j in I:
            S_1 = list(S_p)
#             S_1[j] = np.random.randint(0, 9)
            S_1[j] ^= 1
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
            clf.fit(X_train, S_1)
            pred = clf.predict(X_val)
            acc = accuracy_score(y_val, pred)
            e[j] = 1 - acc
        
        max_value = max(e)
#         print(max_value)
        i_k = e.index(max_value)
        I.remove(i_k) 
        e[i_k] = 0
        S_p.iloc[i_k] ^= 1
    
#     print(S_p.shape)
    print("after perturbation y_train count: ", np.unique(S_p,return_counts=True))
    
    return S_p
    
    
    