import numpy as np

def random_label_perturbation(y_train, change_unit):    
    for index in np.random.randint(0,y_train.size-1, int(y_train.size * change_unit)):
        # y_train[index] = np.random.randint(0, 9)
        y_train.iloc[index] ^= 1
        
#     print(y_train.shape)
    print("after perturbation y_train count: ", np.unique(y_train,return_counts=True))
    
    return y_train