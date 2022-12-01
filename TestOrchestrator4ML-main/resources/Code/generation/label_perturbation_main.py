import time 
import sys
import argparse
from datetime import datetime
import numpy as np
import attack_model
import random_label_perturbation
import loss_based_label_perturbation
import probability_based_label_perturbation



def giveTimeStamp():
    tsObj = time.time()
    strToret = datetime.fromtimestamp(tsObj).strftime('%Y-%m-%d %H:%M:%S')
    return strToret
    
def run_experiment(model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def run_random_perturbation_experiment(change_unit, model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    y_train = random_label_perturbation.random_label_perturbation(y_train, change_unit)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def run_loss_based_perturbation_experiment(change_unit, model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    y_train = loss_based_label_perturbation.label_flip_perturbation(X_train, X_test, X_val, y_train, y_test, y_val, change_unit)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def run_prob_based_perturbation_experiment(change_unit, I, g, model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    X_train, y_train = probability_based_label_perturbation.poisonous_data_perturbation(X_train, X_test, X_val, y_train, y_test, y_val, change_unit, I, g)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    

def run_label_perturbation(model_name):

    print(model_name)

    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    
    change = []
    initial_precision = []
    initial_recall = []
    initial_fscore = []
    initial_auc = []
    random_precision = []
    random_recall = []
    random_fscore = []
    random_auc = []
    loss_precision = []
    loss_recall = []
    loss_fscore = []
    loss_auc = []
    prob_precision = []
    prob_recall = []
    prob_fscore = []
    prob_auc = []
    

    for change_percentage in range(20,30,10):
        change.append(change_percentage)
        change_unit = change_percentage/100
        print("Change: ", change_unit)   
        
        print('*'*100 )
        print("Initial Experiment")
        precision, recall, fscore, auc = run_experiment(model_name)
        initial_precision.append(precision)
        initial_recall.append(recall)
        initial_fscore.append(fscore)
        initial_auc.append(auc)
    
        print('*'*100 )
        print("Random Perturbation")
        precision, recall, fscore, auc = run_random_perturbation_experiment(change_unit, model_name)
        random_precision.append(precision)
        random_recall.append(recall)
        random_fscore.append(fscore)
        random_auc.append(auc)
    
        print('*'*100 )
        print("Loss based Perturbation")
        precision, recall, fscore, auc = run_loss_based_perturbation_experiment(change_unit, model_name)
        loss_precision.append(precision)
        loss_recall.append(recall)
        loss_fscore.append(fscore)
        loss_auc.append(auc)
    
        print('*'*100 )
        print("Probability based Perturbation")
        I = 10
        g = 10
        precision, recall, fscore, auc = run_prob_based_perturbation_experiment(change_unit, I, g, model_name)
        prob_precision.append(precision)
        prob_recall.append(recall)
        prob_fscore.append(fscore)
        prob_auc.append(auc)

        print('*'*300 )
    
    
    print('Ended at:', giveTimeStamp() )
    print('*'*100 )
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print('Duration: {} minutes'.format(time_diff) )
    print( '*'*100  )  
    
    return initial_auc, random_auc, loss_auc, prob_auc