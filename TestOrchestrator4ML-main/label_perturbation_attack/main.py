import time 
import sys
import argparse
from datetime import datetime
import numpy as np
import attack_model
import random_label_perturbation
import loss_based_label_perturbation
import probability_based_label_perturbation
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import cliffsDelta



def giveTimeStamp():
    tsObj = time.time()
    strToret = datetime.fromtimestamp(tsObj).strftime('%Y-%m-%d %H:%M:%S')
    return strToret
    
def run_experiment(model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
#     attack_model.calculate_metrics(X_train, y_train)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def run_random_perturbation_experiment(change_unit, model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    y_train = random_label_perturbation.random_label_perturbation(y_train, change_unit)
#     attack_model.calculate_metrics(X_train, y_train)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def run_loss_based_perturbation_experiment(change_unit, model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    y_train = loss_based_label_perturbation.label_flip_perturbation(X_train, X_test, X_val, y_train, y_test, y_val, change_unit)
#     attack_model.calculate_metrics(X_train, y_train)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def run_prob_based_perturbation_experiment(change_unit, I, g, model_name):
    X_train, X_test, X_val, y_train, y_test, y_val = attack_model.prepare_data()
    X_train, y_train = probability_based_label_perturbation.poisonous_data_perturbation(X_train, X_test, X_val, y_train, y_test, y_val, change_unit, I, g)
#     attack_model.calculate_metrics(X_train, y_train)
    precision, recall, fscore, auc = attack_model.perform_inference(X_train, X_test, y_train, y_test, model_name)
    return precision, recall, fscore, auc
    
def draw_plot(change, initial, random, loss, prob, plot_type):
    plt.figure()
    plt.plot(change, initial, 'r', label = "initial") 
    plt.plot(change, random, 'g', label = "random", alpha=0.5) 
    plt.plot(change, loss, 'b', label = "loss") 
    plt.plot(change, prob, 'c', label = "prob") 
    plt.xlabel("change (%)") 
    plt.ylabel(plot_type)
    plt.legend(loc='lower left')
    
    return plt
    
def call_loss(model_name):
    for change_percentage in range(0,90,10):
        change_unit = change_percentage/100
        print("Change: ", change_unit)  
        print('*'*100 )
        print("Loss based Perturbation")
        start_time = time.time()
        I = 10
        g = 10
        precision, recall, fscore, auc = run_loss_based_perturbation_experiment(change_unit, model_name)
        end_time = time.time()
        time_needed = round( (end_time - start_time) / 60, 5)
        if(auc <= 0.5):
            return precision, recall, fscore, auc, time_needed, change_unit
    return precision, recall, fscore, auc, time_needed, change_unit
    
def call_prob(I, g, model_name):
    for change_percentage in range(0,90,10):
        change_unit = change_percentage/100
        print("Change: ", change_unit)  
        print('*'*100 )
        print("Probability based Perturbation")
        start_time = time.time()
        I = 10
        g = 10
        precision, recall, fscore, auc = run_prob_based_perturbation_experiment(change_unit, I, g, model_name)
        end_time = time.time()
        time_needed = round( (end_time - start_time) / 60, 5)
        if(auc <= 0.5):
            return precision, recall, fscore, auc, time_needed, change_unit
    return precision, recall, fscore, auc, time_needed, change_unit
    
def calculate_stat(baseline_data, experiment_data):
    try:
        TS, p = stats.mannwhitneyu(list(experiment_data), list(baseline_data), alternative='less')
    except ValueError:
        TS, p = 0.0, 1.0 
        print("error")
    cliffs_delta = cliffsDelta.cliffsDelta(list(experiment_data), list(baseline_data))
    print(' p-value:{}, cliffs:{}'.format(p, cliffs_delta) )
    print('='*50)
    
def repeat_experiment():
    model_name = 'KNeighborsClassifier'

    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    
    loss_precision = []
    loss_recall = []
    loss_fscore = []
    loss_auc = []
    prob_precision = []
    prob_recall = []
    prob_fscore = []
    prob_auc = []
    loss_time = []
    prob_time = []
    loss_data = []
    prob_data = []
    
    
    for i in range(0,200,1): 
    
        print('*'*100 )
        print("Loss based Perturbation")
        precision, recall, fscore, auc, time_needed, data_needed = call_loss(model_name)
        loss_precision.append(precision)
        loss_recall.append(recall)
        loss_fscore.append(fscore)
        loss_auc.append(auc)
        loss_data.append(data_needed)
        loss_time.append(time_needed) 
    
        print('*'*100 )
        print("Probability based Perturbation")
        I = 10
        g = 10
        precision, recall, fscore, auc, time_needed, data_needed = call_prob(I, g, model_name)
        prob_precision.append(precision)
        prob_recall.append(recall)
        prob_fscore.append(fscore)
        prob_auc.append(auc)
        prob_data.append(data_needed)
        prob_time.append(time_needed) 

    print('*'*300 )
        
    print("---------data---------")
    calculate_stat(loss_data, prob_data)
    print("----------------------")
    print("---------time---------")
    calculate_stat(loss_time, prob_time)
    print("----------------------")
    print("Loss DATA:::[MEDIAN]:{}, [MEAN]:{}, [MAX]:{}, [MIN]:{}".format(np.median(list(loss_data)), np.mean(list(loss_data)), max(list(loss_data) ), min(list(loss_data) )   ) )
    print("Prob DATA:::[MEDIAN]:{}, [MEAN]:{}, [MAX]:{}, [MIN]:{}".format(np.median(list(prob_data)), np.mean(list(prob_data)),  max(list(prob_data)),  min(list(prob_data)) ) )
    print("Loss Time:::[MEDIAN]:{}, [MEAN]:{}, [MAX]:{}, [MIN]:{}".format(np.median(list(loss_time)), np.mean(list(loss_time)), max(list(loss_time) ), min(list(loss_time) )   ) )
    print("Prob Time:::[MEDIAN]:{}, [MEAN]:{}, [MAX]:{}, [MIN]:{}".format(np.median(list(prob_time)), np.mean(list(prob_time)),  max(list(prob_time)),  min(list(prob_time)) ) )
            
    
    print('Ended at:', giveTimeStamp() )
    print('*'*100 )
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print('Duration: {} minutes'.format(time_diff) )
    print( '*'*100  )  
    

def main(args):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args)

if __name__=='__main__':

    print(main(sys.argv[1:]))
    repeat_experiment()
    # model_name = 'KNeighborsClassifier'
# 
#     t1 = time.time()
#     print('Started at:', giveTimeStamp() )
#     
#     change = []
#     initial_precision = []
#     initial_recall = []
#     initial_fscore = []
#     initial_auc = []
#     random_precision = []
#     random_recall = []
#     random_fscore = []
#     random_auc = []
#     loss_precision = []
#     loss_recall = []
#     loss_fscore = []
#     loss_auc = []
#     prob_precision = []
#     prob_recall = []
#     prob_fscore = []
#     prob_auc = []
#     initial_time = []
#     random_time = []
#     loss_time = []
#     prob_time = []
#     
#     
#     
# 
#     for change_percentage in range(0,90,10):
#         change.append(change_percentage)
#         change_unit = change_percentage/100
#         print("Change: ", change_unit)   
#         
#         print('*'*100 )
#         print("Initial Experiment")
#         start_time = time.time()
#         precision, recall, fscore, auc = run_experiment(model_name)
#         initial_precision.append(precision)
#         initial_recall.append(recall)
#         initial_fscore.append(fscore)
#         initial_auc.append(auc)
#         end_time = time.time()
#         initial_time.append(round( (end_time - start_time) / 60, 5)) 
#     
#         print('*'*100 )
#         print("Random Perturbation")
#         start_time = time.time()
#         precision, recall, fscore, auc = run_random_perturbation_experiment(change_unit, model_name)
#         random_precision.append(precision)
#         random_recall.append(recall)
#         random_fscore.append(fscore)
#         random_auc.append(auc)
#         end_time = time.time()
#         random_time.append(round( (end_time - start_time) / 60, 5)) 
#     
#         print('*'*100 )
#         print("Loss based Perturbation")
#         start_time = time.time()
#         precision, recall, fscore, auc = run_loss_based_perturbation_experiment(change_unit, model_name)
#         loss_precision.append(precision)
#         loss_recall.append(recall)
#         loss_fscore.append(fscore)
#         loss_auc.append(auc)
#         end_time = time.time()
#         loss_time.append(round( (end_time - start_time) / 60, 5)) 
#     
#         print('*'*100 )
#         print("Probability based Perturbation")
#         start_time = time.time()
#         I = 10
#         g = 10
#         precision, recall, fscore, auc = run_prob_based_perturbation_experiment(change_unit, I, g, model_name)
#         prob_precision.append(precision)
#         prob_recall.append(recall)
#         prob_fscore.append(fscore)
#         prob_auc.append(auc)
#         end_time = time.time()
#         prob_time.append(round( (end_time - start_time) / 60, 5)) 
# 
#         print('*'*300 )
#     
#     pp = PdfPages('plots.pdf')
#         
#     precision_plot = draw_plot(change, initial_precision, random_precision, loss_precision, prob_precision, "Precision")
#     pp.savefig()
#     recall_plot = draw_plot(change, initial_recall, random_recall, loss_recall, prob_recall, "Recall")
#     pp.savefig()
#     fscore_plot = draw_plot(change, initial_fscore, random_fscore, loss_fscore, prob_fscore, "F-score")
#     pp.savefig()
#     auc_plot = draw_plot(change, initial_auc, random_auc, loss_auc, prob_auc, "AUC")
#     pp.savefig()
#     time_plot = draw_plot(change, initial_time, random_time, loss_time, prob_time, "TIME")
#     pp.savefig()
# 
#     pp.close()
#     
#     
#     print('Ended at:', giveTimeStamp() )
#     print('*'*100 )
#     t2 = time.time()
#     time_diff = round( (t2 - t1 ) / 60, 5) 
#     print('Duration: {} minutes'.format(time_diff) )
#     print( '*'*100  )  