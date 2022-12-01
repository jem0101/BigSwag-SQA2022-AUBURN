import constants 
import time 
import datetime 
import os 
import pandas as pd
import py_parser 
import numpy as np 
import label_perturbation_main


def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.datetime.fromtimestamp(tsObj).strftime(constants.TIME_FORMAT) 
  return strToret


def generateUnitTest(algo, attack_type):
    file_name = "../../Output/attack_unit_test/test_attack_" + algo + ".py"
    with open(file_name,"w+") as file:
        file.write("import unittest\n")
        file.write("import label_perturbation_main\n")
        file.write("import " + algo + "\n")
        file.write("\n\n")
        file.write("class TestAttack( unittest.TestCase ):\n")
        file.write("\tdef test_attack(self):\n")
        file.write("\t\tchange_unit = 0.5\n")
        file.write("\t\tprecision4model1, recall4model1, fscore4model1, auc4model1 = label_perturbation_main.run_experiment(" + "algo" + ")\n")
        if (attack_type == 'random'):
            file.write("\t\tprecision4model2, recall4model2, fscore4model2, auc4model2 = label_perturbation_main.run_random_perturbation_experiment(change_unit, "+ "algo" + ")\n")
        if (attack_type == 'loss'):
            file.write("\t\tprecision4model2, recall4model2, fscore4model2, auc4model2 = label_perturbation_main.run_loss_based_perturbation_experiment(change_unit, "+ "algo" + ")\n")
        if (attack_type == 'prob'):
            file.write("\t\tprecision4model2, recall4model2, fscore4model2, auc4model2 = label_perturbation_main.run_prob_based_perturbation_experiment(change_unit, 10, 10, "+ "algo" + ")\n")
        file.write("\t\tself.assertEqual(auc4model1, auc4model2, \"DECREASE IN AUC VALUE ... POSSIBLE ATTACK?\"  )\n")


def generateAttack(inp_dir, delta):
    if os.path.exists(inp_dir):
        algo_df = pd.read_csv(inp_dir)
    else:
        return
    for index, row in algo_df.iterrows():
        row['ALGO_NAME'] = row['ALGO_NAME'].replace('[', '')
        row['ALGO_NAME'] = row['ALGO_NAME'].replace(']', '')
        algo_list = row['ALGO_NAME'].split(',')
        for algo in algo_list:
            algo = algo.replace('\'', '')
            algo = algo.replace(' ', '')
            initial_auc, random_auc, loss_auc, prob_auc = label_perturbation_main.run_label_perturbation(algo)
            random_diff = random_auc[0] -  initial_auc[0]
            loss_diff = loss_auc[0] -  initial_auc[0]
            prob_diff = prob_auc[0] -  initial_auc[0]
            if (random_diff < delta): 
                generateUnitTest(algo, 'random')
            elif (loss_diff < delta): 
                generateUnitTest(algo, 'loss')
            elif (prob_diff < delta): 
                generateUnitTest(algo, 'prob')
                

if __name__=='__main__': 

    delta = 0.5
    
    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    print('*'*100 )
    
#     print('*'*100 )
#     algo_input_csv = 'algo_list.csv'
#     generateAttack(algo_input_csv, delta)
#     print('*'*100 )

    print('*'*100 )
    algo_input_csv = '../../Output/ALGO_SUPERVISED_OUTPUT_GITHUB.csv'
    generateAttack(algo_input_csv, delta)
    print('*'*100 )
    
    print('*'*100 )
    algo_input_csv = '../../Output/ALGO_SUPERVISED_OUTPUT_GITLAB.csv'
    generateAttack(algo_input_csv, delta)
    print('*'*100 )
    
    print('*'*100 )
    algo_input_csv = '../../Output/ALGO_SUPERVISED_OUTPUT_MODELZOO.csv'
    generateAttack(algo_input_csv, delta)
    print('*'*100 )

    print('Ended at:', giveTimeStamp() )
    print('*'*100 )
    
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print('Duration: {} minutes'.format(time_diff) )
    print('*'*100 )
