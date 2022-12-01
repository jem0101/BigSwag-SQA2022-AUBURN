import constants 
import time 
import datetime 
import os 
import pandas as pd
import py_parser 
import numpy as np 


def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.datetime.fromtimestamp(tsObj).strftime(constants.TIME_FORMAT) 
  return strToret


def getClassificationAlgoList(test_script):
#     print("script: ", test_script)
    py_tree = py_parser.getPythonParseObject(test_script)
    classification_algo_list = py_parser.getClassificationAlgoNames( py_tree ) 
    return classification_algo_list


def identifyAlgo(inp_dir, algo_output_csv):
#     print(inp_dir)
    algo_list = []
    flag_df = pd.read_csv(inp_dir)
    flag_df['flag_count'] = flag_df.iloc[:, 1:].sum(axis=1)
    for index, row in flag_df.iterrows():
        if (row['flag_count'] > 0):
            algo_list_per_project = []
            project_name = '../' + row['PROJECT']
            print(project_name)
            for root_, dirnames, filenames in os.walk(project_name):
                for file_ in filenames:
                    full_path_file = os.path.join(root_, file_) 
                    if(os.path.exists(full_path_file)):
                        if (file_.endswith('.py')):
                            algo_list_per_project += getClassificationAlgoList(full_path_file)
                            algo_list_per_project = list(set(algo_list_per_project)) 
            if (len(algo_list_per_project) > 0):
                print(algo_list_per_project)
                the_tup = (project_name, algo_list_per_project)
                algo_list.append(the_tup)
     
    if (len(algo_list) > 0):
        algo_list_df = pd.DataFrame( algo_list ) 
        algo_list_df.to_csv(algo_output_csv, header= ["PROJECT","ALGO_NAME"],index=False, encoding= constants.UTF_ENCODING)  


if __name__=='__main__': 

	t1 = time.time()
	print('Started at:', giveTimeStamp() )
	print('*'*100 )
	
# 	print('*'*100 )
# 	flag_input_csv = 'test_flag.csv'
# 	algo_output_csv = 'algo_list.csv'
# 	full_dict  = identifyAlgo(flag_input_csv, algo_output_csv)
# 	print('*'*100 )
# 	
	print('*'*100 )
	flag_input_csv = '../../../Output/FLAG_SUPERVISED_OUTPUT_GITHUB.csv'
	algo_output_csv = '../../../Output/ALGO_SUPERVISED_OUTPUT_GITHUB.csv'
	full_dict  = identifyAlgo(flag_input_csv, algo_output_csv)
	print('*'*100 )
	
	print('*'*100 )
	flag_input_csv = '../../../Output/FLAG_SUPERVISED_OUTPUT_GITLAB.csv'
	algo_output_csv = '../../../Output/ALGO_SUPERVISED_OUTPUT_GITLAB.csv'
	full_dict  = identifyAlgo(flag_input_csv, algo_output_csv)
	print('*'*100 )
	
	print('*'*100 )
	flag_input_csv = '../../../Output/FLAG_SUPERVISED_OUTPUT_MODELZOO.csv'
	algo_output_csv = '../../../Output/ALGO_SUPERVISED_OUTPUT_MODELZOO.csv'
	full_dict  = identifyAlgo(flag_input_csv, algo_output_csv)
	print('*'*100 )
 	
	print('Ended at:', giveTimeStamp() )
	print('*'*100 )
	
	t2 = time.time()
	time_diff = round( (t2 - t1 ) / 60, 5) 
	print('Duration: {} minutes'.format(time_diff) )
	print('*'*100 )
