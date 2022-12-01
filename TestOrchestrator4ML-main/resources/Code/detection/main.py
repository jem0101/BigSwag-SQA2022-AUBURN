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
  
def get_test_details(test_script):
    test_name_list = []
    test_with_assert_list = []
    py_tree = py_parser.getPythonParseObject(test_script)
#     print(test_script)
    func_assert_parameter_list  = py_parser.getTestNames( py_tree ) 
    for func_ in func_assert_parameter_list:
#         print("###############")
#         print(func_) 
#         print("###############")
        the_tup = ( test_script, func_[0])
        test_name_list.append( the_tup )
        if (len(func_) > 2 ):
            the_assert_tup = (test_script, func_[0], tuple([e for e in func_[3]]))
#             print(the_assert_tup)
            test_with_assert_list.append( the_assert_tup )
            
    return test_name_list, test_with_assert_list


def checkClassificationAlgoTest(test_script):
    print("algo check: ", test_script)
    py_tree = py_parser.getPythonParseObject(test_script)
    classification_algo_list = py_parser.getClassificationAlgoNames( py_tree ) 
    if len(classification_algo_list) > 0:
        return 0
    else:
        return 1
  
  
def checkAccuracyTest(test_script):
    print("metric check: ", test_script)
    py_tree = py_parser.getPythonParseObject(test_script)
    metric_list = py_parser.getMetricNames( py_tree ) 
    if len(metric_list) > 0:
        return 0
    else:
        return 1
    
    
def chackAttackTest(test_script, assert_list):
    attack_check = []
    print("attack check: ", test_script)
    py_tree = py_parser.getPythonParseObject(test_script)
    metric_check_list = py_parser.getmetricLHSNames( py_tree )
    for item in metric_check_list:
        for assert_item in assert_list:
            if item in assert_item[2]:
                attack_check.append(item)
    if len(attack_check) > 1:
        return 0
    else:
        return 1

def runDetectionTest(inp_dir, test_output_csv, test_assert_output_csv, flag_output_csv):
    flag_list = []
    df_test_list = []
    df_test_with_assert_list = []
    for root_, dirnames, filenames in os.walk(inp_dir, topdown=False):
        NO_TEST = 1
        NO_ALGO = 1
        NO_ACCURACY = 1
        NO_ATTACK_CHECK = 1
        if (len(root_.split('/')) > 4): 
            repo = inp_dir + "/" + root_.split('/')[5] 
            for file_ in filenames:
                full_path_file = os.path.join(root_, file_) 
                if(os.path.exists(full_path_file)):
                    if ((file_.endswith('test.py')) or (file_.endswith('tests.py')) or (file_.endswith('Test.py')) or (file_.startswith('test')) or (file_.startswith('Test')))  :
                        temp_test_list, temp_test_with_assert_list = get_test_details(full_path_file)
                        df_test_list = df_test_list + temp_test_list 
                        df_test_with_assert_list = df_test_with_assert_list + temp_test_with_assert_list
                        if (len(temp_test_list) > 0) :
                            NO_TEST = 0
                            if (NO_ALGO == 1):
                                NO_ALGO = checkClassificationAlgoTest(full_path_file)
                            if (NO_ACCURACY == 1 and NO_ALGO == 0):
                                NO_ACCURACY = checkAccuracyTest(full_path_file)
                            if (NO_ATTACK_CHECK == 1 and NO_ACCURACY == 0):
                                NO_ATTACK_CHECK = chackAttackTest(full_path_file, temp_test_with_assert_list)
        flag_count = NO_TEST + NO_ALGO + NO_ACCURACY + NO_ATTACK_CHECK
        flag_tup = (repo, NO_TEST, NO_ALGO, NO_ACCURACY, NO_ATTACK_CHECK, flag_count)
        flag_list.append(flag_tup)
    if (len(df_test_list) > 0):
        full_test_df = pd.DataFrame( df_test_list )
        full_test_df.to_csv(test_output_csv, header= ["ML_SCRIPT","TEST_NAME"], index=False, encoding= constants.UTF_ENCODING)
    if (len(df_test_with_assert_list) > 0):
        full_test_with_assert_df = pd.DataFrame( df_test_with_assert_list ) 
        full_test_with_assert_df.to_csv(test_assert_output_csv, header= ["ML_SCRIPT","TEST_NAME","PARAMETER"],index=False, encoding= constants.UTF_ENCODING)  
    flag_df = pd.DataFrame( flag_list, columns= ["PROJECT","NO_TEST","NO_ALGO", "NO_ACCURACY", "NO_ATTACK_CHECK", "FLAG_COUNT"] ) 
    flag_df = flag_df.sort_values('FLAG_COUNT').drop_duplicates('PROJECT', keep='first')
    flag_df = flag_df.drop('FLAG_COUNT', axis=1)
    flag_df.to_csv(flag_output_csv ,index=False, encoding= constants.UTF_ENCODING)  


def runDetectionTestModelzoo(inp_dir, test_output_csv, test_assert_output_csv, flag_output_csv):
    flag_list = []
    df_test_list = []
    df_test_with_assert_list = []
    for root_, dirnames, filenames in os.walk(inp_dir, topdown=False):
        NO_TEST = 1
        NO_ALGO = 1
        NO_ACCURACY = 1
        NO_ATTACK_CHECK = 1
        if (len(root_.split('/')) > 6): 
            repo = inp_dir + "/" + root_.split('/')[5] + "/" + root_.split('/')[6] # for modelzoo
            for file_ in filenames:
                full_path_file = os.path.join(root_, file_) 
                if(os.path.exists(full_path_file)):
                    if ((file_.endswith('test.py')) or (file_.endswith('tests.py')) or (file_.endswith('Test.py')) or (file_.startswith('test')) or (file_.startswith('Test')))  :
                        temp_test_list, temp_test_with_assert_list = get_test_details(full_path_file)
                        df_test_list = df_test_list + temp_test_list 
                        df_test_with_assert_list = df_test_with_assert_list + temp_test_with_assert_list
                        if (len(temp_test_list) > 0) :
                            NO_TEST = 0
                            if (NO_ALGO == 1):
                                NO_ALGO = checkClassificationAlgoTest(full_path_file)
                            if (NO_ACCURACY == 1 and NO_ALGO == 0):
                                NO_ACCURACY = checkAccuracyTest(full_path_file)
                            if (NO_ATTACK_CHECK == 1 and NO_ACCURACY == 0):
                                NO_ATTACK_CHECK = chackAttackTest(full_path_file, temp_test_with_assert_list)
        flag_count = NO_TEST + NO_ALGO + NO_ACCURACY + NO_ATTACK_CHECK
        flag_tup = (repo, NO_TEST, NO_ALGO, NO_ACCURACY, NO_ATTACK_CHECK, flag_count)
        flag_list.append(flag_tup)
    if (len(df_test_list) > 0):
        full_test_df = pd.DataFrame( df_test_list )
        full_test_df.to_csv(test_output_csv, header= ["ML_SCRIPT","TEST_NAME"], index=False, encoding= constants.UTF_ENCODING)
    if (len(df_test_with_assert_list) > 0):
        full_test_with_assert_df = pd.DataFrame( df_test_with_assert_list ) 
        full_test_with_assert_df.to_csv(test_assert_output_csv, header= ["ML_SCRIPT","TEST_NAME","PARAMETER"],index=False, encoding= constants.UTF_ENCODING)  
    flag_df = pd.DataFrame( flag_list, columns= ["PROJECT","NO_TEST","NO_ALGO", "NO_ACCURACY", "NO_ATTACK_CHECK", "FLAG_COUNT"] ) 
    flag_df = flag_df.sort_values('FLAG_COUNT').drop_duplicates('PROJECT', keep='first')
    flag_df = flag_df.drop('FLAG_COUNT', axis=1)
    flag_df.to_csv(flag_output_csv ,index=False, encoding= constants.UTF_ENCODING)  


def runDete(inp_dir, test_output_csv, test_assert_output_csv, flag_output_csv):
    flag_list = []
    df_test_list = []
    df_test_with_assert_list = []
    for root_, dirnames, filenames in os.walk(inp_dir, topdown=False):
        NO_TEST = 1
        NO_ALGO = 1
        NO_ACCURACY = 1
        NO_ATTACK_CHECK = 1
        if (len(root_.split('/')) > 0): 
            repo = inp_dir 
            for file_ in filenames:
                full_path_file = os.path.join(root_, file_) 
                if(os.path.exists(full_path_file)):
                    if ((file_.endswith('test.py')) or (file_.endswith('tests.py')) or (file_.endswith('Test.py')) or (file_.startswith('test')) or (file_.startswith('Test')))  :
                        temp_test_list, temp_test_with_assert_list = get_test_details(full_path_file)
                        df_test_list = df_test_list + temp_test_list 
                        df_test_with_assert_list = df_test_with_assert_list + temp_test_with_assert_list
                        if (len(temp_test_list) > 0) :
                            NO_TEST = 0
                            if (NO_ALGO == 1):
                                NO_ALGO = checkClassificationAlgoTest(full_path_file)
                            if (NO_ACCURACY == 1 and NO_ALGO == 0):
                                NO_ACCURACY = checkAccuracyTest(full_path_file)
                            if (NO_ATTACK_CHECK == 1 and NO_ACCURACY == 0):
                                NO_ATTACK_CHECK = chackAttackTest(full_path_file, temp_test_with_assert_list)
        flag_count = NO_TEST + NO_ALGO + NO_ACCURACY + NO_ATTACK_CHECK
        flag_tup = (repo, NO_TEST, NO_ALGO, NO_ACCURACY, NO_ATTACK_CHECK, flag_count)
        flag_list.append(flag_tup)
    if (len(df_test_list) > 0):
        full_test_df = pd.DataFrame( df_test_list )
        full_test_df.to_csv(test_output_csv, header= ["ML_SCRIPT","TEST_NAME"], index=False, encoding= constants.UTF_ENCODING)
    if (len(df_test_with_assert_list) > 0):
        full_test_with_assert_df = pd.DataFrame( df_test_with_assert_list ) 
        full_test_with_assert_df.to_csv(test_assert_output_csv, header= ["ML_SCRIPT","TEST_NAME","PARAMETER"],index=False, encoding= constants.UTF_ENCODING)  
    flag_df = pd.DataFrame( flag_list, columns= ["PROJECT","NO_TEST","NO_ALGO", "NO_ACCURACY", "NO_ATTACK_CHECK", "FLAG_COUNT"] ) 
    flag_df = flag_df.sort_values('FLAG_COUNT').drop_duplicates('PROJECT', keep='first')
    flag_df = flag_df.drop('FLAG_COUNT', axis=1)
    flag_df.to_csv(flag_output_csv ,index=False, encoding= constants.UTF_ENCODING)  

if __name__=='__main__': 

	t1 = time.time()
	print('Started at:', giveTimeStamp() )
	print('*'*100 )
	
# 	print('*'*100 )
# 	repo_dir   = 'example/'
# 	test_output_csv = 'test_name.csv'
# 	test_assert_output_csv = 'test_assert.csv'
# 	flag_output_csv = 'test_flag.csv'
# 	full_dict  = runDete(repo_dir, test_output_csv, test_assert_output_csv, flag_output_csv)
# 	print('*'*100 )
	
	print('*'*100 )
	repo_dir   = '../../Data/supervised/GITHUB_REPOS/'
	test_output_csv = '../../Output/TEST_NAME_SUPERVISED_OUTPUT_GITHUB.csv'
	test_assert_output_csv = '../../Output/TEST_ASSERT_SUPERVISED_OUTPUT_GITHUB.csv'
	flag_output_csv = '../../Output/FLAG_SUPERVISED_OUTPUT_GITHUB.csv'
	full_dict  = runDetectionTest(repo_dir, test_output_csv, test_assert_output_csv, flag_output_csv)
	print('*'*100 )
	
	print('*'*100 )
	repo_dir   = '../../Data/supervised/GITLAB_REPOS/'
	test_output_csv = '../../Output/TEST_NAME_SUPERVISED_OUTPUT_GITLAB.csv'
	test_assert_output_csv = '../../Output/TEST_ASSERT_SUPERVISED_OUTPUT_GITLAB.csv'
	flag_output_csv = '../../Output/FLAG_SUPERVISED_OUTPUT_GITLAB.csv'
	full_dict  = runDetectionTest(repo_dir, test_output_csv, test_assert_output_csv, flag_output_csv)
	print('*'*100 )
	
	print('*'*100 )
	repo_dir   = '../../Data/supervised/MODELZOO/'
	test_output_csv = '../../Output/TEST_NAME_SUPERVISED_OUTPUT_MODELZOO.csv'
	test_assert_output_csv = '../../Output/TEST_ASSERT_SUPERVISED_OUTPUT_MODELZOO.csv'
	flag_output_csv = '../../Output/FLAG_SUPERVISED_OUTPUT_MODELZOO.csv'
	full_dict  = runDetectionTestModelzoo(repo_dir, test_output_csv, test_assert_output_csv, flag_output_csv)
	print('*'*100 )
 	
	print('Ended at:', giveTimeStamp() )
	print('*'*100 )
	
	t2 = time.time()
	time_diff = round( (t2 - t1 ) / 60, 5) 
	print('Duration: {} minutes'.format(time_diff) )
	print('*'*100 )
