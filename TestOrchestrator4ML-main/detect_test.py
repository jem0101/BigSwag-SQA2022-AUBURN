import os
import fnmatch
import pandas as pd 
import numpy as np
import csv 
import time 
from datetime import datetime


def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.fromtimestamp(tsObj).strftime('%Y-%m-%d %H:%M:%S')
  return strToret
   
    
def checkTestFile(path2dir): 
    """
    Check if project includes tests.
    We look for files that start with test (or tests, Test, Tests) or end with test.py (tests.py,
    Test.py, Tests.py).
    """
    repo_test_dict = {}
    for root_, dirnames, filenames in os.walk(path2dir, topdown=False):
        if (len(root_.split('/')) > 4): 
            # repo = path2dir + "/" + root_.split('/')[4] + "/" + root_.split('/')[5] # for modelzoo
            repo = path2dir + "/" + root_.split('/')[4] 
            for file_ in filenames:
                full_path_file = os.path.join(root_, file_) 
                if(os.path.exists(full_path_file)):
                    if ((file_.endswith('test.py')) or (file_.endswith('tests.py')) or (file_.endswith('Test.py')) or (file_.startswith('test')) or (file_.startswith('Test')))  :
                        repo_test_dict[repo] = 1
        if(not (repo in repo_test_dict)):
            repo_test_dict[repo] = 0
                        
    df = pd.DataFrame.from_dict(repo_test_dict, orient='index', columns=['TEST']) 
    print("Total row: ", len(df))
    print("Test: ", df[df == 1].sum(axis=0))
    df.to_csv('../Output/github_test_detect.csv', encoding='utf-8')              
 

if __name__=='__main__':

    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    print('*'*100 )

    project_with_test = checkTestFile("../Data/supervised/GITHUB_REPOS")
#     project_with_test = checkTestFile("../Data/supervised/GITLAB_REPOS")
#     project_with_test = checkTestFile("../Data/supervised/MODELZOO") 

    print('*'*100 )
    print('Ended at:', giveTimeStamp() )
    print('*'*100 )
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print('Duration: {} minutes'.format(time_diff) )
    print( '*'*100  )  