import os
import pandas as pd 
import numpy as np
import csv 
import time 
from datetime import datetime
import subprocess
import shutil
from git import Repo
from git import exc 


def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.fromtimestamp(tsObj).strftime('%Y-%m-%d %H:%M:%S')
  return strToret

    
def getDevEmailForCommit(repo_path_param, hash_):
    author_emails = []

    cdCommand         = "cd " + repo_path_param + " ; "
    commitCountCmd    = " git log --format='%ae'" + hash_ + "^!"
    command2Run = cdCommand + commitCountCmd

    author_emails = str(subprocess.check_output(['bash','-c', command2Run]))
    author_emails = author_emails.split('\n')
    author_emails = [x_.replace(hash_, '') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    author_emails = [x_.replace('^', '') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    author_emails = [x_.replace('!', '') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    author_emails = [x_.replace('\\n', ',') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    try:
        author_emails = author_emails[0].split(',')
        author_emails = [x_ for x_ in author_emails if len(x_) > 3 ] 
        author_emails = list(np.unique(author_emails) )
    except IndexError as e_:
        pass
    return author_emails  
    
    
def getDevCount(full_path_to_repo, branchName='master', explore=1000):
    repo_emails = []
    if os.path.exists(full_path_to_repo):
        try:
            repo_  = Repo(full_path_to_repo, search_parent_directories=True)
            try:
                all_commits = list(repo_.iter_commits(branchName))   

                for commit_ in all_commits:
                    commit_hash = commit_.hexsha
                    emails = getDevEmailForCommit(full_path_to_repo, commit_hash)
                    repo_emails = repo_emails + emails
                
            except exc.GitCommandError:
                print('Skipping this repo ... due to branch name problem', full_path_to_repo )
        except exc.InvalidGitRepositoryError:
            print('Skipping this repo ... due to not finding the git repo', full_path_to_repo )
    else:
        repo_emails = [ str(x_) for x_ in range(10) ]
    
    return len(repo_emails)
            
            
def getTopRepos(repo_list): 
    repo_dev_dict = {}
    all_list = []
    for repo_ in repo_list:
        print(repo_)
        dev_count = getDevCount(repo_)
        repo_dev_dict[repo_] = dev_count
        print('#'*100 )
    repo_dev_dict = sorted(repo_dev_dict.items(), key=lambda x: x[1], reverse=True)

    df_ = pd.DataFrame( repo_dev_dict ) 
    df_.to_csv('../Output/dev_count.csv', header=['REPO', 'DEVS'] , index=False, encoding='utf-8')    

    

if __name__=='__main__':

    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    print('*'*100 )

    list_ = []
    for root, dirs, files in os.walk("../Data/supervised/GITHUB_REPOS", topdown=False):
        try:
            list_ = np.append(list_, "../Data/supervised/GITHUB_REPOS/" + root.split('/')[4])
        except:
            print()
    
    list_ = np.unique(list_)
    getTopRepos(list_)

    print('*'*100 )
    print('Ended at:', giveTimeStamp() )
    print('*'*100 )
    t2 = time.time()
    time_diff = round( (t2 - t1 ) / 60, 5) 
    print('Duration: {} minutes'.format(time_diff) )
    print( '*'*100  )  
