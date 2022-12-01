Algorithm to detect projects that do not check for flip label attacks 
=================================================

```
1. repo_dirs <- get all projects' repos 
2. for each repo_dir in repo_dirs:
3.     repo_tests <- getTestCasesInRepo()
4.     for each test in repo_tests:

5.         extract test names from tests e.g. you will extract `testKeyExtraction` from  https://github.com/paser-group/KubeSec/blob/e3b59f8d9da88067f0bb6940a7e4952ebf864d9c/TEST_PARSING.py#L13
6.         check for assert() blocks, if there exists such, then extract parameters. For example for https://github.com/paser-group/KubeSec/blob/e3b59f8d9da88067f0bb6940a7e4952ebf864d9c/TEST_PARSING.py#L13, you will extract three things: `assertEqual`, `oracle_value`, and `len(yaml_as_dict)`
7.         identify all of the following: 

8.         7a. tests exist but no tests that focus on classification algorithms (tracking API calls used to build models: knn(), RF(), SVM() etc.), if so assign 'NO_ALGO' flag
9.         7b. tests exist for classification algorithms, but does not track accuracy (basically no tests that check for precision, recall etc.), if so assign 'NO_ACCURACY' flag
10.        7c. tests exist for classification algorithms with accuracy but no use of label flipping (you can use your implementation to see if classification decreases by flipping labels), if so assign 'NO_ATTACK_CHECK' flag
11.    for a project if there are no tests, then assign 'NO_TEST' flag for the project 
12. create a CSV file with all test names from all ML projects for each dataset 
13. create a CSV file with all test names and their assert block elements from all ML projects for each dataset 
14. create a CSV file with all project names and the flags identified in steps 7-10.         
```