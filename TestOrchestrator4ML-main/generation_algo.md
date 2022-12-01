Algorithm to generate test cases to check for flip label attacks 
=================================================

1. from detection algorithm import CSV file with flags for projects 
2. for each project with any 'NO_*' flag:
```
3.     algo_list <- identify classification algorithms used in the projects 
4.     for each algo in algo_list: 
5.         generate label flipping attacks using loss function approach 
6.         generate label flipping attacks using probability-based approach 
7.         generate label flipping attacks using pure random generation approach (the one where you randomly change indices)
8.         calculate accuracy for all approaches for steps#5, 6, 7 
9.         keep track of generated labels from steps#5, 6, 7 using some data structure 
10.        calculate accuracy of algo with non-poisoned dataset 
11.        if accuracy decreases compared to that of non-poisoned dataset by DELTA, then generate test case as follows: 
12.           create a unit test that will 
13.                    create two models: one with label flipping and one with non-poisonous
14.                    calculate accuracy for two models 
15.                    use `self.assertEqual(accuracy4model1, accuracy4model2, "DECREASE IN ACCURACY ... POSSIBLE ATTACK?"  )`
```