### Need to address the following:
- limit scope by applying threat modeling and empirical analysis
- target what attacks you will be focusing on
- address the novelty of your framework ... for this you need to show that existing test generation frameworks don't work for adversarial samples
- make sure that you are not doing anything shallow i.e. doing deep work

### Questions:
- threat modeling on what?
- Metamorphic testing?

### Threat model:   
Select a specific attack, ex: data poisoning (reason for choosing this attack?);  
Find the applicable classifier algorithms: NB, SVM, DT, MLP, DNN (already has this information in my forensic paper);  
Find numerical equation from existing papers;  
Check for values that satisfies the equation;  


## Relevant Paper

### Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing:

Proposed an approach to detect adversarial samples for deep neural networks at runtime.  
Adversarial sample: Generated around 1000 adversarial samples for each of 5 different attacking algorithm. wrongly labeled samples are also considered as adversarial samples.  
Experiment: Tested for adversarial samples during runtime by quantifying the label change rate (LCR) of a DNN.  
Conclusion: Adversarial samples have significantly higher LCR under model mutation than normal samples.  

### AdvDoor: Adversarial Backdoor Attack of Deep Learning System:

Proposed an approach to generate adversarial backdoor attack to confuse the state-of-the-art backdoor detection methodology. 
Adversarial sample: Used Targeted Universal Adversarial Perturbation (TUAP) to generate adversarial samples. (source code is publicly available)
Experiment: Used a threat model;
Conclusion: 

### Paper to read:
- Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing 
- Attack as Defense: Characterizing Adversarial Examples using Robustness
- DeepCrime: Mutation Testing of Deep Learning Systems Based on Real Faults


### metrics:
I looked at the existing literature to find out the metrics used to evaluate the state-of-art test generation techniques, and I am considering the following metrics to evaluate my proposed approach.
- Multiple types of test coverages (for example, statement coverage, branch coverage and function coverage)
- Completion Time (for example, how much time it takes to generate 100 test cases)

- Mutation score
- Importance driven coverage
- Diversity coverage
- Perturbation measure (for example, minimize perturbation to flip the classification label )

All of these are already being used by existing SE research, therefore acceptable by the research community.

### Proposal 1 pager:
- Get top 25 GitHub repositories based on the dev count
- Apply X to say that Y% of the 25 repos dont do testing [X = State of art test case detection techniques]
- Find the least explored attack A from the forensic paper that are related with source code and apply threat model
- Find the state of art test generation techniques and say that technique Z not be able to generate test cases for attack A becasue

#### Conclusion: 
- DSs rarely have test cases
- Attack A is underexplored, while attack b,c and d is explored
- We propose an auto test case generation technique T that will generate test cases for attack A
- We will compare our technique T with other state of art techniques such as Z

### Findings:
- Existing testing techniwues such as, Fuzz testing [1], DeepXplore [2], TensorFuzz [3], Deeptest [4], Deepgauage [5], DeepMutation [6] focus on the quality of DL models and effective in detecting exceptions such as crashes, inconsistencies, nan and inf bugs.
- [7] proposed feature-guided test generation that transformed the problem of finding adversarial examples into a two-player turn-based stochastic game.
- [10] explored Malformed Input attack
- [14] explored Model Stealing attack
- [8, 12] explored Model Poisoning attack
- [9, 11, 13] explored Data Poisoning attack 
- Label Perturbation attack is under-explored.
- The baseline strategy of label perturbation attack is to perturb the labels for a fraction of the training data to reduce the prediction accuracy of supervised learning systems


#### Threat model: 
In our threat model, we assume that the adversaries can access the training data along with the ground truth of the training data, but cannot access the training process. For example, the adversaries can be those who have access to the training data storage or the providers of the training data. We also assume the adversaries know nothing of the model under attack. Besides, we assume the detector has access to the set of benign examples but knows nothing about how the adversary generates adversarial examples. To implement the attack, the adversary perturbs the labels for a fraction of the training data to reduce the prediction accuracy of supervised learning systems. The model would behave as expected on most inputs, but inputs modified by the adversary can be mislabeled, which is different from the actual ground truth. 

After the label is perturbed and shipped to the developer, the developer, as the victim, will use the modified training data to train the model. During the training of the supervised model, the model will learn the wrong input-output relations. After the training and verification, the model will be deployed into the production environment. The model would behave as expected on most inputs, but inputs modified by the adversary can be mislabeled as something different from the actual ground truth.

#### Limitation of existing techniques:
- [15], [3] generates test cases based on fuzzing, which generates random data as inputs. Fuzzing techniques cannot be used to generate test cases for label perturbation attack becasue in label perturbation attack only label of the training data is changed not the actual training data.
- [16], [17] generates test cases based on mutation where a test suite is evaluated on a set of systematically generated artificial faults (mutants). Any surviving mutant that is not detected by the test suite constitutes a concrete test goal, pointing out possible ways to improve the test suite. Mutation techniques cannot be used to generate test cases for label perturbation attack becasue mutation is generated bymodifying the actual model, but in our threat model adversaries do not have access to the model.
- [18] generates test cases based on gunit testing, which is an executable piece of code that validates a functionality of a class or a method under test performing as designed. Unit test techniques cannot be used to generate test cases for label perturbation attack becasue in label perturbation attack adversaries do not have access to the model.
- DeepXplore [2] is a white-box differential test generation technique that uses domain specific constraints on inputs. This technique requires multiple DNN models trained on the same dataset as cross referencing oracles. DeepXplore cannot be used to generate test cases for label perturbation attack becasue in label perturbation attack adversaries do not have access to the model.
- [19] used existing gradient ascent based test generation technique, which also need access to the model
- [20] used a novel technique called Surprise Adequacy for Deep Learning Systems (SADL), which is based on the behaviour of DL systems with respect to their training data. We measure the surprise of an input as the difference in DL system’s behaviour between the input and the training data, and subsequently develop this as an adequacy criterion: a good test input should be sufficiently but not overtly surprising compared to training data. Since it also need access to the model, SADL cannot be used to generate test cases for label perturbation attack. 



1. J. M. Zhang, M. Harman, L. Ma, and Y. Liu, “Machine learning test- ing: Survey, landscapes and horizons,” IEEE Transactions on Software Engineering, 2020.
2. K. Pei, Y. Cao, J. Yang, and S. Jana, “Deepxplore: Automated whitebox testing of deep learning systems,” in Proceedings of the 26th Symposium
on Operating Systems Principles, 2017, pp. 1–18.
3. A. Odena and I. Goodfellow, “Tensorfuzz: Debugging neural networks with coverage-guided fuzzing,” arXiv preprint arXiv:1807.10875, 2018.
4. Y. Tian, K. Pei, S. Jana, and B. Ray, “Deeptest: Automated testing of deep-neural-network-driven autonomous cars,” in Proceedings of the 40th International Conference on Software Engineering, 2018, pp. 303– 314.
5. L. Ma, F. Juefei, Xu, F. Zhang, J. Sun, M. Xue, B. Li, C. Chen, T. Su, L. Li, Y. Liu et al., “Deepgauge: Multi-granularity testing criteria for deep learning systems,” in Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering, 2018, pp. 120–131.
6.  L. Ma, F. Zhang, J. Sun, M. Xue, B. Li, F. Juefei, Xu, C. Xie, L. Li, Y. Liu, J. Zhao et al., “Deepmutation: Mutation testing of deep learning systems,” in IEEE 29th International Symposium on Software Reliability Engineering. IEEE, 2018, pp. 100–111.
7. M. Wicker, X. Huang, and M. Kwiatkowska, “Feature-guided black-box safety testing of deep neural networks,” in International Conference on Tools and Algorithms for the Construction and Analysis of Systems. Springer, 2018, pp. 408–426.
8. Li, Yuanchun, Jiayi Hua, Haoyu Wang, Chunyang Chen, and Yunxin Liu. "DeepPayload: Black-box Backdoor Attack on Deep Learning Models through Neural Payload Injection." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 263-274. IEEE, 2021.
9. Wang, Zan, Hanmo You, Junjie Chen, Yingyi Zhang, Xuyuan Dong, and Wenbin Zhang. "Prioritizing Test Inputs for Deep Neural Networks via Mutation Analysis." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 397-409. IEEE, 2021.
10. Zhang, Peixin, Jingyi Wang, Jun Sun, Guoliang Dong, Xinyu Wang, Xingen Wang, Jin Song Dong, and Ting Dai. "White-box fairness testing through adversarial sampling." In Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering (ICSE), pp. 949-960. 2020.
11. Guiding Deep Learning System Testing using Surprise Adequacy
12. Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing
13. AdvDoor: Adversarial Backdoor Attack of Deep Learning System
14. Li, Yuanchun, Ziqi Zhang, Bingyan Liu, Ziyue Yang, and Yunxin Liu. "ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection." arXiv preprint arXiv:2106.08890 (2021).
15. Luo, Weisi, Dong Chai, Xiaoyue Run, Jiang Wang, Chunrong Fang, and Zhenyu Chen. "Graph-based Fuzz Testing for Deep Learning Inference Engines." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 288-299. IEEE, 2021.
16. Petrović, Goran, Marko Ivanković, Gordon Fraser, and René Just. "Does mutation testing improve testing practices?." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 910-921. IEEE, 2021.
17. Wang, Zan, Hanmo You, Junjie Chen, Yingyi Zhang, Xuyuan Dong, and Wenbin Zhang. "Prioritizing Test Inputs for Deep Neural Networks via Mutation Analysis." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 397-409. IEEE, 2021.
18. Wang, Song, Nishtha Shrestha, Abarna Kucheri Subburaman, Junjie Wang, Moshi Wei, and Nachiappan Nagappan. "Automatic Unit Test Generation for Machine Learning Libraries: How Far Are We?." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 1548-1560. IEEE, 2021.
19. Dola, Swaroopa, Matthew B. Dwyer, and Mary Lou Soffa. "Distribution-aware testing of neural networks using generative models." In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 226-237. IEEE, 2021.
20. Kim, Jinhan, Robert Feldt, and Shin Yoo. "Guiding deep learning system testing using surprise adequacy." In 2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE), pp. 1039-1049. IEEE, 2019.
