<b> Instruction to run: </b>
=================================================

## Run using Docker:

- Docker image: https://hub.docker.com/repository/docker/fbhuiyan42/testing.  
Run:   
docker pull fbhuiyan42/testing:latest   
docker run -it --privileged <IMAGE_ID>   bash [to find `IMAGE_ID` run `docker images -a`]  
cd resources/Code/detection    
python main.py  
cd ..  
cd generation/identify_algo  
python main.py 
cd ..  

## Run using the project folder from Github:

- Create three folders:
  1. Data: keep the suparvised folder (folder that contains all the supervised projects) here
  2. Code: Keep the detection and generation module here
    - Inside the generation modeule create another folder 'data' and keep the 'IST_MIR.csv' file (This data is used to cteate the label perturbation attack)
  4. Output: 
    - all the csv output will be saved here. 
    - Create a folder named 'attack_unit_test' inside the output folder where all the the test file will be saved. 
  
- Detection Module: 
  1. run detection/main.py:   
    Input: It will take input from the data folder   
    Output: It will create csv output files in the output folder
      - a CSV file with all test names from all ML projects for each dataset 
      - a CSV file with all test names and their assert block elements from all ML projects for each dataset 
      - a CSV file with all project names and their corrsponding flags 
- Generation Module:
  1. run generation/identify_algo/main.py (it will get all the algorithms used in the projects):  
    Input: It will use the csv files created by the detection module.   
    Output: It will create csv output file in the output folder
      - a CSV file with the project name and all the classification algorithms used in the project
  2. run generation/main.py:  
    Input: It will use the csv files created by the generation/identify_algo module.   
    Output: It will create unit test files in the output/attack_unit_test folder
