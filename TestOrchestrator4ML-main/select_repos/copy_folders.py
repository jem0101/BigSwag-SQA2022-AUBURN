import os
import pandas
import shutil
import csv

source = "../Output/dev_count.csv"
target = "../top_25_github/"

with open(source, newline='') as f:
    reader = csv.reader(f)
    next(reader) # ignoring header
    fileList = list(reader)

count = 0
for i in fileList: 
    dir = i[0]
    folder = dir.split("/")[4]
    try: 
        target1 = target+folder
        shutil.copytree(str(dir), target1)
    except OSError as error: 
        print("") 
    count = count + 1
    if (count == 25):
        break