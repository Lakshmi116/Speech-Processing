# Logic for filling the annotations


import os
import pandas as pd




"""
Verification of annotations filling
"""
# data_dir = "./home/gdata/narayana/Lakshmi/"


# file = os.path.join(data_dir, "test_annotations.csv") # change test to train to check for duplicates in the train annotaions

# df = pd.read_csv(file)

# print(df)

# print(df.drop_duplicates())

"""
Code to create annotations file
"""
# # Code <=> Word Mapping
f = open("./home/gdata/narayana/Lakshmi/speaker_wordlist.csv", "r")

fl = f.readlines()[1:53]
mf = {}

for line in fl:
    label, code = tuple(line.strip().split(','))
    mf[code] = label

print(f"size of vocabulary: {len(mf)}")

"""
Train - Test - Split - Logic
----------------------------
A random variable with uniform distribution is assigned to each wav file.
With this random variable as the probability each wav file goes into training set
"""
import random as rd

rd.seed(6706)




data_dir = "./home/gdata/narayana/Lakshmi/Data"
SPEAKER = "M05"
TAG = "start51"



train_p = 0.7

train_fp = os.path.join("./home/gdata/narayana/Lakshmi/Data", f"{SPEAKER}_train_annotations_{TAG}.csv" )
test_fp = os.path.join("./home/gdata/narayana/Lakshmi/Data", f"{SPEAKER}_test_annotations_{TAG}.csv")
train_f = open(train_fp, "w")
test_f = open(test_fp, "w")

train_f.write(f"FILE, LABEL\n")
test_f.write(f"FILE, LABEL\n")

speaker_dir = os.path.join(data_dir, SPEAKER)
codes = os.listdir(speaker_dir)

# print("\n","Codes:")
# print(codes)
for code in codes:
    if code not in mf:
        continue
    code_dir = os.path.join(speaker_dir, code)
    files = os.listdir(code_dir)

    # print("\n","Files:")
    # print(files)

    train_flags = [rd.random()<train_p for _ in range(len(files))]

    for idx, file in enumerate(files):
        file_path = os.path.join(code_dir, file) 
        if code in mf:
            label = mf[code]
            if train_flags[idx]:
                train_f.write(f"{file_path}, {label}\n")
            else:
                test_f.write(f"{file_path}, {label}\n")

train_f.close()
test_f.close()
        

        
            

