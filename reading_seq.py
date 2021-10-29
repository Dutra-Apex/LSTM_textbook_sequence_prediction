# Because textbook is written so that it is read somewhat sequentially, we can assume sections 1, 2, 3 are followed by 4. 
# Hence, we can create a good/reasonable sequences by taking sequential sections from each chapter. 
# Bad sequences would be something chosen randomly.

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Loads data
if 1:
    with open('OS_all_M_T_title.p','rb') as f:
        data = pickle.load(f)
    M_OS = data[0]
    T_OS = data[1]
    OS_titles = data[2]
    corpus_category = data[3]
    assert len(corpus_category)==len(OS_titles)
    #print("\n".join(OS_titles[:10]))
    
    corpus_type = ['PHYS','CHEM','BIOL']
    for i in range(len(OS_titles)):
        OS_titles[i] = corpus_type[int(corpus_category[i])] + ' ' + OS_titles[i]

# Make a list of good sequences, using section numbers.
# Assume that the first three sections are good sequences.
good_seq = list()
for i, title in enumerate(OS_titles):
    # For each chapter, take the first few sections as good sequences.
    split_token = title.split('.')
    chapter_num = split_token[0]
    section_num = int(split_token[1][0])
    title_str = split_token[1][2:]
    if section_num==1:
        good_seq.append((i,i+1,i+2))

# Set up a bad seq.
# Assume sections that are far away are likely be bad sequences.
num_seq = len(good_seq)
bad_seq = list()
min_diff = len(good_seq)*0.25
while len(bad_seq) < num_seq:
    permlist = np.random.permutation(range(len(OS_titles)))
    seq = permlist[0:3]
    if (abs(seq[0]-seq[1])>min_diff) and (abs(seq[1]-seq[2])>min_diff) and (abs(seq[2]-seq[0])>min_diff):
        bad_seq.append(seq)
#print(bad_seq)

print("Length of bad and good sequences")
print(len(bad_seq))
print(len(good_seq))

