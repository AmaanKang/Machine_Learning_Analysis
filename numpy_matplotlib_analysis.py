
'''

This script reads data from a file, summarizes it and then, graphs it by using numpy arrays and matplotlib library
'''
from matplotlib import pyplot as plt
import csv
import numpy as np
import random

## READ IT
file = open('frogs_data.csv')
reader = csv.reader(file,delimiter=",")
first_line = next(reader)
feature_list = first_line[0:15]
index = 0
data_list = []
for line in reader:
    line = list(map(int,line)) #convert all elements of a single record into data type-int
    data_list.insert(index,line)
    index+=1
col_count = len(first_line)
row_count1 = int(index*0.75) #for training data
row_count2 = int(index*0.25) #for testing data
index = index-1
#Choose a random number and picks the record on that random index, then creates training data from that record to the end of
#data set. As the training set needs to be 75% of data, if the chosen set is not enough, then it picks some records from 0 index
#to the random index. The left behind data becomes testing data
random_id = np.random.randint(1,index)
num1 = random_id+(row_count1)
#If the data ranging from random id to the end record won't be enough for set
if num1 >= index:
    num2 = num1-index
    split1 = data_list[random_id:(index+1)]
    split2 = data_list[0:num2]
    split3 = data_list[num2:random_id]
    split2.extend(split1)
    training_list = split2
    testing_list = split3
#If the data ranging from random id to the end record would be bigger than what's needed, then pick records from random id to
#the index upto which it is enough for training data, rest can go as testing data
else:
    split1 = data_list[random_id:num1]
    split2 = data_list[0:random_id]
    split3 = data_list[num1:index]
    training_list = split1
    split2.extend(split3)
    testing_list = split2
training_data = np.array(training_list)
testing_data = np.array(testing_list)
feature_names = np.array(feature_list)
training_label = training_data[:,15:22]
training_data = training_data[:,0:15]
testing_label = testing_data[:,15:22]
testing_data = testing_data[:,0:15]

## SUMMARISE IT
training_data = np.transpose(training_data)
testing_data = np.transpose(testing_data)
loop_count = 1
print('******************** TRAINING DATA ********************')
print('     (min)   (max)   (mean)    (median)')
while loop_count < feature_names.shape[0]:
    minimum = training_data[feature_names == feature_names[loop_count]].min()
    maximum = training_data[feature_names == feature_names[loop_count]].max()
    mean = training_data[feature_names == feature_names[loop_count]].mean()
    median = np.median(training_data[feature_names == feature_names[loop_count]])
    print(feature_names[loop_count],': (',minimum,')(',maximum,')(',round(mean,3),')(',median,')')
    loop_count += 1
loop_count = 1
print()
print('******************** TESTING DATA ********************')
print('     (min)   (max)   (mean)    (median)')
while loop_count < feature_names.shape[0]:
    minimum = testing_data[feature_names == feature_names[loop_count]].min()
    maximum = testing_data[feature_names == feature_names[loop_count]].max()
    mean = testing_data[feature_names == feature_names[loop_count]].mean()
    median = np.median(testing_data[feature_names == feature_names[loop_count]])
    print(feature_names[loop_count],': (',minimum,')(',maximum,')(',round(mean,3),')(',median,')')
    loop_count += 1
    
## GRAPH IT
training_data = np.transpose(training_data)
green_frog_label1 = training_data[training_label[:,0] == 0]
green_frog_label2 = training_data[training_label[:,0] == 1]
plt.figure(1)
plt.title('UR vs FR for green frogs')
plt.xlabel('UR - Use of water reservoirs')
plt.ylabel('FR - The presence of fishing')
plt.scatter(green_frog_label1[:,8],green_frog_label1[:,9],c="red",marker="*",s=380)
plt.scatter(green_frog_label2[:,8],green_frog_label2[:,9],c="green",marker=".",s=100)
plt.figure(2)
plt.title('SUR2 vs SUR3 for green frogs')
plt.xlabel('SUR2 - second dominant land surrounding water reservoir')
plt.ylabel('SUR3 - third dominant land surrounding water reservoir')
plt.scatter(green_frog_label1[:,6],green_frog_label1[:,7],c="red",marker="*",s=120)
plt.scatter(green_frog_label2[:,6],green_frog_label2[:,7],c="green",marker=".")
plt.figure(3)
plt.title('RR vs BR for green frogs')
plt.xlabel('RR - Minimum distance from the water reservoir to roads')
plt.ylabel('BR - Minimum distance from the water reservoir to buildings')
plt.scatter(green_frog_label1[:,11],green_frog_label1[:,12],c="red",marker="*",s=120)
plt.scatter(green_frog_label2[:,11],green_frog_label2[:,12],c="green",marker=".")
plt.show()
unique,counts = np.unique(training_label[:,0],return_counts=True)
plt.title('Frequency of green frog label(0,1)')
plt.xlabel('Unique labels')
plt.ylabel('Frequency of labels')
plt.bar(unique,counts,color='teal')
plt.show()





