'''
This part of the code tests the accuracy of k-nearest algorithm over different versions of the data
'''
import csv
import numpy as np
import random
import math
from matplotlib import pyplot as pl
## READ IT
file = open('frogs.csv')
reader = csv.reader(file,delimiter=",")
first_line = next(reader)
feature_list = first_line[0:14]
index = 0
data_list = []
for line in reader:
    line = list(map(int,line)) #convert all elements of a single record into data type-int
    data_list.insert(index,line)
    index+=1
k=int(math.sqrt(index))
k_norm = k
col_count = len(first_line)
row_count1 = int(index*0.75) #for training data
row_count2 = int(index*0.25) #for testing data
index = index-1
run = 0
euc_dis_k1 = 0
euc_dis_k2 = 0
euc_dis_k3 = 0
euc_dis_norm_k1 = 0
euc_dis_norm_k2 = 0
euc_dis_norm_k3 = 0
manhat_dis_k1 = 0
manhat_dis_k2 = 0
manhat_dis_k3 = 0
def find_label(run,k,distance,label_,sort_index,new_label,k1,k2,k3):
    """
    This function loops over the three different values of k and finds out a new class label for the new_data in every loop
    The parameters taken are the: run number, distance array, training label array, sorted indexes of distance array,new label
    taken from the testing label,distance with first k value, distance with second k value, distance with third k value
    """
    if(k%2 == 0):
        k-=1
    if k-2 > 0:
        loop = k-2
    else:
        loop = 1
    k = k+4
    loop_time = 0
    while loop < k:
        if run > 44:
            print('k: ',loop)
        distance_sliced = distance[sort_index[:loop]]
        label_sliced = label_[sort_index[:loop]]
        loop += 2
        label_dict = {}
        for label in label_sliced:
            if label not in label_dict:
                label_dict[label] = 1
            else:
                label_dict[label] += 1
        label_array = np.array(list(label_dict.items()))
        label_array1 = label_array[:,0]
        label_array2 = label_array[:,1]
        new_data_label = label_array1[label_array2.argmax()]
        if new_data_label == new_label:
            score = 1
        else:
            score = 0
        if loop_time == 0:
            k1 += score
        if loop_time == 1:
            k2 += score
        if loop_time == 2:
            k3 += score
        loop_time += 1
    return k1,k2,k3
while run < 50:
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
    training_label = training_data[:,14]
    training_data = training_data[:,0:14]
    testing_label = testing_data[:,14]
    testing_data = testing_data[:,0:14]
    new_data = testing_data[0,:] 
    new_label = testing_label[0]
    
    #Euclidean Distance
    euc_distance = np.sqrt(((np.square(training_data-new_data)).sum(axis=1)))
    euc_distance = np.nan_to_num(euc_distance)
    sort_index = euc_distance.argsort()
    k = k_norm
    euc_dis_k1,euc_dis_k2,euc_dis_k3 = find_label(run,k,euc_distance,training_label,sort_index,new_label,euc_dis_k1,euc_dis_k2,euc_dis_k3)
    if run > 44:
        print('euc_dis_k1: ',euc_dis_k1/run)
        print('euc_dis_k2: ',euc_dis_k2/run)
        print('euc_dis_k3: ',euc_dis_k3/run)
    
    #Normalise the data and find Euclidean distance
    training_data_norm = training_data.astype(float)
    new_data_norm = new_data.astype(float)
    k = k_norm
    column = 0
    while column < 14:
        vmin = training_data_norm[:,column].min()
        r = training_data_norm[:,column].max()
        row = 0
        for element in training_data_norm[:,column]:
            v = element
            normalized = (v-vmin)/r
            training_data_norm[row,column] = normalized
            row+=1
        new_normalized = (new_data_norm[column]-vmin)/r
        new_data_norm[column] = new_normalized
        column += 1
    euc_distance = np.sqrt(((np.square(training_data_norm-new_data_norm)).sum(axis=1)))
    euc_distance = np.nan_to_num(euc_distance)
    sort_index = euc_distance.argsort()
    count = 0
    euc_dis_norm_k1,euc_dis_norm_k2,euc_dis_norm_k3 = find_label(run,k,euc_distance,training_label,sort_index,new_label,euc_dis_norm_k1,euc_dis_norm_k2,euc_dis_norm_k3)
    if run > 44:
        print('euc_dis_norm_k1: ',euc_dis_norm_k1/run)
        print('euc_dis_norm_k2: ',euc_dis_norm_k2/run)
        print('euc_dis_norm_k3: ',euc_dis_norm_k3/run)
    
    #Manhattan Distance
    manhat_dis = np.ones(training_data.shape[0])
    manhat_index = 0
    for element in training_data:
        manhat_dis[manhat_index] = np.sum(abs(element-new_data))
        manhat_index += 1
    manhat_dis = np.nan_to_num(manhat_dis)
    sort_index = manhat_dis.argsort()
    count = 0
    k = k_norm
    manhat_dis_k1,manhat_dis_k2,manhat_dis_k3 = find_label(run,k,manhat_dis,training_label,sort_index,new_label,manhat_dis_k1,manhat_dis_k2,manhat_dis_k3)
    if run > 44:
        print('manhat_dis_k1: ',manhat_dis_k1/run)
        print('manhat_dis_k2: ',manhat_dis_k2/run)
        print('manhat_dis_k3: ',manhat_dis_k3/run)
        
    run += 1
euc_dis_k1_avg = euc_dis_k1/50
euc_dis_k2_avg = euc_dis_k2/50
euc_dis_k3_avg = euc_dis_k3/50
euc_norm_k1_avg = euc_dis_norm_k1/50
euc_norm_k2_avg = euc_dis_norm_k2/50
euc_norm_k3_avg = euc_dis_norm_k3/50
manhat_dis_k1_avg = manhat_dis_k1/50
manhat_dis_k2_avg = manhat_dis_k2/50
manhat_dis_k3_avg = manhat_dis_k3/50
print('euc_dis_k1_avg: ',euc_dis_k1_avg)
print('euc_dis_k2_avg: ',euc_dis_k2_avg)
print('euc_dis_k3_avg: ',euc_dis_k3_avg)
print('euc_norm_k1_avg: ',euc_norm_k1_avg)
print('euc_norm_k2_avg: ',euc_norm_k2_avg)
print('euc_norm_k3_avg: ',euc_norm_k3_avg)
print('manhat_dis_k1_avg: ',manhat_dis_k1_avg)
print('manhat_dis_k2_avg: ',manhat_dis_k2_avg)
print('manhat_dis_k3_avg: ',manhat_dis_k3_avg)

#Draw a bar plot for average accuracies
accuracy_array = np.array([euc_dis_k1_avg,euc_dis_k2_avg,euc_dis_k3_avg,
                          euc_norm_k1_avg,euc_norm_k2_avg,euc_norm_k3_avg,
                          manhat_dis_k1_avg,manhat_dis_k2_avg,manhat_dis_k3_avg])
param_array = np.array(['1_k1','1_k2','1_k3','2_k1','2_k2','2_k3','3_k1','3_k2','3_k3'])
pl.title('Average Accuracy for k-nearest distance metrics')
pl.xlabel('1=Euclidean Distance, 2=Euclidean Distance Norm, 3=Manhattan Distance')
pl.ylabel('Accuracy Score')
pl.bar(param_array,accuracy_array)
pl.show()

