
"""

This script focuses on the use of python lists,sets and arrays to accomplish the task of getting user input and storing it in 
data sets. Then, it uses those data sets to get the desired output.
"""
import numpy as np
import time 
player_count = int(input('How many players? '))
interval_count = int(input('How many intervals? '))
name_set = set()
print('Enter ',player_count,' player names..')
count = 0 #a variable to keep track of the looping so that user can enter only the mentioned number of names
while count < player_count:
    player_name = input()
    player_name = player_name.strip().capitalize()
    if len(player_name) < 1:
        print('Player name cannot be empty')
    else:
        if player_name not in name_set:
            name_set.add(player_name)
            count = count+1
        else:
            print('Name already entered')
            
name_array = np.array(list(name_set))
print()
def compute_time(time2,time1):
    """
    This function calculates the time difference between two times.
    time2 and time1 can be any time value
    """
    return round((time2-time1),3)

loop = 0 #a variable that makes the while loop run as many times as the count of players

nd_array = np.ones((player_count,interval_count)) #multi dimensional array for the interval time of players

while loop < player_count:
    interval_array = np.ones(interval_count)# 1 dimensional array to store the interval times for one player
    print(name_array[loop],"'s turn. Press enter ",(interval_count+1)," times quickly." )
    enter_count = 0 #Keeps the track of as how many times the enter button has been pressed
    while enter_count < (interval_count+1):
        input()
        if enter_count > 0: #When enter is pressed first time, there won't be any time difference to calculate, hence the calculation
                            #will be done after second press of the enter button.
            past_time = current_time
            current_time = time.time()
            time_interval = compute_time(current_time,past_time)
            interval_array[enter_count-1] = time_interval
        else:
            current_time = time.time()
            
        enter_count = enter_count+1
    nd_array[loop] = interval_array   
    loop = loop+1
    
##Now, the array will be sorted on the basis of names array ascending order
sorted_index = name_array.argsort()
name_array = name_array[sorted_index]
nd_array = nd_array[sorted_index]

##Calculates mean of each row in nd_array and makes an array of the means
mean_array = nd_array.mean(axis=1)
mean_array = np.around(mean_array,3)

##Calculates fastest mean time from the array of means
fast_mean_index = mean_array.argmin()
fast_name = name_array[fast_mean_index]
fast_mean = mean_array[fast_mean_index]

##Calculates slowest mean time from the array of means
slow_mean_index = mean_array.argmax()
slow_name = name_array[slow_mean_index]
slow_mean = mean_array[slow_mean_index]

##Calculates fastest single time from nd_array
fast_single_index = nd_array.argmin()
fast_single_name = name_array[fast_single_index//interval_count]#the index of the minimum time divided by the interval count gives the index of 
                                                                #the row that gives the index of the player from names array
fast_single_time = nd_array.min()

##Calculates slowest single time from nd_array
slow_single_index = nd_array.argmax()
slow_single_name = name_array[slow_single_index//interval_count]
slow_single_time = nd_array.max()

print('Names ',name_array)
print('Mean times: ',mean_array)
print('Fastest Average Time: ',fast_mean,' by ',fast_name)
print('Slowest Average Time: ',slow_mean,' by ',slow_name)
print('Fastest Single Time: ',fast_single_time,' by ',fast_single_name)
print('Slowest Single Time: ',slow_single_time,' by ',slow_single_name)
print()
print(name_array)
print(nd_array)

