#%% Practice 3


# find the smallest number in the given list


example_list = [1,3,15,19,20,35,-199,42,-5,-15,2,40]


mini = 1000000

for numbers in example_list:
    if(numbers < mini):
        mini = numbers
    else:
        continue

print(mini)




#%%
