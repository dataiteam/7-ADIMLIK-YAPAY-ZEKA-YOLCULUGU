#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% pandas

# really fast and optimized
# useful for working with data frames
# easier to work with missing data
# reshape data and make it easier to work with
# slicing and indexing
# time series

import pandas as pd



dictionary1 = {"Name"  :["Jeff","Jack","John","Ashley","Jennifer","Jamie"],
              "Age"   :[35,30,40,30,25,40,],
              "Salary":[2000,2500,3000,3500,2000,4000]}


                                                    
dataframe1 = pd.DataFrame(dictionary1)



head1 = dataframe1.head()

tail1 = dataframe1.tail()


#%% pandas basic methods



print(dataframe1.columns)


print(dataframe1.info())

print(dataframe1.dtypes)

print(dataframe1.describe())



#%% indexing and slicing


print(dataframe1["Age"])
print(dataframe1.Age)

dataframe1["new feature"] = [-1,-2,-3,-4,-5,-6]

print(dataframe1["new feature"])
#print(dataframe1.new_feature)


print(dataframe1.loc[:,"Age"])

print(dataframe1.loc[:3,"Age"])

print(dataframe1.loc[0:3,"Name":"Salary"])

print(dataframe1.loc[0:3,["Name","Salary","new feature"]])

print(dataframe1.loc[::-1,:])

print(dataframe1.loc[:,"Name"])


print(dataframe1.iloc[:,0])


#%% filtering


filter1 = dataframe1.Salary > 3000

filteredData = dataframe1[filter1]

filter2 = dataframe1.Age < 30




dataframe1[filter1 & filter2]

#%% list comprehension

import numpy as np

avg_salary = dataframe1.Salary.mean()


avg_salary_np = np.mean(dataframe1.Salary)


dataframe1["salary level"] = ["low" if avg_salary > each else "high" for each in dataframe1.Salary]

#for each in dataframe1.Salary:
#    if(avg_salary > each):
#        print("low")
#    else:
#        print("high")

dataframe1.columns = [each.lower() for each in dataframe1.columns]  





dataframe1.columns = [each.split()[0]+"_"+each.split()[1] if len(each.split())>1 else each for each in dataframe1.columns]  


#%% drop and concatenate


#dataframe1 = dataframe1.drop(["new_feature"],axis = 1)
#dataframe1.drop(["salary"],axis = 1, inplace = True)



data1 = dataframe1.head()
data2 = dataframe1.tail()

# vertical

data_vertical = pd.concat([data1,data2],axis = 0)

#horizontal 

salary_level = dataframe1.salary_level
name = dataframe1.name


data_horizontal = pd.concat([name,salary_level], axis = 1)



#%% transforming data


dataframe1["list_comp"] = [each*2 for each in dataframe1.age] #multiply age by 2
 


#apply()


def multiply(age):
    return age*2

dataframe1["using_apply"] = dataframe1.age.apply(multiply)






