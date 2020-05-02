#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%

i = 0
while(i<5):
    print(i)
    i=i+1


# a = 5/0





#%%  exceptions

a = 10
b = "2"

list1 = [1,2,3,4]

print(sum(list1))

print(str(a)+b)

k = 15
try:
    c = k/0  #15/0
except ZeroDivisionError:
  print("zero division error")

#%% more errors
  


#index error
list1 = [1,2,3,4]

list1[2]


#module not found

#import numpyy


# file not found

import pandas as pd

pd.read_csv("adasdas")


# type error


"2"+2


#value error
int("asdasd")


try:
    1/1
except:
    print("except")
else:
    print("else")
finally:
    print("done")

