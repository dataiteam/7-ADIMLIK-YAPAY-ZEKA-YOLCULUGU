#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% matplotlib library
#Visualizing Data
#line plot, scatter plot, bar plot, subplots, histogram


#%%pandas review

import pandas as pd

df = pd.read_csv("iris.csv")

print(df.columns)


print(df.Species.unique())


print(df.info())



print(df.describe())

setosa = df[df.Species == "Iris-setosa"]
virginica = df[df.Species == "Iris-virginica"]


print(setosa.describe())
print(virginica.describe())

#%% visualizing

import matplotlib.pyplot as plt

df1 = df.drop(["Id"], axis = 1)

df1.plot(grid = True,linestyle = ":",alpha = 1)
plt.show()


setosa = df[df.Species == "Iris-setosa"]
virginica = df[df.Species == "Iris-virginica"]
versicolor = df[df.Species == "Iris-versicolor"]


plt.plot(setosa.Id,setosa.PetalLengthCm, color = "red", label = "setosa - PetalLengthCm")
plt.plot(virginica.Id,virginica.PetalLengthCm, color = "blue", label = "virginica - PetalLengthCm")
plt.plot(versicolor.Id,versicolor.PetalLengthCm, color = "green", label = "versicolor - PetalLengthCm")
plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.legend()
plt.show()


#%% scatter plot




plt.scatter(setosa.Id,setosa.PetalLengthCm, color = "red", label = "setosa - PetalLengthCm")
plt.scatter(virginica.Id,virginica.PetalLengthCm, color = "blue", label = "virginica - PetalLengthCm")
plt.scatter(versicolor.Id,versicolor.PetalLengthCm, color = "green", label = "versicolor - PetalLengthCm")
plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.legend()
plt.show()



#%% histogram


plt.hist(virginica.PetalLengthCm, bins = 50)
plt.xlabel("PetalLengthCm")
plt.ylabel("frequency")
plt.title("histogram")
plt.show



#%% bar plot

import numpy as np

x = np.array([1,2,3,4,5,6])

y = x*3 + 5

plt.bar(x,y)
plt.title("bar plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show



#%% subplots


#df1.plot(grid = True,linestyle = ":",alpha = 1, subplots = True)
#plt.show


plt.subplot(2,1,1)
plt.plot(setosa.Id,setosa.PetalLengthCm, color = "red", label = "setosa - PetalLengthCm")
plt.ylabel("setosa PetalLengthCm" )

plt.subplot(2,1,2)
plt.plot(virginica.Id,virginica.PetalLengthCm, color = "blue", label = "virginica - PetalLengthCm")
plt.ylabel("virginica petallengthcm")
















