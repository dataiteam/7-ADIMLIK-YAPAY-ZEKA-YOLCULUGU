# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:12:16 2018

@author: user
"""

# 1) pandas hizli ve etkili for dataframes
# 2) csv ve text dosyalarına acip inceleyip sonuclarimiza bu dosya tiplerine rahat bir sekilde kaydedbilir.
# 3) pandas bizim isimizi kolaylastiriyor for missing data
# 4) reshape yapip datayi daha etkili bir sekilde kullanabiliriz
# 5) slicing indexing kolay
# 6) time series data analizinde cok yardimci
# 7) ayrica herseyden onemlisi hiz pandas hiz acisindan optimize edilmis hizli bir kutuphane

import pandas as pd

dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren"],
              "AGE":[15,16,17,33,45,66],
              "MAAS": [100,150,240,350,110,220]} 

dataFrame1 = pd.DataFrame(dictionary)


head = dataFrame1.head()
tail = dataFrame1.tail()

# %%
# pandas basic method

print(dataFrame1.columns)

print(dataFrame1.info())

print(dataFrame1.dtypes)

print(dataFrame1.describe())  # numeric feature = columns (age,maas)

# %% indexing and slicing


print(dataFrame1["AGE"])
print(dataFrame1.AGE)

dataFrame1["yeni feature"] = [-1,-2,-3,-4,-5,-6]

print(dataFrame1.loc[:, "AGE"])

print(dataFrame1.loc[:3, "AGE"])

print(dataFrame1.loc[:3, "AGE":"NAME"])

print(dataFrame1.loc[:3, ["AGE","NAME"]])

print(dataFrame1.loc[::-1,:])

print(dataFrame1.loc[:,:"NAME"])

print(dataFrame1.loc[:,"NAME"])

print(dataFrame1.iloc[:,2])

# %% filtering

filtre1 = dataFrame1.MAAS > 200

filtrelenmis_data = dataFrame1[filtre1]

filtre2 = dataFrame1.AGE <20

dataFrame1[filtre1 & filtre2]

print(dataFrame1[dataFrame1.AGE > 60])

# %% list comprehension
import numpy as np

ortalama_maas = dataFrame1.MAAS.mean()

# ortalama_maas_np = np.mean(dataFrame1.MAAS)


dataFrame1["maas_seviyesi"] = ["dusuk" if ortalama_maas > each else "yuksek" for each in dataFrame1.MAAS]

#for each in dataFrame1.MAAS:
#    if(ortalama_maas > each):
#        print("dusuk")
#    else:
#        print("yukse")
        

dataFrame1.columns


dataFrame1.columns = [ each.lower() for each in dataFrame1.columns] 

dataFrame1.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in dataFrame1.columns]

# %% drop and concatenating

dataFrame1.drop(["yeni_feature"],axis=1,inplace = True)

# dataFrame1 = dataFrame1.drop(["yeni_feature"],axis=1)

data1 = dataFrame1.head()
data2 = dataFrame1.tail()

# vertical
data_concat = pd.concat([data1,data2],axis=0)


# horizontal

maas = dataFrame1.maas
age = dataFrame1.age

data_h_concat = pd.concat([maas,age],axis=1)


# %% transforming data

dataFrame1["list_comp"] = [ each*2 for each in dataFrame1.age]

# apply()

def multiply(age):
    return age*2
    
dataFrame1["apply_metodu"] = dataFrame1.age.apply(multiply)




















