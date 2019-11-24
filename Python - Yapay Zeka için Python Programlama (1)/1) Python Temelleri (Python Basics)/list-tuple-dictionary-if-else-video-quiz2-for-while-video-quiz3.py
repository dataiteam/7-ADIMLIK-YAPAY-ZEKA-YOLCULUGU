# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:46:00 2018

@author: user
"""

# %%
# list

liste = [1,2,3,4,5,6]
type(liste)

liste_str = ["ptesi","sali","cars"]
type(liste_str)

value = liste[1]
print(value)

last_value = liste[-1]

liste_divide = liste[0:3]

liste.append(7)
liste.remove(7)
liste.reverse()
liste2 = [1,5,4,3,6,7,2]
liste2.sort()

string_int_liste = [1,2,3,"aa","bb"]

# %% tuple

t = (1,2,3,3,4,5,6)

t.count(3)
t.index(3)

# %% dictionary

dictionary = {"ali":32,"veli":45,"ayse":13}


# ali ,veli ,ayse = keys
# 32,45,13 = values

def deneme():
    dictionary = {"ali":32,"veli":45,"ayse":13}
    return dictionary

dic = deneme()

# %% conditionals
# if else statement

var1 = 10
var2 = 20

if(var1 > var2):
    print("var1 buyuktur var2")
elif(var1 == var2):
    print("var and var2 esitler")
else:
    print("var1 kucuktur var2")


liste = [1,2,3,4,5]

value = 3
if value in liste:
    print("evet {} degeri listenin icinde".format(value))
else:
    print("hayir")


keys = dictionary.keys()

if "can" in keys:
    print("evet")
else:
    print("hayir")


# %% 

# 1640. yil == 17. yuzyil
# 109. yil == 2. yuzyil
# 2000. yil = 20. yuzyil
    
# metod yazin.
    # input integer yillar
    # output integer yuzyil

# input = year  >> 1 <= year <= 2005
    
def year2Century(year):
    """
    year to century
    """
    str_year = str(year)
    
    if(len(str_year)<3):
        return 1
    elif(len(str_year) == 3):
        if(str_year[1:3] == "00"):  # 100 ,200 300, 400 ... 900
            return int(str_year[0])
        else:                       # 190, 250, 450
            return int(str_year[0])+1
    else:                           # 1750, 1700, 1805
        if(str_year[2:4]=="00"):    # 1700, 1900, 1100
            return int(str_year[:2])
        else:                       # 1705, 1645, 1258
            return int(str_year[:2])+1

# %% loops
# for loop
    
for each in range(1,11):
    print(each)
    
for each in "ankara ist":
    print(each)
    
for each in "ankara ist".split(): 
    print(each)
    
liste = [1,4,5,6,8,3,3,4,67]
 
summation = sum(liste)    

count = 0
for each in liste:
    count = count + each
    print(count)
    
# while loop
    
i = 0
while(i <4):
    print(i)
    i = i + 1


sinir = len(liste)   
each = 0
count = 0
while(each < sinir):
    count = count + liste[each]
    each = each + 1 

 
# %%


# liste verecegim
#sizden bu listenin icindeki en kucuk sayiyi bulmanizi istiyorum

liste = [1,2,3,4,5,6,4,23,67,21,-500,23,451,67]

mini = 100000
for each in liste:
    if(each < mini):
        mini = each
    else:
        continue
print(mini)











