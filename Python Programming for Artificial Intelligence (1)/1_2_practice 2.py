#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%Practice 2

# year 1640 is 17th century
# year 109 is 2nd century
# year 200 is 2nd century (actualy not but for sake of learning)
# year 1500 is 15th century (actualy not but for sake of learning)
# year 1550 is 16th century
# year 2000 is 20th century (actualy not but for sake of learning)

#write function
# input year, output century
#input = year, 1<= year <= 2005 


def year2century(year):
    
    str_year = str(year)
    
    if(len(str_year)<3):  # 5, 1, 15, 50 etc
        return 1
    
    elif(len(str_year) == 3): 
        if(str_year[1:3] == "00"):
            return int(str_year[0])  #100,,200,.....700
        else: 
            return int(str_year[0])+1 #150,250
    else:
        if(str_year[2:4] == "00"):
            return int(str_year[:2])
        else:
            return int(str_year[:2])+1









#%%