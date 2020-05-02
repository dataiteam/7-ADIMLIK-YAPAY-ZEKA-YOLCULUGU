#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%%

class Employee:
    
    raiseAmount = 1.8
    counter = 0
    
    def __init__(self,name,lastName,salary): #constructor
        
            
        self.name = name
        self.lastName = lastName
        self.salary = salary
        self.email = name+lastName+"@asd.com"
        
        Employee.counter = Employee.counter+1
        
    def giveNameLastname(self):
        return self.name+" "+self.lastName
   
    def raiseSalary(self):
        self.salary = self.salary*self.raiseAmount
    

employee1 = Employee("jack","daniel",2000)

print("initial salary: ",employee1.salary)


employee1.raiseSalary()

print("new salary: ",employee1.salary)

employee2 = Employee("John","Doe",3000)
employee3 = Employee("John","Asd",3500)
employee4 = Employee("Jane","Sdf",3200)
print(Employee.counter)


#%% oop example


list1 = [employee1,employee2,employee3,employee4]


max_salary = -1
index = -1

for each in list1:
    if(each.salary > max_salary):
        max_salary = each.salary
        index = each

print(max_salary)
print(index.giveNameLastname())









