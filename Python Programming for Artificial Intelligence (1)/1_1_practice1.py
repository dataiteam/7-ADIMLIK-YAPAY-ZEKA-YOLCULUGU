# %%
# Practice 1

#int age
#strings name
#functions
#print(),len(),type(),float()
#*args
#default parameter

age = 15
name = "John"
lastName= "Doe"


def practice_function(age,name,*args,shoeSize = 35):
    print("name:",name,"age:",age,"shoe size:",shoeSize)
    print(type(name))
    print(float(age))
    
    output = args[0]*age
    
    return output

result = practice_function(age,name,lastName)

print("args[0]*age:",result)

























# %%