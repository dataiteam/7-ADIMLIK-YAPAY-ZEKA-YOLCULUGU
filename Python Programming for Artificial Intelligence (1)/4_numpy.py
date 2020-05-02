
#%% numpy basics


import numpy as np


array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # 1 by 15 matrix/vector

print(array.shape)

a = array.reshape(3,5)
print("shape:",a.shape)
print("dimension:",a.ndim)
print("data type:",a.dtype.name)
print("size:",a.size)
print("type:",type(a))



b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])


zeros = np.zeros((3,4))

zeros[0,0] = 1
print(zeros)

empty1 = np.empty((2,3))

d = np.arange(10,51,5)
print(d)

f = np.linspace(10,50,20)
print(f)

#%% basic operations
import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)


print(np.sin(a))

print(a==2)


#element wise product

print(a*b)

#matrix multiplication

c = np.array([[1,2,3],[4,5,6]])
d = np.array([[1,2,3],[4,5,6]])

c.dot(d.T)


exponentialarray = np.exp(c)
print(exponentialarray)



randomMatrix = np.random.random((5,5))


print(randomMatrix.sum())
print(randomMatrix.max())
print(randomMatrix.min())


print(randomMatrix.sum(axis=0))
print(randomMatrix.sum(axis=1))

print(np.sqrt(randomMatrix))
print(np.square(randomMatrix)) #randomMatrix**2


print(np.add(a,a))


#%% indexing and slicing
import numpy as np

array = np.array([1,2,3,4,5,6,7])



print(array[0])


print(array[0:4])

reverse_array = array[::-1]

print(reverse_array)

array1 = np.array([[1,2,3,4],[5,6,7,8,]])

print(array1)

print(array1[1,2])   #row,column

print(array1[:,1])# :,1 all rows and column 1


print(array1[1,:])
print(array1[1,1:3])
print(array1[-1,:])


#%% shape manipulation
import numpy as np


array = np.array([[1,2,3],[4,5,6],[7,8,9]])


#flatten
arrayFlat = array.ravel()


array2 = arrayFlat.reshape(3,3)


arrayT = array2.T


arrayNew = np.array([[1,2],[3,4],[5,6]])

#%% stacking arrays


array1 = np.array([[0,0],[0,0]])
array2 = np.array([[1,1],[1,1]])

#vertical stacking
arrayVertical = np.vstack((array1,array2))

#horizontal stacking
arrayHorizontal = np.hstack((array1,array2))


#%% convert and copy
import numpy as np

list1 = [1,2,3,4]

array1 = np.array(list1)

list2 = list(array1)


a = np.array([1,2,3])

b = a
b[0] = 5
c = a


d = np.array([1,2,3])

e = d.copy()

f = d.copy()

