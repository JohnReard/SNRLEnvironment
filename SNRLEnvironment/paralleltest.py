import random
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

devicecount = jax.device_count()
devices = jax.devices()

seed = 1001
key = jax.random.key(seed)

i=0
print(devices)
def myfunc(arr1, arr2):
    print("arr1 and arr2 = :", arr1.shape, " ", arr2.shape)
    newcoords = jax.vmap(lambda x, y : x + y)(arr1, arr2) #vmap arguments must be jnp types or arrays of jnp types (i.e pytrees with leaves of jnp types)
    print("newcoords are: ", newcoords)
    returnedcoords = jnp.clip(newcoords,min=-600*64, max=600*64)
    print("returned coords are: ", returnedcoords)
    return returnedcoords
a = jnp.array([5, 5])
b = jnp.array([10,10])
#print(myfunc(a,b))

for i in range(10):
    key,subkey = jax.random.split(key)
    output = jax.random.randint(subkey, shape=(2,), minval=-10, maxval=10)
    print("output is: ", output)

#print("4 % -4 =: ", 4 % 3)

#def printtest(x):
#    test = jnp.array((random.randint(1,10)*100,random.randint(1,10)*100))
#    return test
#while i < 10:
#    testarray = jax.vmap(printtest)(jnp.array([1, 2]))
#    print("testarray: ", testarray)
#    i+=1
#    plt.plot(testarray[:,0], testarray[:,1], 'o')
#    plt.show()


