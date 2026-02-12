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
arr1 = jnp.array([1, 2])
arr2 = jnp.array([3, 4])
a = jnp.array([5, 5])
b = jnp.array([10,10])
#print(myfunc(a,b))
addedcoords = jax.vmap(lambda x, y : x + y)(arr1, arr2)
print(addedcoords)
addedcoords = jnp.clip(addedcoords,min=0, max=2)
print(addedcoords)
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


