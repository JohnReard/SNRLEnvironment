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
#Something to test:
# if you add a 2 deep nested array to another and vmap x+y does that vectorised the 1 deep addition as well?


