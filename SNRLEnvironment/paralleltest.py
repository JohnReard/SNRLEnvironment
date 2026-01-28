import random
import jax
import jax.numpy as jnp

devicecount = jax.device_count()
devices = jax.devices()
i=0

def printtest(x):
    test = jnp.array((random.randint(1,10)*100,random.randint(1,10)*100))
    return test
while i < 10:
    testarray = jax.vmap(printtest)(jnp.array([1, 2]))
    print("testarray: ", testarray)
    i+=1


